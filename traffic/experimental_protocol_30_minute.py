# -*- coding: utf-8 -*-
"""
Protocollo Sperimentale (Slide 30 Minuti)
"""

import os
import glob
import time
import datetime
import warnings
import logging
import re

import matplotlib
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import pandas as pd
import numpy as np

from prophet import Prophet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm

from neuralprophet import NeuralProphet

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

import torch

# =========================================================================
# 0. CONFIGURAZIONI GLOBALI E PERCORSI
# =========================================================================
warnings.filterwarnings('ignore')
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
tf.get_logger().setLevel('ERROR')
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_DATI = os.path.join(BASE_DIR, 'dataset_ricostruito')
FILE_NAME = 'traffico_r_completo.csv'

PATH_RISULTATI = os.path.join(BASE_DIR, 'Risultati_Esperimenti')
# Cartella dedicata specificamente all'esperimento con scorrimento a 30 minuti
RESULTS_DIR = os.path.join(PATH_RISULTATI, 'Risultati_Multi_Modello_30Minuti')

# Definizione e creazione delle cartelle strutturate per i grafici
PLOTS_DIR = os.path.join(RESULTS_DIR, 'grafici')
DIR_CURVE_SINGOLE = os.path.join(PLOTS_DIR, 'curve_singole')
DIR_DECADIMENTO_SINGOLO = os.path.join(PLOTS_DIR, 'decadimento_singolo')

# --- CARTELLE PER I GRAFICI CONTINUI E MENSILI ---
DIR_INTEROPERIODO_SINGOLO = os.path.join(PLOTS_DIR, 'curve_interoperiodo_singole')
DIR_METRICA_MENSILE_SINGOLO = os.path.join(PLOTS_DIR, 'metrica_mensile_singola')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(DIR_CURVE_SINGOLE, exist_ok=True)
os.makedirs(DIR_DECADIMENTO_SINGOLO, exist_ok=True)
os.makedirs(DIR_INTEROPERIODO_SINGOLO, exist_ok=True)
os.makedirs(DIR_METRICA_MENSILE_SINGOLO, exist_ok=True)

print(f"Directory risultati: {RESULTS_DIR}")
print(f"Directory grafici globali: {PLOTS_DIR}")
print(f"Directory grafici singole stazioni: {DIR_CURVE_SINGOLE} e {DIR_DECADIMENTO_SINGOLO}")

# =========================================================================
#  FIX DEFINITIVO PER PYTORCH 2.6+
# =========================================================================
_original_torch_load = torch.load

# Funzione di patch per risolvere i problemi di compatibilità con il caricamento in PyTorch 2.6+
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

# =========================================================================
# 1. FUNZIONI DI CARICAMENTO E PRE-PROCESSING
# =========================================================================

# Carica il dataset CSV completo in memoria e converte la colonna del tempo in formato datetime
def carica_dataset_completo(nome_file):
    print("\nLettura del dataset completo in memoria...")
    percorso_completo = os.path.join(PATH_DATI, nome_file)
    if not os.path.exists(percorso_completo):
        raise FileNotFoundError(f"File {nome_file} non trovato nel percorso {PATH_DATI}")
    
    df = pd.read_csv(percorso_completo)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# Filtra i dati per una specifica stazione, esegue il ricampionamento a 30 minuti e crea nuove feature temporali
def pre_processing_stazione(df_completo, id_stazione):
    df_stazione = df_completo[df_completo['id_stazione'] == id_stazione].copy()
    df_stazione.set_index('timestamp', inplace=True)
    df_stazione.sort_index(inplace=True)
    
    df_resampled = df_stazione[['differenza']].resample('30min').mean()
    df_resampled['differenza'] = df_resampled['differenza'].ffill()
    
    df_resampled['hour'] = df_resampled.index.hour
    df_resampled['dayofweek'] = df_resampled.index.dayofweek
    df_resampled['is_weekend'] = (df_resampled.index.dayofweek >= 5).astype(int)
    
    df_resampled.dropna(inplace=True)
    return df_resampled

# =========================================================================
# 2. SLIDING WINDOW DINAMICA
# =========================================================================
from datetime import timedelta

# Generatore per creare finestre temporali di addestramento e test scorrevoli, simulando un flusso dati aggiornato ogni giorno
def sliding_window_generator(df, train_days='auto', test_days=2, step_minutes=30, max_folds=1440, test_on_end=True):
    start_date = df.index.min()
    end_date = df.index.max()
    fold_counter = 1

    if test_on_end:
        minuti_scorrimento = (max_folds - 1) * step_minutes
        inizio_primo_test = end_date - timedelta(minutes=minuti_scorrimento) - timedelta(days=test_days)
        
        if train_days == 'auto':
            train_days = (inizio_primo_test - start_date).days
            
        current_start = inizio_primo_test - timedelta(days=train_days)
    else:
        current_start = start_date

    while True:
        if max_folds is not None and fold_counter > max_folds: break
        
        train_end = current_start + timedelta(days=train_days)
        test_end = train_end + timedelta(days=test_days)
        
        if test_end > end_date: break

        train_data = df[(df.index >= current_start) & (df.index < train_end)]
        test_data = df[(df.index >= train_end) & (df.index < test_end)]

        yield fold_counter, train_data, test_data
        
        current_start += timedelta(minutes=step_minutes)
        fold_counter += 1

# =====================================================================
# 3. WRAPPERS DEI MODELLI
# =====================================================================

class RandomForestWrapper:
    def __init__(self):
        self.model_name = "RandomForest"
        self.params = {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}
        self.model = RandomForestRegressor(**self.params, n_jobs=-1)
        self.features = ['hour', 'dayofweek', 'is_weekend']

    def train(self, train_df):
        self.model.fit(train_df[self.features].values, train_df['differenza'].values)

    def predict(self, train_df, test_df):
        return np.maximum(0, self.model.predict(test_df[self.features].values))

class NeuralProphetWrapper:
    def __init__(self, epochs=50):
        self.model_name = "NeuralProphet"
        self.epochs = epochs
        self.params = {'epochs': epochs, 'yearly_seasonality': False, 'weekly_seasonality': True, 'daily_seasonality': True}

    def train(self, train_df):
        df_np = pd.DataFrame({'ds': train_df.index, 'y': train_df['differenza'].values})
        self.model = NeuralProphet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=True, epochs=self.epochs)
        self.model.fit(df_np, freq="30min", progress=None, minimal=True)

    def predict(self, train_df, test_df):
        df_np = pd.DataFrame({'ds': train_df.index, 'y': train_df['differenza'].values})
        future = self.model.make_future_dataframe(df_np, periods=len(test_df))
        return np.maximum(0, self.model.predict(future)['yhat1'].values)

class StackedLSTMWrapper:
    def __init__(self, look_back=48, epochs=20):
        self.model_name = "Stacked_LSTM"
        self.look_back = look_back
        self.epochs = epochs
        self.params = {'look_back': look_back, 'epochs': epochs, 'batch_size': 64, 'layers': [64, 32, 16, 1]}
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def train(self, train_df):
        tf.keras.backend.clear_session()
        
        scaled = self.scaler.fit_transform(train_df[['differenza']].values)
        X, Y = [], []
        for i in range(len(scaled) - self.look_back):
            X.append(scaled[i:(i + self.look_back), 0])
            Y.append(scaled[i + self.look_back, 0])

        X, Y = np.array(X), np.array(Y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        self.model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.look_back, 1)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        self.model.compile(loss='mse', optimizer='adam')
        self.model.fit(X, Y, epochs=self.epochs, batch_size=64, verbose=0)

    def predict(self, train_df, test_df):
        preds = []
        batch = train_df[['differenza']].values[-self.look_back:]
        batch_scaled = self.scaler.transform(batch).reshape(1, self.look_back, 1)

        for _ in range(len(test_df)):
            p_scaled = self.model.predict(batch_scaled, verbose=0)[0]
            preds.append(max(0, self.scaler.inverse_transform([p_scaled])[0, 0]))
            batch_scaled = np.append(batch_scaled[:, 1:, :], [[[p_scaled[0]]]], axis=1)
        return np.array(preds)

class ProphetWrapper:
    def __init__(self):
        self.model_name = "Prophet"
        self.params = {'yearly_seasonality': False, 'weekly_seasonality': True, 'daily_seasonality': True}

    def train(self, train_df):
        df_p = pd.DataFrame({'ds': train_df.index, 'y': train_df['differenza'].values})
        self.model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=True)
        self.model.fit(df_p)

    def predict(self, train_df, test_df):
        df_future = pd.DataFrame({'ds': test_df.index})
        forecast = self.model.predict(df_future)
        return np.maximum(0, forecast['yhat'].values)

class SVMWrapper:
    def __init__(self, kernel='rbf', C=1.0, epsilon=0.1):
        self.model_name = "SVM"
        self.params = {'kernel': kernel, 'C': C, 'epsilon': epsilon}
        self.scaler = StandardScaler()
        self.model = SVR(kernel=kernel, C=C, epsilon=epsilon)
        self.features = ['hour', 'dayofweek', 'is_weekend']

    def train(self, train_df):
        X = train_df[self.features].values
        y = train_df['differenza'].values
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, train_df, test_df):
        X_test = test_df[self.features].values
        X_test_scaled = self.scaler.transform(X_test)
        return np.maximum(0, self.model.predict(X_test_scaled))

class SARIMAWrapper:
    def __init__(self, order, seasonal_order):
        self.model_name = "SARIMA"
        self.order = order
        self.seasonal_order = seasonal_order
        self.params = {'order': order, 'seasonal_order': seasonal_order}

    def train(self, train_df):
        self.model = sm.tsa.SARIMAX(train_df['differenza'].values,
                                    order=self.order,
                                    seasonal_order=self.seasonal_order,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
        self.fitted = self.model.fit(disp=False)

    def predict(self, train_df, test_df):
        return np.maximum(0, self.fitted.get_forecast(steps=len(test_df)).predicted_mean)

# =====================================================================
# 4. EVALUATION LOOP
# =====================================================================

# Ciclo principale di valutazione: addestra il modello, genera predizioni, calcola metriche di errore e le salva su file
def run_evaluation_loop_modulare(generatore, model_wrapper, path_salvataggio):
    print(f"\nAvvio Loop: {model_wrapper.model_name}")
    tutti_i_risultati = []

    for fold, train_df, test_df in generatore:
        if train_df.empty or test_df.empty: continue
        y_test = test_df['differenza'].values

        try:
            start_train = time.perf_counter()
            model_wrapper.train(train_df)
            train_time = time.perf_counter() - start_train

            start_pred = time.perf_counter()
            y_pred = model_wrapper.predict(train_df, test_df)
            pred_time = time.perf_counter() - start_pred
        except Exception as e:
            print(f"    [!] ERRORE: Il modello {model_wrapper.model_name} ha fallito l'elaborazione della finestra {fold}. Motivo: {e}")
            continue

        mape_array = np.where(y_test > 0, (np.abs(y_test - y_pred) / y_test) * 100, 0)
        
        # --- CALCOLO WAPE ---
        somma_reali = np.sum(y_test)
        wape_val = (np.sum(np.abs(y_test - y_pred)) / somma_reali * 100) if somma_reali > 0 else 0

        fold_result = {
            'modello': model_wrapper.model_name,
            'id_finestra': fold,
            'data_test': test_df.index.min().date(),
            'tempo_training_sec': round(train_time, 4),
            'tempo_predizione_sec': round(pred_time, 4),
            'latenza_ms_per_slot': round((pred_time / len(y_test)) * 1000, 4),
            'rmse_globale': round(np.sqrt(mean_squared_error(y_test, y_pred)), 3),
            'mae_globale': round(mean_absolute_error(y_test, y_pred), 3),
            'mape_globale': round(np.mean(mape_array), 3),
            'wape_globale': round(wape_val, 3) # Salvataggio WAPE nel dizionario
        }

        for i in range(len(y_test)):
            val_reale = y_test[i]
            val_pred = y_pred[i]
            err_ass = abs(val_reale - val_pred)
            err_perc = (err_ass / val_reale * 100) if val_reale > 0 else (100 if val_pred > 0 else 0)

            fold_result[f'reale_slot_{i+1}'] = round(val_reale, 3)
            fold_result[f'predizione_slot_{i+1}'] = round(val_pred, 3)
            fold_result[f'errore_assoluto_slot_{i+1}'] = round(err_ass, 3)
            fold_result[f'errore_percentuale_slot_{i+1}'] = round(err_perc, 3)

        tutti_i_risultati.append(fold_result)
        df_temp = pd.DataFrame([fold_result])

        if not os.path.exists(path_salvataggio):
            df_temp.to_csv(path_salvataggio, index=False)
        else:
            df_temp.to_csv(path_salvataggio, mode='a', header=False, index=False)

    df_risultati = pd.DataFrame(tutti_i_risultati)

    if not df_risultati.empty:
        media_train = df_risultati['tempo_training_sec'].mean()
        media_pred_online = df_risultati['latenza_ms_per_slot'].mean()
        print(f"    [Tempi]: Offline: {media_train:.4f}s | Online: {media_pred_online:.4f}ms/slot")
        print(f"    [Metriche]: MAE: {df_risultati['mae_globale'].mean():.2f} | MAPE: {df_risultati['mape_globale'].mean():.2f}% | WAPE: {df_risultati['wape_globale'].mean():.2f}%")

    return df_risultati

# =====================================================================
# 5. FUNZIONI DI REPORTING & PLOTTING
# =====================================================================
prefissi = ["risultati_RF_Staz_", "risultati_NP_Staz_", "risultati_LSTM_Staz_", "risultati_Prophet_Staz_", "risultati_SVM_Staz_", "risultati_SARIMA_Staz_"]
nomi = ["Random Forest", "Neural Prophet", "Stacked LSTM", "Prophet", "SVM", "SARIMA"]
colori_globali = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# Salva un grafico nella directory principale dedicata ai plot globali
def save_plot(filename):
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"    Immagine salvata (Globale): {path}")

# Salva un grafico in una sotto-directory specifica (es. per grafici di singole stazioni)
def save_single_plot(directory, filename):
    path = os.path.join(directory, filename)
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()

# Raccoglie i risultati di tutti i modelli per ogni stazione, li salva in CSV e genera i grafici a barre riassuntivi
def salva_metriche_per_stazione(stazioni_trovate):
    dati_sommario = []
    
    for st_id in stazioni_trovate:
        for idx, prefisso in enumerate(prefissi):
            file_path = os.path.join(RESULTS_DIR, f"{prefisso}{st_id}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                if 'mae_globale' in df.columns and 'mape_globale' in df.columns:
                    mae_medio = df['mae_globale'].mean()
                    mape_medio = df['mape_globale'].mean()
                    wape_medio = df['wape_globale'].mean() if 'wape_globale' in df.columns else 0 # Aggiunto recupero WAPE
                    
                    dati_sommario.append({
                        'Stazione': str(st_id),
                        'Modello': nomi[idx],
                        'MAE': round(mae_medio, 3),
                        'MAPE_perc': round(mape_medio, 3),
                        'WAPE_perc': round(wape_medio, 3) # Aggiunto WAPE a sommario
                    })
                    
    if dati_sommario:
        df_sommario = pd.DataFrame(dati_sommario)
        
        path_csv = os.path.join(RESULTS_DIR, "sommario_metriche_per_stazione.csv")
        df_sommario.to_csv(path_csv, index=False)
        print(f"    Tabella metriche per stazione salvata in CSV: {path_csv}")
        
        df_pivot_mae = df_sommario.pivot(index='Stazione', columns='Modello', values='MAE')
        df_pivot_mape = df_sommario.pivot(index='Stazione', columns='Modello', values='MAPE_perc')
        df_pivot_wape = df_sommario.pivot(index='Stazione', columns='Modello', values='WAPE_perc') # Pivot WAPE
        
        color_map = {nomi[i]: colori_globali[i] for i in range(len(nomi))}
        color_list = [color_map.get(col) for col in df_pivot_mae.columns]

        df_pivot_mae.plot(kind='bar', figsize=(15, 6), color=color_list, edgecolor='black', alpha=0.8)
        plt.title('MAE (Errore Medio Assoluto) per Singola Stazione', fontweight='bold', fontsize=14)
        plt.xlabel('ID Stazione')
        plt.ylabel('MAE (km/h)')
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle=':', alpha=0.7)
        plt.legend(title='Modello')
        save_plot("06_mae_per_stazione.png")
        
        df_pivot_mape.plot(kind='bar', figsize=(15, 6), color=color_list, edgecolor='black', alpha=0.8)
        plt.title('MAPE (Errore Percentuale) per Singola Stazione', fontweight='bold', fontsize=14)
        plt.xlabel('ID Stazione')
        plt.ylabel('MAPE %')
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle=':', alpha=0.7)
        plt.legend(title='Modello')
        save_plot("07_mape_per_stazione.png")

        # --- GRAFICO WAPE ---
        df_pivot_wape.plot(kind='bar', figsize=(15, 6), color=color_list, edgecolor='black', alpha=0.8)
        plt.title('WAPE (Weighted Absolute Percentage Error) per Singola Stazione', fontweight='bold', fontsize=14)
        plt.xlabel('ID Stazione')
        plt.ylabel('WAPE %')
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle=':', alpha=0.7)
        plt.legend(title='Modello')
        save_plot("07b_wape_per_stazione.png")

# Disegna le curve che confrontano l'andamento reale rispetto alle predizioni per ogni stazione, per una specifica finestra
def plot_curve_singole_stazioni(stazioni_trovate, id_finestra=1):
    asse_x_ore = np.arange(1, 97) * 0.5

    for st_id in stazioni_trovate:
        plt.figure(figsize=(10, 6))
        dati_reali_tracciati = False
        
        for idx, prefisso in enumerate(prefissi):
            file_path = os.path.join(RESULTS_DIR, f"{prefisso}{st_id}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df_finestra = df[df['id_finestra'] == id_finestra]
                
                if not df_finestra.empty:
                    col_reali = [col for col in df.columns if col.startswith('reale_slot_')]
                    col_pred = [col for col in df.columns if col.startswith('predizione_slot_')]
                    
                    if col_reali and col_pred:
                        if not dati_reali_tracciati:
                            plt.plot(asse_x_ore, df_finestra[col_reali].values[0], label='Dato Reale', color='black', linewidth=2.5)
                            dati_reali_tracciati = True
                        plt.plot(asse_x_ore, df_finestra[col_pred].values[0], label=nomi[idx], color=colori_globali[idx], linewidth=1.5, alpha=0.8)

        plt.title(f'Stazione {st_id} - Curve Reali vs Predette (Finestra {id_finestra})', fontweight='bold')
        plt.xlabel("Tempo (Ore)")
        plt.ylabel("Diff. Velocità (km/h)")
        plt.axvline(x=24, color='gray', linestyle='--', alpha=0.5)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.xlim(0, 48)
        plt.legend(fontsize=9, loc='upper right')
        
        save_single_plot(DIR_CURVE_SINGOLE, f"curva_staz_{st_id}_finestra_{id_finestra}.png")
        print(f"    Immagine salvata (Singola Curva): Stazione {st_id}")

# Crea grafici per singola stazione mostrando come l'errore (MAE) aumenta all'aumentare dell'orizzonte di previsione
def plot_decadimento_singole_stazioni(stazioni_trovate):
    for st_id in stazioni_trovate:
        plt.figure(figsize=(10, 6))
        dati_presenti = False

        for idx_mod, prefisso in enumerate(prefissi):
            file_path = os.path.join(RESULTS_DIR, f"{prefisso}{st_id}.csv")
            if os.path.exists(file_path):
                df_ris = pd.read_csv(file_path)
                colonne_errore = [col for col in df_ris.columns if col.startswith('errore_assoluto_slot_')]

                if colonne_errore:
                    dati_presenti = True
                    errore_stazione_per_slot = df_ris[colonne_errore].mean(axis=0).values
                    asse_x_ore = np.arange(1, len(errore_stazione_per_slot) + 1) * 0.5

                    plt.plot(asse_x_ore, errore_stazione_per_slot,
                            label=nomi[idx_mod], color=colori_globali[idx_mod % len(colori_globali)], linewidth=2)

        if dati_presenti:
            plt.title(f"Decadimento Errore - Stazione {st_id}", fontweight='bold')
            plt.xlabel("Orizzonte di Previsione (Ore future)")
            plt.ylabel("MAE (Errore Medio Assoluto)")
            plt.axvline(x=24, color='black', linestyle='--', alpha=0.5)
            plt.axvline(x=48, color='black', linestyle='--', alpha=0.5)
            plt.legend(fontsize=9)
            plt.grid(True, linestyle=':', alpha=0.7)
            plt.xlim(0, 48)
            
            save_single_plot(DIR_DECADIMENTO_SINGOLO, f"decadimento_staz_{st_id}.png")
            print(f"    Immagine salvata (Singolo Decadimento): Stazione {st_id}")
        else:
            plt.close()

# Calcola e visualizza la media globale del decadimento dell'affidabilità (aumento dell'errore nel tempo) per tutta la rete
def analizza_decadimento_globale():
    plt.figure(figsize=(14, 7))
    dati_trovati = False

    for idx, prefisso in enumerate(prefissi):
        pattern = os.path.join(RESULTS_DIR, f"{prefisso}*.csv")
        file_modello = glob.glob(pattern)

        if len(file_modello) > 0:
            dati_trovati = True
            errori_tutte_stazioni = []
            for path_csv in file_modello:
                df_ris = pd.read_csv(path_csv)
                colonne_errore = [col for col in df_ris.columns if col.startswith('errore_assoluto_slot_')]
                errore_stazione_per_slot = df_ris[colonne_errore].mean(axis=0).values
                errori_tutte_stazioni.append(errore_stazione_per_slot)

            errore_medio_globale = np.mean(errori_tutte_stazioni, axis=0)
            asse_x_ore = np.arange(1, len(errore_medio_globale) + 1) * 0.5
            plt.plot(asse_x_ore, errore_medio_globale, label=f"{nomi[idx]}",
                     color=colori_globali[idx % len(colori_globali)], linewidth=2, marker='o', markersize=3)

    if dati_trovati:
        plt.title('Decadimento Globale Affidabilità Predittiva nel Tempo', fontsize=15, fontweight='bold')
        plt.xlabel('Orizzonte di Previsione (Ore future)')
        plt.ylabel('Errore Medio Assoluto (MAE per slot)')
        plt.axvline(x=24, color='black', linestyle='--', alpha=0.6)
        plt.axvline(x=48, color='black', linestyle='--', alpha=0.6)
        plt.legend(fontsize=11)
        plt.grid(True, linestyle=':', alpha=0.7)
        save_plot("01_decadimento_globale.png")

# Estrae i tempi di esecuzione dai CSV per generare grafici sui costi computazionali (training e latenza)
def analizza_tempi_computazionali():
    tempi_train, tempi_latenza, modelli_validi = [], [], []

    for idx, prefisso in enumerate(prefissi):
        pattern = os.path.join(RESULTS_DIR, f"{prefisso}*.csv")
        file_modello = glob.glob(pattern)
        if file_modello:
            t_train, t_lat = [], []
            for path_csv in file_modello:
                df = pd.read_csv(path_csv)
                if 'tempo_training_sec' in df.columns:
                    t_train.append(df['tempo_training_sec'].mean())
                    t_lat.append(df['latenza_ms_per_slot'].mean())
            if t_train:
                tempi_train.append(np.mean(t_train))
                tempi_latenza.append(np.mean(t_lat))
                modelli_validi.append(nomi[idx])

    if modelli_validi:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.bar(modelli_validi, tempi_train, color=colori_globali[:len(modelli_validi)], alpha=0.8, edgecolor='black')
        ax1.set_title('Costo Computazionale: Training Offline')
        ax1.set_ylabel('Secondi')
        ax1.grid(axis='y', linestyle=':', alpha=0.7)

        ax2.bar(modelli_validi, tempi_latenza, color=colori_globali[:len(modelli_validi)], alpha=0.8, edgecolor='black')
        ax2.set_title('Velocità in Produzione: Inferenza Online')
        ax2.set_ylabel('Latenza media (ms/Slot)')
        ax2.grid(axis='y', linestyle=':', alpha=0.7)
        save_plot("02_tempi_computazionali.png")

# Disegna una griglia (subplots) per confrontare visivamente i dati reali e predetti di più stazioni in un'unica immagine
def plot_griglia_curve_reali(stazioni_trovate, id_finestra=1):
    n_stazioni = len(stazioni_trovate)
    cols = 3
    rows = (n_stazioni + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 4 * rows))
    if n_stazioni == 1: axes = [axes]
    else: axes = axes.flatten()
    asse_x_ore = np.arange(1, 97) * 0.5

    for i, st_id in enumerate(stazioni_trovate):
        ax = axes[i]
        dati_reali_tracciati = False
        for idx, prefisso in enumerate(prefissi):
            file_path = os.path.join(RESULTS_DIR, f"{prefisso}{st_id}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df_finestra = df[df['id_finestra'] == id_finestra]
                if not df_finestra.empty:
                    col_reali = [col for col in df.columns if col.startswith('reale_slot_')]
                    col_pred = [col for col in df.columns if col.startswith('predizione_slot_')]
                    if col_reali and col_pred:
                        if not dati_reali_tracciati:
                            ax.plot(asse_x_ore, df_finestra[col_reali].values[0], label='Dato Reale', color='black', linewidth=2.5)
                            dati_reali_tracciati = True
                        ax.plot(asse_x_ore, df_finestra[col_pred].values[0], label=nomi[idx], color=colori_globali[idx], linewidth=1.5, alpha=0.8)

        ax.set_title(f'Stazione {st_id}', fontweight='bold')
        ax.set_xlabel("Tempo (Ore)")
        ax.set_ylabel("Diff. Velocità (km/h)")
        ax.axvline(x=24, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.set_xlim(0, 48)
        if i == 0: ax.legend(fontsize=9, loc='upper right')

    for j in range(i + 1, len(axes)): axes[j].set_visible(False)
    plt.suptitle(f'CURVE REALI VS PREDETTE (Finestra {id_finestra})', fontsize=18, fontweight='bold', y=0.98)
    save_plot(f"03_griglia_curve_reali_finestra_{id_finestra}.png")

# Disegna l'andamento nel tempo delle metriche (MAE/MAPE/WAPE) sull'intera rete
def plot_timeline_globale(metrica='mae_globale'):
    plt.figure(figsize=(15, 6))
    
    date_univoche = set() 
    
    for idx, prefisso in enumerate(prefissi):
        files = glob.glob(os.path.join(RESULTS_DIR, f"{prefisso}*.csv"))
        if files:
            df_lista = [pd.read_csv(f)[['data_test', metrica]] for f in files if metrica in pd.read_csv(f).columns]
            if df_lista:
                df_tot = pd.concat(df_lista)
                df_mediato = df_tot.groupby('data_test')[metrica].mean().reset_index()
                df_mediato['data_test'] = pd.to_datetime(df_mediato['data_test'])
                df_mediato = df_mediato.sort_values('data_test')
                
                plt.plot(df_mediato['data_test'], df_mediato[metrica], label=nomi[idx], color=colori_globali[idx], marker='o', linewidth=2)
                date_univoche.update(df_mediato['data_test'].tolist())

    for data in date_univoche:
        if data.weekday() >= 5: 
            inizio = data - pd.Timedelta(hours=12)
            fine = data + pd.Timedelta(hours=12)
            plt.axvspan(inizio, fine, color='red', alpha=0.15, lw=0)
            
    if date_univoche:
        plt.plot([], [], 's', color='red', alpha=0.15, label='Weekend')

    # Titoli dinamici in base alla metrica scelta
    if metrica == 'mae_globale':
        titolo = "MAE (Errore Medio Assoluto)"
    elif metrica == 'mape_globale':
        titolo = "MAPE (Errore Percentuale %)"
    else:
        titolo = "WAPE (Weighted Absolute Percentage Error %)"

    plt.title(f'RETE GLOBALE: Andamento Medio {titolo}', fontweight='bold', fontsize=14)
    plt.xlabel('Data del Test')
    plt.ylabel(f'Media {titolo}')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    save_plot(f"04_timeline_{metrica}.png")

# Genera un grafico a barre riassuntivo che classifica le performance medie totali di tutti i modelli valutati (Ora esteso a 3 colonne)
def plot_classifica_finale():
    mae_totali, mape_totali, wape_totali, modelli_validi = [], [], [], []
    for idx, prefisso in enumerate(prefissi):
        files = glob.glob(os.path.join(RESULTS_DIR, f"{prefisso}*.csv"))
        if files:
            mae_mod = [pd.read_csv(f)['mae_globale'].mean() for f in files if 'mae_globale' in pd.read_csv(f).columns]
            mape_mod = [pd.read_csv(f)['mape_globale'].mean() for f in files if 'mape_globale' in pd.read_csv(f).columns]
            wape_mod = [pd.read_csv(f)['wape_globale'].mean() for f in files if 'wape_globale' in pd.read_csv(f).columns]
            if mae_mod:
                mae_totali.append(np.mean(mae_mod))
                mape_totali.append(np.mean(mape_mod))
                wape_totali.append(np.mean(wape_mod) if wape_mod else 0)
                modelli_validi.append(nomi[idx])

    if modelli_validi:
        # Griglia 1 riga x 3 colonne (aggiunta la colonna WAPE)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
        
        bars1 = ax1.bar(modelli_validi, mae_totali, color=colori_globali[:len(modelli_validi)], alpha=0.8, edgecolor='black')
        ax1.set_title('Classifica: MAE Globale')
        ax1.set_ylabel('MAE (km/h)')
        ax1.grid(axis='y', linestyle=':', alpha=0.7)

        bars2 = ax2.bar(modelli_validi, mape_totali, color=colori_globali[:len(modelli_validi)], alpha=0.8, edgecolor='black')
        ax2.set_title('Classifica: MAPE Globale %')
        ax2.set_ylabel('MAPE %')
        ax2.grid(axis='y', linestyle=':', alpha=0.7)

        bars3 = ax3.bar(modelli_validi, wape_totali, color=colori_globali[:len(modelli_validi)], alpha=0.8, edgecolor='black')
        ax3.set_title('Classifica: WAPE Globale %')
        ax3.set_ylabel('WAPE %')
        ax3.grid(axis='y', linestyle=':', alpha=0.7)
        
        save_plot("05_classifica_finale.png")

# Disegna una griglia verticale che unisce tutte le finestre per mostrare l'intero periodo di test continuo
def plot_curve_interoperiodo_verticale(stazioni_trovate):
    n_stazioni = len(stazioni_trovate)
    fig, axes = plt.subplots(n_stazioni, 1, figsize=(16, 4 * n_stazioni))
    if n_stazioni == 1: axes = [axes]
    else: axes = axes.flatten()

    for i, st_id in enumerate(stazioni_trovate):
        ax = axes[i]
        dati_reali_tracciati = False

        for idx, prefisso in enumerate(prefissi):
            file_path = os.path.join(RESULTS_DIR, f"{prefisso}{st_id}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                if 'id_finestra' in df.columns:
                    df = df.sort_values('id_finestra')

                col_reali = sorted([col for col in df.columns if col.startswith('reale_slot_')], key=lambda x: int(x.split('_')[-1]))
                col_pred = sorted([col for col in df.columns if col.startswith('predizione_slot_')], key=lambda x: int(x.split('_')[-1]))

                if not col_reali or not col_pred: continue

                reali_seq, pred_seq = [], []
                for r_idx in range(len(df)):
                    # Unisce le prime 24h per evitare sovrapposizioni e tiene le 48h complete solo per l'ultima finestra
                    if r_idx < len(df) - 1:
                        reali_seq.extend(df.iloc[r_idx][col_reali[:48]].values)
                        pred_seq.extend(df.iloc[r_idx][col_pred[:48]].values)
                    else:
                        reali_seq.extend(df.iloc[r_idx][col_reali].values)
                        pred_seq.extend(df.iloc[r_idx][col_pred].values)

                asse_x_giorni = np.arange(len(reali_seq)) / 48.0

                if not dati_reali_tracciati:
                    ax.plot(asse_x_giorni, reali_seq, label='Dato Reale', color='black', linewidth=2, linestyle='-')
                    dati_reali_tracciati = True

                ax.plot(asse_x_giorni, pred_seq, label=nomi[idx], color=colori_globali[idx % len(colori_globali)], linewidth=1.2, alpha=0.8)

        ax.set_title(f'Stazione {st_id}', fontweight='bold')
        ax.set_xlabel("Tempo dall'inizio del test (Giorni)")
        ax.set_ylabel("Diff. Velocità (km/h)")
        ax.grid(True, linestyle=':', alpha=0.7)
        if len(reali_seq) > 0: ax.set_xlim(0, len(reali_seq) / 48.0)
        
        if i == 0: ax.legend(fontsize=10, loc='upper right', ncol=len(nomi)+1)

    plt.suptitle('CURVE REALI VS PREDETTE (INTERO PERIODO DI TEST CONTINUO)', fontsize=18, fontweight='bold', y=0.99)
    save_plot("08_curve_interoperiodo_verticale.png")

# Disegna una griglia 3x3 per visualizzare l'andamento mensile di MAE, MAPE o WAPE per tutte le stazioni
def plot_griglia_metrica_mensile(stazioni_trovate, metrica='mae_globale'):
    n_stazioni = len(stazioni_trovate)
    cols = 3
    rows = (n_stazioni + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 4 * rows))
    if n_stazioni == 1: axes = [axes]
    else: axes = axes.flatten()

    for i, st_id in enumerate(stazioni_trovate):
        ax = axes[i]
        date_univoche = set()

        for idx, prefisso in enumerate(prefissi):
            file_path = os.path.join(RESULTS_DIR, f"{prefisso}{st_id}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                if metrica in df.columns:
                    df['data_test'] = pd.to_datetime(df['data_test'])
                    df = df.sort_values('data_test')
                    ax.plot(df['data_test'], df[metrica], label=nomi[idx], color=colori_globali[idx % len(colori_globali)], marker='o', markersize=4, linewidth=1.5)
                    date_univoche.update(df['data_test'].tolist())

        # Disegniamo i weekend in rosso
        for data in date_univoche:
            if data.weekday() >= 5:
                inizio = data - pd.Timedelta(hours=12)
                fine = data + pd.Timedelta(hours=12)
                ax.axvspan(inizio, fine, color='red', alpha=0.15, lw=0)

        ax.set_title(f'Stazione {st_id}', fontweight='bold')
        ax.set_xlabel("Data del Test")
        
        # Etichette Asse Y dinamiche
        if metrica == 'mae_globale':
            ax.set_ylabel("MAE (km/h)")
        elif metrica == 'mape_globale':
            ax.set_ylabel("MAPE (%)")
        else:
            ax.set_ylabel("WAPE (%)")
            
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
        for tick in ax.get_xticklabels(): tick.set_rotation(45)

        if i == 0:
            ax.plot([], [], 's', color='red', alpha=0.15, label='Weekend')
            ax.legend(fontsize=9, loc='best')

    for j in range(i + 1, len(axes)): axes[j].set_visible(False)

    # Titoli Dinamici
    if metrica == 'mae_globale':
        titolo = "ERRORE MEDIO ASSOLUTO (MAE)"
    elif metrica == 'mape_globale':
        titolo = "ERRORE PERCENTUALE (MAPE %)"
    else:
        titolo = "WAPE (Weighted Absolute Percentage Error %)"

    plt.suptitle(f'ANDAMENTO {titolo} (Tutte le stazioni)', fontsize=18, fontweight='bold', y=0.98)
    save_plot(f"09_griglia_mensile_{metrica}.png")

# Salva un singolo grafico con le curve continue per tutto il periodo di test, per ogni stazione
def plot_curve_interoperiodo_singole_stazioni(stazioni_trovate):
    for st_id in stazioni_trovate:
        plt.figure(figsize=(12, 5))
        dati_reali_tracciati = False
        dati_presenti = False

        for idx, prefisso in enumerate(prefissi):
            file_path = os.path.join(RESULTS_DIR, f"{prefisso}{st_id}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                if 'id_finestra' in df.columns:
                    df = df.sort_values('id_finestra')

                col_reali = sorted([col for col in df.columns if col.startswith('reale_slot_')], key=lambda x: int(x.split('_')[-1]))
                col_pred = sorted([col for col in df.columns if col.startswith('predizione_slot_')], key=lambda x: int(x.split('_')[-1]))

                if not col_reali or not col_pred: continue
                dati_presenti = True

                reali_seq, pred_seq = [], []
                for r_idx in range(len(df)):
                    if r_idx < len(df) - 1:
                        reali_seq.extend(df.iloc[r_idx][col_reali[:48]].values)
                        pred_seq.extend(df.iloc[r_idx][col_pred[:48]].values)
                    else:
                        reali_seq.extend(df.iloc[r_idx][col_reali].values)
                        pred_seq.extend(df.iloc[r_idx][col_pred].values)

                asse_x_giorni = np.arange(len(reali_seq)) / 48.0

                if not dati_reali_tracciati:
                    plt.plot(asse_x_giorni, reali_seq, label='Dato Reale', color='black', linewidth=2, linestyle='-')
                    dati_reali_tracciati = True

                plt.plot(asse_x_giorni, pred_seq, label=nomi[idx], color=colori_globali[idx % len(colori_globali)], linewidth=1.2, alpha=0.8)

        if dati_presenti:
            plt.title(f'Curve Reali vs Predette (Intero Periodo) - Stazione {st_id}', fontweight='bold')
            plt.xlabel("Tempo dall'inizio del test (Giorni)")
            plt.ylabel("Diff. Velocità (km/h)")
            plt.grid(True, linestyle=':', alpha=0.7)
            if len(reali_seq) > 0: plt.xlim(0, len(reali_seq) / 48.0)
            plt.legend(fontsize=9, loc='upper right', ncol=2)
            
            save_single_plot(DIR_INTEROPERIODO_SINGOLO, f"interoperiodo_staz_{st_id}.png")
        else:
            plt.close()

# Salva un singolo grafico con l'andamento del MAPE/MAE/WAPE nel tempo, per ogni stazione
def plot_andamento_mensile_singole_stazioni(stazioni_trovate, metrica='mape_globale'):
    for st_id in stazioni_trovate:
        plt.figure(figsize=(10, 5))
        date_univoche = set()
        dati_presenti = False

        for idx, prefisso in enumerate(prefissi):
            file_path = os.path.join(RESULTS_DIR, f"{prefisso}{st_id}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                if metrica in df.columns:
                    dati_presenti = True
                    df['data_test'] = pd.to_datetime(df['data_test'])
                    df = df.sort_values('data_test')
                    plt.plot(df['data_test'], df[metrica], label=nomi[idx], color=colori_globali[idx % len(colori_globali)], marker='o', markersize=4, linewidth=1.5)
                    date_univoche.update(df['data_test'].tolist())

        if dati_presenti:
            # Aggiunta strisce rosse per il weekend
            for data in date_univoche:
                if data.weekday() >= 5:
                    inizio = data - pd.Timedelta(hours=12)
                    fine = data + pd.Timedelta(hours=12)
                    plt.axvspan(inizio, fine, color='red', alpha=0.15, lw=0)

            plt.plot([], [], 's', color='red', alpha=0.15, label='Weekend')
            
            if metrica == 'mae_globale':
                titolo = "MAE (km/h)"
            elif metrica == 'mape_globale':
                titolo = "MAPE (%)"
            else:
                titolo = "WAPE (%)"
                
            plt.title(f'Andamento Mensile {titolo} - Stazione {st_id}', fontweight='bold')
            plt.xlabel("Data del Test")
            plt.ylabel(titolo)
            plt.grid(True, linestyle=':', alpha=0.7)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
            plt.xticks(rotation=45)
            plt.legend(fontsize=9, loc='best')
            
            save_single_plot(DIR_METRICA_MENSILE_SINGOLO, f"mensile_{metrica}_staz_{st_id}.png")
        else:
            plt.close()

# =========================================================================
# 6. MOTORE DI ESECUZIONE (MAIN ENTRY POINT)
# =========================================================================
if __name__ == "__main__":
    print("\n" + "#"*70)
    print(" INIZIO ELABORAZIONE BATCH MULTI-STAZIONE")
    print("#"*70)

    try:
        dataset_raw = carica_dataset_completo(FILE_NAME)
    except FileNotFoundError as e:
        print(f"\nERRORE: {e}\nPosiziona il file nel percorso corretto o modifica PATH_DATI.")
        exit(1)

    stazioni_disponibili = dataset_raw['id_stazione'].unique()
    
    # -------------------------------------------------------------------------
    # PARAMETRI PER SCORRIMENTO A 30 MINUTI
    # -------------------------------------------------------------------------
    TEST_GIORNI_FUTURI = 2   # Previsione effettuata per le 48 ore successive
    STEP_MINUTI = 30         # Il modello scorre in avanti di 30 minuti
    
    # IMPORTANTE: Avendo settato lo step a 30 minuti, ci sono 48 cicli al giorno.
    # Impostando 1440 come max_folds testiamo esplicitamente 30 GIORNI (48 * 30).
    MAX_ITERAZIONI = 1440 
    
    # Dizionario con gli ordini autoregressivi stagionali presettati per SARIMA
    PARAMETRI_SARIMA_STAZIONI = {
        1: {'order': (3, 0, 1), 'seasonal_order': (0, 1, 1, 48)},
        2: {'order': (3, 0, 1), 'seasonal_order': (0, 1, 1, 48)},
        3: {'order': (3, 0, 1), 'seasonal_order': (0, 1, 1, 48)},
        4: {'order': (3, 0, 1), 'seasonal_order': (0, 1, 1, 48)},
        5: {'order': (3, 0, 1), 'seasonal_order': (0, 1, 1, 48)},
        6: {'order': (3, 0, 1), 'seasonal_order': (0, 1, 1, 48)},
        7: {'order': (3, 0, 1), 'seasonal_order': (0, 1, 1, 48)},
        8: {'order': (3, 0, 1), 'seasonal_order': (0, 1, 1, 48)},
        9: {'order': (3, 0, 1), 'seasonal_order': (0, 1, 1, 48)},
    }
    
    inizio_globale = time.time()

    for st_id in stazioni_disponibili:
        print(f"\n{'='*70}\n--- AVVIO PIPELINE: STAZIONE {st_id} ---\n{'='*70}")
        dataset_stazione = pre_processing_stazione(dataset_raw, st_id)

        if len(dataset_stazione) < 48 * 30:
            print(f" -> Stazione {st_id} saltata (Dati storici insufficienti)")
            continue

        modelli = [
            (RandomForestWrapper(), f"risultati_RF_Staz_{st_id}.csv"),
            (NeuralProphetWrapper(epochs=50), f"risultati_NP_Staz_{st_id}.csv"),
            (StackedLSTMWrapper(epochs=20), f"risultati_LSTM_Staz_{st_id}.csv"),
            (SVMWrapper(), f"risultati_SVM_Staz_{st_id}.csv"),
            (ProphetWrapper(), f"risultati_Prophet_Staz_{st_id}.csv")
        ]

        if st_id in PARAMETRI_SARIMA_STAZIONI:
            p_order = PARAMETRI_SARIMA_STAZIONI[st_id]['order']
            p_seasonal = PARAMETRI_SARIMA_STAZIONI[st_id]['seasonal_order']
            modelli.append((SARIMAWrapper(order=p_order, seasonal_order=p_seasonal), f"risultati_SARIMA_Staz_{st_id}.csv"))

        for modello, file_out in modelli:
            gen = sliding_window_generator(dataset_stazione, train_days='auto', test_days=TEST_GIORNI_FUTURI, step_minutes=STEP_MINUTI, max_folds=MAX_ITERAZIONI, test_on_end=True)
            path_out = os.path.join(RESULTS_DIR, file_out)
            run_evaluation_loop_modulare(gen, modello, path_out)

    # Calcolo del tempo totale
    tempo_totale_secondi = int(time.time() - inizio_globale)
    tempo_totale_formattato = str(datetime.timedelta(seconds=tempo_totale_secondi))
    
    print(f"\nELABORAZIONE COMPLETA. Tempo totale: {tempo_totale_formattato}")

    # --- SALVATAGGIO DEL TEMPO TOTALE SU FILE ---
    percorso_log_tempi = os.path.join(RESULTS_DIR, "tempo_totale_esecuzione.txt")
    with open(percorso_log_tempi, "w") as f:
        f.write("=== REPORT ESECUZIONE ESPERIMENTO ===\n")
        f.write(f"Data completamento: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Tempo totale di esecuzione: {tempo_totale_formattato}\n")
        f.write(f"Totale in secondi: {tempo_totale_secondi} s\n")
        
    print(f" -> Log del tempo salvato con successo in: {percorso_log_tempi}")

    # =========================================================================
    # 7. PLOTTING AUTOMATIZZATO
    # =========================================================================
    print("\nGenerazione Grafici in corso...")
    stazioni_trovate = set()
    for file_path in glob.glob(os.path.join(RESULTS_DIR, "*.csv")):
        match = re.search(r'_Staz_(\d+)\.csv', file_path)
        if match: stazioni_trovate.add(int(match.group(1)))
    stazioni_trovate = sorted(list(stazioni_trovate))

    if stazioni_trovate:
        print("\n -> Creazione grafici globali e calcolo metriche...")
        analizza_decadimento_globale()
        analizza_tempi_computazionali()
        plot_griglia_curve_reali(stazioni_trovate, id_finestra=1)
        
        # Griglie GLOBALI (Aggiunto il salvataggio della griglia per il WAPE)
        plot_curve_interoperiodo_verticale(stazioni_trovate)
        plot_griglia_metrica_mensile(stazioni_trovate, metrica='mae_globale')
        plot_griglia_metrica_mensile(stazioni_trovate, metrica='mape_globale')
        plot_griglia_metrica_mensile(stazioni_trovate, metrica='wape_globale') 
        
        # Timeline andamento temporale (Aggiunto WAPE)
        plot_timeline_globale(metrica='mae_globale')
        plot_timeline_globale(metrica='mape_globale')
        plot_timeline_globale(metrica='wape_globale') 
        
        plot_classifica_finale() # Aggiornata in automatico per generare 3 colonne al posto di 2
        
        salva_metriche_per_stazione(stazioni_trovate) # Aggiornata per salvare il grafico a barre del WAPE
        
        print("\n -> Creazione grafici individuali per singola stazione...")
        plot_curve_singole_stazioni(stazioni_trovate, id_finestra=1)
        plot_decadimento_singole_stazioni(stazioni_trovate)
        
        # Grafici INDIVIDUALI (Aggiunto il salvataggio dei file singoli WAPE)
        plot_curve_interoperiodo_singole_stazioni(stazioni_trovate)
        plot_andamento_mensile_singole_stazioni(stazioni_trovate, metrica='mape_globale')
        plot_andamento_mensile_singole_stazioni(stazioni_trovate, metrica='mae_globale')
        plot_andamento_mensile_singole_stazioni(stazioni_trovate, metrica='wape_globale') 
        
        print(f"\nTutti i grafici sono stati generati e strutturati all'interno della cartella: {PLOTS_DIR}")
    else:
        print("\nNessun dato trovato per la generazione dei grafici.")