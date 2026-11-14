# -*- coding: utf-8 -*-
"""
Protocollo Sperimentale (Avanzamento Giornaliero) - Adattato per Occupazione Colonnine EV
Include: Decadimento Globale dell'Errore, Analisi Storica/Mensile e Percorsi Assoluti
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
import seaborn as sns

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

# Imposta la tua cartella di lavoro assoluta
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Modificato per adattarsi dinamicamente alla cartella del server
PATH_DATI = BASE_DIR

# Usa il file pulito
FILE_NAME = 'stato_colonnine_pulito.csv'

# Cartella dedicata per salvare tutti i risultati
PATH_RISULTATI = os.path.join(BASE_DIR, 'Risultati_Esperimenti_Occupazione')
RESULTS_DIR = os.path.join(PATH_RISULTATI, 'Risultati_Multi_Modello_Giornaliero')

PLOTS_DIR = os.path.join(RESULTS_DIR, 'grafici')
DIR_CURVE_SINGOLE = os.path.join(PLOTS_DIR, 'curve_singole')
DIR_DECADIMENTO_SINGOLO = os.path.join(PLOTS_DIR, 'decadimento_singolo')
DIR_INTEROPERIODO_SINGOLO = os.path.join(PLOTS_DIR, 'curve_interoperiodo_singole')
DIR_METRICA_MENSILE_SINGOLO = os.path.join(PLOTS_DIR, 'metrica_mensile_singola')

# Creazione automatica delle cartelle se non esistono
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(DIR_CURVE_SINGOLE, exist_ok=True)
os.makedirs(DIR_DECADIMENTO_SINGOLO, exist_ok=True)
os.makedirs(DIR_INTEROPERIODO_SINGOLO, exist_ok=True)
os.makedirs(DIR_METRICA_MENSILE_SINGOLO, exist_ok=True)

print(f"Directory di lavoro: {BASE_DIR}")
print(f"Directory salvataggio risultati: {RESULTS_DIR}")

# =========================================================================
#  FIX DEFINITIVO PER PYTORCH 2.6+
# =========================================================================
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

# =========================================================================
# 1. FUNZIONI DI CARICAMENTO E PRE-PROCESSING
# =========================================================================

# Mappatura esclusiva sulle colonnine selezionate con alto score informativo
dati_iniziali_colonnine = [
    (1, "VUN6Vmn62SrL2bTEgNj1LQ"), # Bocconi
    (3, "KFuZUwMchLahhkOWC7JWCQ"), # Viale Certosa
    (4, "hjcQ0Y4ew_Q87GS1SbrGBw"), # Tangenziale Ovest
    (5, "1_bXSX1bKKga0akzOrO9qw"), # Stazione Centrale 1
    (7, "_lf0j_lBN8bPRGUiuGDGPg"), # Navigli
    (8, "-rSGGo3ox0NoCjAAdMt_Nw"), # Corso Buenos Aires
    (9, "v6t0SuVogqQw8AgNAA2bVA"), # Porta Romana
    (2, "bFF9ywyXJxYkCB2-iD_enQ")  # Citylife (Inserita per ultima perchè potrebbe dare problemi)
]

def carica_dataset_completo(nome_file):
    print("\nLettura e calcolo del target (Tasso Occupazione) in memoria...")
    percorso_completo = os.path.join(PATH_DATI, nome_file)
    if not os.path.exists(percorso_completo):
        raise FileNotFoundError(f"File {nome_file} non trovato nel percorso: {percorso_completo}")
    
    df = pd.read_csv(percorso_completo)
    df = df[df['note_stato'] == 'OK'].copy()
    df['timestamp'] = pd.to_datetime(df['giorno'] + ' ' + df['orario'])
    
    mappa_stazioni = {col_id: st_id for st_id, col_id in dati_iniziali_colonnine}
    df['id_stazione'] = df['id_colonnina'].map(mappa_stazioni)
    df.dropna(subset=['id_stazione'], inplace=True)
    df['id_stazione'] = df['id_stazione'].astype(int)
    
    df['tasso_occupazione'] = (df['prese_occupate'] / df['prese_totali']) * 100
    df['tasso_occupazione'] = df['tasso_occupazione'].fillna(0)
    
    return df

def pre_processing_stazione(df_completo, id_stazione):
    df_stazione = df_completo[df_completo['id_stazione'] == id_stazione].copy()
    df_stazione.set_index('timestamp', inplace=True)
    df_stazione.sort_index(inplace=True)
    
    df_resampled = df_stazione[['tasso_occupazione']].resample('30min').mean()
    df_resampled['tasso_occupazione'] = df_resampled['tasso_occupazione'].ffill().fillna(0)
    
    df_resampled['hour'] = df_resampled.index.hour
    df_resampled['dayofweek'] = df_resampled.index.dayofweek
    df_resampled['is_weekend'] = (df_resampled.index.dayofweek >= 5).astype(int)
    
    df_resampled.dropna(inplace=True)
    return df_resampled

# =========================================================================
# 2. SLIDING WINDOW DINAMICA
# =========================================================================
from datetime import timedelta

def sliding_window_generator(df, train_days='auto', test_days=2, step_minutes=1440, max_folds=30, test_on_end=True):
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
# 3. WRAPPERS DEI MODELLI (Con Clipping 0-100)
# =====================================================================

class RandomForestWrapper:
    def __init__(self):
        self.model_name = "RandomForest"
        self.params = {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}
        self.model = RandomForestRegressor(**self.params, n_jobs=-1)
        self.features = ['hour', 'dayofweek', 'is_weekend']

    def train(self, train_df):
        self.model.fit(train_df[self.features].values, train_df['tasso_occupazione'].values)

    def predict(self, train_df, test_df):
        return np.clip(self.model.predict(test_df[self.features].values), 0, 100)

class NeuralProphetWrapper:
    def __init__(self, epochs=50):
        self.model_name = "NeuralProphet"
        self.epochs = epochs

    def train(self, train_df):
        df_np = pd.DataFrame({'ds': train_df.index, 'y': train_df['tasso_occupazione'].values})
        self.model = NeuralProphet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=True, epochs=self.epochs)
        self.model.fit(df_np, freq="30min", progress=None, minimal=True)

    def predict(self, train_df, test_df):
        df_np = pd.DataFrame({'ds': train_df.index, 'y': train_df['tasso_occupazione'].values})
        future = self.model.make_future_dataframe(df_np, periods=len(test_df))
        return np.clip(self.model.predict(future)['yhat1'].values, 0, 100)

class StackedLSTMWrapper:
    def __init__(self, look_back=48, epochs=20):
        self.model_name = "Stacked_LSTM"
        self.look_back = look_back
        self.epochs = epochs
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def train(self, train_df):
        tf.keras.backend.clear_session()
        scaled = self.scaler.fit_transform(train_df[['tasso_occupazione']].values)
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
        batch = train_df[['tasso_occupazione']].values[-self.look_back:]
        batch_scaled = self.scaler.transform(batch).reshape(1, self.look_back, 1)

        for _ in range(len(test_df)):
            p_scaled = self.model.predict(batch_scaled, verbose=0)[0]
            valore_riportato = self.scaler.inverse_transform([p_scaled])[0, 0]
            preds.append(np.clip(valore_riportato, 0, 100))
            batch_scaled = np.append(batch_scaled[:, 1:, :], [[[p_scaled[0]]]], axis=1)
        return np.array(preds)

class ProphetWrapper:
    def __init__(self):
        self.model_name = "Prophet"

    def train(self, train_df):
        df_p = pd.DataFrame({'ds': train_df.index, 'y': train_df['tasso_occupazione'].values})
        self.model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=True)
        self.model.fit(df_p)

    def predict(self, train_df, test_df):
        df_future = pd.DataFrame({'ds': test_df.index})
        forecast = self.model.predict(df_future)
        return np.clip(forecast['yhat'].values, 0, 100)

class SVMWrapper:
    def __init__(self, kernel='rbf', C=1.0, epsilon=0.1):
        self.model_name = "SVM"
        self.scaler = StandardScaler()
        self.model = SVR(kernel=kernel, C=C, epsilon=epsilon)
        self.features = ['hour', 'dayofweek', 'is_weekend']

    def train(self, train_df):
        X = train_df[self.features].values
        y = train_df['tasso_occupazione'].values
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, train_df, test_df):
        X_test = test_df[self.features].values
        X_test_scaled = self.scaler.transform(X_test)
        return np.clip(self.model.predict(X_test_scaled), 0, 100)

class SARIMAWrapper:
    def __init__(self, order, seasonal_order):
        self.model_name = "SARIMA"
        self.order = order
        self.seasonal_order = seasonal_order

    def train(self, train_df):
        self.model = sm.tsa.SARIMAX(train_df['tasso_occupazione'].values,
                                    order=self.order,
                                    seasonal_order=self.seasonal_order,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
        self.fitted = self.model.fit(disp=False)

    def predict(self, train_df, test_df):
        return np.clip(self.fitted.get_forecast(steps=len(test_df)).predicted_mean, 0, 100)

# =====================================================================
# 4. EVALUATION LOOP
# =====================================================================

def run_evaluation_loop_modulare(generatore, model_wrapper, path_salvataggio):
    print(f"Avvio Loop: {model_wrapper.model_name}")
    tutti_i_risultati = []

    for fold, train_df, test_df in generatore:
        if train_df.empty or test_df.empty: continue
        y_test = test_df['tasso_occupazione'].values

        try:
            start_train = time.perf_counter()
            model_wrapper.train(train_df)
            train_time = time.perf_counter() - start_train

            start_pred = time.perf_counter()
            y_pred = model_wrapper.predict(train_df, test_df)
            pred_time = time.perf_counter() - start_pred
        except Exception as e:
            print(f"    [!] ERRORE: {model_wrapper.model_name} (finestra {fold}): {e}")
            continue

        mape_array = np.where(y_test > 0, (np.abs(y_test - y_pred) / y_test) * 100, 0)
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
            'wape_globale': round(wape_val, 3)
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
        print(f"    [Metriche Medie]: MAE: {df_risultati['mae_globale'].mean():.2f}% | WAPE: {df_risultati['wape_globale'].mean():.2f}%")

    return df_risultati

# =====================================================================
# 5. FUNZIONI DI REPORTING & PLOTTING
# =====================================================================
prefissi = ["risultati_RF_Staz_", "risultati_NP_Staz_", "risultati_LSTM_Staz_", "risultati_Prophet_Staz_", "risultati_SVM_Staz_", "risultati_SARIMA_Staz_"]
nomi = ["Random Forest", "Neural Prophet", "Stacked LSTM", "Prophet", "SVM", "SARIMA"]
colori_globali = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

def save_plot(filename):
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()

def save_single_plot(directory, filename):
    path = os.path.join(directory, filename)
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()

def salva_metriche_per_stazione(stazioni_trovate):
    dati_sommario = []
    for st_id in stazioni_trovate:
        for idx, prefisso in enumerate(prefissi):
            file_path = os.path.join(RESULTS_DIR, f"{prefisso}{st_id}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                if 'mae_globale' in df.columns:
                    dati_sommario.append({
                        'Stazione': str(st_id),
                        'Modello': nomi[idx],
                        'MAE': round(df['mae_globale'].mean(), 3),
                        'MAPE_perc': round(df['mape_globale'].mean(), 3),
                        'WAPE_perc': round(df['wape_globale'].mean(), 3)
                    })
    if dati_sommario:
        df_sommario = pd.DataFrame(dati_sommario)
        df_sommario.to_csv(os.path.join(RESULTS_DIR, "sommario_metriche.csv"), index=False)
        
        df_pivot_mae = df_sommario.pivot(index='Stazione', columns='Modello', values='MAE')
        color_list = [colori_globali[nomi.index(col)] for col in df_pivot_mae.columns]

        df_pivot_mae.plot(kind='bar', figsize=(15, 6), color=color_list, edgecolor='black', alpha=0.8)
        plt.title('MAE (Errore Medio in % di Occupazione) per Zona', fontweight='bold')
        plt.ylabel('MAE (%)')
        plt.xticks(rotation=0)
        save_plot("06_mae_per_stazione.png")
        
        df_pivot_wape = df_sommario.pivot(index='Stazione', columns='Modello', values='WAPE_perc')
        df_pivot_wape.plot(kind='bar', figsize=(15, 6), color=color_list, edgecolor='black', alpha=0.8)
        plt.title('WAPE (Occupazione Globale) per Zona', fontweight='bold')
        plt.ylabel('WAPE %')
        plt.xticks(rotation=0)
        save_plot("07b_wape_per_stazione.png")

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
                            plt.plot(asse_x_ore, df_finestra[col_reali].values[0], label='Reale', color='black', linewidth=2.5)
                            dati_reali_tracciati = True
                        plt.plot(asse_x_ore, df_finestra[col_pred].values[0], label=nomi[idx], color=colori_globali[idx])
        plt.title(f'Zona {st_id} - Curve Predette vs Reali (Finestra {id_finestra})')
        plt.ylabel("Tasso Occupazione (%)")
        plt.xlim(0, 48)
        plt.legend()
        save_single_plot(DIR_CURVE_SINGOLE, f"curva_staz_{st_id}_finestra_{id_finestra}.png")

def plot_decadimento_singole_stazioni(stazioni_trovate):
    for st_id in stazioni_trovate:
        plt.figure(figsize=(10, 6))
        dati_presenti = False
        for idx, prefisso in enumerate(prefissi):
            file_path = os.path.join(RESULTS_DIR, f"{prefisso}{st_id}.csv")
            if os.path.exists(file_path):
                df_ris = pd.read_csv(file_path)
                colonne_errore = [col for col in df_ris.columns if col.startswith('errore_assoluto_slot_')]
                if colonne_errore:
                    dati_presenti = True
                    errore_medio = df_ris[colonne_errore].mean(axis=0).values
                    plt.plot(np.arange(1, len(errore_medio) + 1) * 0.5, errore_medio, label=nomi[idx], color=colori_globali[idx])
        if dati_presenti:
            plt.title(f"Decadimento Errore - Zona {st_id}")
            plt.ylabel("MAE (%)")
            plt.xlim(0, 48)
            plt.legend()
            save_single_plot(DIR_DECADIMENTO_SINGOLO, f"decadimento_staz_{st_id}.png")
        else:
            plt.close()

def plot_classifica_finale():
    mae_totali, mape_totali, wape_totali, modelli_validi = [], [], [], []
    for idx, prefisso in enumerate(prefissi):
        files = glob.glob(os.path.join(RESULTS_DIR, f"{prefisso}*.csv"))
        if files:
            mae_mod = [pd.read_csv(f)['mae_globale'].mean() for f in files if 'mae_globale' in pd.read_csv(f).columns]
            wape_mod = [pd.read_csv(f)['wape_globale'].mean() for f in files if 'wape_globale' in pd.read_csv(f).columns]
            if mae_mod:
                mae_totali.append(np.mean(mae_mod))
                wape_totali.append(np.mean(wape_mod) if wape_mod else 0)
                modelli_validi.append(nomi[idx])
    if modelli_validi:
        fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(16, 5))
        ax1.bar(modelli_validi, mae_totali, color=colori_globali[:len(modelli_validi)], alpha=0.8, edgecolor='black')
        ax1.set_title('Classifica: MAE Globale')
        ax1.set_ylabel('MAE (%)')
        
        ax3.bar(modelli_validi, wape_totali, color=colori_globali[:len(modelli_validi)], alpha=0.8, edgecolor='black')
        ax3.set_title('Classifica: WAPE Globale')
        ax3.set_ylabel('WAPE %')
        save_plot("05_classifica_finale.png")

# --- NUOVE FUNZIONI AVANZATE ---

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
                
                if colonne_errore:
                    errore_stazione_per_slot = df_ris[colonne_errore].mean(axis=0).values
                    errori_tutte_stazioni.append(errore_stazione_per_slot)

            if errori_tutte_stazioni:
                errore_medio_globale = np.mean(errori_tutte_stazioni, axis=0)
                asse_x_ore = np.arange(1, len(errore_medio_globale) + 1) * 0.5
                
                plt.plot(asse_x_ore, errore_medio_globale, label=f"{nomi[idx]}",
                         color=colori_globali[idx % len(colori_globali)], linewidth=2.5, marker='o', markersize=4)

    if dati_trovati:
        plt.title('Decadimento Globale dell\'Affidabilità Predittiva (Fino a 48 Ore)', fontsize=15, fontweight='bold')
        plt.xlabel('Orizzonte di Previsione Futuro (Ore)')
        plt.ylabel('Errore Medio Assoluto - MAE (%)')
        plt.axvline(x=12, color='gray', linestyle='--', alpha=0.5, label='12 Ore')
        plt.axvline(x=24, color='black', linestyle='--', alpha=0.8, label='24 Ore')
        plt.legend(fontsize=11)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.xlim(0, 48)
        save_plot("08_decadimento_errore_slot_globale.png")

def analizza_momenti_dell_anno():
    df_lista = []
    
    for idx, prefisso in enumerate(prefissi):
        files = glob.glob(os.path.join(RESULTS_DIR, f"{prefisso}*.csv"))
        for f in files:
            df_temp = pd.read_csv(f)
            if 'data_test' in df_temp.columns and 'mae_globale' in df_temp.columns:
                df_temp['Modello'] = nomi[idx]
                df_lista.append(df_temp[['data_test', 'Modello', 'mae_globale']])
                
    if not df_lista:
        return
        
    df_totale = pd.concat(df_lista)
    df_totale['data_test'] = pd.to_datetime(df_totale['data_test'])
    df_totale['Mese'] = df_totale['data_test'].dt.to_period('M') 
    
    # Grafico 1: Timeline Storica dell'Errore
    plt.figure(figsize=(15, 6))
    df_mediato_giorno = df_totale.groupby(['data_test', 'Modello'])['mae_globale'].mean().reset_index()
    
    for idx, modello in enumerate(df_mediato_giorno['Modello'].unique()):
        dati_modello = df_mediato_giorno[df_mediato_giorno['Modello'] == modello]
        dati_modello = dati_modello.sort_values('data_test')
        plt.plot(dati_modello['data_test'], dati_modello['mae_globale'], 
                 label=modello, color=colori_globali[idx % len(colori_globali)], marker='o')
                 
    plt.title('Andamento dell\'Errore (MAE) nel Tempo: Quali sono i giorni peggiori?', fontweight='bold', fontsize=14)
    plt.xlabel('Data')
    plt.ylabel('Errore Predittivo (MAE %)')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    save_plot("09a_timeline_storica_errore.png")
    
    # Grafico 2: Errore aggregato per Mese 
    plt.figure(figsize=(12, 6))
    df_mese = df_totale.groupby(['Mese', 'Modello'])['mae_globale'].mean().reset_index()
    df_mese['Mese'] = df_mese['Mese'].astype(str)
    
    sns.barplot(data=df_mese, x='Mese', y='mae_globale', hue='Modello', palette=colori_globali)
    plt.title('Difficoltà di Predizione per Mese (Mesi con errore più alto = più difficili)', fontweight='bold', fontsize=14)
    plt.xlabel('Mese dell\'Anno')
    plt.ylabel('Errore Medio Assoluto (MAE %)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    save_plot("09b_errore_aggregato_per_mese.png")

# =========================================================================
# 6. MOTORE DI ESECUZIONE (MAIN)
# =========================================================================
if __name__ == "__main__":
    print("\n" + "#"*70)
    print(" INIZIO PREDIZIONE OCCUPAZIONE COLONNINE EV")
    print("#"*70)

    try:
        dataset_raw = carica_dataset_completo(FILE_NAME)
    except FileNotFoundError as e:
        print(f"\nERRORE: {e}")
        exit(1)

    # Estraiamo le stazioni presenti ma forziamo l'ID 2 (Citylife) in ultima posizione
    stazioni_disponibili = list(dataset_raw['id_stazione'].unique())
    if 2 in stazioni_disponibili:
        stazioni_disponibili.remove(2)
        stazioni_disponibili.append(2) 
    
    TEST_GIORNI_FUTURI = 2   
    STEP_MINUTI = 1440    # Avanzamento giornaliero di 24 ore (24 * 60)
    MAX_ITERAZIONI = 30   # Massimo 30 step in avanti (quindi test su 30 giorni consecutivi a blocchi di 2)
    
    PARAMETRI_SARIMA_STAZIONI = {i: {'order': (3, 0, 1), 'seasonal_order': (0, 1, 1, 48)} for i in range(1, 10)}
    
    inizio_globale = time.time()

    for st_id in stazioni_disponibili:
        print(f"\n{'='*70}\n--- PIPELINE: ZONA {st_id} ---\n{'='*70}")
        dataset_stazione = pre_processing_stazione(dataset_raw, st_id)

        if len(dataset_stazione) < 48 * 30:
            print(f" -> Zona {st_id} saltata (Meno di 30 giorni di storico continuo)")
            continue

        modelli = [
            (RandomForestWrapper(), f"risultati_RF_Staz_{st_id}.csv"),
            (NeuralProphetWrapper(epochs=50), f"risultati_NP_Staz_{st_id}.csv"),
            (StackedLSTMWrapper(epochs=20), f"risultati_LSTM_Staz_{st_id}.csv"), # LSTM Riattivato
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

    tempo_totale_secondi = int(time.time() - inizio_globale)
    print(f"\nELABORAZIONE COMPLETA. Tempo totale: {datetime.timedelta(seconds=tempo_totale_secondi)}")

    # =========================================================================
    # 7. PLOTTING 
    # =========================================================================
    print("\nGenerazione Grafici...")
    stazioni_trovate = set()
    for file_path in glob.glob(os.path.join(RESULTS_DIR, "*.csv")):
        match = re.search(r'_Staz_(\d+)\.csv', file_path)
        if match: stazioni_trovate.add(int(match.group(1)))
    
    stazioni_trovate = sorted(list(stazioni_trovate))

    if stazioni_trovate:
        plot_classifica_finale()
        salva_metriche_per_stazione(stazioni_trovate)
        plot_curve_singole_stazioni(stazioni_trovate, id_finestra=1)
        plot_decadimento_singole_stazioni(stazioni_trovate)
        
        analizza_decadimento_globale()
        analizza_momenti_dell_anno()
        
        print(f"\nGrafici generati in: {PLOTS_DIR}")
    else:
        print("\nNessun dato trovato per i grafici.")