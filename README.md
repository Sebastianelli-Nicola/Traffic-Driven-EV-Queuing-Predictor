# Study and Comparison of Techniques for Predicting the Occupancy Rate of Electric Vehicle Charging Stations 

[![UNIVPM Thesis](https://img.shields.io/badge/UNIVPM-Master's_Thesis_(110L)-blue.svg)](https://tesi.univpm.it/handle/20.500.12075/29223)
[![Unitesi Archive](https://img.shields.io/badge/Unitesi-Official_Handle-005596.svg)](https://tesi.univpm.it/handle/20.500.12075/29223)
[![Thesis PDF](https://img.shields.io/badge/Thesis_Full_Text-PDF_(ITA)-red.svg)](./Tesi_Sebastianelli_Nicola.pdf)
[![Presentation Slides](https://img.shields.io/badge/Defense_Slides-PDF_(ITA)-orange.svg)](./Presentazione_Tesi_Sebastianelli.pdf)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](#tech-stack--environment)

Official repository for the Master's Degree Thesis in **Computer and Automation Engineering** at **Università Politecnica delle Marche (UNIVPM)**, developed within the **SUPERSONIC** research project (*Charging policies and routing of electric vehicles for medium-range logistics towards an ecological transition*).

>  **Official Links & Documents:**
> *  **UNIVPM Institutional Archive (Unitesi):** [https://tesi.univpm.it/handle/20.500.12075/29223](https://tesi.univpm.it/handle/20.500.12075/29223)
> *  **Full Master's Thesis (PDF):** [Tesi_Sebastianelli_Nicola.pdf](./Tesi_Sebastianelli_Nicola.pdf)
> *  **Thesis Defense Presentation (Slides PDF):** [Presentazione_Tesi_Sebastianelli.pdf](./Presentazione_Tesi_Sebastianelli.pdf)
> *  **Companion Data Logger Repository:** [automated-ev-station-tracker](https://github.com/Sebastianelli-Nicola/automated-ev-station-tracker)

---

##  Project Overview & Research Objectives

The large-scale adoption of Electric Vehicles (EVs) in urban environments is hindered by **range anxiety** and uncertainty regarding charging station availability. However, directly forecasting the occupancy rate of individual charging stations is challenging due to the highly stochastic, discrete, and impulsive ("step-function") nature of human charging behavior.

This thesis designs, implements, and evaluates an end-to-end **Intelligent Transportation System (ITS)** predictive pipeline across **8 strategic urban zones in Milan, Italy**, structured into three sequential research phases:
1. **Macroscopic Baseline (Urban Road Traffic Forecasting):** Validating and tuning 6 forecasting architectures on continuous, highly periodic vehicular traffic flow data ($\Delta v$).
2. **Direct Domain Translation (EV Charging Occupancy Forecasting):** Evaluating the exact same architectures on discrete EV charging occupancy rates under dynamic sliding-window protocols (24-hour batch vs. 30-minute real-time nowcasting).
3. **Indirect Inference & Spatio-Temporal Causality:** Proving via **Spearman Rank Correlation**, **Cross-Correlation Time-Lags (0–3h)**, and **Granger Causality Tests** that local road traffic congestion acts as a statistically significant **leading indicator (1–2 hours ahead)** for EV charging station saturation.

---

##  End-to-End Architecture & Data Pipeline

<pre>
 [TomTom Traffic API] (10-min) ──┐
                                 ├──> [MySQL Server DB] ──> [EDA & Info Score] ──> [Gap-Filling & Time-Shifting]
 [TomTom EV API] (15-min RR)   ──┘                                                            │
                                                                                              ▼
 [Granger Causality & Lag Analysis] ◄── [MAE / RMSE / MAPE + Latency] ◄── [30-min Resampling & Sliding Window]
</pre>

### 1. Automated Data Acquisition & Fault-Tolerant Scraping
Real-world telemetry was continuously collected in **Milan** via **TomTom REST APIs** and stored in a normalized **MySQL** relational database:
* **Road Traffic Dataset (`traffico.csv` — >653,000 records):** Sampled every **10 minutes** across 9 monitoring stations (March 2025 – September 2026). To abstract from local road speed limits, the target congestion metric is defined as the speed differential:
  $$\Delta v_t = \max(0, \; v_{\text{free\_flow}, t} - v_{\text{current}, t}) \quad [\text{km/h}]$$
* **EV Charging Dataset (`stato_colonnine.csv` — >179,000 records):** Sampled every **15 minutes** within a **150-meter radius** of each traffic node (April – September 2026). Implemented a custom **Round-Robin API Key Rotation** engine in Python to handle HTTP `403/429` rate limits and network faults (`"Nessun connettore trovato"`).
* **Information Score Selection:** Formulated a normalized **Information Score (0–100)** combining mean occupancy, standard deviation, state-transition dynamism, and saturation penalties to select the optimal reference charging station for each urban zone (providers: *A2A*, *Plenitude On The Road*).

### 2. Monitored Urban Zones (Milan Case Study)

| Zone ID | Urban Area (Milan) | Latitude | Longitude | EV Provider | EV Info Score |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | **Bocconi** (*Via Ferdinando Bocconi*) | 45.447862 | 9.190672 | A2A | 54.74 |
| **2** | **Citylife** (*Via Senofonte*) | 45.475285 | 9.157795 | A2A | 0.00 *(Excluded in EV)* |
| **3** | **Viale Certosa** | 45.495426 | 9.140054 | A2A | 78.99 |
| **4** | **Tangenziale Ovest** | 45.429429 | 9.080622 | Plenitude | 5.51 |
| **5** | **Stazione Centrale 1** | 45.484478 | 9.204951 | A2A | 33.69 |
| **6** | **Stazione Centrale 2** *(Traffic only)* | 45.484552 | 9.204714 | — | — |
| **7** | **Navigli** | 45.452137 | 9.169741 | A2A | 18.45 |
| **8** | **Corso Buenos Aires** | 45.476918 | 9.197998 | A2A | 74.82 |
| **9** | **Porta Romana** | 45.454743 | 9.205632 | Plenitude | **80.92** |

### 3. Advanced Pre-Processing & Missing Data Reconstruction
To guarantee chronological continuity for autoregressive and recurrent models without flattening variance:
* **Traffic Dataset — Similarity-Based Sampling:** Reconstructed server maintenance gaps using a 4-step algorithm: local Grid Search optimization on pre/post-gap windows, bidirectional hierarchical scoring (`+100` exact hour, `+50` day of week, `+20` temporal recency), Gaussian noise injection calibrated on local standard deviation, and boundary linear smoothing (achieving monthly pattern correlation $r \ge 0.94$).
* **EV Dataset — 21-Day Time-Shifting & 7-Day Autocorrelation:** Addressed a 3-week API quota outage in July by shifting the post-interruption block backward by exactly **21 days (3 weeks)**, preserving circadian rhythms and weekday/weekend transitions, followed by 7-day lag imputation and discrete integer interpolation.

---

##  Forecasting Models & Experimental Protocol

All series were resampled to a unified **30-minute grid (48 slots/day)**. Six models spanning four methodological families were encapsulated in object-oriented Python **Wrapper classes** (enforcing `StandardScaler`/`MinMaxScaler` and `[0, 100]%` output clipping):

| Family | Model | Key Hyperparameters & Configuration |
| :--- | :--- | :--- |
| **Classical Statistical** | **SARIMA** | `order=(3,0,1)`, `seasonal_order=(0,0,1,48)`. Seasonal differencing forced to $D=0$ for numerical stability on zero-inflated night series. |
| **Additive / Hybrid** | **Prophet** | Daily and weekly Fourier seasonalities enabled; yearly seasonality disabled; L-BFGS optimizer. |
| **Neural Hybrid** | **NeuralProphet** | PyTorch AR-Net backend, `epochs=50`, explicit daily and weekly seasonalities. |
| **Ensemble ML** | **Random Forest** | `n_estimators=100`, `max_depth=10`, `n_jobs=-1`. Exogenous features: `hour`, `dayofweek`, `is_weekend`, `lag_1` (30m), `lag_2` (1h), `lag_48` (24h). |
| **Kernel ML** | **SVM (SVR)** | Radial Basis Function (`rbf`) kernel, $C=1.0$, $\epsilon=0.1$, `StandardScaler` normalization. |
| **Deep Learning** | **Stacked LSTM** | 48-slot look-back window (24h), 3 sequential layers (`64 -> 32 -> 16` units), `Dropout=0.2`, `epochs=20`, `batch_size=64`, `Adam` optimizer. |

### Dynamic Sliding Window Validation
Implemented via a memory-efficient Python generator (`sliding_window_generator` using `yield`) to prevent data leakage and combat concept drift, predicting a **48-hour future horizon** under two comparative update regimes:
1. **Daily Sliding Window (24h Step — Batch Production):** Weights updated once per day (`30 folds` on Traffic, `7 folds` on EV).
2. **High-Frequency Sliding Window (30-min Step — Real-Time Nowcasting):** Weights retrained at every new 30-minute sample (`1,440 folds` on Traffic, `336 folds` on EV).

---

##  Key Experimental Results

### 1. Urban Road Traffic Forecasting ($\Delta v$)
Macroscopic traffic flow is smooth and strongly autocorrelated. Non-linear ML models (**SVM** and **Random Forest**) achieved the best accuracy, keeping the global error around **1.06–1.07 km/h**.

| Model | Global MAE (24h Step) | Global MAPE (24h Step) | Offline Training Time | Online Inference Latency |
| :--- | :---: | :---: | :---: | :---: |
| **SVM (SVR)** | **1.06 km/h** | 25.8% | ~8.0 s | **< 0.5 ms/slot** |
| **Random Forest** | **1.07 km/h** | **22.1%** | **~0.4 s** | **~0.6 ms/slot** |
| **NeuralProphet** | 1.10 km/h | 23.4% | ~30.0 s | ~0.7 ms/slot |
| **Prophet** | 1.11 km/h | 23.2% | ~4.0 s | ~0.5 ms/slot |
| **Stacked LSTM** | 1.25 km/h | 24.3% | ~203.0 s (CPU) | ~76.0 ms/slot |
| **SARIMA** | 1.39 km/h | 30.5% | ~31.0 s | < 0.5 ms/slot |

### 2. EV Charging Station Occupancy Forecasting (%)
Due to the discrete step-function behavior of single charging sessions, additive and linear models smooth out predictions around the historical mean, whereas **SVM** and **Random Forest** react sharply to step transitions:

| Model | EV Global MAE (24h Step) | EV Global MAPE (24h Step) | EV Global MAE (30-min Step) | EV Global MAPE (30-min Step) |
| :--- | :---: | :---: | :---: | :---: |
| **SVM (SVR)** | **4.77 %** | **9.29 %** | **5.44 %** | **10.80 %** |
| **Random Forest** | 5.37 % | 9.38 % | 6.13 % | 10.83 % |
| **Prophet** | 13.67 % | 19.89 % | 15.51 % | 22.82 % |
| **NeuralProphet** | 15.42 % | 24.16 % | 15.28 % | 24.28 % |
| **SARIMA** | 16.73 % | 23.00 % | *Excluded (OOM)* | *Excluded (OOM)* |

>  **Architectural Engineering Takeaway (24h vs. 30-min Trade-off):**
> Retraining models every 30 minutes increased computation time from **21 hours to over 5.5 days** on Traffic (and from **36 minutes to >9.3 hours** on EV) while slightly **degrading accuracy** (SVM MAE rose from `4.77%` to `5.44%`) due to overfitting on short-term noise. **Nightly 24-hour batch retraining** with sub-millisecond online inference is proven to be the optimal, scalable architecture for Smart City deployments.

---

### 3. Spatio-Temporal Correlation & Granger Causality (Traffic $\rightarrow$ EV)
By aligning Traffic ($\Delta v$) and EV Occupancy on a 1-hour grid within each 150m zone:
* **Synchronous Correlation (Lag 0):** Statistically significant positive correlation ($p < 0.05$) in **6 out of 7 active zones**, peaking in **Zone 1 — Bocconi** ($\rho_{\text{Spearman}} = 0.569$, $r_{\text{Pearson}} = 0.518$, rising to $\rho = 0.685$ on weekends) and **Zone 8 — Corso Buenos Aires** ($\rho_{\text{Spearman}} = 0.422$, rising to $\rho = 0.478$ on weekdays).
* **Cross-Correlation Time-Lag (1h–3h):** In commuter hubs like **Zone 5 — Stazione Centrale**, Spearman correlation increases from `0.197` (Lag 0) to **`0.284` at a +1 hour lag**, proving that traffic build-up precedes charging saturation.
* **Granger Causality Validation:** Confirmed statistically significant predictive causality from road traffic to future EV occupancy across 1h, 2h, and 3h horizons in **Zone 1** ($p < 0.0041$) and **Zone 5** (peak significance at **+2 hours**, $p = 0.0031$).

---

##  Tech Stack & Environment

* **Language:** Python 3.10+
* **Data Manipulation & Storage:** `pandas`, `numpy`, `mysql-connector-python`, `requests`
* **Machine Learning & Deep Learning:** `scikit-learn` (*RandomForestRegressor, SVR, StandardScaler, MinMaxScaler*), `tensorflow` / `keras` (*Stacked LSTM*), `pytorch`
* **Time Series & Econometrics:** `statsmodels` (*SARIMAX, Granger Causality*), `prophet`, `neuralprophet`, `scipy.stats` (*Pearson, Spearman*)
* **Visualization:** `matplotlib`, `seaborn`

---

##  Author & Academic Supervision

* **Author:** **Nicola Sebastianelli** — [GitHub](https://github.com/Sebastianelli-Nicola)
* **Supervisors:** **Prof. Domenico Potena**, **Dr. Alice Moretti**
* **Institution:** Università Politecnica delle Marche (UNIVPM) — Department of Information Engineering (DII)
* **Degree:** M.Sc. in Computer and Automation Engineering (*Ingegneria Informatica e dell'Automazione*), A.Y. 2025–2026
