# AI-Powered Wave Height Forecasting for the North Sea

A deep learning project that predicts ocean wave heights using neural networks and weather data.

## Project Overview

This project uses the NHITS architecture to forecast wave heights 96 hours == 4 days ahead for a point in the North Sea (56.0°N, 3.0°E). By combining weather data as pressure, wind speed with historical wave measurements, the model achieves prediction accuracy that could be useful for maritime operations.

**Best model performance: MAE 0.325m**\\

## Motivation

Wave height prediction is crucial for:
- Maritime navigation and safety
- Offshore operations planning
- Coastal infrastructure management

Traditional physics-based models are complex and computationally expensive. This project explores whether modern deep learning can provide fast, accurate predictions.

## Dataset

The project uses two open APIs for 2023 data:

**Weather Data** here we use Open-Meteo archive api:
- Atmospheric pressure at sea level
- Wind speed at 10m height
- Wind direction at 10m in degrees

**Marine Data** here we use open-meteo marine api:
- Wave height (meters) - our target variable

All data is hourly resolution for the full year 2023, with the last 96 hours reserved for testing.

## Experiments Conducted

### 1. Baseline model (MAE: 0.377m)
Simple NHITS with basic features
- **Input size:** 48 hours
- **Horizon:** 96 hours
- **Loss function:** MAE
- **Training steps:** 2000
- **Features:** pressure, wind_speed
- **Result:** Established baseline performance

<img width="1004" height="564" alt="image" src="https://github.com/user-attachments/assets/2ff241a9-d853-46f6-aa6d-72ea5a56f56d" />

### 2. Optimized Loss Function - Huber Loss (MAE: 0.325m)
**BEST PERFORMING MODEL** - Switched from MAE to Huber loss to handle outliers better.
- Added `wind_rolling` feature (3-hour moving average)
- **Training steps:** 4000
- **Features:** pressure, wind_speed, wind_rolling (3h MA)
- **Result:** Best test performance, robust to outliers while maintaining accuracy
- **Training loss:** 0.013-0.014

<img width="1005" height="566" alt="image" src="https://github.com/user-attachments/assets/fc630db0-afa6-4288-8eac-afc8f2b221df" />


This model achieved the lowest test MAE by combining robust loss function with engineered wind features.

### 3. Physics-informed features and ensemble (MAE: 0.334m)
Created a bagging ensemble of 3 models with added physics-based feature
- **New feature:** `wind_energy = wind_speed²` (kinetic energy proxy)
- **Ensemble:** 3 NHITS models with different random seeds (1, 42, 999)
- **Prediction:** Average of all 3 models
- **Training losses:** 0.113-0.127 (individual models)
- **Result:** Reduced model variance but slightly higher error than Huber approach

<img width="1007" height="566" alt="image" src="https://github.com/user-attachments/assets/b348230e-4447-4cfa-a0bb-194b372b27ce" />

The gray lines show individual model predictions, demonstrating how averaging reduces prediction noise.

### 4. Target smoothing (MAE: 0.327m)
Applied 3-hour centered rolling average to the target variable to reduce sensor noise
- **Rationale:** Real wave measurements contain high-frequency noise
- **Training loss:** 0.105-0.110
- **Test MAE:** 0.327m
- **Result:** Smoother predictions but measured against smoothed targets (not directly comparable to other experiments)

<img width="1008" height="564" alt="image" src="https://github.com/user-attachments/assets/52fcac02-37e0-46b0-bb41-f7ca00c1c55e" />

**Note:** This experiment smooths both the training target AND test target, so the MAE of 0.327m is against smoothed data. When tested against original unsmoothed data, MAE increases to 0.381m.

### 5. Random forest baseline (MAE: 0.332m)
Built a traditional ML model with manual lag features to understand wind-wave relationship.
- **Features:** Wind and pressure at lags [1, 3, 6, 12, 24 hours] + wind energy
- **Model:** Random Forest (200 trees, depth 15)

<img width="1027" height="565" alt="image" src="https://github.com/user-attachments/assets/78569c70-1485-4c9b-9ca2-17083066aa2f" />

### 6. Gradient boosting (LightGBM)
Tested gradient boosting as an alternative to neural networks.
- **Model:** LightGBM with automatic lag features [1, 2, 3, 6, 12, 24, 48]
- **Features:** Rolling means, hour/month cyclical features

<img width="982" height="485" alt="image" src="https://github.com/user-attachments/assets/71788e08-7f22-4f38-a780-974570d68547" />

### 7. Early warning system, lead time analysis
Built a specialized model to predict wave height 6 hours in advance for early warning applications
- **Model:** Gradient Boosting Regressor (200 trees, depth 5)
- **Target:** Wave height 6 hours in the future (shifted target)
- **New features:**
  - `pressure_drop`: 3-hour pressure differential (rapid drops indicate storms)
  - `wind_energy`: Wind speed squared (kinetic energy)
  - `wind_rolling`: 6-hour wind moving average
- **Purpose:** Give maritime operators reaction time before dangerous conditions arrive

<img width="1010" height="565" alt="image" src="https://github.com/user-attachments/assets/7a997e60-baa6-4f11-8992-85da70c0fbbb" />

This visualization shows the critical **lead time** - the gap between current conditions and what's coming 6 hours ahead. Red shaded areas indicate periods where the forecast gives advance warning of rising wave heights, allowing time for vessels to seek shelter or adjust operations.

## Final Results & Model Comparison

| Model | Test MAE (meters) | Training Time | Features | Complexity |
|-------|------------------|---------------|----------|------------|
| **NHITS + Huber** | **0.325** | High | pressure, wind_speed, wind_rolling | High |
| NHITS Smoothed | 0.327* | Medium | pressure, wind_speed | High |
| NHITS Ensemble | 0.334 | Very High | pressure, wind_speed, wind_energy | Very High |
| Random Forest | 0.332 | Low | wind/pressure lags, wind_energy | Low |
| NHITS Baseline | 0.377 | Medium | pressure, wind_speed | High |
| NHITS Smoothed (unsmoothed test) | 0.381 | Medium | pressure, wind_speed | High |

*smoothed model's 0.327m is measured against smoothed test data. Against original data it's 0.381m.

### Risk analysis visualization

The final visualization includes risk area detection for threshold-based alerts:

<img width="982" height="486" alt="image" src="https://github.com/user-attachments/assets/26de0273-3719-49e9-9501-c10309f72d23" />

Red shaded areas indicate where actual waves exceeded predictions - critical for safety-conscious applications like storm surge warnings.

## Key Learnings

1. **Loss function matters!** Huber loss outperformed MAE by 14% (0.325m vs 0.377m)
2. **Feature engineering** - Simple 3-hour rolling averages of wind improved predictions
3. **Ensemble methods don't always win** - Single well-tuned model beat the 3-model ensemble
4. **Smoothing is a double-edged sword** - Reduces noise but can hide extreme events
5. **Traditional ML remains competitive** - Random Forest (0.332m) nearly matched neural networks with much faster training
6. **Physics-informed features help** - Wind energy (v²) provides meaningful signal but wasn't enough to beat good engineering

## Why we chose NHITS?

NHITS (Neural Hierarchical Interpolation for Time Series) is perfect for this application because:
- Captures both short-term wind fluctuations and longer pressure systems
- Only 2.9-3.0M parameters compared to transformers
- Critical for real-time marine forecasting
- Handles missing data and irregular patterns well
---
