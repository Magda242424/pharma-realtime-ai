# PHARMA – Real-Time AI Platform

A real-time artificial intelligence platform for pharmaceutical process monitoring, anomaly detection, dissolved oxygen forecasting, streaming inference, and operational visualization.

The project combines machine learning, real-time event processing, REST API serving, experiment tracking, monitoring dashboards, and Power BI analytics.

> **Project status:** Fully implemented and validated locally.
> **Azure deployment:** Target architecture designed but not deployed due to the absence of an Azure subscription.

---

## 1. Project Overview

The objective of this project is to design and implement a real-time AI pipeline capable of monitoring pharmaceutical process data and producing machine learning predictions.

The platform processes sensor measurements such as:

* Temperature
* pH
* Dissolved oxygen
* Agitation speed
* Pressure
* Flow rate

Two machine learning tasks are implemented:

1. **Anomaly detection** using Isolation Forest
2. **Dissolved oxygen forecasting** using Random Forest Regressor

The resulting predictions are exposed through a REST API, processed through a streaming inference pipeline, stored as prediction events, and visualized through Streamlit and Power BI.

---

## 2. Architecture

The local architecture is designed as a cloud-ready architecture.

```text
                    PHARMA REAL-TIME AI

       ┌───────────────────────┐
       │  Simulated IoT Data   │
       │  producer.py          │
       └───────────┬───────────┘
                   │
                   ▼
       ┌───────────────────────┐
       │ Streaming Layer       │
       │ Python event stream   │
       └───────────┬───────────┘
                   │
                   ▼
       ┌───────────────────────┐
       │ ML Inference          │
       │ Isolation Forest      │
       │ Random Forest         │
       └───────────┬───────────┘
                   │
          ┌────────┴────────┐
          ▼                 ▼
 ┌─────────────────┐ ┌─────────────────┐
 │ FastAPI         │ │ Prediction      │
 │ REST API        │ │ JSONL / CSV     │
 └────────┬────────┘ └────────┬────────┘
          │                   │
          ▼                   ▼
 ┌─────────────────┐ ┌─────────────────┐
 │ Streamlit       │ │ Power BI        │
 │ Monitoring      │ │ Analytics       │
 └─────────────────┘ └─────────────────┘
```

A corresponding Azure target architecture was also designed using Azure IoT Hub, Azure Event Hubs, Azure Stream Analytics, Azure Machine Learning, Azure Data Lake Storage Gen2, Power BI, Azure Monitor and API Management.

The Azure architecture is a **target architecture only** and was not deployed during the project.

---

## 3. Machine Learning

### Anomaly Detection

The anomaly detection component uses:

**Isolation Forest**

The model analyses the selected process and lag features and identifies observations that differ from normal operating behaviour.

### Dissolved Oxygen Forecasting

The forecasting component uses:

**Random Forest Regressor**

The model predicts the next dissolved oxygen value based on the selected process variables and historical lag features.

---

## 4. Streaming Pipeline

The streaming layer simulates real-time sensor events.

The main components are:

```text
src/streaming/producer.py
src/streaming/inference.py
src/streaming/consumer.py
```

### Producer

`producer.py` generates simulated pharmaceutical process events.

Each event contains equipment information, timestamps, process measurements and dissolved oxygen lag features.

### Inference

`inference.py` validates each event and performs:

* anomaly prediction;
* anomaly scoring;
* anomaly classification;
* dissolved oxygen forecasting.

### Consumer

`consumer.py` persists prediction events in JSON Lines format.

Output:

```text
exports/stream_predictions.jsonl
```

---

## 5. REST API

The project exposes the machine learning functionality through FastAPI.

Main endpoints include:

```text
GET /
GET /health
POST /predict
```

Example health response:

```json
{
  "status": "healthy",
  "anomaly_model": true,
  "forecasting_model": true
}
```

Example prediction output contains:

```text
anomaly_prediction
is_anomaly
anomaly_score
predicted_dissolved_oxygen
```

---

## 6. Monitoring Dashboard

A Streamlit dashboard provides operational monitoring of the platform.

The dashboard displays:

* total analysed rows;
* number of anomalies;
* number of normal observations;
* anomaly rate;
* model availability;
* streaming status;
* latest streaming prediction.

---

## 7. Power BI Dashboard

Prediction data is exported for Power BI through:

```text
exports/powerbi_predictions.csv
```

The Power BI dashboard contains:

* Total Events
* Anomalies
* Anomaly Rate
* Average Anomaly Score
* Average Predicted Dissolved Oxygen
* Normal vs Anomalous Events
* Events by Equipment
* Anomaly Score Over Time
* Predicted Dissolved Oxygen Over Time
* Predicted Dissolved Oxygen by Equipment

Power BI Desktop was used locally to build the analytical dashboard.

---

## 8. Experiment Tracking

MLflow is used locally to track machine learning experiments.

The tracked experiment includes parameters and metrics for the training run.

Example metrics include:

```text
MAE
RMSE
```

The experiment also records the model configuration, including:

```text
IsolationForest
RandomForestRegressor
```

---

## 9. Results

The complete pipeline was validated locally.

### Automated tests

```text
30 passed
```

The test suite covers:

* API endpoints
* data availability
* feature contracts
* machine learning models
* model prediction
* streaming producer
* streaming inference
* streaming consumer
* end-to-end streaming

### Batch inference

The streaming generator produced:

```text
4,910 events
```

The complete inference pipeline processed all:

```text
4,910 events
```

Results:

```text
Anomalies:           99
Normal events:       4,811
Anomaly rate:        2.02%
Average anomaly score: 0.0775
Average predicted DO: 7.5233
```

The results were exported to:

```text
exports/powerbi_predictions.csv
```

---

## 10. Project Structure

```text
pharma/
│
├── src/
│   ├── api/
│   │   └── main.py
│   │
│   ├── dashboard/
│   │   ├── app.py
│   │   └── export_powerbi.py
│   │
│   ├── features/
│   │
│   ├── models/
│   │
│   └── streaming/
│       ├── producer.py
│       ├── inference.py
│       └── consumer.py
│
├── tests/
│   ├── test_api.py
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_ml.py
│   └── test_streaming.py
│
├── data/
│   ├── raw/
│   ├── clean/
│   └── features/
│
├── models/
│   ├── anomaly_model.joblib
│   └── forecasting_model.joblib
│
├── exports/
│   ├── stream_predictions.jsonl
│   └── powerbi_predictions.csv
│
├── notebooks/
│
├── requirements.txt
└── README.md
```

---

## 11. Local Installation

Clone the repository:

```bash
git clone <repository-url>
cd pharma
```

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 12. Running the API

Start FastAPI with:

```powershell
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

The API is then available locally at:

```text
http://127.0.0.1:8000
```

---

## 13. Running the Dashboard

Start Streamlit with:

```powershell
streamlit run src/dashboard/app.py --server.port 8502
```

The dashboard can then be accessed locally through the Streamlit URL displayed in the terminal.

---

## 14. Running the Tests

Run the complete test suite:

```powershell
python -m pytest tests -v
```

Expected result:

```text
30 passed
```

---

## 15. Azure Target Architecture

The project was designed with a possible Azure industrialisation path.

| Local component         | Azure target                                 |
| ----------------------- | -------------------------------------------- |
| `producer.py`           | Azure IoT Hub                                |
| Python streaming        | Azure IoT Hub / Event Hubs                   |
| Streaming processing    | Azure Stream Analytics                       |
| Isolation Forest        | Azure Machine Learning                       |
| Random Forest Regressor | Azure Machine Learning                       |
| `.joblib` models        | Azure ML Model Registry                      |
| FastAPI                 | Azure ML endpoint / API Management           |
| `exports/`              | Azure Data Lake Storage Gen2                 |
| Streamlit monitoring    | Cloud monitoring application / Azure Monitor |
| Power BI Desktop        | Power BI                                     |
| MLflow                  | Azure Machine Learning / MLflow              |

This architecture represents a **proposed cloud deployment architecture**. No Azure resources were provisioned for this project.

---

## 16. Limitations

The main limitation is the absence of an Azure subscription during development.

Consequently:

* no Azure resources were provisioned;
* no cloud endpoint was deployed;
* no Azure IoT Hub was connected to physical equipment;
* no Azure Machine Learning endpoint was deployed;
* no Azure Data Lake was used for production storage.

Instead, all functional components were implemented and validated locally.

This approach allowed the complete data and machine learning pipeline to be tested before a potential cloud deployment.

---

## 17. Future Improvements

Possible future extensions include:

* deployment to Azure Machine Learning;
* integration with real IoT devices;
* Azure IoT Hub ingestion;
* Azure Event Hubs for scalable event streaming;
* Azure Stream Analytics processing;
* Azure Data Lake Storage Gen2;
* automated CI/CD deployment;
* model versioning and monitoring;
* real-time Power BI integration;
* production-grade authentication and API security.

---

## 18. Conclusion

PHARMA Real-Time AI demonstrates an end-to-end machine learning architecture for pharmaceutical process monitoring.

The project integrates data generation, feature engineering, anomaly detection, forecasting, streaming inference, REST API serving, experiment tracking, operational monitoring and Business Intelligence.

The complete local implementation has been tested successfully, with **30 automated tests passing** and **4,910 events processed through the inference pipeline**.

The resulting architecture is designed to be transferable to Microsoft Azure when the required cloud resources become available.
