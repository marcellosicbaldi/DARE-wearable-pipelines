# DARE Wearable Data Processing 
This repository contains pipelines for processing wearable sensor data of the DARE (Digital Lifelong Prevention) project. 
Functionalities across three domains:
- **Sleep**
- **Heart Rate**
- **Physical Activity**

## Repository Structure

```
wearable-data-processing/
│── io/                             # Data input/output handling
|   |── embraceplus/                # Read EmbracePlus data
|   |── geneactiv/                  # Read GENEActiv data
|   |── rootirx/                    # Read RootiRx data
|   |── veritysense/                # Read VeritySense recorded with VEGA
|
│── sleep/                          # Sleep processing pipelines
│   ├── GGIR.R                      # GGIR call to process raw accelerometer data 
│   ├── process_GGIR_output.py      # Extract sleep-related features from GGIR output
│   ├── utils.py                    # Helper functions
|   |── detect_acc_bursts.py        # Detect accelerometer bursts during the night
│
│── heart_rate/                     # Heart rate processing pipelines
|─  |── beliefppg/                  # HR estimation from PPG
|       |── ..                 
│   ├── extract_summary_metrics.py  # Aggregation of HR and HRV metrics
│   ├── heart_rate_fragmentation.py # Compute heart rate fragmentation
│   ├── hrv.py                      # Compute hrv (RMSSD, SDNN)
│   ├── ppg_beat_detection.py       # Extract systolic peaks and feet from PPG
|   |── kubios.py                   # Artifact detection in RR/PP intervals time series
│
│── physical_activity/              # Physical activity pipelines
│   WIP
│
|── nonwear/  
|   |── DETACH.py                   # Non-wear algorithm
|
|
│── visualization/                  # Plot functions (bokeh interactive viz)
│
│── notebooks/                      # Jupyter Notebooks - playground, data exploration
|
│── docs/                           # Documentation
|
│── tests/                          # Unit tests (@help wanted)

```
