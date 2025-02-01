# DARE Wearable Data Processing 
This repository contains pipelines for processing wearable sensor data of the DARE (Digital Lifelong Prevention) project. 
Functionalities across three domains:
- **Sleep**
- **Heart Rate**
- **Physical Activity**

## Repository Structure

```
wearable-data-processing/
│── sleep/                          # Sleep processing pipelines
│   ├── GGIR.R                      # GGIR call to process raw accelerometer data
│   ├── process_GGIR_output.py      # Extract sleep-related features from GGIR output
│   ├── utils.py                    # Helper functions
│
│── heart_rate/                     # Heart rate processing pipelines
│   ├── preprocessing.py            # Preprocess HR data
│   ├── feature_extraction.py       # Extract HRV features
│   ├── models.py                   # HR analysis models
│   ├── utils.py                    # Helper functions
│
│── physical_activity/              # Physical activity pipelines
│   ├── preprocessing.py            # Preprocess movement data
│   ├── feature_extraction.py       # Extract activity features
│   ├── models.py                   # Activity classification models
│   ├── utils.py                    # Helper functions
│
│── common/                         # Shared functions across domains
│   ├── io.py                       # Data input/output handling
│   ├── visualization.py            # Plot functions
│
│── notebooks/                      # Jupyter Notebooks for analysis
│── docs/                           # Documentation
│── tests/                          # Unit tests
│── README.md                       # Project overview
│── requirements.txt                # Dependencies
│── setup.py                        # Python package setup
│── .gitignore                      # Files to ignore in Git
```
