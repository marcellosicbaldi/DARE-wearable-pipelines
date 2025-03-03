# Data Input/Output (`data_io`) Module

The `data_io` module is responsible for handling **raw wearable sensor data** from multiple devices.  
It provides **efficient methods** to load, preprocess, and structure data for downstream analysis.

---

## 📌 Features
✔ Supports multiple **wearable devices**:  
   - **EmbracePlus** (`embraceplus/`)
   - **GENEActiv** (`geneactiv/`)
   - **RootiRx** (`rootirx/`)
   - **VeritySense** recorded with **VEGA** (`veritysense/`)  
✔ Handles **multiple file formats** (`.bin`, `.avro`, `.csv`, `.json`).  
✔ Ensures **consistent timestamp formats** across devices.  
✔ Provides **structured DataFrames** for further processing.

---

## 📂 Folder Structure
```
data_io/
│── embraceplus/                # Read EmbracePlus data
│── geneactiv/                  # Read GENEActiv data
│── rootirx/                    # Read RootiRx data
│── veritysense/                # Read VeritySense recorded with VEGA
```

---

## 🚀 Supported Devices & Usage

### **:one: EmbracePlus (Empatica)**
EmbracePlus `.bin` files contain raw **PPG, ACC, EDA, and temperature** data.  
📌 **Usage Example**
```python
from data_io.embraceplus.load_embraceplus import load_embraceplus_data

# Load and preview EmbracePlus data
df = load_embraceplus_data("path/to/embraceplus_data.bin")
print(df.head())
```

---

### **2️⃣ GENEActiv**
GENEActiv devices record **high-frequency accelerometer data** for sleep and activity analysis.  
📌 **Usage Example**
```python
from data_io.geneactiv.load_geneactiv import load_geneactiv_data

# Load and preview GENEActiv data
df = load_geneactiv_data("path/to/geneactiv_data.csv")
print(df.describe())
```

---

### **3️⃣ RootiRx**
RootiRx devices provide **PPG-based HR and HRV measurements**.  
📌 **Usage Example**
```python
from data_io.rootirx.load_rootirx import load_rootirx_data

# Load and preview RootiRx data
df = load_rootirx_data("path/to/rootirx_data.hdf5")
df.plot()
```

---

### **4️⃣ VeritySense (VEGA)**
Polar VeritySense data recorded using VEGA software includes **HR and motion data**.  
📌 **Usage Example**
```python
from data_io.veritysense.load_veritysense import load_veritysense_data

# Load and preview VeritySense data
df = load_veritysense_data("path/to/veritysense_data.json")
df.info()
```

---

## 🛠 Handling Large Data Files
If datasets are **too large** for memory:
- Use **chunk-based processing** (`pandas.read_csv(..., chunksize=10000)`)
- Convert raw data to **HDF5 or Parquet** for fast retrieval:
  ```python
  df.to_parquet("data.parquet")
  df = pd.read_parquet("data.parquet")
  ```

---

## ⚙ Future Enhancements
- ✅ **Unified Data Loader** across all devices.
- ✅ **Automatic Sensor Synchronization** (e.g., aligning timestamps).
- ✅ **Improved File Format Support** (adding `.edf` for sleep EEG).

---

## 🎯 Summary
The `data_io` module provides **structured methods to load wearable device data** efficiently.  
It **standardizes sensor outputs**, making data **easier to analyze and visualize**.

🔹 **See also**: [API Reference](api_reference.md) for function details.  

---
```

---

## **📌 What’s Next?**
Would you like me to:
1️⃣ **Add an API Reference (`api_reference.md`)** detailing each function?  
2️⃣ **Improve data handling examples** (e.g., merging multimodal data)?  
3️⃣ **Generate an interactive documentation website** with `MkDocs` or `Sphinx`?  

Let me know! 🚀