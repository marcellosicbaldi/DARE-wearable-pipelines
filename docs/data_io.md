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
EmbracePlus `.bin` files contain raw **PPG, ACC, EDA, temperature** data, as well as **systolic peaks time**

📌 **Usage Example**

Reading a single 30 min file:
```python
from data_io.embraceplus.read_avro import ReadEmpaticaAvro
empatica_reader = ReadEmpaticaAvro()
data = empatica_reader.read(file=avro_file)
print(data.keys())
```
dict_keys(['systolic_peaks', 'steps', 'acc', 'time', 'fs', 'bvp', 'time_temp', 'fs_temp', 'temp', 'time_eda', 'fs_eda', 'eda'])

---

### **:two: GENEActiv**
GENEActiv devices record **high-frequency accelerometer data** for sleep and activity analysis.  
📌 **Usage Example**
```python
from data_io.geneactiv.load_geneactiv import load_geneactiv_data

# Load and preview GENEActiv data
df = load_geneactiv_data("path/to/geneactiv_data.csv")
print(df.describe())
```

---

### **:three: RootiRx**
RootiRx devices provide **PPG-based HR and HRV measurements**.  
📌 **Usage Example**
```python
from data_io.rootirx.load_rootirx import load_rootirx_data

# Load and preview RootiRx data
df = load_rootirx_data("path/to/rootirx_data.hdf5")
df.plot()
```

---

### **:four: VeritySense (VEGA)**
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
- Convert raw data to **Parquet** for fast retrieval:
  ```python
  df.to_parquet("data.parquet")
  df = pd.read_parquet("data.parquet")
  ```

---

## ⚙ Future Enhancements
- ✅ **Unified Data Loader** across all devices.
- ✅ **Automatic Sensor Synchronization** for recordings using multi-sensor set-ups.

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
