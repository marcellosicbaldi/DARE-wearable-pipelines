---

### **3️⃣ `docs/heart_rate.md` (Heart Rate Processing)**
```md
# Heart Rate Processing

This module handles heart rate and heart rate variability (HRV) computations.

## 📌 Features
- **HR Estimation**: Uses the **BeliefPPG** model to estimate HR from PPG.
- **HRV Analysis**: Computes **RMSSD, SDNN, and other HRV metrics**.
- **Artifact Detection**: Uses **Kubios algorithms** to detect artifacts in RR intervals.

## 🛠 Usage Example
```python
from heart_rate.hrv import compute_hrv_metrics

hr_data = compute_hrv_metrics("path/to/hr_data.csv")
print(hr_data)