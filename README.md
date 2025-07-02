# DeepMedScan: AI-Powered Chest X-Ray Classifier with Grad-CAM

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)


This project classifies chest X-rays as NORMAL or PNEUMONIA and explains its decision using Grad-CAM.

## Files Included

- `app.py`: Streamlit app for predictions and Grad-CAM visualization.
- `train_model.py`: Model training script using TensorFlow Functional API.
- `deepmedscan_model.h5`: Your trained Keras model.
- `README.md`: Instructions and overview.

## Expected Folder Structure

Before running the app or training, make sure you have:

deepmedscan/
├── app.py
├── train_model.py
├── deepmedscan_model.h5
├── README.md
└── data/
└── chest_xray/
├── train/
├── val/
└── test/


## Run It

```bash
streamlit run app.py


---

### ✅ Once You Have All Files

1. Create a folder named `deepmedscan/`
2. Place `app.py`, `train_model.py`, `deepmedscan_model.h5`, and `README.md` inside
3. Create your `data/chest_xray/train`, `val`, and `test` folders
4. Run the app!

---

Let me know if you’d like help pushing this to GitHub or deploying it live — you’re now fully project-ready. ​:contentReference[oaicite:0]{index=0}​
