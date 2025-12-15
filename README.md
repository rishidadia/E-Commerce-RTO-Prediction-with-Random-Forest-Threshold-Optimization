# E-Commerce-RTO-Prediction-with-Random-Forest-Threshold-Optimization
Predicting e-commerce Return-to-Origin (RTO) risk using Random Forest, threshold optimization, and SHAP-based explainability.

# E-Commerce RTO Prediction with Random Forest

This project builds a machine learning pipeline to predict **Return-to-Origin (RTO)** risk for e-commerce orders before delivery.  
The focus is on **real-world ML practices** such as leakage detection, threshold optimization, and explainability, rather than inflated accuracy.

---

## Problem Statement
Given order-level attributes (payment method, discount, category, location, time), predict whether an order will result in **RTO**.

Reducing RTO helps lower:
- reverse logistics costs  
- inventory blocking  
- delivery inefficiencies  

---

## Key Insights (Directional Patterns)
- **Cash-on-Delivery (COD) orders show higher RTO risk**, likely due to lower delivery commitment.
- **High discounts increase RTO likelihood non-linearly**, especially when combined with COD.
- **Certain product categories exhibit structural RTO risk**, driven by size/fit or quality expectations.
- **Geographic context matters more than order value alone**, highlighting operational challenges.

These insights were validated using **SHAP explainability**.

---

## Model & Approach
- **Model:** RandomForestClassifier  
- **Encoding:** One-hot encoding  
- **Tuning:** RandomizedSearchCV  
- **Decision Rule:** Custom probability threshold (not default 0.5)  
- **Explainability:** SHAP (global + local)

The model outputs probabilities, which are converted into final predictions using a **business-aligned threshold**.

---

## Evaluation Results (Test Set)
After removing leakage and tuning the decision threshold:

- **ROC-AUC:** ~0.65
- **Accuracy:** ~0.68 
- **F1 (Returned / RTO):** ~0.60
- **Recall (Returned / RTO):** ~0.60

These results reflect a **challenging, low-signal problem** and demonstrate honest model behavior rather than overfitting.

---

## Key Learnings
- Feature engineering matters more than hyperparameter tuning  
- Threshold tuning exposes weak models early  
- Static order-level features alone provide limited signal  
- Interpretability is critical for operational decision-making  

---

## Future Improvements
- Add customer-level and location-level historical aggregates  
- Introduce time-based behavioral features  
- Compare with gradient boosting models (XGBoost / LightGBM)  
- Deploy as an internal logistics risk-scoring tool  

---

## Disclaimer
This project demonstrates methodology and decision logic using anonymized or public data and is intended for learning and portfolio purposes.
