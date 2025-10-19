# ⚙️ GPU Time Prediction using Machine Learning  

## 🚀 Overview  
This project predicts **GPU computation time** based on system and kernel parameters using advanced regression models.  
Built from scratch with deep analytical rigor, it explores **1.3M+ hyperparameter combinations** and achieves a **stellar R² of 0.9994** with **Random Forest Regression**.  

---

## 🧠 Key Highlights  
- 📊 **Dataset Size:** 1,327,104 possible configurations  
- ⚡ **Models Tried:** Linear Regression, Decision Tree, Random Forest, SVM, XGBoost, Polynomial Regression  
- 🧪 **Best Model:** Random Forest  
  - MSE: **0.0007**  
  - R²: **0.9994**  
- ⏱️ **Total GPU Time Spent:** 210,261,535 ms (~58.4 hours)  
- 🔍 **Estimated Time for Remaining Grid Search:** ~94 hours  
- 🧰 **Tech Stack:** Python, NumPy, Pandas, Matplotlib, Scikit-Learn  

---

## 🧩 Problem Formulation  
Predict GPU execution time for unseen configurations, enabling smarter **resource scheduling**, **cost optimization**, and **hardware efficiency** in large-scale ML environments.  

---

## 💡 Features Used  
Each feature corresponds to hardware or kernel parameters that define the computational workload and memory access patterns:  

- **MWG, NWG:** 2D tiling at workgroup level  
- **KWG:** Inner dimension of workgroup tiling  
- **MDIMC, NDIMC:** Local workgroup sizes  
- **MDIMA, NDIMB:** Local memory shapes  
- **KWI:** Kernel loop unrolling factor  
- **VWM, VWN:** Vector widths for loading/storing  
- **STRM, STRN:** Stride flags for memory access  
- **SA, SB:** Caching flags for 2D tiles  

Together, these parameters form the core configuration influencing GPU runtime.  

---

## 🧪 Results Summary  

| Model | MSE | R² | Remarks |
|:------|:----|:---|:---------|
| Linear Regression | 0.548 | 0.5608 | Poor generalization |
| Decision Tree | 0.00104 | 0.9991 | Strong, slightly overfit |
| Random Forest | **0.0007** | **0.9994** | Optimal balance of bias-variance |

---

## 🔥 What Makes This Project Stand Out  
- Explored **>1M parameter combinations**  
- Analytical runtime estimation using **log-transformed regression**  
- Computation-aware modeling with real-time scaling  
- Built with **minimal resources, maximum performance** attitude 💪  

---

## 🧭 Future Work  
- Integrate **SHAP** for feature interpretability  
- Add **parallelized hyperparameter tuning**  
- Experiment with **neural networks** and **AutoML frameworks**  
- Develop a **web interface** for live GPU time prediction  

---

## 💬 Closing Note  
> “This isn’t just a regression model — it’s a statement on efficiency, precision, and persistence.” ⚡  

---

## 📁 Run Locally  

```bash
git clone https://github.com/<your-username>/gpu-time-prediction.git
cd gpu-time-prediction
pip install -r requirements.txt
python main.py
