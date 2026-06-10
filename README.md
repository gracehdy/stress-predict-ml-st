# Student Stress Predictor

> Predicts stress level category (High / Medium / Low) for students based on lifestyle, academic, and mental health indicators.

**Task:** Classification

**Framework:** Pandas, SKLearn, LightGBM, Streamlit

---

## Results

During model development, we trained several other models for model selection. As a baseline, below are the metrics for all the models we trained:

| Model | Accuracy | F1-Macro | Time |
|--------|-------|-----|------|
| Logistic Regression |0.7559 ± 0.0017 | 0.7566 ± 0.0017 | 36.8s |
| Random Forest | 0.7470 ± 0.0012 | 0.7491 ± 0.0012 | 1,317.8s |
| SVM (with 100k subsample) | 0.7384 ± 0.0025 | 0.7256 ± 0.0032 | 6,744.5s |
| LightGBM | 0.7560 ± 0.0013 | 0.7578 ± 0.0013 | 143.5s |
| XGBoost (best) | 0.7559 ± 0.0015 | 0.7578 ± 0.0015| 242.4s |

The LightGBM model we chose has the following performance in the test set.

| Metric | Test |
|--------|------|
| Accuracy | 75.51% |
| F1-Macro | 75.69% |
| Precision | 75.93% |
| Recall | 75.51% |
| ROC-AUC | 90.73% |

The per-class performance of the chosen model on 200,000 test samples is documented below:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Low Stress | 0.83 | 0.80 | 0.81 | 66,667 |
| Moderate Stress | 0.63 | 0.67 | 0.65 | 66,667 |
| High Stress | 0.83 | 0.80 | 0.81 | 66,666 |
| Macro Avg | 0.76 | 0.75 | 0.76 | 200,000 |

Meanwhile, the performance of the chosen model on the test set is documented below:
| | Pred: Low | Pred: Moderate | Pred: High |
|-|-----------|----------------|------------|
| Actual: Low | **53,125** | 3,346 | 196 |
| Actual: Moderate | 11,056 | **44,774** | 10,837 |
| Actual: High | 238 | 13,353 | **53,075** |


---

## Setup

**Requirements:** Python `3.12+`

```bash
git clone https://github.com/gracehdy/stress-predict-ml-st
cd stress-predict-ml-st
pip install -r requirements.txt
```

---

## Data

Our dataset comes as a Kaggle dataset, titled [Student Mental Health and Burnout](https://www.kaggle.com/datasets/sharmajicoder/student-mental-health-and-burnout)

### EDA

The dataset has the following properties:
- Balanced classes through quantile binning (each consisting 33.3% of the dataset)
- 16 out of 20 features are used for model training
- 0 missing values in dataset

Additionally, we note following key correlations with the target variable:
- `mental_health_index` ↔ `stress_level`: r = −0.95
- `burnout_score` ↔ `stress_level`: r = +0.75
- `anxiety_score` ↔ `stress_level`: r = +0.76
- `sleep_hours` ↔ `stress_level`: r = −0.26 

### Format

The data contains 1,000,000 records, with a total of 20 features:
- **Demographic**: Age, Sex, Academic Year
- **Academic**: Hours studying per day, academic test pressure (Likert), academic performance (GPA)
- **Mental Health**: Mental_Health_Index, Anxiety score, depression score, burnout score (Likert)
- **Lifestyle**: Sleep duration, physical activity duration, daily screentime, daily internet use (hours)
- **Social/Economic**: Social support, financial stress, family expectations
- **Output**: Stress Category (High, Medium, Low)

### Splits

We train our model with Stratified K-Fold strategy, using `k=5`, splitting the dataset into:
 `80%` Train / `20%` Test 

---

## Usage

To launch the Streamlit app, run the following command:

```bash
streamlit run app.py
# or
docker build -t stress-predict . && docker run -p 8501:8501 stress-predict
```

---

## Model

The LightGBM model we chose has the following parameters:

| Params | Values |
|--------|--------|
| `verbosity` | -1 |
| `random_state` | 42 |
| `n_jobs` | -1 |
| `class_weight` | `balanced` |
| `max_depth` | 10 |
| ` learning_ratet` | 0.05 |
| `n_estimators` | 500 |


LightGBM was selected for its superior efficiency and ability to handle large, complex datasets, offering distinct advantages over Random Forest, SVM, and XGBoost. SVM was eliminated early due to subsampling constraints (100K of 800K rows). Random Forest, which builds trees independently and often consumes excessive memory, LightGBM utilizes a gradient-boosting approach that iteratively corrects errors, resulting in higher predictive accuracy for non-linear relationships.LightGBM and XGBoost performed comparably (Δ accuracy < 0.1%); however LightGBM was prioritized since it provided the most stable and robust decision boundaries for distinguishing nuanced "Moderate" stress levels in a resource-constrained deployment environment.

**Pretrained weights:** We store the resulting model in the file `model_stress3.pkl` (see below)

---

## Project Structure

```
.
├── .streamlit/           # Streamlit config directory      
│   └── config.toml
├── pages/                # Streamlit pages
│   ├── hasilprediksi.py
│   └── prediksi.py
├── .gitignore
├── README.md
├── api.py                # FastAPI app
├── app.py                # Streamlit entry point
├── Dockerfile 
├── model_stress3.pkl      # Serialised, trained model
├── requirements.txt
└── scaler2.pkl            # Serialised, fitted scaler instance

```

---

## Limitations

Throughout the course of the development of this project, we observe 4 main limitations:

### [High] Performance Gaps in 'Moderate' Class
There is a noticeable disparity in classification performance. While the `Low` and `High` classes achieve high Precision/Recall scores (0.80 – 0.83), the `Moderate` class lags behind at 0.63 – 0.67. This likely occurs because the feature space for the 'Moderate' category overlaps significantly with both 'Low' and 'High' boundaries, making it harder for the model to distinguish.

### [High]  Synthetic / Simulated Dataset
The Kaggle dataset (`student_mental_health_burnout_1M.csv`) appears to be synthetically generated. Near-perfect class balance and smooth distributions suggest data generation artifacts. Results may not generalize to real student populations.

### [Medium]  Target Construction via Quantile Binning
Stress categories are defined by tertile splits of a continuous score, not clinically validated thresholds. `High` stress in this dataset means top-33% `stress_level`, not a clinically significant condition. Labels have no ground-truth psychological validation.

### [Medium]  SVM Trained on Subsample Only
SVM was trained on 100K of 800K training rows due to computational cost. Its reported performance (73.84% CV accuracy) may not reflect full-dataset behavior and cannot be fairly compared to models trained on the complete data.

