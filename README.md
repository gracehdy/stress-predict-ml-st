# Student Stress Predictor

> Predicts stress level category (High / Medium / Low) for students based on lifestyle, academic, and mental health indicators.

**Task:** Classification

**Framework:** Pandas, SKLearn, XGBoost, Streamlit

---

## Results

During model development, we trained several other models for model selection. As a baseline, below are the metrics for all the models we trained:

| Model | Accuracy | F1-Macro | Time |
|--------|-------|-----|------|
| Logistic Regression | 0.9986 ± 0.0001 | 0.9986 ± 0.0001 | 36.8s |
| Random Forest | 0.9478 ± 0.0003 | 0.9481 ± 0.0003 | 1,317.8s |
| SVM (with 100k subsample) | 0.9681 ± 0.0014 | 0.9682 ± 0.0014 | 6,744.5s |
| LightGBM | 0.9907 ± 0.0002 | 0.9907 ± 0.0002 | 143.5s |
| XGBoost (best) | 0.9900 ± 0.0003 | 0.9900 ± 0.0003 | 242.4s |

The XGBoost model we chose has the following performance in the test set.

| Metric | Test |
|--------|------|
| Accuracy | 98.40% |
| F1-Macro | 98.41% |
| Precision | 98.41% |
| Recall | 98.40% |
| ROC-AUC | 99.94% |

The per-class performance of the chosen model on 200,000 test samples is documented below:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Low Stress | 0.99 | 0.98 | 0.99 | 66,667 |
| Moderate Stress | 0.97 | 0.98 | 0.98 | 66,667 |
| High Stress | 0.99 | 0.99 | 0.99 | 66,667 |
| Macro Avg | 0.98 | 0.98 | 0.98 | 200,000 |

Meanwhile, the performance of the chosen model on the test set is documented below:
| | Pred: Low | Pred: Moderate | Pred: High |
|-|-----------|----------------|------------|
| Actual: Low | **65,571** | 1,096 | 0 |
| Actual: Moderate | 360 | **65,511** | 796 |
| Actual: High | 0 | 940 | **65,726** |

> WARNING: `mental_health_index` contributes to 73.5% of feature importance. The variable was calculated from `stress_level`, which also constitutes the target variable. **Reported accuracy may be higher than actual results due to data leakage.**

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
- 17 out of 20 features are used for model training
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
- **Mental Health**: Mental health index, anxiety score, depression score, burnout score (Likert)
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

The XGBoost model we chose has the following parameters:

| Params | Values |
|--------|--------|
| `learning_rate` | 0.1 |
| `max_depth` | 6 |
| `n_estimators` | 100 |
| `eval_metric` | `mlogloss` |

XGBoost was selected for its sequential residual-based learning, which handles non-linear relationships in structured data more explicitly than Random Forest's parallel ensemble approach. SVM was eliminated early due to subsampling constraints (100K of 800K rows). LightGBM and XGBoost performed comparably (Δ accuracy < 0.1%); no strong metric-based justification distinguishes the two given the data leakage present in this dataset.

**Pretrained weights:** We store the resulting model in the file `model_stress.pkl` (see below)

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
├── model_stress.pkl      # Serialised, trained model
├── requirements.txt
└── scaler.pkl            # Serialised, fitted scaler instance

```

---

## Limitations

Throughout the course of the development of this project, we observe 4 main limitations:

### [CRITICAL] Data Leakage — `mental_health_index`
`mental_health_index` is computed from stress_level, which is also the source of the target variable (via quantile binning). This creates a circular dependency: 73.5% of the model's decisions rely on a feature that encodes the answer. True accuracy without this feature would be lower.

### [High]  Synthetic / Simulated Dataset
The Kaggle dataset (`student_mental_health_burnout_1M.csv`) appears to be synthetically generated. Near-perfect class balance and smooth distributions suggest data generation artifacts. Results may not generalize to real student populations.

### [Medium]  Target Construction via Quantile Binning
Stress categories are defined by tertile splits of a continuous score, not clinically validated thresholds. `High` stress in this dataset means top-33% `stress_level`, not a clinically significant condition. Labels have no ground-truth psychological validation.

### [Medium]  SVM Trained on Subsample Only
SVM was trained on 100K of 800K training rows due to computational cost. Its reported performance (96.8% CV accuracy) may not reflect full-dataset behavior and cannot be fairly compared to models trained on the complete data.

