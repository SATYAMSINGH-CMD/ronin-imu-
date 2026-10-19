# Production ML Lifecycle & Learning Roadmap

> **Core Philosophy**: A real production ML system does not stop at `train → evaluate → benchmark → Streamlit`. It continues across the entire lifecycle:
> $$\text{Data Ingestion} \rightarrow \text{Validation} \rightarrow \text{Leakage-Free Splitting} \rightarrow \text{Baseline} \rightarrow \text{HPO (Optuna)} \rightarrow \text{Regularization} \rightarrow \text{Calibration} \rightarrow \text{Threshold Tuning} \rightarrow \text{Error Analysis} \rightarrow \text{MLflow Registry} \rightarrow \text{FastAPI} \rightarrow \text{Docker} \rightarrow \text{CI/CD} \rightarrow \text{Cloud} \rightarrow \text{Monitoring} \rightarrow \text{Retraining}$$

---

## 1. First Understand What "Improving an ML Model" Actually Means

Right now the default thought process is often:
```
Try XGBoost ──► Performance bad? ──► Try LightGBM ──► Still bad? ──► Try Random Forest
```
That is **only model selection**.

A real ML workflow decomposes model performance across three core pillars:

```
                 MODEL PERFORMANCE
                        │
      ┌─────────────────┼─────────────────┐
      ↓                 ↓                 ↓
    DATA             FEATURES          MODEL
      │                 │                 │
 cleaning          engineering      hyperparameters
 labels            selection         regularization
 splitting         transformations   architecture
 leakage                             objective
                                          │
                                          ↓
                                    PREDICTIONS
                                          │
                         ┌────────────────┼───────────────┐
                         ↓                ↓               ↓
                     threshold        calibration      errors
                         │                │               │
                         └────────────────┴───────────────┘
                                          ↓
                                      evaluation
```

So when somebody says *"Tune the model"*, they might mean only one small part of this loop.

---

## 2. The Complete Production Lifecycle

```
RAW SIGNAL / DATA
        ↓
DATA INGESTION
        ↓
DATA VALIDATION
        ↓
DATA SPLIT (Leakage-Safe)
        ↓
PREPROCESSING
        ↓
FEATURE ENGINEERING
        ↓
BASELINE MODEL
        ↓
HYPERPARAMETER OPTIMIZATION (Optuna)
        ↓
CROSS-VALIDATION
        ↓
ERROR ANALYSIS
        ↓
MODEL SELECTION
        ↓
PROBABILITY CALIBRATION
        ↓
THRESHOLD OPTIMIZATION
        ↓
FINAL TEST (Frozen)
        ↓
MLFLOW EXPERIMENT TRACKING
        ↓
MODEL REGISTRY
        ↓
FASTAPI
        ↓
DOCKER
        ↓
GITHUB ACTIONS (CI/CD)
        ↓
AWS / CLOUD
        ↓
MONITORING
        ↓
DRIFT DETECTION
        ↓
RETRAINING
        ↓
NEW MODEL VERSION / ROLLBACK
```

---

## 3. STEP 1 — Define Exactly What the Model Does

Before training anything, write the contract:
- **Input**: Sensor signal or tabular measurements.
- **Output**: Binary classification (e.g., Stable vs Anomaly / Chatter).
- **Class Definition**:
  - Positive class = Event of Interest (e.g., Chatter / Failure / Fraud)
  - Negative class = Normal / Baseline (e.g., Stable / Non-fraud)

### Cost Asymmetry & Business Question
- **False Negative (FN)**: Event happened, but model predicted normal. (Machine damaged, scrap part produced, fraud missed).
- **False Positive (FP)**: Machine was fine, but model raised false alarm. (Operator pauses machine, check required).

This cost asymmetry dictates your metric strategy (e.g., whether to optimize Recall vs Precision).

---

## 4. STEP 2 — Define Metrics (Beyond Accuracy)

Don't just use accuracy. For imbalanced or critical binary classification problems:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **ROC-AUC**
- **PR-AUC**

### Confusion Matrix Basics:
```
                    ACTUAL
                 Positive  Negative
               ┌─────────┬─────────┐
PREDICT Positive│   TP    │   FP    │
               ├─────────┼─────────┤
PREDICT Negative│   FN    │   TN    │
               └─────────┴─────────┘
```
- **TP (True Positive)**: Model says Positive, Actually Positive.
- **TN (True Negative)**: Model says Negative, Actually Negative.
- **FP (False Positive)**: Model says Positive, Actually Negative (False alarm).
- **FN (False Negative)**: Model says Negative, Actually Positive (Missed detection).

---

## 5. Precision

Precision asks: *"When my model predicts positive, how often is it right?"*

$$\text{Precision} = \frac{TP}{TP + FP}$$

---

## 6. Recall

Recall asks: *"Of all actual positive events, how many did we catch?"*

$$\text{Recall} = \frac{TP}{TP + FN}$$

---

## 7. F1 Score

F1 combines Precision and Recall into a harmonic mean:

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

When someone says *"Optimize F1"*, they mean search for the model, features, and decision threshold that maximizes validation F1.

---

## 8. STEP 3 — Split Your Data Correctly (Prevent Data Leakage)

You must define what **unseen data** really means:
- Unseen observation window?
- Unseen workpiece / session?
- Unseen machine / tool?
- Unseen environment / dynamic setup?

If observations from the same machine/run are randomly shuffled into both Train and Test, the model simply memorizes the machine setup rather than learning transferable physics.
Use **GroupKFold** or **Leave-One-Group-Out (LODO)** to test real generalization.

---

## 9. STEP 4 — Create a Simple Baseline

Before hyperparameter tuning or complex models, establish an un-tuned baseline:
```python
model = XGBClassifier(random_state=42)
```
Now you have a reference benchmark:
$$\text{F1} = 0.89, \quad \text{Recall} = 0.87, \quad \text{Precision} = 0.91$$
Every subsequent change must answer: *"Did this actually improve upon the baseline?"*

---

## 10 & 11. STEP 5 — Hyperparameters (XGBoost Knobs)

- **Parameter**: Learned internally by the model from data (e.g., tree split values, weights).
- **Hyperparameter**: Set by the engineer prior to training.

Key XGBoost Knobs:
- `max_depth`: How deep trees grow (3 = shallow, 6 = moderate, 12 = deep/complex).
- `n_estimators`: Number of sequential boosting rounds/trees.
- `learning_rate` ($\eta$): Step size shrinkage per tree (smaller $\eta$ + more trees is often best).
- `subsample`: Fraction of rows randomly sampled per tree iteration.
- `colsample_bytree`: Fraction of features randomly sampled per tree.
- `min_child_weight`: Minimum sum of instance weight (hessian) needed in a child node to allow splitting.

---

## 12 & 13. Regularization

> **Regularization** = Penalty/restriction on unnecessary model complexity to prevent overfitting.

In XGBoost:
- `reg_alpha` $\rightarrow$ L1 regularization (promotes sparsity).
- `reg_lambda` $\rightarrow$ L2 regularization (shrinks weights smoothly).
- Indirect regularization: `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`.

---

## 14 & 15. Optuna (Automated Bayesian HPO)

Optuna systematically evaluates configurations across trials to maximize a specific objective:

```python
def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    model = XGBClassifier(**params, random_state=42)
    # Evaluate across GroupKFold CV
    return mean_cv_f1
```

---

## 16. STEP 6 — The Full Diagnostic Loop

Don't jump algorithms blindly. Ask:
1. Is the data clean and valid?
2. Are labels consistent?
3. Is the splitting strategy realistic?
4. Are features informative?
5. Are hyperparameters reasonable?
6. Is the model overfitting?
7. Is the threshold tuned?
8. Are probabilities calibrated?
9. Where are errors concentrated?

---

## 17. STEP 7 — Systematic Error Analysis

After generating cross-validated predictions, record:
- Sample ID
- Ground truth label
- Predicted probability
- Predicted class
- Operational condition metadata (e.g., speed, depth, tool ID, dataset group)

Slice and inspect **False Positives** and **False Negatives**. Discover which physical or domain regimes cause failures.

---

## 18 & 19. STEP 8 & 9 — Feature Pipeline & Ablation

- Maintain **one unified feature extraction function** across both training and inference.
- Run ablation experiments: All features, Selected features, Physics-only, Signal-only, Combined.

---

## 20. STEP 10 — Neural Network Regularization

For deep architectures / MLPs:
- **Dropout**: Randomly deactivates nodes during training.
- **Weight Decay**: L2 regularization on network weights.
- **Early Stopping**: Halts training when validation loss stops improving.

---

## 21, 22, 23 & 24. STEP 11 — Probability Calibration

Calibration asks: *"When the model outputs 0.90, does that event actually happen ~90% of the time?"*

- **Platt Scaling (Sigmoid)**: Learns a logistic transformation of raw scores.
- **Isotonic Regression**: Non-parametric isotonic (monotonic step) transformation.
- **Key Insight**: Calibration transforms scores into accurate probabilities; it does **not** change the underlying ranking.

---

## 25. STEP 12 — Operating Threshold Optimization

The default threshold $0.5$ is arbitrary.
Sweep thresholds from $0.01$ to $0.99$ to find the optimal cut-off based on business goals:
- Maximum $F_1$ score.
- Guaranteed Recall (e.g. $\ge 95\%$) while maximizing Precision.

---

## 26. STEP 13 — The Frozen Final Test Set

Never optimize on the test set.
```
TRAIN ──► VALIDATION / CV ──► HPO ──► CALIBRATION ──► THRESHOLD ──► FREEZE ──► FINAL TEST
```
The test set is touched once for unbiased validation.

---

## 27 & 28. STEP 14 & 15 — MLflow & Model Registry

- **MLflow Tracking**: Logs Git commit, parameters, CV metrics, artifacts, ROC curves.
- **MLflow Model Registry**: Versions models (`v1`, `v2`, `v3`) with lineage and deployment staging.

---

## 29. STEP 16 — FastAPI Production Serving

Expose model predictions via a REST microservice:
- `GET /health`
- `GET /model-info`
- `POST /predict` with Pydantic schema validation.

---

## 30 & 31. STEP 17 — Docker Containerization

Package Python, libraries, feature extractor, and model into a reproducible container image:
```dockerfile
# Build image
docker build -t ml-chatter-api:v1 .
# Run image
docker run -p 8000:8000 ml-chatter-api:v1
```

---

## 32, 33 & 34. STEP 18 & 19 — CI/CD with GitHub Actions

Automate quality and deployment on every git push:
```
git push ──► Lint & Unit Tests ──► Model Regression Check ──► Docker Build ──► Push to Registry
```

---

## 35 & 36. STEP 20 — Cloud Architecture (AWS / SageMaker)

- **S3**: Model artifacts and versioned data.
- **ECR**: Docker container repository.
- **EC2 / ECS / SageMaker**: Managed endpoint for real-time inference.
- **CloudWatch**: Logs, latency alarms, and system metrics.

---

## 37, 38 & 39. STEP 21 & 22 — Monitoring & Drift Detection

Monitor three distinct layers in production:
1. **Infrastructure**: Latency (p50, p95, p99), CPU, memory, error rates.
2. **Input Data Drift**: Feature distributions shifting over time (KS test, Wasserstein distance).
3. **Prediction Drift**: Ratio of positive predictions and confidence shift.

---

## 40 & 41. STEP 23 & 24 — Automated Retraining & Rollback

- Ingest new ground truth data.
- Train candidate model against production champion.
- Promote if performance criteria are met; roll back automatically if degradation is detected.

---

## 42. STEP 25 — Infrastructure as Code (Terraform)

Define cloud infrastructure (S3, ECR, IAM, ECS) declaratively in code (`main.tf`).

---

## Implementation Order

```
LEVEL 1: Improve the ML (Metrics, HPO, Regularization, Thresholds, Calibration, Error Analysis)
LEVEL 2: Experiment Engineering (MLflow Tracking & Model Registry)
LEVEL 3: Serving (FastAPI)
LEVEL 4: Packaging (Docker)
LEVEL 5: Automation (GitHub Actions CI/CD)
LEVEL 6: Cloud (AWS / SageMaker / IaC)
LEVEL 7: Production ML (Monitoring, Drift Detection, Retraining & Rollbacks)
```
