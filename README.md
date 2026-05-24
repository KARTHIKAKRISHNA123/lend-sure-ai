<!-- Banner -->
<div align="center">

<h1>🏦 LendSure AI/h1>
<h3><em>LendSure AI — ML-Powered Loan Approval Intelligence Platform</em></h3>

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-1.x-013243?logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Deployed-HuggingFace%20Spaces-FFD21E?logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/Models-LR%20·%20KNN%20·%20NB-blueviolet" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

<p>
  <strong>End-to-end ML loan approval system</strong> with three trained classifiers, a full preprocessing pipeline, live risk scoring, and a polished Streamlit UI — deployed on Hugging Face Spaces.
</p>

</div>

---

## 📌 Problem Statement

Traditional loan approval processes rely heavily on manual assessment, prone to bias, inconsistency, and delays. Financial institutions need a transparent, data-driven system that evaluates loan eligibility based on applicant financial signals — without human subjectivity.

**CreditWise Loan System** addresses this by automating loan eligibility prediction using three scikit-learn classifiers trained on real applicant data, with a complete preprocessing pipeline and a production-grade Streamlit interface with live risk scoring and explainability.

---

## ✅ Solution Overview

A three-model ML system where each classifier is trained on the same deterministic pipeline:

```
Raw applicant data → Imputation → Encoding → Feature Engineering → Scaling → Classifier → Decision
```

The UI surfaces predictions with model confidence, a live risk scorecard, and actionable improvement tips — making the system interpretable to end users, not just data scientists.

---

## 🚀 Key Features

| Feature | Description |
|---|---|
| 🤖 **3 ML Models** | Logistic Regression (best overall), KNN, Naive Bayes (best precision) — switchable in UI |
| ⚡ **Live Risk Scorecard** | Real-time scoring of Credit Score, DTI, Income, Savings, Collateral as user inputs |
| 📊 **Model Comparison Dashboard** | Side-by-side accuracy, precision, F1 comparison of all three models |
| 🔧 **Feature Engineering** | Squared transformations for DTI Ratio and Credit Score for non-linear signal capture |
| 🎨 **Navy-Blue Branded UI** | Production-grade Streamlit UI with custom CSS, gradient header, pill badges |
| 💡 **Eligibility Tips** | Actionable suggestions for rejected applicants to improve their profile |
| 📈 **EDA Insights Page** | Credit band approval rates, dataset overview, key factor analysis |
| 📦 **Pickle Artifacts** | All preprocessing objects (scaler, encoder) and models serialized for exact reproduction |

---

## 🏗️ Overall Architecture

```mermaid
graph TB
  subgraph User["👤 User Layer"]
    U[Applicant Browser]
  end

  subgraph App["🌐 Streamlit App — app.py"]
    UI["UI Pages\nPredict · Insights · Models · About"]
    SIDE["Sidebar\nModel Selector · Nav"]
    RISK["Live Risk\nScorecard"]
  end

  subgraph Pipeline["⚙️ ML Pipeline Layer"]
    IMP["SimpleImputer\nmean + most_frequent"]
    LE["LabelEncoder\nEducation_Level, Loan_Approved"]
    OHE["OneHotEncoder\ndrop=first · 6 cols"]
    FE["Feature Engineering\nDTI_sq · Credit_Score_sq"]
    SC["StandardScaler\n27 features"]
  end

  subgraph Models["🤖 Model Layer"]
    LR["model_lr.pkl\nLogistic Regression\nAcc 87.5%"]
    KNN["model_knn.pkl\nKNN n=5\nAcc 75.5%"]
    NB["model_nb.pkl\nGaussian NB\nAcc 86.5%"]
  end

  subgraph Artifacts["📦 Serialized Artifacts"]
    SC_PKL["scaler.pkl"]
    ENC_PKL["encoder.pkl"]
    CSV["loan_approval_data.csv\n1,000 rows · 20 cols"]
    NB_IPYNB["Credit_Wise_Loan_Approval_System.ipynb"]
  end

  U -->|"Form Inputs"| UI
  UI --> SIDE
  UI --> RISK
  UI -->|"raw dict"| IMP
  IMP --> LE --> OHE --> FE --> SC
  SC -->|"27-col scaled array"| LR
  SC -->|"27-col scaled array"| KNN
  SC -->|"27-col scaled array"| NB
  LR -->|"predict + proba"| UI
  KNN -->|"predict + proba"| UI
  NB -->|"predict + proba"| UI
  Artifacts -->|"loaded at startup via @st.cache_resource"| Pipeline
```

---

## 🧠 System Architecture

```mermaid
graph LR
  subgraph INPUT["Input Layer"]
    RAW["Raw CSV\n1000 rows × 20 cols"]
    FORM["Streamlit Form\n18 input fields"]
  end

  subgraph PREPROCESS["Preprocessing Layer — Notebook Cells 12–65"]
    C12["Cell 12\nNumeric Imputer\nstrategy=mean"]
    C14["Cell 14\nCategorical Imputer\nstrategy=most_frequent"]
    C41["Cell 41\nDrop Applicant_ID"]
    C46["Cell 46\nLabelEncode\nEducation_Level + Loan_Approved"]
    C48["Cell 48\nOneHotEncode\n6 cols · drop=first"]
    C80["Cell 80\nFeature Engineering\nDTI_sq · Credit_sq\nDrop originals"]
    C65["Cell 65\nStandardScaler\n→ 27 features"]
  end

  subgraph TRAIN["Training Layer — Cells 69–86"]
    LR2["LogisticRegression\ndefault hyperparams"]
    KNN2["KNeighborsClassifier\nn_neighbors=5"]
    NB2["GaussianNB\ndefault"]
    EVAL["Precision · Recall\nF1 · Accuracy · CM"]
  end

  subgraph SERVE["Serving Layer — app.py"]
    LOAD["@st.cache_resource\nLoad pkl artifacts"]
    PROC["preprocess() function\nMirrors notebook pipeline exactly"]
    PRED["model.predict()\nmodel.predict_proba()"]
    DISP["Result UI\nApproved · Rejected · Confidence · Scorecard"]
  end

  RAW --> C12 --> C14 --> C41 --> C46 --> C48 --> C80 --> C65
  C65 --> LR2 & KNN2 & NB2
  LR2 & KNN2 & NB2 --> EVAL
  LR2 -->|"pickle.dump"| TRAIN
  FORM --> LOAD --> PROC --> PRED --> DISP
```

---

## 🧰 Technology Stack — Complete Breakdown

> Every library, framework, tool, and artifact used across all layers of the system.

### Core Stack

| Technology | Version | Category | Purpose in Project | Why Chosen | Key Features Used |
|---|---|---|---|---|---|
| **Python** | 3.10+ | Language | Entire ML pipeline and Streamlit app | Universal ML ecosystem, Streamlit compatibility | f-strings, pathlib, dataclasses |
| **Streamlit** | 1.x | UI Framework | Multi-page web app with sidebar nav, forms, metrics | Fastest way to deploy ML apps without frontend code | `st.cache_resource`, `st.columns`, `st.progress`, `st.dataframe`, custom CSS injection |
| **scikit-learn** | 1.x | ML Framework | All preprocessing and model training/inference | Industry standard, unified API across all classifiers | `SimpleImputer`, `LabelEncoder`, `OneHotEncoder`, `StandardScaler`, `LogisticRegression`, `KNeighborsClassifier`, `GaussianNB`, `train_test_split`, confusion_matrix, classification_report |
| **pandas** | 2.x | Data Manipulation | DataFrame operations, OHE column merging, CSV loading | Vectorized operations, seamless scikit-learn integration | `pd.read_csv`, `pd.concat`, `pd.DataFrame`, `df.drop`, `df.select_dtypes`, `df.reindex` |
| **NumPy** | 1.x | Numerical Computing | Array operations, squared feature computation | Foundation for all scikit-learn data structures | `np.ndarray`, `np.log1p` (explored), squared transforms |
| **matplotlib** | 3.x | Visualization | EDA plots in notebook (histograms, boxplots) | Standard Python plotting, seaborn backend | `plt.figure`, `plt.pie`, `plt.show` |
| **seaborn** | 0.x | Statistical Viz | Distribution and correlation analysis in notebook | High-level API on matplotlib, beautiful defaults | `sns.barplot`, `sns.histplot`, `sns.boxplot`, `sns.heatmap` |
| **pickle** | stdlib | Serialization | Serialize all trained models and preprocessing artifacts | Zero-dependency, built-in Python serialization | `pickle.dump`, `pickle.load` for 5 artifacts |
| **pathlib** | stdlib | File I/O | Cross-platform path handling for pkl file loading | More Pythonic than os.path, works on HF Spaces | `Path.exists()` for graceful missing-artifact handling |

### ML Artifacts (Serialized)

| Artifact | Type | Purpose | Fitted On |
|---|---|---|---|
| `model_lr.pkl` | `LogisticRegression()` | Primary classifier — best overall | `x_train_scaled` (800 rows × 27 cols) |
| `model_knn.pkl` | `KNeighborsClassifier(n_neighbors=5)` | Pattern-similarity classifier | `x_train_scaled` (800 rows × 27 cols) |
| `model_nb.pkl` | `GaussianNB()` | Best precision — minimizes false approvals | `x_train_scaled` (800 rows × 27 cols) |
| `scaler.pkl` | `StandardScaler` | Feature normalization (exact 27-col order) | `x_train` (27 features, post-engineering) |
| `encoder.pkl` | `OneHotEncoder(drop='first', sparse_output=False)` | Categorical encoding for 6 columns | Full categorical block pre-split |

### Dataset

| Property | Value |
|---|---|
| File | `loan_approval_data.csv` |
| Rows | 1,000 |
| Raw Features | 20 columns |
| Final Features | 27 (post-engineering + OHE expansion) |
| Target | `Loan_Approved` (binary: Yes/No) |
| Class Distribution | ~29.8% Approved / ~65.2% Rejected (imbalanced) |

---

## 🔄 Request Lifecycle

### Flow 1: Loan Eligibility Prediction

```
1. USER INTERACTION
   └── Applicant fills 18 input fields across 3 sections:
       → Applicant Profile (age, gender, marital, dep, edu, emp, employer)
       → Financial Details (income, co-income, savings, credit_score, dti, ex_loans)
       → Loan & Property (loan_amount, loan_term, purpose, collateral, property_area)
       → Clicks: "🔍 Predict Loan Eligibility"

2. STREAMLIT EVENT HANDLING
   └── predict_btn → True
       → Builds raw dict of 18 key-value pairs
       → Selects pkl path from MODEL_OPTIONS dict based on sidebar radio

3. ARTIFACT LOADING (cached)
   └── @st.cache_resource: load_scaler() → scaler.pkl
       @st.cache_resource: load_encoder() → encoder.pkl
       @st.cache_resource: load_model(sel_pkl) → model_lr/knn/nb.pkl
       → Cached after first load — zero reload cost on re-predictions

4. PREPROCESSING — preprocess(raw, scaler, encoder)
   ├── LabelEncode Education_Level: Graduate=0, Not Graduate=1
   ├── Build numerical dict (12 features including DTI_sq, Credit_sq)
   │     DTI_Ratio_sq  = raw["DTI_Ratio"] ** 2     ← Cell 80
   │     Credit_Score_sq = raw["Credit_Score"] ** 2  ← Cell 80
   │     Credit_Score, DTI_Ratio originals → NOT included
   ├── Build categorical DataFrame for 6 OHE columns
   ├── encoder.transform(cat_df) → OHE array (15 columns, drop=first)
   ├── Concatenate num_df + ohe_df → 27-column DataFrame
   └── full_df.reindex(EXACT_COLS) → ensures column order matches scaler
       └── scaler.transform(full_df) → normalized 27-col array

5. MODEL INFERENCE
   └── model.predict(X)        → binary label (0=Rejected, 1=Approved)
       model.predict_proba(X)  → confidence score (max of class probabilities)

6. RESULT RENDERING
   └── Approved → green .res-ok div with ✅
       Rejected → red .res-no div with ❌
       Confidence → st.progress bar + percentage
       Live Risk Scorecard → 5 metrics (Credit, Income, Savings, DTI, Collateral)
                             each scored 0–100 and rendered as st.progress bars
```

### Flow 2: Live Risk Scorecard (Real-Time)

```
1. USER ADJUSTS any input widget (no button click needed)
   └── Streamlit re-runs app.py top-to-bottom on each interaction

2. SCORECARD COMPUTATION (no ML inference — pure arithmetic)
   └── cr_pct   = (credit_score - 300) / 600 × 100
       inc_pct  = min(income / 200_000 × 100, 100)
       sav_pct  = min(savings / 500_000 × 100, 100)
       dti_pct  = max(0, (1 - dti) × 100)       ← inverted: low DTI = good
       coll_pct = min(collateral / loan_amt × 100, 100)

3. SCORECARD RENDER
   └── 5 × [label row + st.progress bar] → instant visual feedback
```

---

## 📡 Data Flow Explanation

```
loan_approval_data.csv
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│                  NOTEBOOK PIPELINE                       │
│                                                         │
│  Raw (1000 × 20)                                        │
│      │                                                  │
│      ├── SimpleImputer (mean)      → numeric columns    │
│      ├── SimpleImputer (mode)      → categorical cols   │
│      ├── Drop Applicant_ID         → 1000 × 19          │
│      ├── LabelEncode (2 cols)      → Education, Target  │
│      ├── OneHotEncode (6 cols)     → +15 cols, -6 cols  │
│      │                             → 1000 × 28          │
│      ├── Feature Eng (Cell 80)     → +DTI_sq, +Cr_sq    │
│      │                               -DTI_orig, -Cr_orig│
│      │                             → 1000 × 28          │
│      ├── Drop Loan_Approved        → X: 1000 × 27       │
│      ├── train_test_split 80/20    → X_train: 800 × 27  │
│      │                               X_test:  200 × 27  │
│      └── StandardScaler           → mean=0, std=1       │
│                                                         │
│  X_train_scaled → fit 3 classifiers                     │
│  X_test_scaled  → evaluate 3 classifiers                │
│                                                         │
│  pickle.dump → 5 artifacts                              │
└─────────────────────────────────────────────────────────┘
        │
        ▼ (pkl files)
┌─────────────────────────────────────────────────────────┐
│                  STREAMLIT SERVING                       │
│                                                         │
│  Form inputs (18 values)                                │
│      │                                                  │
│      └── preprocess(raw, scaler, encoder)               │
│              │                                          │
│              ├── Manual LabelEncode (Education_Level)   │
│              ├── Build 12 numerical features            │
│              ├── encoder.transform(6 cat cols)          │
│              ├── Concatenate → 27-col DataFrame         │
│              ├── reindex(EXACT_COLS)  ← critical step   │
│              └── scaler.transform()  → X: 1 × 27        │
│                                                         │
│  model.predict(X)       → 0 or 1                        │
│  model.predict_proba(X) → [p0, p1]                      │
│      │                                                  │
│      └── Render: Approved/Rejected + Confidence         │
└─────────────────────────────────────────────────────────┘
```

**Critical data transformation at serving time:**
The `reindex(EXACT_COLS)` step in `preprocess()` is the most critical line in the entire application. Without it, column order mismatch between inference-time DataFrame and the fitted scaler causes silent wrong predictions. The exact 27-column order is: 10 numeric features → 15 OHE columns (in `encoder.get_feature_names_out()` order) → DTI_Ratio_sq → Credit_Score_sq.

---

<details>
<summary>📐 UML Diagram Suite (9 Diagrams)</summary>

### 1. Use Case Diagram

```mermaid
graph TD
  subgraph Actors["Actors"]
    APP["👤 Applicant"]
    DS["👩‍💻 Data Scientist"]
  end

  subgraph System["CreditWise Loan System"]
    UC1["Submit Loan Application"]
    UC2["View Prediction Result"]
    UC3["View Risk Scorecard"]
    UC4["Switch ML Model"]
    UC5["View Insights Dashboard"]
    UC6["Train and Export Models"]
    UC7["Configure Preprocessing Pipeline"]
    UC8["View Model Comparison"]
  end

  APP --> UC1
  APP --> UC2
  APP --> UC3
  APP --> UC4
  APP --> UC5
  APP --> UC8
  DS --> UC6
  DS --> UC7
  UC1 -->|"includes"| UC2
  UC2 -->|"extends"| UC3
```

### 2. Class Diagram

```mermaid
classDiagram
  class StreamlitApp {
    +load_scaler() StandardScaler
    +load_encoder() OneHotEncoder
    +load_model(name: str) Classifier
    +preprocess(raw: dict, scaler, encoder) ndarray
    +credit_band(score: int) tuple
    +fmt(v: float) str
  }

  class PreprocessingPipeline {
    -num_imputer: SimpleImputer
    -cat_imputer: SimpleImputer
    -label_encoder: LabelEncoder
    -ohe_encoder: OneHotEncoder
    -scaler: StandardScaler
    +fit(X_raw: DataFrame) void
    +transform(X_raw: DataFrame) ndarray
  }

  class LoanRecord {
    +Applicant_ID: str
    +Applicant_Income: float
    +Coapplicant_Income: float
    +Age: int
    +Dependents: int
    +Credit_Score: int
    +DTI_Ratio: float
    +Savings: float
    +Collateral_Value: float
    +Loan_Amount: float
    +Loan_Term: int
    +Education_Level: str
    +Employment_Status: str
    +Marital_Status: str
    +Loan_Purpose: str
    +Property_Area: str
    +Gender: str
    +Employer_Category: str
    +Loan_Approved: str
  }

  class Classifier {
    <<interface>>
    +fit(X, y) void
    +predict(X) ndarray
    +predict_proba(X) ndarray
  }

  class LogisticRegressionModel {
    +solver: str
    +max_iter: int
    +fit(X, y) void
    +predict(X) ndarray
    +predict_proba(X) ndarray
  }

  class KNNModel {
    +n_neighbors: int = 5
    +fit(X, y) void
    +predict(X) ndarray
    +predict_proba(X) ndarray
  }

  class NaiveBayesModel {
    +var_smoothing: float
    +fit(X, y) void
    +predict(X) ndarray
    +predict_proba(X) ndarray
  }

  Classifier <|.. LogisticRegressionModel
  Classifier <|.. KNNModel
  Classifier <|.. NaiveBayesModel
  StreamlitApp --> PreprocessingPipeline
  StreamlitApp --> Classifier
  PreprocessingPipeline --> LoanRecord
```

### 3. Sequence Diagram — Prediction Request

```mermaid
sequenceDiagram
  actor User
  participant UI as Streamlit UI
  participant Cache as Cache Resource
  participant Preproc as preprocess()
  participant Model as ML Model

  User->>UI: Fill 18 input fields
  User->>UI: Click Predict button
  UI->>Cache: load_scaler()
  Cache-->>UI: StandardScaler
  UI->>Cache: load_encoder()
  Cache-->>UI: OneHotEncoder
  UI->>Cache: load_model(sel_pkl)
  Cache-->>UI: Classifier
  UI->>Preproc: preprocess(raw, scaler, encoder)
  Preproc->>Preproc: LabelEncode Education_Level
  Preproc->>Preproc: Compute DTI_sq, Credit_sq
  Preproc->>Preproc: encoder.transform(cat_df)
  Preproc->>Preproc: pd.concat + reindex(EXACT_COLS)
  Preproc->>Preproc: scaler.transform(full_df)
  Preproc-->>UI: X (1x27 ndarray)
  UI->>Model: model.predict(X)
  Model-->>UI: label (0 or 1)
  UI->>Model: model.predict_proba(X)
  Model-->>UI: [p_reject, p_approve]
  UI-->>User: Approved or Rejected + Confidence + Scorecard
```

### 4. Activity Diagram — ML Training Pipeline

```mermaid
graph TD
  A1["Load loan_approval_data.csv"] --> A2["Identify numeric and categorical columns"]
  A2 --> A3["SimpleImputer: mean for numerics"]
  A3 --> A4["SimpleImputer: most_frequent for categoricals"]
  A4 --> A5["Drop Applicant_ID"]
  A5 --> A6["LabelEncode Education_Level and Loan_Approved"]
  A6 --> A7["OneHotEncode 6 categorical columns\ndrop=first"]
  A7 --> A8["Concat encoded columns into main DataFrame"]
  A8 --> A9["EDA: Histograms, boxplots, correlation heatmap"]
  A9 --> A10["Feature Engineering\nDTI_Ratio_sq and Credit_Score_sq"]
  A10 --> A11["Drop Credit_Score and DTI_Ratio originals"]
  A11 --> A12["train_test_split 80-20 random_state=42"]
  A12 --> A13["StandardScaler: fit on X_train, transform both"]
  A13 --> A14{"Train 3 Models"}
  A14 -->|"LR"| A15["LogisticRegression().fit()"]
  A14 -->|"KNN"| A16["KNeighborsClassifier(5).fit()"]
  A14 -->|"NB"| A17["GaussianNB().fit()"]
  A15 & A16 & A17 --> A18["Evaluate: Precision, Recall, F1, Accuracy, CM"]
  A18 --> A19["pickle.dump 5 artifacts"]
```

### 5. Component Diagram

```mermaid
graph TB
  subgraph HFSpaces["☁️ Hugging Face Spaces"]
    subgraph AppLayer["Application Layer"]
      CMP1A["app.py\nStreamlit Entry Point"]
    end
    subgraph PagesLayer["Page Components"]
      CMP2A["Predict Page\nForm + Result + Scorecard"]
      CMP3A["Insights Page\nCredit Bands + EDA Summary"]
      CMP4A["Models Page\nComparison Table + Pipeline"]
      CMP5A["About Page\nAuthor + Stack"]
    end
    subgraph MLLayer["ML Artifact Layer"]
      CMP6A["model_lr.pkl"]
      CMP7A["model_knn.pkl"]
      CMP8A["model_nb.pkl"]
      CMP9A["scaler.pkl"]
      CMP10A["encoder.pkl"]
    end
    subgraph DataLayer["Data Layer"]
      CMP11A["loan_approval_data.csv"]
      CMP12A["Credit_Wise_Loan_Approval_System.ipynb"]
    end
  end

  CMP1A --> CMP2A & CMP3A & CMP4A & CMP5A
  CMP2A --> CMP6A & CMP7A & CMP8A
  CMP2A --> CMP9A & CMP10A
  CMP12A -->|"trains + exports"| CMP6A & CMP7A & CMP8A & CMP9A & CMP10A
  CMP11A -->|"source data"| CMP12A
```

### 6. Deployment Diagram

```mermaid
graph TB
  subgraph Client["Client"]
    BR["Web Browser"]
  end
  subgraph HF["Hugging Face Spaces (Free Tier)"]
    ST["Streamlit SDK Runtime\nPython 3.10+"]
    FILES["Space Files\napp.py + pkl + csv"]
  end
  subgraph Dev["Developer Machine"]
    JUPYTER["Jupyter Notebook\nTraining + EDA"]
    GIT["Git Repository\nGitHub: KARTHIKAKRISHNA123"]
  end

  BR -->|"HTTPS"| HF
  ST --> FILES
  JUPYTER -->|"git push"| GIT
  GIT -->|"sync"| HF
```

### 7. State Diagram — Prediction Lifecycle

```mermaid
stateDiagram-v2
  [*] --> Idle: App loads
  Idle --> FormFilling: User interacts with widgets
  FormFilling --> FormFilling: Live scorecard updates
  FormFilling --> Loading: User clicks Predict
  Loading --> ArtifactsReady: pkl files found in cache
  Loading --> DemoMode: pkl files missing
  ArtifactsReady --> Preprocessing: preprocess() runs
  Preprocessing --> Inference: scaler.transform complete
  Inference --> Approved: model.predict == 1
  Inference --> Rejected: model.predict == 0
  DemoMode --> Error: Model file not found
  Approved --> FormFilling: User modifies inputs
  Rejected --> FormFilling: User modifies inputs
  Error --> Idle: User uploads pkl files
```

### 8. ER Diagram — Loan Record Schema

```mermaid
erDiagram
  LOAN_RECORD {
    string Applicant_ID PK
    float Applicant_Income
    float Coapplicant_Income
    int Age
    int Dependents
    int Credit_Score
    float DTI_Ratio
    float Savings
    float Collateral_Value
    float Loan_Amount
    int Loan_Term
    string Education_Level
    string Employment_Status
    string Marital_Status
    string Loan_Purpose
    string Property_Area
    string Gender
    string Employer_Category
    string Loan_Approved
  }

  ENGINEERED_FEATURES {
    float DTI_Ratio_sq
    float Credit_Score_sq
  }

  ENCODED_RECORD {
    int Education_Level_enc
    int Loan_Approved_enc
    int Employment_Status_OHE
    int Marital_Status_OHE
    int Loan_Purpose_OHE
    int Property_Area_OHE
    int Gender_OHE
    int Employer_Category_OHE
  }

  LOAN_RECORD ||--|| ENGINEERED_FEATURES : "transforms into"
  LOAN_RECORD ||--|| ENCODED_RECORD : "encodes into"
```

### 9. Package Diagram — Module Dependencies

```mermaid
flowchart TD
  subgraph AppModule["app.py"]
    MAIN["main app logic"]
    PREPROC["preprocess()"]
    LOADERS["load_scaler, load_encoder, load_model"]
    PAGES["4 pages: Predict, Insights, Models, About"]
  end

  subgraph SklearnPkg["scikit-learn"]
    IMPUTE["sklearn.impute.SimpleImputer"]
    PREPKG["sklearn.preprocessing\nLabelEncoder, OHE, StandardScaler"]
    MODPKG["sklearn.linear_model.LogisticRegression\nsklearn.neighbors.KNN\nsklearn.naive_bayes.GaussianNB"]
    METRICS["sklearn.metrics\naccuracy, precision, recall, f1, cm"]
  end

  subgraph DataPkg["Data Layer"]
    PANDAS["pandas"]
    NUMPY["numpy"]
    PICKLE["pickle stdlib"]
    PATHLIB["pathlib stdlib"]
  end

  MAIN --> PREPROC
  MAIN --> LOADERS
  MAIN --> PAGES
  PREPROC --> PREPKG
  PREPROC --> PANDAS
  PREPROC --> NUMPY
  LOADERS --> PICKLE
  LOADERS --> PATHLIB
  LOADERS --> MODPKG
```

</details>

---

<details>
<summary>📊 Data Flow Diagrams (DFD Level 0 + Level 1)</summary>

### DFD Level 0 — Context Diagram

```mermaid
flowchart LR
  E1["👤 Loan Applicant"]
  E2["👩‍💻 Data Scientist"]
  P1(("0.0\nCreditWise\nLoan System"))
  E3["🏦 Decision Output"]

  E1 -->|"Applicant Profile\nFinancial Details\nLoan Details"| P1
  E2 -->|"Training Data CSV\nNotebook Pipeline"| P1
  P1 -->|"Approval Decision\nConfidence Score\nRisk Scorecard"| E3
  P1 -->|"Model Comparison\nEDA Insights"| E1
```

### DFD Level 1 — System Decomposition

```mermaid
flowchart TD
  E1["👤 Applicant"]
  E2["👩‍💻 Data Scientist"]

  P1(("1.0\nCollect\nApplicant Data"))
  P2(("2.0\nPreprocess\nInput Data"))
  P3(("3.0\nRun ML\nInference"))
  P4(("4.0\nScore\nRisk Profile"))
  P5(("5.0\nTrain and\nExport Models"))

  D1[("D1: loan_approval_data.csv")]
  D2[("D2: Serialized Artifacts\nscaler encoder models")]
  D3[("D3: Prediction Cache\nStreamlit cache_resource")]

  E1 -->|"18 form inputs"| P1
  P1 -->|"raw applicant dict"| P2
  E2 -->|"raw CSV"| P5
  P5 -->|"trained pkl files"| D2
  D1 -->|"training dataset"| P5
  P2 -->|"loads encoder + scaler"| D2
  P2 -->|"27-col scaled array"| P3
  D2 -->|"model pkl"| P3
  D3 -->|"cached scaler encoder model"| P3
  P3 -->|"label + proba"| D3
  P3 -->|"0 or 1 + confidence"| P4
  P4 -->|"Approval decision\nScorecard metrics"| E1
```

</details>

---

## 📁 Folder Structure

```
CreditWise_Loan_System/mine/
│
├── app.py                              ← Streamlit application entry point
│     ├── preprocess()                  ← Mirrors notebook pipeline exactly
│     ├── load_scaler/encoder/model()   ← @st.cache_resource loaders
│     ├── Pages: Predict, Insights, Models, About
│     └── Custom CSS: Navy-blue theme, pill badges, result cards
│
├── Credit_Wise_Loan_Approval_System.ipynb  ← Full ML training notebook (87 cells)
│     ├── Cells 0–16: Data loading + imputation
│     ├── Cells 17–40: EDA (distributions, boxplots, heatmap)
│     ├── Cells 41–65: Encoding + scaling pipeline
│     ├── Cells 69–77: Baseline model training + evaluation
│     ├── Cells 79–86: Feature engineering + retrain + pickle export
│     └── Cell 85: pickle.dump → 5 artifacts
│
├── loan_approval_data.csv              ← Source dataset (1,000 rows × 20 cols)
│
├── model_lr.pkl                        ← Trained LogisticRegression
├── model_knn.pkl                       ← Trained KNeighborsClassifier(n=5)
├── model_nb.pkl                        ← Trained GaussianNB (best precision)
├── scaler.pkl                          ← StandardScaler (fitted on 27-col X_train)
├── encoder.pkl                         ← OneHotEncoder(drop=first) for 6 cols
│
├── requirements.txt                    ← Python dependencies for HF Spaces
├── README.md                           ← HF Spaces metadata (YAML frontmatter)
├── CreditWise Loan System.pdf          ← Project report / documentation
└── .gitattributes                      ← Git LFS tracking for .pkl files
```

---

## ⚙️ ML Pipeline — Exact Notebook Recreation

The preprocessing pipeline in `preprocess()` is a deterministic reproduction of the notebook cells. Any deviation causes silent column-order mismatches → wrong predictions.

```
Notebook Cell → app.py Equivalent

Cell 12: SimpleImputer(mean) on numerics     → done at training time; inference uses form defaults
Cell 14: SimpleImputer(most_frequent) on cat → done at training time; inference uses form defaults
Cell 41: Drop Applicant_ID                   → never collected in the form
Cell 46: LabelEncode Education_Level         → manual mapping: Graduate=0, Not Graduate=1
Cell 46: LabelEncode Loan_Approved           → target only; not needed at inference
Cell 48: OneHotEncode 6 cols, drop=first     → encoder.transform(cat_df)
Cell 80: DTI_Ratio_sq = DTI_Ratio ** 2       → raw["DTI_Ratio"] ** 2
Cell 80: Credit_Score_sq = Credit_Score ** 2 → raw["Credit_Score"] ** 2
Cell 80: Drop Credit_Score, DTI_Ratio        → originals NOT included in num dict
Cell 65: StandardScaler                      → scaler.transform(full_df)
```

### Final 27-Column Feature Order (Critical)

```
Columns 0–9  (numeric):
  Applicant_Income, Coapplicant_Income, Age, Dependents,
  Existing_Loans, Savings, Collateral_Value, Loan_Amount,
  Loan_Term, Education_Level

Columns 10–24 (OHE — exact encoder.get_feature_names_out() order):
  Employment_Status_*, Marital_Status_*, Loan_Purpose_*,
  Property_Area_*, Gender_*, Employer_Category_*
  (15 cols after drop=first from 6 categorical columns)

Columns 25–26 (engineered):
  DTI_Ratio_sq, Credit_Score_sq
```

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | Best Use Case |
|---|---|---|---|---|---|
| **Logistic Regression ⭐** | ~87.5% | ~79% | ~80% | ~80% | Default — balanced overall |
| K-Nearest Neighbours | ~75.5% | ~62% | ~56% | ~56% | Pattern-similarity matching |
| **Gaussian Naive Bayes** | ~86.5% | ~78% | ~78% | ~78% | Minimizing false approvals |

> ⚠️ *Dataset is class-imbalanced (~29.8% approved). F1 Score and Precision are primary metrics, not just Accuracy.*

---

## 🔒 Dataset — Feature Dictionary

| Feature | Type | Description | Impact on Approval |
|---|---|---|---|
| `Applicant_Income` | float | Monthly income of primary applicant (₹) | Higher → positive |
| `Coapplicant_Income` | float | Monthly income of co-applicant (₹) | Higher → positive |
| `Age` | int | Age of applicant | Mid-range preferred |
| `Dependents` | int | Number of financial dependents | More → negative |
| `Credit_Score` | int | CIBIL-style score (300–900) | **Strongest predictor** |
| `DTI_Ratio` | float | Debt-to-Income ratio (0.0–1.0) | Lower → positive |
| `Savings` | float | Current savings balance (₹) | Higher → positive |
| `Collateral_Value` | float | Asset value offered as security (₹) | Higher → positive |
| `Loan_Amount` | float | Requested loan amount (₹) | Lower relative to collateral → positive |
| `Loan_Term` | int | Repayment period in months | — |
| `Education_Level` | cat | Graduate / Not Graduate | Graduate → positive |
| `Employment_Status` | cat | Salaried / Self-employed / Contract / Unemployed | Salaried → best |
| `Marital_Status` | cat | Married / Single | — |
| `Loan_Purpose` | cat | Home / Car / Business / Education / Personal | — |
| `Property_Area` | cat | Urban / Semiurban / Rural | Urban → positive |
| `Gender` | cat | Male / Female | — |
| `Employer_Category` | cat | Government / MNC / Private / Business / Unemployed | Govt → best |

---

## 🛠️ Installation & Setup

### Prerequisites

```bash
Python 3.10+
pip
Git
```

### Local Setup

```bash
# 1. Clone repository
git clone https://github.com/KARTHIKAKRISHNA123/CreditWise_Loan_System.git
cd CreditWise_Loan_System

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py
```

The app opens at `http://localhost:8501`

### Retrain Models from Scratch

```bash
# Open the notebook
jupyter notebook Credit_Wise_Loan_Approval_System.ipynb

# Run all cells sequentially (Cell 0 → 86)
# Cell 85 auto-exports all 5 pkl artifacts to the working directory
```

---

## 📦 Requirements

```
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
```

> All pinned at latest stable at time of deployment. No version locking — HF Spaces resolves latest compatible set.

---

## 🚀 Deployment

### Hugging Face Spaces

The project is deployed as a Streamlit Space on Hugging Face. The `README.md` YAML frontmatter configures the Space:

```yaml
---
title: LendSure AI
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: streamlit
app_file: app.py
pinned: false
---
```

Push to the HF Space repository to trigger auto-deploy:

```bash
git remote add space https://huggingface.co/spaces/KarthikaKrishna123/CreditWise
git push space main
```

### Local Docker (optional)

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 🔐 Security Considerations

| Risk | Mitigation |
|---|---|
| **Model file tampering** | `.gitattributes` enforces Git LFS tracking for `.pkl` files — binary integrity preserved |
| **Column order mismatch** | `reindex(EXACT_COLS)` with `fill_value=0` prevents silent wrong predictions on column mismatches |
| **Missing artifacts** | `Path.exists()` guards + warning banner in demo mode — no crash |
| **Input validation** | Streamlit widget constraints (`min_value`, `max_value`, `step`) prevent nonsensical inputs |
| **No PII storage** | All form inputs are ephemeral — no database, no logging, no user data persistence |

---

## ⚡ Performance

| Concern | Approach |
|---|---|
| **Model reload cost** | `@st.cache_resource` caches all pkl artifacts on first load — zero reload on re-predictions |
| **Preprocessing speed** | Pure NumPy/pandas operations — sub-millisecond for single record |
| **UI re-render** | Streamlit re-runs app top-to-bottom on each interaction — scorecard updates are lightweight arithmetic-only |
| **Memory** | Three models + scaler + encoder cached in memory — total ~5MB for all artifacts |


---

## 🐛 Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `⚠️ scaler.pkl not found` | pkl files not in working directory | Re-run Cell 85 in notebook; ensure artifacts are in same folder as app.py |
| `Prediction error: ...` | Column mismatch at inference | Retrain and re-export scaler with current feature engineering config |
| Streamlit runs but predictions always 0 | Scaler fitted on different feature set | Verify 27-column order matches `EXACT_COLS` list in `preprocess()` |
| HF Space fails to start | Dependency conflict | Pin versions in `requirements.txt`; check HF build logs |
| Wrong confidence values | Model doesn't support `predict_proba` | App defaults to 75.0% fallback — this is expected for some sklearn estimators |

---

## ❓ FAQ

**Q: Why are three separate models included?**  
A: To demonstrate the tradeoff between precision and recall — Naive Bayes minimizes false approvals (high precision), Logistic Regression balances both, KNN serves as a baseline for comparison.

**Q: Why squared transformations for DTI and Credit Score?**  
A: Credit score influence on approval is non-linear — a jump from 650→700 matters far more than 750→800. Squaring amplifies this signal without requiring tree-based models.

**Q: Why is `Applicant_Income_log` commented out?**  
A: Explored during EDA (Cell 80) but removed — income distribution wasn't sufficiently log-normal to warrant the transformation after imputation.

**Q: Why `reindex(EXACT_COLS)` instead of just concatenating?**  
A: `pd.concat` column order depends on DataFrame construction order, which can vary. `reindex` enforces deterministic column order matching `scaler.feature_names_in_`.

**Q: Is this suitable for production loan decisions?**  
A: No. This is an educational demonstration. Production systems require regulatory compliance (RBI guidelines), bias auditing, explainability standards (LIME/SHAP), and human-in-the-loop review.

---

## 📜 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/shap-explainability`
3. Make changes and test locally with `streamlit run app.py`
4. Ensure all pkl artifacts are reproducible from the notebook
5. Open a Pull Request with a description of changes

---

## 📄 License

MIT License — Free to use for educational and non-commercial purposes.

---

🤗 [KarthikaKrishna123 on Hugging Face](https://huggingface.co/KarthikaKrishna123)

*For educational and demonstration purposes only — not for real loan decisions.*

</div>
