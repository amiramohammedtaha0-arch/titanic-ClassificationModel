# Titanic Survival Prediction

A beginner-friendly machine learning project on the Kaggle Titanic dataset. It covers the full workflow: data quality checks, statistical exploration, preprocessing, a **Logistic Regression implemented from scratch with NumPy**, and a scikit-learn pipeline for comparison.

The goal of this project is learning: understanding each step, not only getting a high score.

## Project Structure

```
.
├── Data/
│   ├── train.csv              # Original Kaggle data (891 rows)
│   └── dataAfterPrep.csv      # Output of preperation.py (890 rows)
├── src/
│   ├── dataExp.ipynb          # Exploration and statistical tests
│   ├── preperation.py         # Cleaning + feature engineering
│   ├── mlModel.py             # Logistic Regression from scratch (NumPy)
│   └── sklearnModel.py        # scikit-learn pipeline + feature experiments
├── requirements.txt
└── README.md
```

> The scripts read data from `../Data/`, so run them from inside `src/`.

## Workflow

### 1. Data quality and exploration (`dataExp.ipynb`, `preperation.py`)
- Checked missing values per column.
- `Cabin` is mostly missing, so it was converted to a binary `hasCabin` feature, and its relation to survival was plotted.
- Ran a **chi-square test** between `hasCabin` and `Pclass` (significant relationship, p < 0.001).
- `Age` is right-skewed, so missing values were filled with the **median**.

### 2. Preprocessing (`preperation.py`)
- `Sex` encoded as binary (female = 1).
- New feature `familySize = SibSp + Parch + 1`.
- Dropped `Name`, `PassengerId`, `Ticket`, `SibSp`, `Parch`.
- One-hot encoding for `Embarked` (`drop_first=True`).

Final features: `Pclass, Sex, Age, Fare, hasCabin, familySize, Embarked_Q, Embarked_S`.

### 3. Logistic Regression from scratch (`mlModel.py`)
Implemented with NumPy only: sigmoid, binary cross-entropy cost (with clipping for numerical stability), gradient computation, and batch gradient descent (`alpha = 0.1`, 1000 iterations). Numeric features are standardized.

### 4. scikit-learn pipeline (`sklearnModel.py`)
`PolynomialFeatures(degree=2) → RobustScaler → RidgeClassifier(alpha=10)`, evaluated with F1 score and a classification report on a train / validation / test split.

## Results

| Model | Evaluation | F1 |
|---|---|---|
| Logistic Regression (from scratch) | Training data (no split) | ~0.74 |
| Ridge pipeline (scikit-learn) | Validation (81 samples) | ~0.78 |
| Ridge pipeline (scikit-learn) | Test (89 samples) | ~0.82 |

Reference baseline: predicting "female survives, male dies" gives about **78.7% accuracy** on this data.

### Feature experiments
Different combinations of `Age`, `Fare`, `Pclass` and `hasCabin` were compared (results are in the comments of `sklearnModel.py`). The differences were small (about 1–4 F1 points) and inconsistent between the validation and test sets. With only ~80–90 samples per set, changing the prediction of a couple of passengers moves F1 noticeably, so these differences are within random noise. **No feature was dropped based on these numbers.**

## Known Limitations
- The from-scratch model is evaluated on its own training data, so its score is not a measure of generalization. It was built to learn the algorithm.
- A single train/valid/test split on a small dataset gives noisy estimates.
- Median imputation and scaling for the from-scratch model are computed on the full dataset before any split.
- `preperation.py` drops the row with index 152 (`df.drop(152)`), which was meant for a missing `Fare` value. The Kaggle `train.csv` has no missing `Fare`, so this removes one valid passenger (891 → 890 rows).

## Future Work
- Cross-validation (repeated stratified k-fold) for more reliable comparisons.
- Imputation inside the pipeline to avoid data leakage.
- Extract passenger titles (Mr / Mrs / Miss / Master) from `Name` and use them to impute `Age`.
- Compare against the sex-only baseline and tree-based models (Random Forest, Gradient Boosting).
- Hyperparameter tuning with `GridSearchCV`.

## How to Run

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install -r requirements.txt

cd src
python preperation.py     # creates Data/dataAfterPrep.csv
python mlModel.py         # Logistic Regression from scratch
python sklearnModel.py    # scikit-learn pipeline
```

## Dataset
[Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic) (Kaggle).
