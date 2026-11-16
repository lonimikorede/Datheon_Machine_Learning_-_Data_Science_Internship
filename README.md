# Datheon Machine Learning & Data Science Internship

Project submissions for the Datheon Machine Learning & Data Science Internship.

## Task 1: Multi-Class Benchmark Classification Pipeline (Iris)

An end-to-end supervised classification pipeline on the Iris dataset (150 flowers, 4 features, 3 species) using Scikit-Learn.

### What the notebook covers
1. **Data loading** with Pandas, plus data-quality checks (missing values, duplicates, class balance)
2. **Exploratory analysis**: Seaborn pairplot, boxplots and a correlation heatmap
3. **Preprocessing**: label encoding, a stratified 80/20 train/test split, and `StandardScaler` fitted on the training set only (no data leakage)
4. **Models**: `LogisticRegression` and `DecisionTreeClassifier` (max_depth=4)
5. **Benchmarking**: accuracy, precision, recall and F1 (macro), full classification reports, and 5-fold stratified cross-validation
6. **Visualisation**: confusion matrix heatmaps, 2D decision boundary plots, coefficient and feature-importance charts, and the trained tree diagram

### Results

| Model | Accuracy | Precision | Recall | F1 (macro) | 5-fold CV F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.933 | 0.933 | 0.933 | 0.933 | 0.953 ± 0.045 |
| Decision Tree | 0.933 | 0.933 | 0.933 | 0.933 | 0.953 ± 0.034 |

- Both models classified 28 of 30 test flowers correctly. *Setosa* was always classified correctly; the only errors were between *Versicolor* and *Virginica*, whose measurements overlap.
- Petal length and petal width were the most informative features for both models.
- Performance is tied. Logistic Regression gives smoother linear boundaries, while the Decision Tree is easier to interpret as simple rules.

### Repository structure
```
Datheon_Machine_Learning_&_Data_Science_Internship/
├── README.md
├── requirements.txt
├── data/
│   └── iris.csv
└── notebooks/
    └── Task1_Iris_Classification.ipynb
```

### How to run
```bash
pip install -r requirements.txt
jupyter notebook notebooks/Task1_Iris_Classification.ipynb
```
The notebook loads the dataset from the URL in the task brief and falls back to `data/iris.csv` if offline.

### Tools
Python, Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, Jupyter
