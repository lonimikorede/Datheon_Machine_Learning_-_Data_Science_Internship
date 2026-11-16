# Datheon Machine Learning & Data Science Internship

My project work for the Datheon ML & Data Science internship.

## Task 1 - Iris Classification

Notebook: `notebooks/Task1_Iris_Classification.ipynb`

I used the Iris dataset (150 flowers, 3 species) to train and compare a Logistic Regression model and a Decision Tree.

Steps:
- loaded the data with pandas and did some EDA (pairplot, correlation heatmap)
- split the data 80/20 and scaled the features with StandardScaler
- trained Logistic Regression and a Decision Tree
- compared accuracy, precision, recall and F1-score
- plotted confusion matrices and decision boundaries

Results on the test set:

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.933 | 0.933 | 0.933 | 0.933 |
| Decision Tree | 0.933 | 0.933 | 0.933 | 0.933 |

Both models got the same score. Setosa was always predicted correctly, and the only mistakes were between versicolor and virginica.

## How to run

```
pip install -r requirements.txt
jupyter notebook
```

Then open the notebook in the `notebooks` folder.
