import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)
from features import PhysicsFeatureEngineer

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
df = pd.read_csv(url)

print("Dataset Shape:", df.shape)
print("Missing Values:\n", df.isnull().sum())
print("Duplicate Rows:", df.duplicated().sum())
print("\nTarget Distribution:\n", df['Machine failure'].value_counts())

cols_to_drop = ['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
X = df.drop(columns=cols_to_drop)
y = df['Machine failure']

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.20, random_state=42, stratify=y_train_full
)

categorical_features = ['Type']
numerical_features = [
    'Air temperature [K]', 'Process temperature [K]',
    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
    'Temp_Diff [K]', 'Power [W]', 'Overstrain [min*Nm]'
]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ]
)

def build_pipeline(classifier):
    return Pipeline([
        ('physics_features', PhysicsFeatureEngineer()),
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

models = {
    'Logistic Regression': build_pipeline(LogisticRegression(max_iter=1000, random_state=42)),
    'Decision Tree': build_pipeline(DecisionTreeClassifier(random_state=42)),
    'Random Forest': build_pipeline(RandomForestClassifier(random_state=42, n_jobs=-1))
}

print("\n--- Validation Set Comparison ---")
val_results = []
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)
    probs = pipe.predict_proba(X_val)[:, 1]
    val_results.append({
        'Model': name,
        'Accuracy': round(accuracy_score(y_val, preds), 4),
        'Precision': round(precision_score(y_val, preds), 4),
        'Recall': round(recall_score(y_val, preds), 4),
        'F1-Score': round(f1_score(y_val, preds), 4),
        'ROC-AUC': round(roc_auc_score(y_val, probs), 4)
    })
print(pd.DataFrame(val_results).sort_values(by='F1-Score', ascending=False).to_string(index=False))

print("\n--- 5-Fold Stratified Cross-Validation (F1-Score) ---")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, pipe in models.items():
    scores = cross_val_score(pipe, X_train_full, y_train_full, cv=cv, scoring='f1', n_jobs=-1)
    print(f"{name}: {np.round(scores, 4)} | Mean: {scores.mean():.4f}")

param_grids = {
    'Logistic Regression': {
        'classifier__C': [0.01, 0.1, 1, 10],
        'classifier__class_weight': [None, 'balanced']
    },
    'Decision Tree': {
        'classifier__max_depth': [6, 10, 15, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__class_weight': [None, 'balanced']
    },
    'Random Forest': {
        'classifier__n_estimators': [150, 300],
        'classifier__max_depth': [12, 20, None],
        'classifier__min_samples_split': [2, 5],
        'classifier__class_weight': [None, 'balanced']
    }
}

print("\n--- Hyperparameter Tuning (GridSearchCV) ---")
best_estimators = {}
tuning_summary = []
for name, pipe in models.items():
    grid = GridSearchCV(pipe, param_grids[name], cv=cv, scoring='f1', n_jobs=-1)
    grid.fit(X_train_full, y_train_full)
    best_estimators[name] = grid.best_estimator_
    tuning_summary.append({
        'Model': name,
        'Best CV F1': round(grid.best_score_, 4),
        'Best Params': grid.best_params_
    })

tuning_df = pd.DataFrame(tuning_summary).sort_values(by='Best CV F1', ascending=False)
print(tuning_df.to_string(index=False))

best_model_name = tuning_df.iloc[0]['Model']
best_model = best_estimators[best_model_name]

y_test_pred = best_model.predict(X_test)
y_test_prob = best_model.predict_proba(X_test)[:, 1]

test_metrics = {
    'model_name': best_model_name,
    'accuracy': round(accuracy_score(y_test, y_test_pred) * 100, 2),
    'precision': round(precision_score(y_test, y_test_pred) * 100, 2),
    'recall': round(recall_score(y_test, y_test_pred) * 100, 2),
    'f1_score': round(f1_score(y_test, y_test_pred) * 100, 2),
    'roc_auc': round(roc_auc_score(y_test, y_test_prob) * 100, 2)
}

print(f"\n--- Final Untouched Test Evaluation ({best_model_name}) ---")
print(classification_report(y_test, y_test_pred, target_names=['No Failure (0)', 'Failure (1)']))

cm = confusion_matrix(y_test, y_test_pred)
ConfusionMatrixDisplay(cm, display_labels=['No Failure', 'Failure']).plot(cmap='Greens')
plt.title(f'Test Confusion Matrix - {best_model_name}')
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

joblib.dump(best_model, 'industrial_failure_best_pipeline.joblib')
with open('model_metrics.json', 'w') as f:
    json.dump(test_metrics, f, indent=2)

print("\nSaved 'industrial_failure_best_pipeline.joblib', 'model_metrics.json', and 'confusion_matrix.png'.")
