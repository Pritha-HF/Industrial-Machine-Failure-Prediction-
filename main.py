!pip install -q gradio

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import gradio as gr
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    RocCurveDisplay, PrecisionRecallDisplay
)

sns.set_theme(style="whitegrid", context="notebook", palette="muted")

url = "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance"
df = pd.read_csv(url)

print("=" * 60)
print("STEP 1: DATASET OVERVIEW")
print("=" * 60)
print("Dataset Shape:", df.shape)
display(df.head())
df.info()

print("\n" + "=" * 60)
print("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 60)
print("Missing Values:\n", df.isnull().sum())
print("Duplicate Rows:", df.duplicated().sum())
display(df.describe().T.round(2))

failure_counts = df['Machine failure'].value_counts()
failure_pct = df['Machine failure'].value_counts(normalize=True) * 100
display(pd.DataFrame({'Count': failure_counts, 'Percentage (%)': failure_pct.round(2)}))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.countplot(x='Machine failure', data=df, hue='Machine failure', legend=False, palette=['#2ecc71', '#e74c3c'], ax=axes[0])
axes[0].set_title('Target Class Distribution', fontweight='bold')
axes[0].set_xticklabels(['No Failure (0)', 'Failure (1)'])
sns.countplot(x='Type', hue='Machine failure', data=df, palette=['#2ecc71', '#e74c3c'], ax=axes[1])
axes[1].set_title('Machine Failure by Quality Category', fontweight='bold')
plt.tight_layout()
plt.show()

sensor_cols = [
    'Air temperature [K]', 'Process temperature [K]',
    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Machine failure'
]
plt.figure(figsize=(8.5, 6))
sns.heatmap(df[sensor_cols].corr(), annot=True, cmap='RdBu_r', center=0, fmt=".2f", linewidths=0.5)
plt.title('Sensor Feature Correlation Matrix', fontweight='bold')
plt.show()

fig, axes = plt.subplots(1, 5, figsize=(21, 4))
for idx, col in enumerate(sensor_cols[:-1]):
    sns.boxplot(x='Machine failure', y=col, data=df, hue='Machine failure', legend=False, palette=['#3498db', '#e74c3c'], ax=axes[idx])
    axes[idx].set_title(col, fontsize=10, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("STEP 3 & 4: DATA PREPARATION & TRAIN / VALIDATION / TEST SPLIT")
print("=" * 60)
cols_to_drop = ['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
X = df.drop(columns=cols_to_drop)
y = df['Machine failure']

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.20, random_state=42, stratify=y_train_full
)

print(f"Training Set:   {X_train.shape[0]} rows ({y_train.mean()*100:.2f}% failure)")
print(f"Validation Set: {X_val.shape[0]} rows ({y_val.mean()*100:.2f}% failure)")
print(f"Untouched Test: {X_test.shape[0]} rows ({y_test.mean()*100:.2f}% failure)")

def add_physics_features(data):
    d = data.copy()
    d['Temp_Diff [K]'] = d['Process temperature [K]'] - d['Air temperature [K]']
    d['Power [W]'] = d['Torque [Nm]'] * (d['Rotational speed [rpm]'] * 2 * np.pi / 60)
    d['Overstrain [min*Nm]'] = d['Tool wear [min]'] * d['Torque [Nm]']
    return d

feature_engineer = FunctionTransformer(add_physics_features, validate=False)

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
        ('physics_features', feature_engineer),
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

models = {
    'Logistic Regression': build_pipeline(LogisticRegression(max_iter=1000, random_state=42)),
    'Decision Tree': build_pipeline(DecisionTreeClassifier(random_state=42)),
    'Random Forest': build_pipeline(RandomForestClassifier(random_state=42, n_jobs=-1))
}

print("\n" + "=" * 60)
print("STEP 5 & 6: BASELINE TRAINING & VALIDATION COMPARISON")
print("=" * 60)
val_results = []
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

for idx, (name, pipeline) in enumerate(models.items()):
    pipeline.fit(X_train, y_train)
    y_val_pred = pipeline.predict(X_val)
    y_val_prob = pipeline.predict_proba(X_val)[:, 1]
    val_results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_val, y_val_pred),
        'Precision': precision_score(y_val, y_val_pred),
        'Recall': recall_score(y_val, y_val_pred),
        'F1-Score': f1_score(y_val, y_val_pred),
        'ROC-AUC': roc_auc_score(y_val, y_val_prob)
    })
    cm = confusion_matrix(y_val, y_val_pred)
    ConfusionMatrixDisplay(cm, display_labels=['No Failure', 'Failure']).plot(
        ax=axes[idx], colorbar=False, cmap='Blues'
    )
    axes[idx].set_title(f'{name}\n(Validation Set)', fontweight='bold')

plt.tight_layout()
plt.show()

val_df = pd.DataFrame(val_results).sort_values(by='F1-Score', ascending=False)
display(val_df.round(4))

print("\n" + "=" * 60)
print("STEP 7: 5-FOLD STRATIFIED CROSS-VALIDATION (ON 80% TRAIN DATA)")
print("=" * 60)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

for name, pipeline in models.items():
    scores = cross_val_score(pipeline, X_train_full, y_train_full, cv=cv, scoring='f1', n_jobs=-1)
    cv_results.append({
        'Model': name,
        'Fold 1': scores[0], 'Fold 2': scores[1], 'Fold 3': scores[2],
        'Fold 4': scores[3], 'Fold 5': scores[4],
        'Mean CV F1': scores.mean(), 'Std Dev': scores.std()
    })

cv_df = pd.DataFrame(cv_results).sort_values(by='Mean CV F1', ascending=False)
display(cv_df.round(4))

print("\n" + "=" * 60)
print("STEP 8: HYPERPARAMETER TUNING (GRIDSEARCHCV)")
print("=" * 60)
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

best_estimators = {}
tuning_summary = []

for name, pipeline in models.items():
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grids[name],
        cv=cv,
        scoring='f1',
        n_jobs=-1
    )
    grid.fit(X_train_full, y_train_full)
    best_estimators[name] = grid.best_estimator_
    tuning_summary.append({
        'Model': name,
        'Best CV F1-Score': grid.best_score_,
        'Best Parameters': grid.best_params_
    })

tuning_df = pd.DataFrame(tuning_summary).sort_values(by='Best CV F1-Score', ascending=False)
display(tuning_df)

print("\n" + "=" * 60)
print("STEP 9: FINAL EVALUATION ON UNTOUCHED TEST SET")
print("=" * 60)
best_model_name = tuning_df.iloc[0]['Model']
best_model = best_estimators[best_model_name]

y_test_pred = best_model.predict(X_test)
y_test_prob = best_model.predict_proba(X_test)[:, 1]

print(f"Selected Best Model: {best_model_name}")
print(f"Test Accuracy:  {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Test Precision: {precision_score(y_test, y_test_pred):.4f}")
print(f"Test Recall:    {recall_score(y_test, y_test_pred):.4f}")
print(f"Test F1-Score:  {f1_score(y_test, y_test_pred):.4f}")
print(f"Test ROC-AUC:   {roc_auc_score(y_test, y_test_prob):.4f}\n")
print(classification_report(y_test, y_test_pred, target_names=['No Failure (0)', 'Failure (1)']))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_pred), display_labels=['No Failure', 'Failure']).plot(
    ax=axes[0], colorbar=False, cmap='Greens'
)
axes[0].set_title(f'Test Confusion Matrix\n({best_model_name})', fontweight='bold')

RocCurveDisplay.from_predictions(y_test, y_test_prob, ax=axes[1], name=best_model_name, color='#2980b9')
axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[1].set_title('ROC Curve (Test Set)', fontweight='bold')

PrecisionRecallDisplay.from_predictions(y_test, y_test_prob, ax=axes[2], name=best_model_name, color='#8e44ad')
axes[2].set_title('Precision-Recall Curve (Test Set)', fontweight='bold')
plt.tight_layout()
plt.show()

if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
    ohe_names = best_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_feature_names = numerical_features + list(ohe_names)
    importances = pd.Series(
        best_model.named_steps['classifier'].feature_importances_,
        index=all_feature_names
    ).sort_values(ascending=True)
    plt.figure(figsize=(9, 5))
    importances.plot(kind='barh', color='#16a085')
    plt.title(f'Feature Importances ({best_model_name})', fontweight='bold')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.show()

print("\n" + "=" * 60)
print("STEP 10 & 11: SAVE PIPELINE & LAUNCH PREDICTION SYSTEM")
print("=" * 60)
model_filename = "industrial_failure_best_pipeline.joblib"
joblib.dump(best_model, model_filename)
loaded_pipeline = joblib.load(model_filename)
print(f"Pipeline saved and verified: {model_filename}")

def predict_machine_failure(m_type, air_temp, proc_temp, rot_speed, torque, tool_wear):
    input_df = pd.DataFrame([{
        'Type': m_type,
        'Air temperature [K]': float(air_temp),
        'Process temperature [K]': float(proc_temp),
        'Rotational speed [rpm]': float(rot_speed),
        'Torque [Nm]': float(torque),
        'Tool wear [min]': float(tool_wear)
    }])
    pred = loaded_pipeline.predict(input_df)[0]
    prob = loaded_pipeline.predict_proba(input_df)[0][1] * 100
    status = "Machine Failure: Yes" if pred == 1 else "Machine Failure: No"
    risk = "HIGH RISK" if prob >= 50 else ("MODERATE RISK" if prob >= 20 else "LOW RISK")
    return f"{status}\nFailure Probability: {prob:.2f}%\nRisk Level: {risk}"

demo = gr.Interface(
    fn=predict_machine_failure,
    inputs=[
        gr.Radio(["L", "M", "H"], label="Machine Quality Type", value="M"),
        gr.Slider(290.0, 310.0, value=300.0, step=0.1, label="Air temperature [K]"),
        gr.Slider(300.0, 320.0, value=310.0, step=0.1, label="Process temperature [K]"),
        gr.Slider(1000, 3000, value=1500, step=10, label="Rotational speed [rpm]"),
        gr.Slider(0.0, 80.0, value=40.0, step=0.5, label="Torque [Nm]"),
        gr.Slider(0, 260, value=100, step=1, label="Tool wear [min]")
    ],
    outputs=gr.Textbox(label="Prediction Result", lines=3),
    title="Industrial Machine Failure Prediction System",
    description=f"Powered by Tuned {best_model_name} with Physics-Based Feature Engineering (AI4I 2020 Dataset).",
    examples=[
        ["M", 300.0, 310.0, 1500, 40.0, 100],
        ["L", 302.5, 310.5, 1350, 62.0, 210]
    ],
    theme=gr.themes.Soft()
)

demo.launch(share=True)
