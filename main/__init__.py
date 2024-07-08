import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import time

print("Libraries imported successfully.")

sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 14, 8

RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]

# Load data
data_path = 'C:/Users/hadad/PycharmProjects/Fraud_detection/data/creditcard.csv'
data = pd.read_csv(data_path)
print("Data loaded successfully.")

# Preprocess data
features = data.columns.drop(['Class', 'Time'])
X = data[features]
y = data['Class']
print("Data preprocessing completed.")

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Data standardized.")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=RANDOM_SEED)
print("Data split into train and test sets.")

# Define models
models = {
    "Logistic Regression": LogisticRegression(random_state=RANDOM_SEED),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_SEED),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_SEED, n_estimators=10, max_depth=5),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_SEED, n_estimators=10),
    "Support Vector Machine": SVC(probability=True, random_state=RANDOM_SEED),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=RANDOM_SEED, n_estimators=10),
    "LightGBM": LGBMClassifier(random_state=RANDOM_SEED, n_estimators=10)
}
print("Models defined.")

# Define scoring metrics
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'f1': make_scorer(f1_score),
    'roc_auc': make_scorer(roc_auc_score)
}
print("Scoring metrics defined.")

# Evaluate models
results = {}
for model_name, model in models.items():
    start_time = time.time()
    print(f"Evaluating {model_name}...")
    grid = GridSearchCV(estimator=model, param_grid={}, scoring=scoring, refit='roc_auc', cv=2)
    grid_result = grid.fit(X_train, y_train)
    results[model_name] = grid_result
    end_time = time.time()
    print(f"{model_name} evaluated in {end_time - start_time:.2f} seconds.")

# Summarize results
summary = []
for model_name, grid_result in results.items():
    summary.append({
        "Model": model_name,
        "Best Score (ROC AUC)": grid_result.best_score_,
        "Best Params": grid_result.best_params_
    })
    print(f"Model: {model_name}")
    print(f"Best Score (ROC AUC): {grid_result.best_score_}")
    print(f"Best Params: {grid_result.best_params_}")
    print("-" * 30)

# Create a DataFrame for summary
summary_df = pd.DataFrame(summary)

# Print the summary DataFrame
print(summary_df)

# Plot ROC AUC scores
plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="Best Score (ROC AUC)", hue="Model", data=summary_df, palette="muted", dodge=False)
plt.title("Comparison of ROC AUC Scores for Different Models")
plt.ylabel("Best Score (ROC AUC)")
plt.xlabel("Model")
plt.legend([], [], frameon=False)  # حذف کردن legend
plt.xticks(rotation=45)
plt.show()

# Plot Best Params as a heatmap
summary_df['Best Score (ROC AUC)'] = summary_df['Best Score (ROC AUC)'].astype(float)
pivot_table = summary_df.pivot_table(values="Best Score (ROC AUC)", index="Model")

plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, annot=True, cmap="viridis", fmt=".4f")
plt.title("Best ROC AUC Scores for Different Models")
plt.show()

print("Script finished.")
