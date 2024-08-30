pip install pandas scikit-learn imbalanced-learn xgboost seaborn matplotlib

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('creditcard.csv')
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])
df['Time'] = scaler.fit_transform(df[['Time']])

X = df.drop('Class', axis=1)
y = df['Class']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)
gb_predictions = gb_model.predict(X_test)

xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)

def evaluate_model(model_name, predictions, y_test):
    print(f"Performance of {model_name}:")
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    print("\n")

evaluate_model("Random Forest", rf_predictions, y_test)
evaluate_model("Gradient Boosting", gb_predictions, y_test)
evaluate_model("XGBoost", xgb_predictions, y_test)

importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({'feature': X_train.columns, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_df)
plt.title('Feature Importance - Random Forest')
plt.show()

cv_scores_rf = cross_val_score(rf_model, X_resampled, y_resampled, cv=5)
print(f"Random Forest CV Accuracy: {cv_scores_rf.mean()}")
cv_scores_gb = cross_val_score(gb_model, X_resampled, y_resampled, cv=5)
print(f"Gradient Boosting CV Accuracy: {cv_scores_gb.mean()}")
cv_scores_xgb = cross_val_score(xgb_model, X_resampled, y_resampled, cv=5)
print(f"XGBoost CV Accuracy: {cv_scores_xgb.mean()}")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='f1')
grid_search.fit(X_train, y_train)

print(f"Best parameters for Random Forest: {grid_search.best_params_}")

joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(gb_model, 'gradient_boosting_model.pkl')
joblib.dump(xgb_model, 'xgboost_model.pkl')
