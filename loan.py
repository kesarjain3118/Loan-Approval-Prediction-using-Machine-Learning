# -----------------------------
# Loan Approval Prediction - Classification Mode
# -----------------------------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

# 1. Load datasets
train = pd.read_csv('/kaggle/input/playground-series-s4e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e10/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s4e10/sample_submission.csv')

# 2. Handle missing values
for col in train.select_dtypes(include='number').columns:
    train[col] = train[col].fillna(train[col].median())
    if col in test.columns:
        test[col] = test[col].fillna(train[col].median())

for col in train.select_dtypes(include='object').columns:
    train[col] = train[col].fillna(train[col].mode()[0])
    if col in test.columns:
        test[col] = test[col].fillna(train[col].mode()[0])

# 3. Encode categorical columns
le = LabelEncoder()
for col in train.select_dtypes(include='object').columns:
    train[col] = le.fit_transform(train[col])
    if col in test.columns:
        test[col] = le.transform(test[col])

# 4. Main Data Exploration Graphs
plt.figure(figsize=(6,4))
sns.countplot(x='loan_status', data=train)
plt.title("Target Distribution (Loan Status)")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x='loan_status', y='person_income', data=train)
plt.title("Applicant Income vs Loan Status")
plt.show()

# 5. Split features & target
X = train.drop('loan_status', axis=1)
y = train['loan_status']

# 6. Train-validation split (stratified to preserve class balance)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(test[X.columns])

# 8. Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_split=5,
                               min_samples_leaf=3, random_state=42)
model.fit(X_train_scaled, y_train)

# 9. Evaluate Model
y_val_pred = model.predict(X_val_scaled)
print("=== Validation Metrics ===")
print("Accuracy:", accuracy_score(y_val, y_val_pred))
print("\nClassification Report:\n", classification_report(y_val, y_val_pred))

# Confusion Matrix Graph
cm = confusion_matrix(y_val, y_val_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# 10. Feature Importance (compact)
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh', figsize=(8,6), color='skyblue')
plt.title("Feature Importance")
plt.show()

# 11. Predict on Test Set and Save Submission
test_pred = model.predict(X_test_scaled)
submission['loan_status'] = test_pred
submission.to_csv('loan_submission.csv', index=False)
print("Submission saved as 'loan_submission.csv'")
