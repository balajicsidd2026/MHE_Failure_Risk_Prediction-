# Generated from: code.ipynb
# Converted at: 2026-06-01T11:00:25.197Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# **1. Import Libraries**


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# **2. Upload the Dataset**


dataset=pd.read_csv("dataset/Airport_MHE_Final_Dataset.csv")
dataset

noise_idx = dataset.sample(
    frac=0.05,
    random_state=42
).index

# Available classes
classes = ['Low', 'Medium', 'High']

# Replace selected rows with random classes
dataset.loc[noise_idx, 'Failure_Risk_Level'] = np.random.choice(
    classes,
    size=len(noise_idx)
)

# **3. Basic Checkups**


dataset['Failure_Risk_Level'].value_counts()

# Shape
print("Shape of the dataset:")
print(dataset.shape)
print("\n")


# Columns
print("Columns in the dataset:")
print(dataset.columns)

#check the null values in the dataset
dataset.isnull().sum()

#information about the dataset
dataset.info()

#statistical summary of the dataset
dataset.describe()

# **4. Exploratory Data Analysis**


# 4.1 Target Distribution


plt.figure(figsize=(8,5))

ax = sns.countplot(
    x='Failure_Risk_Level',
    data=dataset,
    order=['Low','Medium','High']
)

for p in ax.patches:
    ax.annotate(
        f'{int(p.get_height())}',
        (p.get_x() + p.get_width()/2, p.get_height()),
        ha='center',
        va='bottom'
    )

plt.title('Failure Risk Distribution')
plt.xlabel('Risk Level')
plt.ylabel('Count')

plt.show()

# 4.2 Equipment type vs failure risk


# ====================================
# 4.2 EQUIPMENT TYPE VS FAILURE RISK
# ====================================

plt.figure(figsize=(12,6))

ax = sns.countplot(
    x='Equipment_Type',
    hue='Failure_Risk_Level',
    data=dataset
)

for container in ax.containers:
    ax.bar_label(container)

plt.title('Equipment Type vs Failure Risk')

plt.xticks(rotation=30)

plt.show()

# 4.3 Operating Hours by Failure Risk
# 


plt.figure(figsize=(8,5))

sns.barplot(
    x='Failure_Risk_Level',
    y='Daily_Operating_Hours',
    data=dataset,
    order=['Low','Medium','High']
)

plt.title('Operating Hours by Failure Risk')
plt.xlabel('Failure Risk Level')
plt.ylabel('Daily Operating Hours')

plt.show()

# 4.4 Equipment Age vs Failure risk


plt.figure(figsize=(8,5))

sns.barplot(
    x='Failure_Risk_Level',
    y='Equipment_Age_Year',
    data=dataset,
    order=['Low','Medium','High']
)

plt.title('Equipment Age by Failure Risk')
plt.xlabel('Failure Risk Level')
plt.ylabel('Equipment Age (Years)')

plt.show()

# 4.5 Month-Wise Failure risk trend
# 


# Convert date column
dataset['Record_Date'] = pd.to_datetime(
    dataset['Record_Date'],
    format='mixed',
    dayfirst=True,
    errors='coerce'
)

# Create Month-Year column
dataset['Month_Year'] = dataset['Record_Date'].dt.strftime('%Y-%m')

# Count risk levels by month
monthly_risk = pd.crosstab(
    dataset['Month_Year'],
    dataset['Failure_Risk_Level']
)

# Plot
ax = monthly_risk.plot(
    figsize=(15,6),
    marker='o'
)

plt.title('Monthly Failure Risk Trend')
plt.xlabel('Month')
plt.ylabel('Record Count')

# Show values on line points
for line in ax.lines:
    for x, y in zip(line.get_xdata(), line.get_ydata()):
        plt.annotate(
            str(int(y)),
            (x, y),
            textcoords="offset points",
            xytext=(0,5),
            ha='center',
            fontsize=7
        )

plt.xticks(rotation=90)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()

# **5. Feature Engineering**


# 5.1 Dataset Splitting
# 


dataset['Record_Date'] = pd.to_datetime(
    dataset['Record_Date']
)
dataset['Year'] = dataset['Record_Date'].dt.year
dataset['Month'] = dataset['Record_Date'].dt.month
dataset['Quarter'] = dataset['Record_Date'].dt.quarter


# Columns to Drop
drop_columns = [
    'Manufacturer',
    'Failure_Risk_Level',
    'Record_Date',
    'Fuel_Type',
    'Month_Year',
    'Fleet_Category',
    'Equipment_Type',
    'Warehouse_Zone'
]

# Features
X = dataset.drop(drop_columns, axis=1)

# Target
y = dataset['Failure_Risk_Level']

# Check Shape
print("X Shape :", X.shape)
print("y Shape :", y.shape)

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)

print("X_train :", X_train.shape)
print("X_val   :", X_val.shape)
print("X_test  :", X_test.shape)

# Backup for CatBoost
X_train_cat = X_train.copy()
X_val_cat = X_val.copy()
X_test_cat = X_test.copy()

y_train_cat = y_train.copy()
y_val_cat = y_val.copy()
y_test_cat = y_test.copy()

X_train_cat

categorical_features = X_train.select_dtypes(
    include=['object','category']
).columns.tolist()

print("Categorical Features:")
print(categorical_features)

print("\nTotal Categorical Features:")
print(len(categorical_features))

feature_encoders = {}

for column in categorical_features:
    le = LabelEncoder()
    X_train[column] = le.fit_transform(X_train[column])
    X_val[column] = le.transform(X_val[column])
    X_test[column] = le.transform(X_test[column])
    feature_encoders[column] = le

print("Input Features Encoded Successfully")

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)
y_test_encoded = label_encoder.transform(y_test)
print(label_encoder.classes_)

X_train

# **6. Model Building**


# 6.1 Random Forest


rf_model = RandomForestClassifier(
    n_estimators=250,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

# Train Model

rf_model.fit(X_train, y_train_encoded)

# Validation Prediction

y_val_pred_rf = rf_model.predict(X_val)

print("Random Forest Trained Successfully")

# Accuracy
rf_accuracy = accuracy_score(
    y_val_encoded,
    y_val_pred_rf
)

# F1 Score
rf_f1 = f1_score(
    y_val_encoded,
    y_val_pred_rf,
    average='weighted'
)
rf_recall = recall_score(
    y_val_encoded,
    y_val_pred_rf,
    average='weighted'
)
rf_precision = precision_score(
    y_val_encoded,
    y_val_pred_rf,
    average='weighted'
)

print("Accuracy :", rf_accuracy)
print("Precision :", rf_precision)
print("Recall :", rf_recall)
print("F1 Score :", rf_f1)

print("\nConfusion Matrix:\n")

print(
    confusion_matrix(
        y_val_encoded,
        y_val_pred_rf
    )
)

print("\nClassification Report:\n")

print(
    classification_report(
        y_val_encoded,
        y_val_pred_rf
    )
)

# 6.2 Xgboost 


# Initialize Model
xgb_model = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    random_state=42
)

# Train Model
xgb_model.fit(
    X_train,
    y_train_encoded
)

# Validation Prediction
y_val_pred_xgb = xgb_model.predict(X_val)

print("XGBoost Model Trained Successfully")

# Accuracy
xgb_accuracy = accuracy_score(
    y_val_encoded,
    y_val_pred_xgb
)

# F1 Score
xgb_f1 = f1_score(
    y_val_encoded,
    y_val_pred_xgb,
    average='weighted'
)
xgb_recall = recall_score(
    y_val_encoded,
    y_val_pred_xgb,
    average='weighted'
)
xgb_precision = precision_score(
    y_val_encoded,
    y_val_pred_xgb,
    average='weighted'
)

print("Accuracy :", xgb_accuracy)
print("Precision :", xgb_precision)
print("Recall :", xgb_recall)
print("F1 Score :", xgb_f1)

print("\nConfusion Matrix:\n")

print(
    confusion_matrix(
        y_val_encoded,
        y_val_pred_xgb
    )
)

print("\nClassification Report:\n")

print(
    classification_report(
        y_val_encoded,
        y_val_pred_xgb
    )
)

# 6.3 LightGBM 


# Initialize Model
lgbm_model = LGBMClassifier(
    objective='multiclass',
    num_class=3,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

# Train Model
lgbm_model.fit(
    X_train,
    y_train_encoded
)

# Validation Prediction
y_val_pred_lgbm = lgbm_model.predict(X_val)

print("LightGBM Model Trained Successfully")

lgbm_accuracy = accuracy_score(
    y_val_encoded,
    y_val_pred_lgbm
)

lgbm_f1 = f1_score(
    y_val_encoded,
    y_val_pred_lgbm,
    average='weighted'
)
lgbm_recall = recall_score(
    y_val_encoded,
    y_val_pred_lgbm,
    average='weighted'
)
lgbm_precision = precision_score(
    y_val_encoded,
    y_val_pred_lgbm,
    average='weighted'
)

print("Accuracy :", lgbm_accuracy)
print("Precision :", lgbm_precision)
print("Recall :", lgbm_recall)
print("F1 Score :", lgbm_f1)

print("\nConfusion Matrix:\n")

print(
    confusion_matrix(
        y_val_encoded,
        y_val_pred_lgbm
    )
)

print("\nClassification Report:\n")

print(
    classification_report(
        y_val_encoded,
        y_val_pred_lgbm
    )
)

# 6.4 Catboost


cat_features = X_train_cat.select_dtypes(
    include=['object', 'category']
).columns.tolist()

print(cat_features)

# Initialize Model
cat_model = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    loss_function='MultiClass',
    random_state=42,
    verbose=0
)

# Train Model
cat_model.fit(
    X_train_cat,
    y_train_cat,
    cat_features=cat_features
)

# Validation Prediction
y_val_pred_cat = cat_model.predict(X_val_cat)

# Convert prediction shape
y_val_pred_cat = y_val_pred_cat.flatten()

print("CatBoost Model Trained Successfully")

# =====================================
# CHECK TRAINING FEATURES
# =====================================

print("Shape:")
print(X_train_cat.shape)

print("\nColumns:")
print(X_train_cat.columns.tolist())

print("\nCategorical Feature Indexes:")
print(cat_features)

print(len(X_train_cat.columns.tolist()))

# Accuracy
cat_accuracy = accuracy_score(
    y_val_cat,
    y_val_pred_cat
)

# Precision
cat_precision = precision_score(
    y_val_cat,
    y_val_pred_cat,
    average='weighted'
)

# Recall
cat_recall = recall_score(
    y_val_cat,
    y_val_pred_cat,
    average='weighted'
)

# F1 Score
cat_f1 = f1_score(
    y_val_cat,
    y_val_pred_cat,
    average='weighted'
)

print("Accuracy :", cat_accuracy)
print("Precision :", cat_precision)
print("Recall :", cat_recall)
print("F1 Score :", cat_f1)

print("\nConfusion Matrix:\n")

print(
    confusion_matrix(
        y_val_cat,
        y_val_pred_cat
    )
)

print("\nClassification Report:\n")

print(
    classification_report(
        y_val_cat,
        y_val_pred_cat
    )
)

# **7. Model Comparision**
# 



results = pd.DataFrame({
    'Model': [
        'Random Forest',
        'XGBoost',
        'LightGBM',
        'CatBoost'
    ],

    'Accuracy': [rf_accuracy, xgb_accuracy, lgbm_accuracy, cat_accuracy],
    'Precision': [rf_precision, xgb_precision, lgbm_precision, cat_precision],

    'Recall': [rf_recall, xgb_recall, lgbm_recall, cat_recall],

    'F1 Score': [rf_f1, xgb_f1, lgbm_f1, cat_f1]})

results = results.sort_values(
    by='F1 Score',
    ascending=False
)

results

# **8. Feature Importance**


feature_importance = pd.DataFrame({
    'Feature': X_train_cat.columns,
    'Importance': cat_model.get_feature_importance()
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

feature_importance

top_features = feature_importance.head(10)

plt.figure(figsize=(10,6))

ax = sns.barplot(
    data=top_features,
    x='Importance',
    y='Feature'
)

# Add values on bars
for p in ax.patches:

    width = p.get_width()

    ax.annotate(
        f'{width:.2f}',
        (
            width,
            p.get_y() + p.get_height()/2
        ),
        ha='left',
        va='center',
        fontsize=9,
        fontweight='bold'
    )

plt.title('Top 15 Important Features - CatBoost')
plt.xlabel('Importance Score')
plt.ylabel('Features')

plt.tight_layout()
plt.show()

# **9. Test the model**


y_test_pred_cat = cat_model.predict(X_test_cat)

y_test_pred_cat = y_test_pred_cat.flatten()

test_accuracy = accuracy_score(
    y_test_cat,
    y_test_pred_cat
)

test_precision = precision_score(
    y_test_cat,
    y_test_pred_cat,
    average='weighted'
)

test_recall = recall_score(
    y_test_cat,
    y_test_pred_cat,
    average='weighted'
)

test_f1 = f1_score(
    y_test_cat,
    y_test_pred_cat,
    average='weighted'
)

print("Test Accuracy :", test_accuracy)
print("Test Precision :", test_precision)
print("Test Recall :", test_recall)
print("Test F1 Score :", test_f1)

sample = X_test_cat.iloc[[3000]]

prediction = cat_model.predict(sample)

print("Prediction :", prediction)

# **10. Save the Model**


with open('model/feature_encoders.pkl', 'wb') as file:
    pickle.dump(feature_encoders, file)

with open('model/catboost_mhe_model.pkl', 'wb') as file:
    pickle.dump(cat_model, file)

print("CatBoost Model Saved Successfully")