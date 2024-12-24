# Student ID: 12345678
# Name: John Doe

# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Load dataset
df = pd.read_csv('winequality-white.csv', sep=';')

# Classify wine quality into categories
def classify_quality(score):
    if score <= 5:
        return 'Low'
    elif score == 6:
        return 'Medium'
    else:
        return 'High'

df['quality_label'] = df['quality'].apply(classify_quality)

# Split data into features and labels
X = df.drop(['quality', 'quality_label'], axis=1)
y = df['quality_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define classification models
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM': SVC(random_state=42, probability=True)
}

# Train and evaluate models
model_results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    model_results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'F1 Score': f1_score(y_test, y_pred, average='weighted')
    }

# Streamlit App
st.title('White Wine Quality Classification')
st.write('Predict the quality of white wine based on chemical attributes.')

# User input for wine attributes
def user_input_features():
    with st.sidebar:
        st.header('Input Wine Attributes')
        fixed_acidity = st.slider('Fixed Acidity', 0.0, 15.0, 7.0)
        volatile_acidity = st.slider('Volatile Acidity', 0.0, 2.0, 0.5)
        citric_acid = st.slider('Citric Acid', 0.0, 1.0, 0.3)
        residual_sugar = st.slider('Residual Sugar', 0.0, 50.0, 20.0)
        chlorides = st.slider('Chlorides', 0.0, 1.0, 0.05)
        free_sulfur_dioxide = st.slider('Free Sulfur Dioxide', 0.0, 100.0, 30.0)
        total_sulfur_dioxide = st.slider('Total Sulfur Dioxide', 0.0, 400.0, 150.0)
        density = st.slider('Density', 0.9, 1.1, 1.0)
        pH = st.slider('pH', 0.0, 14.0, 3.3)
        sulphates = st.slider('Sulphates', 0.0, 2.0, 0.5)
        alcohol = st.slider('Alcohol', 0.0, 20.0, 10.0)
    
    data = {
        'fixed_acidity': fixed_acidity,
        'volatile_acidity': volatile_acidity,
        'citric_acid': citric_acid,
        'residual_sugar': residual_sugar,
        'chlorides': chlorides,
        'free_sulfur_dioxide': free_sulfur_dioxide,
        'total_sulfur_dioxide': total_sulfur_dioxide,
        'density': density,
        'pH': pH,
        'sulphates': sulphates,
        'alcohol': alcohol
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Model selection
model_name = st.selectbox('Choose a Model', list(models.keys()))
model = models[model_name]

# Evaluate selected model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')

st.write(f'### Model Performance')
st.write(f'**Accuracy:** {accuracy:.2f}')
st.write(f'**F1 Score:** {f1:.2f}')
st.write(f'**Precision:** {precision:.2f}')

# Display Classification Report
st.write('### Classification Report')
st.text(classification_report(y_test, y_pred))

# Feature Importance Visualization (Random Forest only)
if model_name == 'Random Forest':
    feature_importances = model.feature_importances_
    plt.figure(figsize=(10, 5))
    sns.barplot(x=X.columns, y=feature_importances)
    plt.title('Feature Importance')
    plt.xticks(rotation=45)
    st.pyplot(plt)

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
st.pyplot(plt)

# Predict user input
if st.button('Predict Quality'):
    prediction = model.predict(input_df)
    st.write(f'## Predicted Wine Quality: **{prediction[0]}**')

st.write('### Model Comparison')
st.write(pd.DataFrame(model_results).T)
