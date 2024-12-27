# 11890054
# 李彥緯 Rex

# Import required libraries
import pandas as pd  # For data manipulation and analysis
from sklearn.model_selection import train_test_split  # For splitting dataset into training and testing sets
from sklearn.preprocessing import StandardScaler  # For standardizing features
import matplotlib.pyplot as plt  # For creating plots
import seaborn as sns  # For advanced visualizations
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score  # For evaluating model performance
import streamlit as st  # For building the interactive web application
from sklearn.ensemble import RandomForestClassifier  # Random Forest classifier model
from sklearn.linear_model import LogisticRegression  # Logistic Regression model
from sklearn.svm import SVC  # Support Vector Classifier model

# Load dataset
df = pd.read_csv('winequality-white.csv', sep=';')  # Load white wine dataset with semicolon separator

# Classify wine quality into categories
def classify_quality(score):
    if score <= 5:
        return 'Low'  # Scores 5 and below are categorized as 'Low'
    elif score == 6:
        return 'Medium'  # Score of 6 is categorized as 'Medium'
    else:
        return 'High'  # Scores above 6 are categorized as 'High'

df['quality_label'] = df['quality'].apply(classify_quality)  # Apply classification to the 'quality' column

# Split data into features and labels
X = df.drop(['quality', 'quality_label'], axis=1)  # Features exclude 'quality' and 'quality_label'
y = df['quality_label']  # Target labels are the 'quality_label'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)  # Stratified split into training and test sets

# Standardize data
scaler = StandardScaler()  # Initialize the scaler
X_train_scaled = scaler.fit_transform(X_train)  # Fit scaler on training data and transform it
X_test_scaled = scaler.transform(X_test)  # Transform test data using the same scaler

# Define classification models
models = {
    'Random Forest': RandomForestClassifier(random_state=42),  # Random Forest model
    'Logistic Regression': LogisticRegression(max_iter=1000),  # Logistic Regression with increased iterations
    'SVM': SVC(random_state=42, probability=True)  # Support Vector Classifier with probability enabled
}

# Train and evaluate models
model_results = {}  # Dictionary to store evaluation results
for name, model in models.items():
    model.fit(X_train_scaled, y_train)  # Train each model
    y_pred = model.predict(X_test_scaled)  # Make predictions on test data
    model_results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),  # Calculate accuracy
        'Precision': precision_score(y_test, y_pred, average='weighted'),  # Calculate precision
        'F1 Score': f1_score(y_test, y_pred, average='weighted')  # Calculate F1 score
    }

# Streamlit App
st.title('White Wine Quality Classification')  # App title
st.write('Predict the quality of white wine based on chemical attributes.')  # App description

# User input for wine attributes
def user_input_features():
    with st.sidebar:
        st.header('Input Wine Attributes')  # Sidebar header
        # Collect user inputs with sliders
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
    
    # Store inputs in a dictionary and return as a DataFrame
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

input_df = user_input_features()  # Get user inputs

# Model selection
model_name = st.selectbox('Choose a Model', list(models.keys()))  # Select model from dropdown
model = models[model_name]  # Load selected model

# Evaluate selected model
y_pred = model.predict(X_test_scaled)  # Predict on test data
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')

# Display model performance
st.write(f'### Model Performance')
st.write(f'**Accuracy:** {accuracy:.2f}')
st.write(f'**F1 Score:** {f1:.2f}')
st.write(f'**Precision:** {precision:.2f}')

# Display classification report
st.write('### Classification Report')
st.text(classification_report(y_test, y_pred))

# Feature Importance (for Random Forest only)
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

# Display Model Comparison
st.write('### Model Comparison')
st.write(pd.DataFrame(model_results).T)  # Display comparison table
