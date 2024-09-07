import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Load your data here
# For example:
# df = pd.read_csv('your_data.csv')
# features = df.drop(columns=['target_column'])
# label = df['target_column']

# Example data loading (replace with your own logic)
def load_data():
    # Replace this with actual data loading logic
    df = pd.read_csv('yield_df.csv')
    features = df.drop(columns=['target_column'])
    label = df['target_column']
    return features, label

# Data preprocessing
features, label = load_data()
train_data, test_data, train_labels, test_labels = train_test_split(features, label, test_size=0.3, random_state=42)

# Create a DataFrame for predictions (example)
test_df = pd.DataFrame(test_data)
test_df['target'] = test_labels

def compare_models(model):
    model_name = model.__class__.__name__
    fit = model.fit(train_data, train_labels)
    y_pred = fit.predict(test_data)
    r2 = r2_score(test_labels, y_pred)
    return [model_name, r2]

# Define models
models = [
    GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=0),
    RandomForestRegressor(n_estimators=200, max_depth=3, random_state=0),
    svm.SVR(gamma='auto'),
    DecisionTreeRegressor()
]

# Compare models
model_train = list(map(compare_models, models))

st.title('Model Comparison & Selection')

st.subheader('Model Comparison Results')
st.write(pd.DataFrame(model_train, columns=['Model', 'R^2 Score']))

# Fit the Decision Tree model
clf = DecisionTreeRegressor()
model = clf.fit(train_data, train_labels)

# Predictions
test_df["yield_predicted"] = model.predict(test_data)
test_df["yield_actual"] = test_labels.tolist()

# Scatter plot of actual vs predicted
st.subheader('Actual vs Predicted Plot')
fig, ax = plt.subplots()
ax.scatter(test_df["yield_actual"], test_df["yield_predicted"], edgecolors=(0, 0, 0))
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title("Actual vs Predicted")
st.pyplot(fig)

# Calculate and display R^2 scores by group
test_group = test_df.groupby("Item")
r2_by_group = test_group.apply(lambda x: r2_score(x.yield_actual, x.yield_predicted))
st.subheader('R^2 Scores by Group')
st.write(r2_by_group)

# Adjusted R^2 function
def adjusted_r_squared(y, yhat, x):
    score = 1 - (((1 - r2_score(y, yhat)) * (len(y) - 1)) / (len(y) - x.shape[1] - 2))
    return score

adjusted_r2_by_group = test_group.apply(lambda x: adjusted_r_squared(x.yield_actual, x.yield_predicted, x))
st.subheader('Adjusted R^2 Scores by Group')
st.write(adjusted_r2_by_group)

# Feature importance
varimp = {'importances': model.feature_importances_, 'names': features.columns}

# Plot feature importances
st.subheader('Feature Importances')
a4_dims = (10, 30)
fig, ax = plt.subplots(figsize=a4_dims)
df = pd.DataFrame.from_dict(varimp)
df.sort_values(ascending=False, by=["importances"], inplace=True)
df = df.dropna()
sns.barplot(x="importances", y="names", palette="vlag", data=df, orient="h", ax=ax)
st.pyplot(fig)

# Plot top 7 important features
st.subheader('Top 7 Important Features')
a4_dims = (16.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
df = pd.DataFrame.from_dict(varimp)
df.sort_values(ascending=False, by=["importances"], inplace=True)
df = df.dropna()
df = df.nlargest(7, 'importances')
sns.barplot(x="importances", y="names", palette="vlag", data=df, orient="h", ax=ax)
st.pyplot(fig)
