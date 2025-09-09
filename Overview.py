# page 1
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("C:\\Users\\dell\\OneDrive\\Desktop\\project\\application_train.csv")
df['AGE_YEARS'] = abs(df['DAYS_BIRTH']) // 365  # Age
df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True) 

num_features = df.select_dtypes(include=[np.number]).columns
cat_features = df.select_dtypes(exclude=[np.number]).columns

# KPIs
total_applicants = df['SK_ID_CURR'].nunique()
default_rate = df['TARGET'].mean() * 100
repaid_rate = (1 - df['TARGET'].mean()) * 100
total_features = df.shape[1]
avg_missing_per_feature = df.isnull().mean().mean() * 100
num_features_count = len(num_features)
cat_features_count = len(cat_features)
median_age = df['AGE_YEARS'].median()
median_income = df['AMT_INCOME_TOTAL'].median()
avg_credit = df['AMT_CREDIT'].mean()

kpis = {
    "Total Applicants": total_applicants,
    "Default Rate (%)": round(default_rate, 2),
    "Repaid Rate (%)": round(repaid_rate, 2),
    "Total Features": total_features,
    "Avg Missing per Feature (%)": round(avg_missing_per_feature, 2),
    "Numerical Features": num_features_count,
    "Categorical Features": cat_features_count,
    "Median Age (Years)": median_age,
    "Median Annual Income": median_income,
    "Average Credit Amount": avg_credit
}

# ---------------- Streamlit App ----------------
st.title("Overview & Data Quality")

# Show KPIs
st.metric("Total Applicants", total_applicants)
st.metric("Default Rate (%)", round(default_rate, 2))
st.metric("Repaid Rate (%)", round(repaid_rate, 2))
st.metric("Total Features", total_features)
st.metric("Avg Missing per Feature (%)", round(avg_missing_per_feature, 2))
st.metric("Numerical Features", num_features_count)
st.metric("Categorical Features", cat_features_count)
st.metric("Median Age (Years)", median_age)
st.metric("Median Annual Income", median_income)
st.metric("Average Credit Amount", avg_credit)

# ---------------- Plots ----------------

# 1. Target distribution (Pie)
fig, ax = plt.subplots()
df['TARGET'].value_counts().plot.pie(
    autopct='%1.1f%%',
    labels=['Repaid', 'Default'],
    ax=ax
)
ax.set_title("Target Distribution")
st.pyplot(fig)

# 2. Missing values (Top 20 features)
missing = df.isnull().mean().sort_values(ascending=False)[:20] * 100
fig, ax = plt.subplots(figsize=(10,5))
missing.plot(kind='bar', ax=ax)
ax.set_title("Top 20 Features by Missing %")
ax.set_ylabel("% Missing")
st.pyplot(fig)

# 3. Histogram - Age
fig, ax = plt.subplots()
sns.histplot(df['AGE_YEARS'], bins=30, kde=False, ax=ax)
ax.set_title("Age Distribution")
st.pyplot(fig)

# 4. Histogram - Income
fig, ax = plt.subplots()
sns.histplot(df['AMT_INCOME_TOTAL'], bins=50, ax=ax)
ax.set_title("Income Distribution")
ax.set_xlim(0, 500000)
st.pyplot(fig)

# 5. Histogram - Credit Amount
fig, ax = plt.subplots()
sns.histplot(df['AMT_CREDIT'], bins=50, ax=ax)
ax.set_title("Credit Amount Distribution")
ax.set_xlim(0, 2000000)
st.pyplot(fig)

# 6. Boxplot - Income
fig, ax = plt.subplots()
sns.boxplot(x=df['AMT_INCOME_TOTAL'], ax=ax)
ax.set_title("Income Boxplot")
ax.set_xlim(0, 500000)
st.pyplot(fig)

# 7. Boxplot - Credit Amount
fig, ax = plt.subplots()
sns.boxplot(x=df['AMT_CREDIT'], ax=ax)
ax.set_title("Credit Amount Boxplot")
ax.set_xlim(0, 2000000)
st.pyplot(fig)

# 8. Countplot - Gender
fig, ax = plt.subplots()
sns.countplot(x='CODE_GENDER', data=df, ax=ax)
ax.set_title("Applicants by Gender")
st.pyplot(fig)

# 9. Countplot - Family Status
fig, ax = plt.subplots()
sns.countplot(
    x='NAME_FAMILY_STATUS',
    data=df,
    order=df['NAME_FAMILY_STATUS'].value_counts().index,
    ax=ax
)
ax.set_title("Applicants by Family Status")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
st.pyplot(fig)

# 10. Countplot - Education
fig, ax = plt.subplots()
sns.countplot(
    x='NAME_EDUCATION_TYPE',
    data=df,
    order=df['NAME_EDUCATION_TYPE'].value_counts().index,
    ax=ax
)
ax.set_title("Applicants by Education Level")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
st.pyplot(fig)