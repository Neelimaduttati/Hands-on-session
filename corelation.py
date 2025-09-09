# page 5 — Correlation Analysis in Streamlit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

sns.set(style="whitegrid")

# --- Load dataset ---
df = pd.read_csv("C:\\Users\\dell\\OneDrive\\Desktop\\project\\application_train.csv")

# --- Feature engineering ---
df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365.25
df['EMPLOYMENT_YEARS'] = df['DAYS_EMPLOYED'].clip(upper=0) / -365.25

numeric_df = df.select_dtypes(include=['int64', 'float64']).copy()
numeric_df['AGE_YEARS'] = df['AGE_YEARS']
numeric_df['EMPLOYMENT_YEARS'] = df['EMPLOYMENT_YEARS']

# --- Correlations ---
corr_series = numeric_df.corr()['TARGET'].drop('TARGET').sort_values(ascending=False)

top5_pos = corr_series[corr_series > 0].nlargest(5)
top5_neg = corr_series.nsmallest(5)

corr_with_income = numeric_df.corr()['AMT_INCOME_TOTAL'].drop('AMT_INCOME_TOTAL').abs().sort_values(ascending=False)
most_corr_income = corr_with_income.idxmax()

corr_with_credit = numeric_df.corr()['AMT_CREDIT'].drop('AMT_CREDIT').abs().sort_values(ascending=False)
most_corr_credit = corr_with_credit.idxmax()

corr_income_credit = numeric_df['AMT_INCOME_TOTAL'].corr(numeric_df['AMT_CREDIT'])
corr_age_target = numeric_df['AGE_YEARS'].corr(numeric_df['TARGET'])
corr_emp_target = numeric_df['EMPLOYMENT_YEARS'].corr(numeric_df['TARGET'])
family_col = 'CNT_FAM_MEMBERS' if 'CNT_FAM_MEMBERS' in numeric_df.columns else None
corr_family_target = numeric_df[family_col].corr(numeric_df['TARGET']) if family_col else np.nan

abs_corr = corr_series.abs().sort_values(ascending=False)
top5_features = abs_corr.index[:5]
variance_explained_proxy = (corr_series[top5_features] ** 2).sum()

high_corr_count = (corr_series.abs() > 0.5).sum()

# --- Streamlit UI ---
st.title("Correlation Insights & KPIs")

# KPIs (metrics)
col1, col2, col3 = st.columns(3)
col1.metric("Most correlated with Income", most_corr_income)
col2.metric("Most correlated with Credit", most_corr_credit)
col3.metric("Corr(Income, Credit)", round(corr_income_credit, 4))

col4, col5, col6 = st.columns(3)
col4.metric("Corr(Age, TARGET)", round(corr_age_target, 4))
col5.metric("Corr(Employment Years, TARGET)", round(corr_emp_target, 4))
col6.metric("Corr(Family Size, TARGET)", round(corr_family_target, 4))

col7, col8 = st.columns(2)
col7.metric("Variance explained (Top 5 R² proxy)", round(variance_explained_proxy, 4))
col8.metric("# Features with |corr| > 0.5", int(high_corr_count))

st.markdown("---")

# --- Correlation Tables ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("Top 5 Positive Correlations with TARGET")
    st.table(top5_pos)

with col2:
    st.subheader("Top 5 Negative Correlations with TARGET")
    st.table(top5_neg)

st.markdown("---")

# --- Correlation Heatmap ---
st.subheader("Heatmap of Key Correlations")
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(
    numeric_df[['TARGET','AGE_YEARS','EMPLOYMENT_YEARS','AMT_INCOME_TOTAL','AMT_CREDIT']].corr(),
    annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax
)
st.pyplot(fig)

# --- Correlation distribution ---
st.subheader("Distribution of Feature Correlations with TARGET")
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(corr_series, bins=30, kde=False, ax=ax)
ax.set_title("Distribution of Correlations with TARGET")
ax.set_xlabel("Correlation with TARGET")
ax.set_ylabel("Feature Count")
st.pyplot(fig)