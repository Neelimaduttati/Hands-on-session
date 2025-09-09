import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils.load_data import load_data

sns.set(style="whitegrid")
st.set_page_config(page_title="Home Credit Dashboard", layout="wide")

# --- Load dataset ---
csv_path =("C:\\Users\\dell\\OneDrive\\Desktop\\project\\application_train.csv")
df = load_data(csv_path)

# --- Feature engineering (common) ---
df['AGE_YEARS'] = (df['DAYS_BIRTH'].abs() / 365).astype(int)
df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
df['EMPLOYMENT_YEARS'] = (df['DAYS_EMPLOYED'].abs() / 365).replace([np.inf, 0], np.nan)
df['TARGET'] = df['TARGET'].astype(int)
df['CNT_CHILDREN'] = df['CNT_CHILDREN'].apply(lambda x: 0 if pd.isna(x) or x < 0 else int(x))
df['CNT_FAM_MEMBERS'] = pd.to_numeric(df['CNT_FAM_MEMBERS'], errors='coerce')

# --- Tabs ---
tabs = st.tabs([
    "Overview & Data Quality",
    "Default Risk Segmentation",
    "Demographics & Employment",
    "Financial Health",
    "Correlation Analysis"
])

# ------------------------------
# Tab 1: Overview & Data Quality
# ------------------------------
with tabs[0]:
    st.title("Overview & Data Quality")
    total_applicants = df['SK_ID_CURR'].nunique()
    default_rate = df['TARGET'].mean() * 100
    repaid_rate = 100 - default_rate
    total_features = df.shape[1]
    avg_missing_per_feature = df.isnull().mean().mean() * 100
    num_features_count = len(df.select_dtypes(include=[np.number]).columns)
    cat_features_count = len(df.select_dtypes(exclude=[np.number]).columns)
    median_age = df['AGE_YEARS'].median()
    median_income = df['AMT_INCOME_TOTAL'].median()
    avg_credit = df['AMT_CREDIT'].mean()

    kpi_cols = st.columns(5)
    kpi_cols[0].metric("Total Applicants", total_applicants)
    kpi_cols[1].metric("Default Rate (%)", round(default_rate,2))
    kpi_cols[2].metric("Repaid Rate (%)", round(repaid_rate,2))
    kpi_cols[3].metric("Total Features", total_features)
    kpi_cols[4].metric("Avg Missing per Feature (%)", round(avg_missing_per_feature,2))

    st.subheader("Distribution Plots")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        df['TARGET'].value_counts().plot.pie(
            autopct='%1.1f%%',
            labels=['Repaid','Default'],
            ax=ax
        )
        ax.set_title("Target Distribution")
        st.pyplot(fig)

    with col2:
        missing = df.isnull().mean().sort_values(ascending=False)[:20] * 100
        fig, ax = plt.subplots(figsize=(8,4))
        missing.plot(kind='bar', ax=ax)
        ax.set_title("Top 20 Features by Missing %")
        ax.set_ylabel("% Missing")
        st.pyplot(fig)
# Tab 2: Default Risk Segmentation
# ------------------------------
with tabs[1]:
    st.title("Target & Risk Segmentation")
    total_defaults = int(df['TARGET'].sum())
    default_rate_pct = df['TARGET'].mean()*100
    avg_income_defaulters = df.loc[df['TARGET']==1,'AMT_INCOME_TOTAL'].mean()
    avg_credit_defaulters = df.loc[df['TARGET']==1,'AMT_CREDIT'].mean()
    avg_annuity_defaulters = df.loc[df['TARGET']==1,'AMT_ANNUITY'].mean()
    avg_emp_years_defaulters = df.loc[df['TARGET']==1,'EMPLOYMENT_YEARS'].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Defaults", total_defaults)
    col2.metric("Default Rate (%)", f"{round(default_rate_pct,2)}%")
    col3.metric("Avg Income (Defaulters)", f"{round(avg_income_defaulters,2)}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Avg Credit (Defaulters)", f"{round(avg_credit_defaulters,2)}")
    col5.metric("Avg Annuity (Defaulters)", f"{round(avg_annuity_defaulters,2)}")
    col6.metric("Avg Employment Years (Defaulters)", f"{round(avg_emp_years_defaulters,2)}")

    st.markdown("---")
    st.subheader("Default Rate by Demographics")
    col1, col2 = st.columns(2)
    default_by_gender = df.groupby('CODE_GENDER')['TARGET'].mean()*100
    default_by_education = df.groupby('NAME_EDUCATION_TYPE')['TARGET'].mean()*100
    default_by_family = df.groupby('NAME_FAMILY_STATUS')['TARGET'].mean()*100
    default_by_housing = df.groupby('NAME_HOUSING_TYPE')['TARGET'].mean()*100

    with col1:
        fig, ax = plt.subplots()
        default_by_gender.sort_values(ascending=False).plot(kind='bar', ax=ax)
        ax.set_title("Default Rate by Gender (%)")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        default_by_education.sort_values(ascending=False).plot(kind='bar', ax=ax)
        ax.set_title("Default Rate by Education (%)")
        st.pyplot(fig)

# ------------------------------
# Tab 3: Demographics & Employment
# ------------------------------
with tabs[2]:
    st.title("Demographic Insights")
    avg_age_def = df.loc[df['TARGET']==1,'AGE_YEARS'].mean()
    avg_age_nondef = df.loc[df['TARGET']==0,'AGE_YEARS'].mean()
    pct_with_children = (df['CNT_CHILDREN'].gt(0).mean()*100).round(2)
    avg_family_size = df['CNT_FAM_MEMBERS'].mean()
    edu_ser = df['NAME_EDUCATION_TYPE'].fillna('Unknown')
    pct_higher_edu = (edu_ser.isin(['Higher education','Academic degree'])).mean()*100
    pct_with_parents = (df['NAME_HOUSING_TYPE']=='With parents').mean()*100
    pct_currently_working = df['EMPLOYMENT_YEARS'].notna() & (df['EMPLOYMENT_YEARS']>0)
    avg_employment_years = df['EMPLOYMENT_YEARS'].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Age - Defaulters", round(avg_age_def,2))
    col2.metric("Avg Age - Non-Defaulters", round(avg_age_nondef,2))
    col3.metric("% With Children", pct_with_children)

# ------------------------------
# Tab 4: Financial Health & Affordability
# ------------------------------
with tabs[3]:
    st.title("Financial Health & Affordability")
    fin_cols = ["AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY","AMT_GOODS_PRICE","TARGET"]
    fin = df[fin_cols].copy()
    fin.replace({0: np.nan}, inplace=True)
    fin["DTI"] = fin["AMT_ANNUITY"]/fin["AMT_INCOME_TOTAL"]
    fin["LTI"] = fin["AMT_CREDIT"]/fin["AMT_INCOME_TOTAL"]

    st.subheader("Key Financial KPIs")
    kpis = {
        "Avg Income": fin["AMT_INCOME_TOTAL"].mean(),
        "Avg Credit": fin["AMT_CREDIT"].mean(),
        "Avg Annuity": fin["AMT_ANNUITY"].mean(),
        "Avg DTI": fin["DTI"].mean(),
        "Avg LTI": fin["LTI"].mean()
    }
    st.table(pd.DataFrame.from_dict(kpis, orient="index", columns=["Value"]))

# ------------------------------
# Tab 5: Correlation Analysis
# ------------------------------
with tabs[4]:
    st.title("Correlation Insights & KPIs")
    numeric_df = df.select_dtypes(include=['int64','float64']).copy()
    numeric_df['AGE_YEARS'] = df['AGE_YEARS']
    numeric_df['EMPLOYMENT_YEARS'] = df['EMPLOYMENT_YEARS']

    corr_series = numeric_df.corr()['TARGET'].drop('TARGET').sort_values()
    st.subheader("Top Correlations with TARGET")
    st.table(pd.concat([corr_series.head(5), corr_series.tail(5)]))
    
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(numeric_df[['TARGET','AGE_YEARS','EMPLOYMENT_YEARS','AMT_INCOME_TOTAL','AMT_CREDIT']].corr(), 
                annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    st.pyplot(fig)