# page 3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

sns.set(style="whitegrid", rc={"figure.figsize": (8,5)})

# Load dataset
df = pd.read_csv("C:\\Users\\dell\\OneDrive\\Desktop\\project\\application_train.csv")

# --- Data Cleaning / Feature Engineering ---
if 'AGE_YEARS' not in df.columns:
    df['AGE_YEARS'] = (df['DAYS_BIRTH'].abs() / 365).astype(int)

df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
df['EMPLOYMENT_YEARS'] = (df['DAYS_EMPLOYED'].abs() / 365).replace([np.inf, 0], np.nan)
df['TARGET'] = df['TARGET'].astype(int)
df['CNT_CHILDREN'] = df['CNT_CHILDREN'].apply(lambda x: 0 if (pd.isna(x) or x < 0) else int(x))
df['CNT_FAM_MEMBERS'] = pd.to_numeric(df['CNT_FAM_MEMBERS'], errors='coerce')

# --- KPIs ---
gender_counts = df['CODE_GENDER'].value_counts(dropna=False)
pct_gender = (gender_counts / gender_counts.sum() * 100).round(2)
avg_age_def = df.loc[df['TARGET'] == 1, 'AGE_YEARS'].mean()
avg_age_nondef = df.loc[df['TARGET'] == 0, 'AGE_YEARS'].mean()
pct_with_children = (df['CNT_CHILDREN'].gt(0).mean() * 100).round(2)
avg_family_size = df['CNT_FAM_MEMBERS'].mean()
family_counts = df['NAME_FAMILY_STATUS'].value_counts(dropna=False)
pct_family_status = (family_counts / family_counts.sum() * 100).round(2)
edu_ser = df['NAME_EDUCATION_TYPE'].fillna('Unknown')
higher_mask = edu_ser.isin(['Higher education', 'Academic degree']) | edu_ser.str.contains('Bachelor|Master|Post', case=False, na=False)
pct_higher_edu = (higher_mask.mean() * 100).round(2)
pct_with_parents = (df['NAME_HOUSING_TYPE'].eq('With parents').mean() * 100).round(2)
is_working = df['EMPLOYMENT_YEARS'].notna() & (df['EMPLOYMENT_YEARS'] > 0)
is_working = is_working | df['OCCUPATION_TYPE'].notna()
pct_currently_working = (is_working.mean() * 100).round(2)
avg_employment_years = df.loc[df['EMPLOYMENT_YEARS'].notna(), 'EMPLOYMENT_YEARS'].mean()

# --- Streamlit UI ---
st.title("Demographic Insights")

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Avg Age - Defaulters", round(avg_age_def, 2))
col2.metric("Avg Age - Non-Defaulters", round(avg_age_nondef, 2))
col3.metric("% With Children", pct_with_children)

col4, col5, col6 = st.columns(3)
col4.metric("Avg Family Size", round(avg_family_size, 2))
col5.metric("% Higher Education", pct_higher_edu)
col6.metric("% Living With Parents", pct_with_parents)

col7, col8 = st.columns(2)
col7.metric("% Currently Working", pct_currently_working)
col8.metric("Avg Employment Years", round(avg_employment_years, 2))

st.subheader("Gender Distribution (%)")
st.write(pct_gender.to_dict())

st.subheader("Family Status Distribution (%)")
st.write(pct_family_status.to_dict())

# --- Plots in grid (2 per row) ---

# 1 & 2
col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots()
    sns.histplot(df['AGE_YEARS'].dropna(), bins=30, kde=False, ax=ax)
    ax.set_title('Age Distribution (all)')
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.histplot(df.loc[df['TARGET']==0,'AGE_YEARS'].dropna(), bins=30, label='Repaid (0)', alpha=0.6, ax=ax)
    sns.histplot(df.loc[df['TARGET']==1,'AGE_YEARS'].dropna(), bins=30, label='Default (1)', alpha=0.6, ax=ax)
    ax.legend()
    ax.set_title('Age Distribution by Target')
    st.pyplot(fig)

# 3 & 4
col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots()
    sns.countplot(x='CODE_GENDER', data=df, order=df['CODE_GENDER'].value_counts().index, ax=ax)
    ax.set_title('Gender distribution')
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    order = df['NAME_FAMILY_STATUS'].value_counts().index
    sns.countplot(x='NAME_FAMILY_STATUS', data=df, order=order, ax=ax)
    ax.set_title('Family Status distribution')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    st.pyplot(fig)

# 5 & 6
col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots()
    order = df['NAME_EDUCATION_TYPE'].value_counts().index
    sns.countplot(x='NAME_EDUCATION_TYPE', data=df, order=order, ax=ax)
    ax.set_title('Education distribution')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(8,5))
    occ_counts = df['OCCUPATION_TYPE'].value_counts().nlargest(10)
    sns.barplot(x=occ_counts.values, y=occ_counts.index, ax=ax)
    ax.set_title('Top 10 Occupations')
    st.pyplot(fig)

# 7 & 8
col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots(figsize=(6,6))
    housing = df['NAME_HOUSING_TYPE'].value_counts()
    ax.pie(housing.values, labels=housing.index, autopct='%1.1f%%', startangle=90, pctdistance=0.75)
    ax.set_title('Housing Type Distribution')
    st.pyplot(fig)

with col2:
    df['CNT_CHILDREN_CAPPED'] = df['CNT_CHILDREN'].clip(upper=5)
    fig, ax = plt.subplots()
    sns.countplot(x='CNT_CHILDREN_CAPPED', data=df, order=sorted(df['CNT_CHILDREN_CAPPED'].unique()), ax=ax)
    ax.set_title('Children per Applicant (capped at 5)')
    st.pyplot(fig)

# 9 & 10
col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots()
    sns.boxplot(x='TARGET', y='AGE_YEARS', data=df, ax=ax)
    ax.set_xticks([0,1])
    ax.set_xticklabels(['Repaid (0)', 'Default (1)'])
    ax.set_title('Age vs Target (boxplot)')
    st.pyplot(fig)

with col2:
    corr_df = df[['AGE_YEARS', 'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'TARGET']].copy()
    corr_mat = corr_df.corr()
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    ax.set_title('Correlation Heatmap')
    st.pyplot(fig)