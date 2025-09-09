# page 2 - Streamlit version
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

sns.set(style="whitegrid")

# --- Load dataset ---
df = pd.read_csv("C:\\Users\\dell\\OneDrive\\Desktop\\project\\application_train.csv")

# --- Feature engineering / cleaning ---
df['AGE_YEARS'] = (df['DAYS_BIRTH'].abs() / 365).astype(int)
df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
df['EMPLOYMENT_YEARS'] = (df['DAYS_EMPLOYED'].abs() / 365).replace(np.inf, np.nan)
df['TARGET'] = df['TARGET'].astype(int)

# --- KPIs ---
total_defaults = int(df['TARGET'].sum())
default_rate_pct = df['TARGET'].mean() * 100
default_by_gender = df.groupby('CODE_GENDER')['TARGET'].mean().multiply(100).round(2)
default_by_education = df.groupby('NAME_EDUCATION_TYPE')['TARGET'].mean().multiply(100).round(2)
default_by_family = df.groupby('NAME_FAMILY_STATUS')['TARGET'].mean().multiply(100).round(2)
default_by_housing = df.groupby('NAME_HOUSING_TYPE')['TARGET'].mean().multiply(100).round(2)
avg_income_defaulters = df.loc[df['TARGET']==1, 'AMT_INCOME_TOTAL'].mean()
avg_credit_defaulters = df.loc[df['TARGET']==1, 'AMT_CREDIT'].mean()
avg_annuity_defaulters = df.loc[df['TARGET']==1, 'AMT_ANNUITY'].mean()
avg_emp_years_defaulters = df.loc[df['TARGET']==1, 'EMPLOYMENT_YEARS'].mean()

# --- Streamlit UI ---
st.title("Target & Risk Segmentation")

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Total Defaults", total_defaults)
col2.metric("Default Rate (%)", f"{round(default_rate_pct,2)}%")
col3.metric("Avg Income (Defaulters)", f"{round(avg_income_defaulters,2)}")

col4, col5, col6 = st.columns(3)
col4.metric("Avg Credit (Defaulters)", f"{round(avg_credit_defaulters,2)}")
col5.metric("Avg Annuity (Defaulters)", f"{round(avg_annuity_defaulters,2)}")
col6.metric("Avg Employment Years (Defaulters)", f"{round(avg_emp_years_defaulters,2)}")

st.markdown("---")

# --- Charts in 2 per row ---

# 1 & 2
col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots()
    sns.countplot(x='TARGET', data=df, order=[0,1], ax=ax)
    ax.set_xticklabels(['Repaid (0)', 'Default (1)'])
    ax.set_title('Counts: Repaid vs Default')
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    default_by_gender.sort_values(ascending=False).plot(kind='bar', ax=ax)
    ax.set_title('Default Rate (%) by Gender')
    ax.set_ylabel('Default Rate (%)')
    st.pyplot(fig)

# 3 & 4
col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots()
    default_by_education.sort_values(ascending=False).plot(kind='bar', ax=ax)
    ax.set_title('Default Rate (%) by Education')
    ax.set_ylabel('Default Rate (%)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    default_by_family.sort_values(ascending=False).plot(kind='bar', ax=ax)
    ax.set_title('Default Rate (%) by Family Status')
    ax.set_ylabel('Default Rate (%)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    st.pyplot(fig)

# 5 & 6
col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots()
    default_by_housing.sort_values(ascending=False).plot(kind='bar', ax=ax)
    ax.set_title('Default Rate (%) by Housing Type')
    ax.set_ylabel('Default Rate (%)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.boxplot(x='TARGET', y='AMT_INCOME_TOTAL', data=df, ax=ax)
    ax.set_yscale('log')
    ax.set_xticklabels(['Repaid (0)', 'Default (1)'])
    ax.set_title('Income Distribution by Target (log scale)')
    st.pyplot(fig)

# 7 & 8
col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots()
    sns.boxplot(x='TARGET', y='AMT_CREDIT', data=df, ax=ax)
    ax.set_yscale('log')
    ax.set_xticklabels(['Repaid (0)', 'Default (1)'])
    ax.set_title('Credit Amount by Target (log scale)')
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.violinplot(x='TARGET', y='AGE_YEARS', data=df, inner='quartile', ax=ax)
    ax.set_xticklabels(['Repaid (0)', 'Default (1)'])
    ax.set_title('Age Distribution by Target')
    st.pyplot(fig)

# 9 & 10
col1, col2 = st.columns(2)
with col1:
    bins = [0,1,3,5,10,20,40,100]
    df['EMPLOY_BIN'] = pd.cut(df['EMPLOYMENT_YEARS'], bins=bins, include_lowest=True)
    emp_counts = df.groupby(['EMPLOY_BIN','TARGET']).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(8,5))
    emp_counts.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title('Employment Years (binned) by Target')
    ax.set_xlabel('Employment Years (bins)')
    ax.set_ylabel('Count')
    ax.legend(title='TARGET', labels=['Repaid (0)','Default (1)'])
    st.pyplot(fig)

with col2:
    contract_counts = df.groupby(['NAME_CONTRACT_TYPE','TARGET']).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(6,5))
    contract_counts.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title('Contract Type vs Target (stacked)')
    ax.set_ylabel('Count')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title='TARGET', labels=['Repaid (0)','Default (1)'])
    st.pyplot(fig)