import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from utils.load_data import load_data

df=load_data()

st.title("📊 4.Financial Insights")

# -------------------------
# KPTs
# -------------------------
df["DTI"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
df["LTI"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]

avg_income = df["AMT_INCOME_TOTAL"].mean()
median_income = df["AMT_INCOME_TOTAL"].median()
avg_credit = df["AMT_CREDIT"].mean()
avg_annuity = df["AMT_ANNUITY"].mean()
avg_goods_price = df["AMT_GOODS_PRICE"].mean()
avg_dti = df["DTI"].mean()
avg_lti = df["LTI"].mean()

income_gap = df[df["TARGET"] == 0]["AMT_INCOME_TOTAL"].mean() - df[df["TARGET"] == 1]["AMT_INCOME_TOTAL"].mean()
credit_gap = df[df["TARGET"] == 0]["AMT_CREDIT"].mean() - df[df["TARGET"] == 1]["AMT_CREDIT"].mean()
high_credit_pct = (df["AMT_CREDIT"] > 1_000_000).mean() * 100

# -------------------------
# KPIs Display
# -------------------------
st.title("Financial Health & Affordability Dashboard")

col1, col2, col3 = st.columns(3)
col1.metric("Avg Annual Income", f"{avg_income:,.0f}")
col2.metric("Median Annual Income", f"{median_income:,.0f}")
col3.metric("Avg Credit Amount", f"{avg_credit:,.0f}")

col4, col5, col6 = st.columns(3)
col4.metric("Avg Annuity", f"{avg_annuity:,.0f}")
col5.metric("Avg Goods Price", f"{avg_goods_price:,.0f}")
col6.metric("Avg DTI", f"{avg_dti:.2f}")

col7, col8, col9 = st.columns(3)
col7.metric("Avg LTI", f"{avg_lti:.2f}")
col8.metric("Income Gap (Non-def − Def)", f"{income_gap:,.0f}")
col9.metric("Credit Gap (Non-def − Def)", f"{credit_gap:,.0f}")

st.metric("% High Credit (>1M)", f"{high_credit_pct:.2f}%")


# -------------------------
# Charts
# -------------------------
st.subheader("📊 Financial Distributions & Relationships")

# Histogram Income
st.write("### Income Distribution")
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df["AMT_INCOME_TOTAL"].dropna(), bins=50, alpha=1, color="#1f77b4", label="Income")
ax.set_xlabel("Income")
ax.set_ylabel("Count")
ax.legend()
st.pyplot(fig)

# Histogram Credit
st.write("### Credit Distribution")
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df["AMT_CREDIT"].dropna(), bins=50, alpha=1, color="#ff7f0e", label="Credit")
ax.set_xlabel("Credit")
ax.set_ylabel("Count")
ax.legend()
st.pyplot(fig)

# Histogram Annuity
st.write("### Annuity Distribution")
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df["AMT_ANNUITY"].dropna(), bins=50, alpha=1, color="#2ca02c", label="Annuity")
ax.set_xlabel("Annuity")
ax.set_ylabel("Count")
ax.legend()
st.pyplot(fig)

# Scatter Income vs Credit
st.write("### Income vs Credit")
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(df["AMT_INCOME_TOTAL"], df["AMT_CREDIT"], alpha=0.3, color="#3e82b3", label="Applicants")
ax.set_xlabel("Income")
ax.set_ylabel("Credit")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# Scatter Income vs Annuity
st.write("### Income vs Annuity")
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(df["AMT_INCOME_TOTAL"], df["AMT_ANNUITY"], alpha=0.3, color="#ff7f0e", label="Applicants")
ax.set_xlabel("Income")
ax.set_ylabel("Annuity")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Boxplot Credit by Target
st.write("### Credit by Target")
fig, ax = plt.subplots(figsize=(10, 5))
ax.boxplot([df[df["TARGET"] == 0]["AMT_CREDIT"].dropna(), df[df["TARGET"] == 1]["AMT_CREDIT"].dropna()],
           labels=["Repaid (0)", "Default (1)"])
ax.set_ylabel("Credit")
ax.grid(True)
st.pyplot(fig)

# Boxplot Income by Target
st.write("### Income by Target")
fig, ax = plt.subplots(figsize=(10, 5))
ax.boxplot([df[df["TARGET"] == 0]["AMT_INCOME_TOTAL"].dropna(), df[df["TARGET"] == 1]["AMT_INCOME_TOTAL"].dropna()],
           labels=["Repaid (0)", "Default (1)"])
ax.set_ylabel("Income")
ax.grid(True)
st.pyplot(fig)

# KDE approximation with histogram overlay
st.write("### Joint Income–Credit (Density Approximation)")
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist2d(df["AMT_INCOME_TOTAL"], df["AMT_CREDIT"], bins=50, cmap="Blues")
ax.set_xlabel("Income")
ax.set_ylabel("Credit")
st.pyplot(fig)

# Bar — Income Brackets vs Default Rate
st.write("### Income Brackets vs Default Rate")
df["Income_Bracket"] = pd.qcut(df["AMT_INCOME_TOTAL"], q=10, duplicates="drop")
default_rate_by_bracket = df.groupby("Income_Bracket")["TARGET"].mean() * 100
fig, ax = plt.subplots(figsize=(10, 5))
default_rate_by_bracket.plot(kind="bar", ax=ax, color="#1f77b4", alpha=1)
ax.set_xlabel("Income Bracket")
ax.set_ylabel("Default Rate (%)")
st.pyplot(fig)

# Heatmap — Correlations
st.write("### Correlation Heatmap (Financial Variables)")
fin_vars = df[["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "DTI", "LTI", "TARGET"]]
corr = fin_vars.corr()
fig, ax = plt.subplots(figsize=(10, 5))
cax = ax.matshow(corr, cmap="coolwarm")
fig.colorbar(cax)
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45)
ax.set_yticklabels(corr.columns)
st.pyplot(fig)

# -------------------------
# Narrative
# -------------------------
st.subheader("📝 Insights")
st.markdown("""
- Higher **LTI (>6)** and **DTI (>0.35)** are potential affordability stress thresholds.  
- Defaults tend to rise in lower income brackets despite smaller loans.  
- Large credits (>1M) form a small % but contribute significantly to overall exposure.  
- Income–Credit joint density shows concentration in lower–mid ranges with scattered outliers.  
""")