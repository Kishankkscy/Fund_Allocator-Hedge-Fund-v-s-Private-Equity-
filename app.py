import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to simulate realistic returns
def simulate_returns(investment_type, size, interest_rate, gdp_growth, inflation_rate):
    if investment_type == "Hedge Fund":
        base_return = np.random.choice(
            np.linspace(8, 14, 100),
            size=size,
            p=np.clip(np.exp(-np.linspace(0, 1, 100)), 0.01, None) / np.exp(-np.linspace(0, 1, 100)).sum()
        )
        adjustment = 0.1 * gdp_growth - 0.05 * inflation_rate
        return base_return + adjustment
    elif investment_type == "Private Equity":
        base_return = np.random.normal(loc=11, scale=1.5, size=size).clip(8, 13)
        adjustment = 0.15 * gdp_growth - 0.1 * inflation_rate + 0.05 * interest_rate
        return base_return + adjustment

# ROI Calculation Function
def calculate_roi(investment, annual_return, years):
    return investment * ((1 + annual_return / 100) ** years)

# Streamlit App Title
st.title("💰 Hedge Fund vs Private Equity Investment Analyzer")
st.write("Analyze the expected ROI and risks for Hedge Funds and Private Equity investments based on economic conditions.")

# User Inputs
st.sidebar.header("📊 Investment Inputs")
investment = st.sidebar.number_input("Investment Amount ($)", min_value=1000, value=100000, step=1000)
risk_tolerance = st.sidebar.selectbox("Risk Tolerance", ["Low", "Medium", "High"])
investment_term = st.sidebar.slider("Investment Term (Years)", 1, 20, 10)
interest_rate = st.sidebar.number_input("Current Interest Rate (%)", 0.0, 20.0, 3.0)
gdp_growth = st.sidebar.number_input("GDP Growth Rate (%)", -10.0, 20.0, 5.0)
inflation_rate = st.sidebar.number_input("Inflation Rate (%)", 0.0, 20.0, 3.0)

# Simulate Data
np.random.seed(42)
data_size = 1000
investment_types = np.random.choice(["Hedge Fund", "Private Equity"], size=data_size)
annual_returns = np.array([
    simulate_returns(it, 1, interest_rate, gdp_growth, inflation_rate)[0] for it in investment_types
])
years = np.random.randint(1, 20, size=data_size)
roi = [calculate_roi(investment, ar, yr) for ar, yr in zip(annual_returns, years)]

# Create DataFrame
df = pd.DataFrame({
    "Investment_Type": investment_types,
    "Years": years,
    "Annual_Return": annual_returns,
    "ROI": roi,
})

# Filter Data for User's Investment Term
filtered_df = df[df["Years"] == investment_term]

# Calculate Aggregated Statistics
hedge_fund_data = filtered_df[filtered_df["Investment_Type"] == "Hedge Fund"]
private_equity_data = filtered_df[filtered_df["Investment_Type"] == "Private Equity"]

hedge_fund_avg_return = hedge_fund_data["ROI"].mean()
private_equity_avg_return = private_equity_data["ROI"].mean()

hedge_fund_risk = hedge_fund_data["Annual_Return"].std()
private_equity_risk = private_equity_data["Annual_Return"].std()

# Decision Logic
if risk_tolerance == "Low":
    recommendation = "Hedge Fund" if hedge_fund_risk < private_equity_risk else "Private Equity"
elif risk_tolerance == "Medium":
    recommendation = "Hedge Fund" if hedge_fund_avg_return > private_equity_avg_return else "Private Equity"
elif risk_tolerance == "High":
    recommendation = "Private Equity" if private_equity_avg_return > hedge_fund_avg_return else "Hedge Fund"
else:
    recommendation = "Invalid risk tolerance input. Please choose Low, Medium, or High."

# Display Results
st.subheader("📈 Investment Analysis Results")
st.write(f"**Investment Term:** {investment_term} years")
st.write(f"**Average ROI (Hedge Funds):** ${hedge_fund_avg_return:,.2f}")
st.write(f"**Average ROI (Private Equity):** ${private_equity_avg_return:,.2f}")
st.write(f"**Risk (Hedge Funds - Standard Deviation):** {hedge_fund_risk:.2f}")
st.write(f"**Risk (Private Equity - Standard Deviation):** {private_equity_risk:.2f}")
st.success(f"✅ **Recommended Investment:** {recommendation}")

# Line Graph: ROI over Time for Hedge Funds and Private Equity
st.subheader("📊 ROI Over Time by Investment Type")
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(data=df[df["Investment_Type"] == "Hedge Fund"], x="Years", y="ROI", label="Hedge Fund")
sns.lineplot(data=df[df["Investment_Type"] == "Private Equity"], x="Years", y="ROI", label="Private Equity")
plt.title("ROI Over Time by Investment Type")
plt.xlabel("Years")
plt.ylabel("ROI ($)")
plt.legend()
st.pyplot(fig)

# Boxplot: ROI Distribution by Investment Type
st.subheader(f"📊 ROI Distribution for {investment_term}-Year Investment")
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(x="Investment_Type", y="ROI", data=filtered_df)
plt.title(f"ROI Distribution for {investment_term}-Year Investment")
plt.ylabel("ROI ($)")
plt.xlabel("Investment Type")
st.pyplot(fig)

# Bar Graph: Average ROI vs Investment Type
st.subheader(f"📊 Average ROI for {investment_term}-Year Investment by Investment Type")
fig, ax = plt.subplots(figsize=(8, 5))
avg_roi = filtered_df.groupby("Investment_Type")["ROI"].mean().reset_index()
sns.barplot(x="Investment_Type", y="ROI", data=avg_roi, palette="viridis")
plt.title(f"Average ROI for {investment_term}-Year Investment")
plt.ylabel("Average ROI ($)")
plt.xlabel("Investment Type")
st.pyplot(fig)
