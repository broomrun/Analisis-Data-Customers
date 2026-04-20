import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

st.set_page_config(layout="wide")

# Load Data
df = pd.read_csv("main_data.csv")
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])


st.title("📊 E-Commerce Data Dashboard")

# Buat filter tahun
st.sidebar.header("Filter")

year = st.sidebar.multiselect(
    "Pilih Tahun",
    options=df['year'].unique(),
    default=df['year'].unique()
)

df = df[df['year'].isin(year)]

# KPI
total_revenue = df['payment_value'].sum()
total_orders = df['order_id'].nunique()
total_customers = df['customer_unique_id'].nunique()

col1, col2, col3 = st.columns(3)

col1.metric("Total Revenue", f"{total_revenue:,.0f}")
col2.metric("Total Orders", total_orders)
col3.metric("Total Customers", total_customers)

# SALES TREND
st.subheader("📈 Monthly Sales Trend")

monthly_sales = df.groupby('month')['payment_value'].sum()
monthly_sales.index = monthly_sales.index.astype(str)

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(monthly_sales.index, monthly_sales.values, marker='o')

ax.set_title("Monthly Revenue")
ax.set_xlabel("Month")
ax.set_ylabel("Revenue")

ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
plt.xticks(rotation=45)

st.pyplot(fig)

# TOP CUSTOMERS
st.subheader("👑 Top Customers")

top_customers = (
    df.groupby('customer_unique_id')['payment_value']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

top_df = pd.DataFrame({
    'Revenue': top_customers
})

# Ganti label
top_df.index = [f'Customer {i+1}' for i in range(len(top_df))]

fig, ax = plt.subplots(figsize=(10,4))
sns.barplot(x=top_df['Revenue'], y=top_df.index, ax=ax)

ax.set_title("Top 10 Customers")
ax.set_xlabel("Revenue")
ax.set_ylabel("Customer")

ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

st.pyplot(fig)

# RFM SEGMENTATION
st.subheader("🎯 Customer Segmentation (RFM)")

snapshot_date = df['order_purchase_timestamp'].max() + pd.Timedelta(days=1)

rfm = df.groupby('customer_unique_id').agg({
    'order_purchase_timestamp': lambda x: (snapshot_date - x.max()).days,
    'order_id': 'nunique',
    'payment_value': 'sum'
})

rfm.columns = ['Recency','Frequency','Monetary']

# Scoring
rfm['R_score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1])
rfm['F_score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
rfm['M_score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5])

# Segment
def segment(row):
    if row['R_score']==5 and row['F_score']==5 and row['M_score']==5:
        return 'Champions'
    elif row['F_score']>=4 and row['M_score']>=4:
        return 'Loyal'
    elif row['R_score']>=4:
        return 'Recent'
    elif row['R_score']<=2:
        return 'At Risk'
    else:
        return 'Others'

rfm['Segment'] = rfm.apply(segment, axis=1)

segment_counts = rfm['Segment'].value_counts()

fig, ax = plt.subplots()
sns.barplot(x=segment_counts.index, y=segment_counts.values, ax=ax)

ax.set_title("Customer Segments")
st.pyplot(fig)
