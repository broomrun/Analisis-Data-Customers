import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

st.set_page_config(page_title="E-Commerce Dashboard", layout="wide")
sns.set(style='whitegrid')

df = pd.read_csv("dashboard/main_data.csv")
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])

df['year'] = df['order_purchase_timestamp'].dt.year
df['month'] = df['order_purchase_timestamp'].dt.to_period('M').astype(str)

st.sidebar.header("Filter Data")

year = st.sidebar.multiselect(
    "Pilih Tahun",
    options=sorted(df['year'].unique()),
    default=sorted(df['year'].unique())
)

filtered_df = df[df['year'].isin(year)]

month = st.sidebar.multiselect(
    "Pilih Bulan",
    options=sorted(filtered_df['month'].unique()),
    default=sorted(filtered_df['month'].unique())
)

filtered_df = filtered_df[filtered_df['month'].isin(month)]

st.title("📊 E-Commerce Data Dashboard")
st.markdown("Analisis Customer Behavior & Revenue (Olist Dataset)")

st.divider()
st.header("Overview")

total_revenue = filtered_df['payment_value'].sum()
total_orders = filtered_df['order_id'].nunique()
total_customers = filtered_df['customer_unique_id'].nunique()

col1, col2, col3 = st.columns(3)

col1.metric("Total Revenue", f"{total_revenue:,.0f}")
col2.metric("Total Orders", total_orders)
col3.metric("Total Customers", total_customers)

st.divider()
st.header("Customer Contribution Analysis")

customer_revenue = (
    filtered_df.groupby('customer_unique_id')['payment_value']
    .sum()
    .sort_values(ascending=False)
)

customer_pct = (customer_revenue / customer_revenue.sum()) * 100
cumulative_pct = customer_pct.cumsum()

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(cumulative_pct.values, linewidth=2)
ax.axhline(y=80, linestyle='--', color='red')

ax.set_title("Pareto Analysis (Customer Contribution)")
ax.set_xlabel("Customer Rank")
ax.set_ylabel("Cumulative % Revenue")

st.pyplot(fig)

top10 = customer_pct.head(10).sum()
top100 = customer_pct.head(100).sum()
pareto_cutoff = (cumulative_pct <= 80).sum()

st.info(f"""
Top 10 customers: **{top10:.2f}%**  
Top 100 customers: **{top100:.2f}%**  
~**{pareto_cutoff:,} customers** contribute 80% revenue  

➡️ Insight: Revenue **tidak terpusat pada sedikit pelanggan**, melainkan tersebar luas (long-tail distribution).  
➡️ Implikasi: Fokus bisnis sebaiknya pada **mass customer retention & repeat purchase**, bukan hanya VIP.
""")

st.divider()
st.header("Sales Trend Analysis")

monthly_sales = (
    filtered_df
    .groupby(filtered_df['order_purchase_timestamp'].dt.to_period('M'))['payment_value']
    .sum()
)

monthly_sales.index = monthly_sales.index.astype(str)

peak_month = monthly_sales.idxmax()
lowest_month = monthly_sales.idxmin()

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(monthly_sales.index, monthly_sales.values, marker='o')

ax.scatter(peak_month, monthly_sales.max(), color='green')
ax.text(peak_month, monthly_sales.max(), "Peak")

ax.scatter(lowest_month, monthly_sales.min(), color='red')
ax.text(lowest_month, monthly_sales.min(), "Lowest")

ax.set_title("Monthly Revenue Trend")
ax.set_xlabel("Month")
ax.set_ylabel("Revenue")

ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
plt.xticks(rotation=45)

st.pyplot(fig)

st.info(f"""
Peak revenue: **{peak_month}**  
Lowest revenue: **{lowest_month}**  

➡️ Insight: Revenue menunjukkan **tren meningkat dengan pola musiman (seasonality)**, terutama pada periode tertentu seperti akhir tahun.  
➡️ Implikasi: Perlu optimalisasi campaign marketing pada periode peak.
""")

st.divider()
st.header("Customer Segmentation (RFM)")

snapshot_date = filtered_df['order_purchase_timestamp'].max() + pd.Timedelta(days=1)

rfm = filtered_df.groupby('customer_unique_id').agg({
    'order_purchase_timestamp': lambda x: (snapshot_date - x.max()).days,
    'order_id': 'nunique',
    'payment_value': 'sum'
})

rfm.columns = ['Recency','Frequency','Monetary']

rfm['R_score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1], duplicates='drop')
rfm['F_score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5], duplicates='drop')
rfm['M_score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5], duplicates='drop')

rfm[['R_score','F_score','M_score']] = rfm[['R_score','F_score','M_score']].astype(int)

def segment(row):
    if row['R_score']==5 and row['F_score']==5 and row['M_score']==5:
        return 'Champions'
    elif row['F_score']>=4 and row['M_score']>=4:
        return 'Loyal Customers'
    elif row['R_score']>=4:
        return 'Recent Customers'
    elif row['R_score']<=2:
        return 'At Risk'
    else:
        return 'Others'

rfm['Segment'] = rfm.apply(segment, axis=1)

segment_counts = rfm['Segment'].value_counts()

col1, col2 = st.columns(2)

# Distribusi customer
with col1:
    fig, ax = plt.subplots()
    sns.barplot(x=segment_counts.index, y=segment_counts.values, ax=ax)
    ax.set_title("Customer Distribution by Segment")
    st.pyplot(fig)

# Revenue per segment
rfm_reset = rfm.reset_index()
df_rfm = filtered_df.merge(rfm_reset[['customer_unique_id','Segment']], on='customer_unique_id')

segment_revenue = df_rfm.groupby('Segment')['payment_value'].sum()

with col2:
    fig, ax = plt.subplots()
    sns.barplot(x=segment_revenue.index, y=segment_revenue.values, ax=ax)
    ax.set_title("Revenue Contribution by Segment")
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    st.pyplot(fig)

st.info("""
➡️ Mayoritas pelanggan berada di segmen **Others** → engagement rendah  
➡️ Segmen **Champions & Loyal Customers** memberikan kontribusi revenue terbesar  
➡️ Terdapat segmen **At Risk** yang berpotensi churn  

🎯 Strategi:
- Retain high-value customers (Champions & Loyal)
- Re-engage At Risk customers
- Tingkatkan engagement segmen Others untuk mendorong repeat purchase
""")
