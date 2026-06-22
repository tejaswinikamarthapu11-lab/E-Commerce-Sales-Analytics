# =========================================================
# E-COMMERCE SALES ANALYTICS PROJECT
# Tools  : Python, Pandas, NumPy, Matplotlib, Seaborn
# =========================================================


# =========================================================
# 1. IMPORTING LIBRARIES
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt


# =========================================================
# 2. LOADING DATASETS
# =========================================================

orders = pd.read_csv(
    r"C:\Users\kamar\Downloads\archive\olist_orders_dataset.csv"
)

customers = pd.read_csv(
    r"C:\Users\kamar\Downloads\archive\olist_customers_dataset.csv"
)

order_items = pd.read_csv(
    r"C:\Users\kamar\Downloads\archive\olist_order_items_dataset.csv"
)

payments = pd.read_csv(
    r"C:\Users\kamar\Downloads\archive\olist_order_payments_dataset.csv"
)

products = pd.read_csv(
    r"C:\Users\kamar\Downloads\archive\olist_products_dataset.csv"
)


# =========================================================
# 3. INITIAL DATA EXPLORATION
# =========================================================

print("\nOrders Dataset")
print(orders.head())

print("\nDataset Information")
print(orders.info())

print("\nStatistical Summary")
print(orders.describe())


# =========================================================
# 4. DATA CLEANING & PREPROCESSING
# =========================================================

# -----------------------------
# Checking Missing Values
# -----------------------------

print("\nMissing Values")

print(orders.isnull().sum())
print(customers.isnull().sum())
print(order_items.isnull().sum())
print(payments.isnull().sum())
print(products.isnull().sum())


# -----------------------------
# Creating Copies of Datasets
# -----------------------------

df_orders = orders.copy()
df_customers = customers.copy()
df_order_items = order_items.copy()
df_payments = payments.copy()
df_products = products.copy()


# =========================================================
# 5. HANDLING MISSING VALUES
# =========================================================

# -----------------------------
# Orders Table
# -----------------------------

# Filling missing approved dates
df_orders['order_approved_at'] = (
    df_orders['order_approved_at']
    .fillna(df_orders['order_purchase_timestamp'])
)

# Convert delivery date columns
df_orders['order_delivered_carrier_date'] = pd.to_datetime(
    df_orders['order_delivered_carrier_date']
)

df_orders['order_delivered_customer_date'] = pd.to_datetime(
    df_orders['order_delivered_customer_date']
)


# -----------------------------
# Products Table
# -----------------------------

# Fill missing product categories
df_products['product_category_name'] = (
    df_products['product_category_name']
    .fillna("Unknown")
)

# Fill missing product details
cols = [
    'product_name_lenght',
    'product_description_lenght',
    'product_photos_qty'
]

for col in cols:
    df_products[col] = df_products[col].fillna(0)

# Remove remaining null values
df_products = df_products.dropna()


# =========================================================
# 6. DATE CONVERSIONS
# =========================================================

df_orders['order_purchase_timestamp'] = pd.to_datetime(
    df_orders['order_purchase_timestamp']
)

df_orders['order_approved_at'] = pd.to_datetime(
    df_orders['order_approved_at']
)

df_orders['order_estimated_delivery_date'] = pd.to_datetime(
    df_orders['order_estimated_delivery_date']
)


# =========================================================
# 7. REMOVING DUPLICATES
# =========================================================

df_orders = df_orders.drop_duplicates()
df_customers = df_customers.drop_duplicates()
df_order_items = df_order_items.drop_duplicates()
df_payments = df_payments.drop_duplicates()
df_products = df_products.drop_duplicates()


# =========================================================
# 8. MERGING DATASETS
# =========================================================

df = df_orders.merge(
    df_customers,
    on='customer_id',
    how='left'
)

df = df.merge(
    df_order_items,
    on='order_id',
    how='left'
)

df = df.merge(
    df_payments,
    on='order_id',
    how='left'
)

df = df.merge(
    df_products,
    on='product_id',
    how='left'
)

print("\nMerged Dataset")
print(df.head())


# =========================================================
# 9. FEATURE ENGINEERING
# =========================================================

# -----------------------------
# Revenue Calculation
# -----------------------------

df['total_value'] = (
    df['price'] + df['freight_value']
)


# -----------------------------
# Time Features
# -----------------------------

df['order_month'] = (
    df['order_purchase_timestamp'].dt.month
)

df['order_year'] = (
    df['order_purchase_timestamp'].dt.year
)

df['order_day'] = (
    df['order_purchase_timestamp'].dt.day
)


# -----------------------------
# Delivery Time
# -----------------------------

df['delivery_time'] = (
    df['order_delivered_customer_date']
    - df['order_purchase_timestamp']
).dt.days


# =========================================================
# 10. HANDLING REMAINING NULL VALUES
# =========================================================

df['delivery_time'] = df['delivery_time'].fillna(0)

df['payment_value'] = df['payment_value'].fillna(0)

print("\nRemaining Null Values")
print(df.isnull().sum())


# =========================================================
# 11. DATA VALIDATION
# =========================================================

# Negative payment values
print("\nNegative Payment Values")
print(df[df['payment_value'] < 0])

# Duplicate rows
print("\nDuplicate Rows")
print(df.duplicated().sum())


# =========================================================
# 12. RFM ANALYSIS
# =========================================================

# -----------------------------
# Recency
# -----------------------------

snapshot_date = (
    df['order_purchase_timestamp'].max()
    + dt.timedelta(days=1)
)

recency = (
    df.groupby('customer_unique_id')
    ['order_purchase_timestamp']
    .max()
    .reset_index()
)

recency['Recency'] = (
    snapshot_date
    - recency['order_purchase_timestamp']
).dt.days


# -----------------------------
# Frequency
# -----------------------------

frequency = (
    df.groupby('customer_unique_id')
    ['order_id']
    .nunique()
    .reset_index()
)

frequency.columns = [
    'customer_unique_id',
    'Frequency'
]


# -----------------------------
# Monetary Value
# -----------------------------

monetary = (
    df.groupby('customer_unique_id')
    ['payment_value']
    .sum()
    .reset_index()
)

monetary.columns = [
    'customer_unique_id',
    'Monetary'
]


# -----------------------------
# Creating RFM Table
# -----------------------------

rfm = recency.merge(
    frequency,
    on='customer_unique_id'
)

rfm = rfm.merge(
    monetary,
    on='customer_unique_id'
)

print("\nRFM Table")
print(rfm.head())


# =========================================================
# 13. CUSTOMER SEGMENTATION
# =========================================================

rfm['Customer_Type'] = np.where(
    rfm['Monetary'] > rfm['Monetary'].median(),
    'High Value',
    'Low Value'
)

print("\nCustomer Segmentation")
print(rfm.head())


# =========================================================
# 14. CHURN ANALYSIS
# =========================================================

rfm['Churn_Risk'] = np.where(
    rfm['Recency'] > 90,
    'High Risk',
    'Low Risk'
)

print("\nChurn Analysis")
print(rfm.head())


# =========================================================
# 15. EXPLORATORY DATA ANALYSIS (EDA)
# =========================================================

print("\nDataset Overview")
print(df.head())

print(df.info())

print(df.describe())


# =========================================================
# 16. DATA VISUALIZATIONS
# =========================================================

# -----------------------------
# Monthly Sales Trend
# -----------------------------

monthly_sales = (
    df.groupby('order_month')
    ['payment_value']
    .sum()
)

monthly_sales.plot(kind='line')

plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Revenue')

plt.show()


# -----------------------------
# Top Product Categories
# -----------------------------

top_products = (
    df.groupby('product_category_name')
    ['payment_value']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

top_products.plot(kind='bar')

plt.title('Top Product Categories')
plt.xlabel('Product Category')
plt.ylabel('Revenue')

plt.show()


# -----------------------------
# Payment Type Analysis
# -----------------------------

sns.countplot(
    x='payment_type',
    data=df
)

plt.title('Payment Methods')

plt.show()


# -----------------------------
# Delivery Time Analysis
# -----------------------------

sns.histplot(
    df['delivery_time'],
    bins=20
)

plt.title('Delivery Time Distribution')
plt.xlabel('Delivery Time (Days)')
plt.ylabel('Frequency')

plt.show()


# =========================================================
# 17. KPI CALCULATIONS
# =========================================================

# -----------------------------
# Total Revenue
# -----------------------------

total_revenue = df['payment_value'].sum()

print("\nTotal Revenue:", total_revenue)


# -----------------------------
# Total Orders
# -----------------------------

total_orders = df['order_id'].nunique()

print("Total Orders:", total_orders)


# -----------------------------
# Total Customers
# -----------------------------

total_customers = (
    df['customer_unique_id'].nunique()
)

print("Total Customers:", total_customers)


# -----------------------------
# Average Order Value
# -----------------------------

aov = total_revenue / total_orders

print("Average Order Value:", aov)


# -----------------------------
# Repeat Customer Rate
# -----------------------------

repeat_customers = (
    df.groupby('customer_unique_id')
    ['order_id']
    .nunique()
)

repeat_rate = (
    (repeat_customers > 1).sum()
    / total_customers
) * 100

print("Repeat Customer Rate:", repeat_rate)


# -----------------------------
# Monthly Revenue
# -----------------------------

monthly_revenue = (
    df.groupby('order_month')
    ['payment_value']
    .sum()
)

print("\nMonthly Revenue")
print(monthly_revenue)


# =========================================================
# 18. EXPORTING CLEANED DATA
# =========================================================

df.to_csv(
    "cleaned_ecommerce_data.csv",
    index=False
)

print("\nCleaned dataset exported successfully.")