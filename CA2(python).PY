import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#STYLE
plt.style.use('ggplot')

df = pd.read_csv(r"C:\Users\yashika\Downloads\Merged_Annually_Quarterly.csv")

print("=" * 50)
print("DATA LOADED")
print("=" * 50)
print("Shape:", df.shape)
print(df.head())
print("\nColumns:")
print(df.columns.tolist())

#CLEANING

df = df.drop_duplicates().copy()

cat_cols = ['revision', 'industry', 'subindustry', 'institutional_sector', 'quarter']
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")

num_cols = ['current_price', 'constant_price']
for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

df['year'] = df['year'].astype(str).str.strip()
df['year_start'] = pd.to_numeric(df['year'].str[:4], errors='coerce')

df['price_gap'] = df['current_price'] - df['constant_price']

#SHORT NAMES

industry_map = {
    'Agriculture, forestry and fishing': 'Agriculture',
    'Manufacturing': 'Mfg',
    'Construction': 'Const',
    'Financial and insurance activities': 'Finance',
    'Public administration and defence': 'Public Admin',
    'Wholesale and retail trade; repair of motor vehicles and motorcycles': 'Retail',
    'Wholesale and retail trade': 'Retail',
    'Information and communication': 'IT',
    'Real estate activities': 'Real Estate',
    'Mining and quarrying': 'Mining',
    'Electricity, gas, water supply and other utility services': 'Utilities',
    'Transport, storage, communication & services related to broadcasting': 'Transport',
    'Trade, hotels, transport, communication and services related to broadcasting': 'Trade+Transport',
    'Professional, scientific and technical activities': 'Professional',
    'Public administration, defence and other services': 'Public Services',
    'Financial, real estate and professional services': 'Finance+RE',
    'Unknown': 'Unknown'
}

df['industry_short'] = df['industry'].replace(industry_map)
df['industry_short'] = df['industry_short'].fillna(df['industry'])

sector_map = {
    'Non-financial corporations': 'Non-Fin Corp',
    'Financial corporations': 'Fin Corp',
    'General government': 'Government',
    'Households': 'Households',
    'Households including Non-Profit Institutions Serving Households': 'HH + NPISH',
    'Unknown': 'Unknown'
}

df['sector_short'] = df['institutional_sector'].replace(sector_map)
df['sector_short'] = df['sector_short'].fillna(df['institutional_sector'])


#EDA

print("\n" + "=" * 50)
print("MISSING VALUES AFTER CLEANING")
print("=" * 50)
print(df.isnull().sum())

print("\n" + "=" * 50)
print("SUMMARY STATS")
print("=" * 50)
print(df[['current_price', 'constant_price', 'price_gap']].describe())

# Q1 LINE CHART
# Yearly trend of current price
#This chart shows yearly current price trends

q1 = df.groupby('year_start', as_index=False)['current_price'].sum().sort_values('year_start')

plt.figure(figsize=(10, 5))
plt.plot(q1['year_start'], q1['current_price'], marker='o')
plt.title("1. Yearly Current Price Trend")
plt.xlabel("Year")
plt.ylabel("Current Price")
plt.tight_layout()
plt.show()


# Q2 AREA CHART
# Yearly trend of constant price
#This shows constant price trend over years.

q2 = df.groupby('year_start', as_index=False)['constant_price'].sum().sort_values('year_start')

plt.figure(figsize=(10, 5))
plt.fill_between(q2['year_start'], q2['constant_price'], alpha=0.4)
plt.plot(q2['year_start'], q2['constant_price'], marker='o')
plt.title("2. Yearly Constant Price Trend")
plt.xlabel("Year")
plt.ylabel("Constant Price")
plt.tight_layout()
plt.show()

# Q3 BAR CHART
# Top industries
#These are top-performing industries

q3 = (
    df[df['industry_short'] != 'Unknown']
    .groupby('industry_short', as_index=False)['current_price']
    .sum()
    .sort_values('current_price', ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 5))
plt.bar(q3['industry_short'], q3['current_price'])
plt.title("3. Top 10 Industries")
plt.xlabel("Industry")
plt.ylabel("Current Price")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# Q4 HORIZONTAL BAR CHART
# Bottom industries
#These are low-performing industries.

q4 = (
    df[df['industry_short'] != 'Unknown']
    .groupby('industry_short', as_index=False)['current_price']
    .sum()
    .sort_values('current_price', ascending=True)
    .head(10)
)

plt.figure(figsize=(10, 5))
plt.barh(q4['industry_short'], q4['current_price'])
plt.title("4. Bottom 10 Industries")
plt.xlabel("Current Price")
plt.ylabel("Industry")
plt.tight_layout()
plt.show()


# Q5 STACKED BAR CHART
# Frequency comparison
#This compares annual and quarterly values.

q5 = df.groupby('frequency')[['current_price', 'constant_price']].sum()

q5.plot(kind='bar', stacked=True, figsize=(8, 5))
plt.title("5. Annual vs Quarterly")
plt.xlabel("Frequency")
plt.ylabel("Total Value")
plt.tight_layout()
plt.show()


# Q6 PIE CHART
# Shows Indicator contribution
# Keep only positive values

q6 = df.groupby('indicator')['current_price'].sum()

#keep only positive values
q6 = q6[q6 > 0]

#sort descending
q6 = q6.sort_values(ascending=False)

#top 5 indicators
top5 = q6.head(5)

#combine remaining as Others
others = q6.iloc[5:].sum()

if others > 0:
    top5['Others'] = others

# plot clean pie
#To show indicator wise contribution
plt.figure(figsize=(8, 8))
plt.pie(
    top5,
    labels=top5.index,
    autopct='%1.1f%%',
    startangle=140,
    wedgeprops={'edgecolor': 'white'}
)

plt.title("6. Indicator Share")
plt.tight_layout()
plt.show()

# Q7 QUARTER BAR CHART
#This compares quarter performance

q7 = (
    df[df['quarter'].isin(['Q1', 'Q2', 'Q3', 'Q4', 'Quarter 1', 'Quarter 2', 'Quarter 3', 'Quarter 4'])]
    .groupby('quarter', as_index=False)['current_price']
    .sum()
)

quarter_order = {
    'Q1': 1, 'Quarter 1': 1,
    'Q2': 2, 'Quarter 2': 2,
    'Q3': 3, 'Quarter 3': 3,
    'Q4': 4, 'Quarter 4': 4
}

q7['order'] = q7['quarter'].map(quarter_order)
q7 = q7.sort_values('order')

plt.figure(figsize=(8, 5))
plt.bar(q7['quarter'], q7['current_price'])
plt.title("7. Quarter-wise Current Price")
plt.xlabel("Quarter")
plt.ylabel("Current Price")
plt.tight_layout()
plt.show()


# Q8 SCATTER PLOT
#To check relationship between current and constant price.

plt.figure(figsize=(8, 5))
plt.scatter(df['current_price'], df['constant_price'], alpha=0.4)
plt.title("8. Current vs Constant Price")
plt.xlabel("Current Price")
plt.ylabel("Constant Price")
plt.tight_layout()
plt.show()

# Q9 HISTOGRAM
#To show distribution of current price

plt.figure(figsize=(8, 5))
plt.hist(df['current_price'], bins=25)
plt.title("9. Current Price Distribution")
plt.xlabel("Current Price")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# Q10 BOXPLOT
#To detect outliers

plt.figure(figsize=(8, 5))
plt.boxplot(df['current_price'].dropna())
plt.title("10. Current Price Boxplot")
plt.ylabel("Current Price")
plt.tight_layout()
plt.show()

# DONUT CHART
#Sector contribution

q11 = (
    df[df['sector_short'] != 'Unknown']
    .groupby('sector_short')['current_price']
    .sum()
)
q11 = q11[q11 > 0]

if len(q11) > 0:
    plt.figure(figsize=(8, 8))
    wedges, texts, autotexts = plt.pie(q11, labels=q11.index, autopct='%1.1f%%', startangle=90)
    centre_circle = plt.Circle((0, 0), 0.60, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.title("Extra 1. Sector Share")
    plt.tight_layout()
    plt.show()
else:
    print("\nExtra 1 donut chart skipped because no positive values were available.")

# PRICE GAP CHART
#To compare value difference across industries

q12 = (
    df[df['industry_short'] != 'Unknown']
    .groupby('industry_short', as_index=False)['price_gap']
    .mean()
    .sort_values('price_gap', ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 5))
plt.bar(q12['industry_short'], q12['price_gap'])
plt.title("Extra 2. Avg Price Gap by Industry")
plt.xlabel("Industry")
plt.ylabel("Avg Price Gap")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#SIMPLE CORRELATION VIEW

corr_data = df[['current_price', 'constant_price', 'price_gap']].corr()

plt.figure(figsize=(6, 5))
plt.imshow(corr_data, aspect='auto')
plt.colorbar()
plt.xticks(range(len(corr_data.columns)), corr_data.columns, rotation=45, ha='right')
plt.yticks(range(len(corr_data.index)), corr_data.index)

for i in range(len(corr_data.index)):
    for j in range(len(corr_data.columns)):
        plt.text(j, i, round(corr_data.iloc[i, j], 2), ha='center', va='center')

plt.title("Extra 3. Correlation Matrix")
plt.tight_layout()
plt.show()

#SAVE DATA

df.to_csv("Cleaned_Merged_Annually_Quarterly.csv", index=False)

print("\nProject completed successfully.")
print("Cleaned file saved as: Cleaned_Merged_Annually_Quarterly.csv")
