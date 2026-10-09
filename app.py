from pathlib import Path
from io import StringIO

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


st.set_page_config(page_title='Amazon EC2 Cost Analysis Dashboard', layout='centered')
st.title('Amazon EC2 Cost Analysis Dashboard')
st.write('This dashboard analyzes Amazon EC2 instance costs and performance data.')

cost_columns = [
    'On Demand', 'Linux Reserved cost', 'Linux Spot Minimum cost',
    'Windows On Demand cost', 'Windows Reserved cost'
]

# Use the CSV next to this file, regardless of the working directory.
file_path = Path(__file__).resolve().parent / 'ec2dataset.csv'
if not file_path.exists():
    st.error('Place ec2dataset.csv in the same folder as app.py, then rerun the app.')
    st.stop()

raw_data = pd.read_csv(file_path)
data = raw_data.copy()
for column in cost_columns:
    data[column] = pd.to_numeric(
        data[column].str.replace('[$, hourly]', '', regex=True),
        errors='coerce'
    )

sns.set(style='whitegrid')


def cost_boxplot(frame, title, palette, showmeans=False):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=frame[cost_columns], palette=palette,
                showmeans=showmeans, ax=ax)
    ax.set_title(title, fontsize=16)
    ax.set_ylabel('Cost (USD per hour)', fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def detect_outliers(column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] < lower_bound) |
                (data[column] > upper_bound)]


def filter_instance_family(family):
    # Use the prefix filter exactly as provided in Part 1 Step 8.
    return data[data['Name'].str.startswith(family)]



st.subheader('Dataset Overview')
st.write(f'Number of rows: {len(raw_data)}')
st.write(f'Number of columns: {len(raw_data.columns)}')
st.subheader('Dataset Preview')
st.dataframe(raw_data.head())
with st.expander('Dataset information and cleaning'):
    info_buffer = StringIO()
    raw_data.info(buf=info_buffer)
    st.text(info_buffer.getvalue())

    st.write('Cost cleaning')
    st.write('Remove price formatting and convert costs to numbers. '
             'Unavailable prices become missing values rather than zero.')
    st.dataframe(data[cost_columns].isnull().sum().rename('Missing values').to_frame())


st.subheader('Cost Summary')
a, b, c = st.columns(3)
a.metric('Average On-Demand Cost', f"${data['On Demand'].mean():.2f}")
b.metric('Lowest On-Demand Cost', f"${data['On Demand'].min():.4f}")
c.metric('Highest On-Demand Cost', f"${data['On Demand'].max():.2f}")
with st.expander('Full cost summary statistics'):
    st.dataframe(data[cost_columns].describe())

st.subheader('Cost Distribution')
cost_boxplot(data, 'Cost Comparison of Amazon EC2 Instances (Hourly)', 'Set2')

st.subheader('On-Demand Cost Outliers')
