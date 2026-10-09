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
outliers_on_demand = detect_outliers('On Demand')
with st.expander('IQR calculation'):
    Q1 = data['On Demand'].quantile(0.25)
    Q3 = data['On Demand'].quantile(0.75)
    IQR = Q3 - Q1
    st.write(f'Q1: ${Q1:.4f}; Q3: ${Q3:.4f}; IQR: ${IQR:.4f}')
    st.write(f'Lower bound: ${Q1 - 1.5 * IQR:.4f}; '
             f'upper bound: ${Q3 + 1.5 * IQR:.4f}')
st.write(f'Number of On-Demand outliers: {len(outliers_on_demand)}')
st.dataframe(outliers_on_demand, hide_index=True)
st.caption('The IQR rule flags unusual prices; it does not prove the records are errors.')

st.subheader('Top 10 Lowest-Cost EC2 Instances')
cost_comparison = (
    data[['Name', 'On Demand', 'Linux Reserved cost']]
    .dropna().sort_values('On Demand')
)
cost_comparison['Hourly savings'] = (
    cost_comparison['On Demand'] - cost_comparison['Linux Reserved cost']
)
cost_comparison['Savings percent'] = (
    cost_comparison['Hourly savings'] / cost_comparison['On Demand'] * 100
)
st.dataframe(cost_comparison[['Name', 'On Demand', 'Linux Reserved cost']].head(10))
with st.expander('Reserved cost savings'):
    st.dataframe(cost_comparison.head(10), hide_index=True)
st.caption('Savings percent = (On-Demand − Reserved) / On-Demand × 100. '
           'Check memory, vCPUs and workload requirements before selecting an instance.')

st.subheader('On-Demand vs Reserved Cost')
# Render with the lab's Matplotlib library instead of st.bar_chart/Altair.
price_chart = cost_comparison.head(10).set_index('Name')[
    ['On Demand', 'Linux Reserved cost']
]
fig, ax = plt.subplots(figsize=(10, 5))
price_chart.plot.bar(ax=ax, color=['#4c78a8', '#f58518'], width=0.8)
ax.set_title('On-Demand vs Reserved Cost')
ax.set_xlabel('Instance')
ax.set_ylabel('Cost (USD per hour)')
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader('T2 vs T3 Instance Family Comparison')
t2_instances = filter_instance_family('T2')
t3_instances = filter_instance_family('T3')
st.write(f'T2 instances: {len(t2_instances)}; T3 instances: {len(t3_instances)}')
st.caption('The handout uses a prefix filter: T3 also includes names beginning with T3A.')
st.write('T2 Instance Cost Summary')
st.dataframe(t2_instances[cost_columns].describe())
st.write('T3 Instance Cost Summary')
st.dataframe(t3_instances[cost_columns].describe())
cost_boxplot(t2_instances, 'Cost Distribution for T2 Instances', 'Blues', True)
cost_boxplot(t3_instances, 'Cost Distribution for T3 Instances', 'Greens', True)
comparison = pd.concat([
    t2_instances[['Name', 'On Demand', 'Linux Reserved cost']],
    t3_instances[['Name', 'On Demand', 'Linux Reserved cost']]
])
comparison_sorted = comparison.dropna().sort_values('On Demand')
with st.expander('T2 and T3 price comparison'):
    st.dataframe(comparison_sorted.head(10), hide_index=True)

st.subheader('EC2 On-Demand Cost Prediction Model')
with st.expander('Feature preparation and model details'):
    st.write('Numeric features')
    # Work on a separate copy so the Part 1 data stays available.
    regression_data = data.copy()
    regression_data['Instance Memory'] = pd.to_numeric(
        regression_data['Instance Memory'].str.replace(' GiB', '', regex=False)
    )
    regression_data['vCPUs'] = pd.to_numeric(
        regression_data['vCPUs'].str.extract(r'(\d+)', expand=False)
    )
    st.write('Memory in GiB and the first number in each vCPU description become numeric features.')
    st.dataframe(regression_data[['Instance Memory', 'vCPUs']].head(), hide_index=True)

    st.write('Missing data')
    data_cleaned = regression_data.dropna(
        subset=['On Demand', 'Instance Memory', 'vCPUs']
    )
    st.write(f'Rows before filtering: {len(regression_data)}; '
             f'rows available for regression: {len(data_cleaned)}')
    st.dataframe(data_cleaned.isnull().sum().rename('Remaining missing values').to_frame())
    st.caption('Only the target and two predictor columns must be complete. '
               'Other price columns can still contain missing values.')

    st.write('Training and testing split')
    X = data_cleaned[['Instance Memory', 'vCPUs']]
    y = data_cleaned['On Demand']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    st.write(f'80/20 split with random_state=42: '
             f'{len(X_train)} training rows and {len(X_test)} testing rows.')

    st.write('Model coefficients')
    model = LinearRegression()
    model.fit(X_train, y_train)
    st.write(f'Intercept: {model.intercept_:.7f}')
    st.dataframe(pd.DataFrame({
        'Feature': X.columns, 'Coefficient': model.coef_
    }), hide_index=True)
    st.code(
        f'Predicted hourly cost = {model.intercept_:.7f}'
        f' + {model.coef_[0]:.7f} × memory in GiB'
        f' + {model.coef_[1]:.7f} × vCPUs', language='text'
    )


st.write(f'Training samples: {len(X_train)}')
st.write(f'Testing samples: {len(X_test)}')

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
a, b, c = st.columns(3)
a.metric('MAE', f'{mae:.4f}')
b.metric('MSE', f'{mse:.4f}')
c.metric('RMSE', f'{rmse:.4f}')
st.caption('MAE and RMSE are in dollars per hour; MSE is in squared dollars per hour. '
           'RMSE gives greater weight to large errors than MAE.')

st.subheader('Actual vs Predicted On-Demand Costs')
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_pred, alpha=0.7, color='b')
ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],
        color='red', linestyle='--')
ax.set_title('Actual vs Predicted On-Demand Costs')
ax.set_xlabel('Actual On-Demand Cost ($/hour)')
ax.set_ylabel('Predicted On-Demand Cost ($/hour)')
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.caption('The dashed red line represents a perfect prediction.')

st.subheader('Predict EC2 On-Demand Cost')
st.caption('Default configuration: 4 GiB memory and 2 vCPUs, as in the activity.')
with st.form('cost_prediction'):
    memory = st.number_input('Instance Memory (GiB)', min_value=0.5,
                             value=4.0, step=0.5)
    vcpus = st.number_input('Number of vCPUs', min_value=1, value=2, step=1)
    submitted = st.form_submit_button('Predict Cost', type='primary')

if submitted:
    new_instance = pd.DataFrame([[memory, vcpus]], columns=X.columns)
    predicted_cost = model.predict(new_instance)[0]
    st.metric('Predicted On-Demand cost ($/hour)', f'{predicted_cost:.4f}')
    if predicted_cost < 0:
        st.warning('This negative prediction is not a valid EC2 price. '
                   'The instructed linear model is not constrained to positive values. '
                   'Memory and vCPUs alone do not fully explain instance pricing. '
                   'The result is preserved to demonstrate this limitation.')
    else:
        st.caption('This is a model estimate, not an AWS price quote. '
                   'The model uses only memory and vCPUs.')

st.caption('All prices come from the supplied dataset, not live AWS pricing.')
