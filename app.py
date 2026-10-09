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


st.set_page_config(page_title='EC2 Instance Cost Analysis', layout='wide')
st.title('EC2 Instance Cost Analysis')
st.caption('Analysis of the supplied dataset. Prices are dataset values, not current AWS quotes.')

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
    # An exact first-word match keeps T3A separate from T3.
    return data[data['Name'].str.split().str[0] == family]


part1, part2 = st.tabs(['Part 1 — Cost Analysis', 'Part 2 — Regression'])

with part1:
    st.header('Part 1 — Analyze EC2 costs')

    st.subheader('Step 2 — Load and explore the dataset')
    a, b = st.columns(2)
    a.metric('Dataset rows', len(raw_data))
    b.metric('Dataset columns', len(raw_data.columns))
    st.write('First five rows')
    st.dataframe(raw_data.head(), hide_index=True)
    info_buffer = StringIO()
    raw_data.info(buf=info_buffer)
    st.text(info_buffer.getvalue())

    st.subheader('Step 3 — Clean the cost columns')
    st.write('Remove price formatting and convert costs to numbers. '
             'Unavailable prices become missing values rather than zero.')
    st.dataframe(data[cost_columns].isnull().sum().rename('Missing values').to_frame())

    st.subheader('Step 4 — Summary statistics')
    st.dataframe(data[cost_columns].describe())

    st.subheader('Step 5 — Visualize the cost distributions')
    cost_boxplot(data, 'Cost Comparison of Amazon EC2 Instances (Hourly)', 'Set2')

    st.subheader('Step 6 — Identify On-Demand outliers')
    outliers_on_demand = detect_outliers('On Demand')
    Q1 = data['On Demand'].quantile(0.25)
    Q3 = data['On Demand'].quantile(0.75)
    IQR = Q3 - Q1
    st.write(f'Q1: ${Q1:.4f}; Q3: ${Q3:.4f}; IQR: ${IQR:.4f}')
    st.write(f'Lower bound: ${Q1 - 1.5 * IQR:.4f}; '
             f'upper bound: ${Q3 + 1.5 * IQR:.4f}')
    st.metric('On-Demand outliers', len(outliers_on_demand))
    st.dataframe(outliers_on_demand, hide_index=True)
    st.caption('The IQR rule flags unusual prices; it does not prove the records are errors.')

    st.subheader('Step 7 — Compare On-Demand and Reserved costs')
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
    st.write('Ten lowest On-Demand prices with an available Linux Reserved price')
    st.dataframe(cost_comparison.head(10), hide_index=True)
    st.caption('Savings percent = (On-Demand − Reserved) / On-Demand × 100. '
               'Check memory, vCPUs and workload requirements before selecting an instance.')

    st.subheader('Step 8 — Compare T2 and T3 families')
    t2_instances = filter_instance_family('T2')
    t3_instances = filter_instance_family('T3')
    st.write(f'T2 instances: {len(t2_instances)}; T3 instances: {len(t3_instances)}')
    st.caption('T3A is excluded by matching the family name exactly.')
    st.write('T2 cost summary')
    st.dataframe(t2_instances[cost_columns].describe())
    st.write('T3 cost summary')
    st.dataframe(t3_instances[cost_columns].describe())
    cost_boxplot(t2_instances, 'Cost Distribution for T2 Instances', 'Blues', True)
    cost_boxplot(t3_instances, 'Cost Distribution for T3 Instances', 'Greens', True)
    comparison = pd.concat([
        t2_instances[['Name', 'On Demand', 'Linux Reserved cost']],
        t3_instances[['Name', 'On Demand', 'Linux Reserved cost']]
    ])
    comparison_sorted = comparison.dropna().sort_values('On Demand')
    st.write('Ten lowest On-Demand prices in the combined T2 and T3 comparison')
    st.dataframe(comparison_sorted.head(10), hide_index=True)

with part2:
    st.header('Part 2 — Predict EC2 costs using linear regression')
    st.subheader('Steps 2 and 3 — Clean costs and convert features')
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

    st.subheader('Step 4 — Handle missing data')
    data_cleaned = regression_data.dropna(
        subset=['On Demand', 'Instance Memory', 'vCPUs']
    )
    st.write(f'Rows before filtering: {len(regression_data)}; '
             f'rows available for regression: {len(data_cleaned)}')
    st.dataframe(data_cleaned.isnull().sum().rename('Remaining missing values').to_frame())
    st.caption('Only the target and two predictor columns must be complete. '
               'Other price columns can still contain missing values.')

    st.subheader('Step 5 — Split into training and testing sets')
    X = data_cleaned[['Instance Memory', 'vCPUs']]
    y = data_cleaned['On Demand']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    st.write(f'80/20 split with random_state=42: '
             f'{len(X_train)} training rows and {len(X_test)} testing rows.')

    st.subheader('Step 6 — Train the linear regression model')
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

    st.subheader('Step 7 — Evaluate on the test set')
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    a, b, c = st.columns(3)
    a.metric('MAE ($/hour)', f'{mae:.4f}')
    b.metric('MSE (($/hour)²)', f'{mse:.4f}')
    c.metric('RMSE ($/hour)', f'{rmse:.4f}')
    st.caption('Lower error values indicate better predictions. '
               'RMSE gives greater weight to large errors than MAE.')

    st.subheader('Step 8 — Actual versus predicted costs')
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

    st.subheader('Step 9 — Predict a new configuration')
    st.write('The default inputs match the handout example: 4 GiB and 2 vCPUs.')
    a, b = st.columns(2)
    memory = a.number_input('Instance memory (GiB)', min_value=0.5,
                            value=4.0, step=0.5)
    vcpus = b.number_input('vCPUs', min_value=1, value=2, step=1)
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
