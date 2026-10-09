# %% Part 1 Step 2 Load and explore
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'ec2dataset.csv'
data = pd.read_csv(file_path)
data.info()
print(data.head().to_string(index=False))
print('Dataset shape:', data.shape)


# %% Part 1 Step 3 Clean costs
cost_columns = ['On Demand', 'Linux Reserved cost',
                'Linux Spot Minimum cost', 'Windows On Demand cost',
                'Windows Reserved cost']

for column in cost_columns:
    data[column] = pd.to_numeric(
        data[column].str.replace('[$, hourly]', '', regex=True),
        errors='coerce'
    )

print('Missing values after cost conversion:')
print(data[cost_columns].isnull().sum())


# %% Part 1 Step 4 Summary statistics
cost_summary = data[cost_columns].describe()
print(cost_summary.to_string())


# %% Part 1 Step 5 Cost boxplot
sns.set(style='whitegrid')
plt.figure(figsize=(12, 6))
sns.boxplot(data=data[cost_columns], palette='Set2')
plt.title('Cost Comparison of Amazon EC2 Instances (Hourly)', fontsize=16)
plt.ylabel('Cost (USD)', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.tight_layout()
plt.show()


# %% Part 1 Step 6 IQR outliers
def detect_outliers(column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] < lower_bound) |
                (data[column] > upper_bound)]

outliers_on_demand = detect_outliers('On Demand')
print(outliers_on_demand.to_string(index=False))
print('Number of On-Demand outliers:', len(outliers_on_demand))


# %% Part 1 Step 7 Compare reserved and On-Demand
cost_comparison = (
    data[['Name', 'On Demand', 'Linux Reserved cost']]
    .dropna().sort_values('On Demand')
)
print(cost_comparison.head(10).to_string(index=False))

# Calculate savings to answer the question in Step 7.
cost_comparison['Hourly savings'] = (
    cost_comparison['On Demand'] - cost_comparison['Linux Reserved cost']
)
cost_comparison['Savings percent'] = (
    cost_comparison['Hourly savings'] / cost_comparison['On Demand'] * 100
)
print(cost_comparison.head(10).to_string(index=False))


# %% Part 1 Step 8 Filter families and summarize
def filter_instance_family(family):
    # Match the first word exactly, so T3 does not also include T3A.
    return data[data['Name'].str.split().str[0] == family]

t2_instances = filter_instance_family('T2')
t3_instances = filter_instance_family('T3')

t2_summary = t2_instances[cost_columns].describe()
t3_summary = t3_instances[cost_columns].describe()
print('T2 rows:', len(t2_instances), 'T3 rows:', len(t3_instances))
print('T2 Instance Costs Summary:\n', t2_summary.to_string())
print('\nT3 Instance Costs Summary:\n', t3_summary.to_string())


# %% Part 1 Step 8 Family boxplots
plt.figure(figsize=(12, 6))
sns.boxplot(data=t2_instances[cost_columns], palette='Blues', showmeans=True)
plt.title('Cost Distribution for T2 Instances', fontsize=16)
plt.ylabel('Cost (USD)', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=t3_instances[cost_columns], palette='Greens', showmeans=True)
plt.title('Cost Distribution for T3 Instances', fontsize=16)
plt.ylabel('Cost (USD)', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.tight_layout()
plt.show()


# %% Part 1 Step 8 Family price comparison
comparison = pd.concat([
    t2_instances[['Name', 'On Demand', 'Linux Reserved cost']],
    t3_instances[['Name', 'On Demand', 'Linux Reserved cost']]
])
comparison_sorted = comparison.dropna().sort_values('On Demand')
print(comparison_sorted.head(10).to_string(index=False))


# %% Part 2 Step 2 Reload and clean
# Reload so Part 2 can also be run independently.
data = pd.read_csv(file_path)
cost_columns = ['On Demand', 'Linux Reserved cost',
                'Linux Spot Minimum cost', 'Windows On Demand cost',
                'Windows Reserved cost']
for column in cost_columns:
    data[column] = pd.to_numeric(
        data[column].str.replace('[$, hourly]', '', regex=True),
        errors='coerce'
    )


# %% Part 2 Step 3 Convert features
data['Instance Memory'] = pd.to_numeric(
    data['Instance Memory'].str.replace(' GiB', '', regex=False)
)
data['vCPUs'] = pd.to_numeric(
    data['vCPUs'].str.extract(r'(\d+)', expand=False)
)
print(data[['Instance Memory', 'vCPUs']].head())


# %% Part 2 Step 4 Handle missing values
data_cleaned = data.dropna(
    subset=['On Demand', 'Instance Memory', 'vCPUs']
)
print(data_cleaned.isnull().sum())
print('Rows before cleaning:', len(data))
print('Rows available for regression:', len(data_cleaned))


# %% Part 2 Step 5 Train test split
from sklearn.model_selection import train_test_split

X = data_cleaned[['Instance Memory', 'vCPUs']]
y = data_cleaned['On Demand']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f'Training samples: {len(X_train)}, Testing samples: {len(X_test)}')


# %% Part 2 Step 6 Train linear regression
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
print(f'Intercept: {model.intercept_}')
print(f'Coefficients: {model.coef_}')
print('Coefficient order: Instance Memory, vCPUs')


# %% Part 2 Step 7 Evaluate
from sklearn.metrics import mean_absolute_error, mean_squared_error

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')


# %% Part 2 Step 8 Actual versus predicted plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='b')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],
         color='red', linestyle='--')
plt.title('Actual vs Predicted On-Demand Costs')
plt.xlabel('Actual On-Demand Cost')
plt.ylabel('Predicted On-Demand Cost')
plt.tight_layout()
plt.show()


# %% Part 2 Step 9 New prediction
# Same example as the handout, with feature names to avoid a warning.
new_instance = pd.DataFrame([[4, 2]], columns=X.columns)
predicted_cost = model.predict(new_instance)
print(f'Predicted On-Demand Cost for 4 GiB, 2 vCPUs: ${predicted_cost[0]:.4f}')
