import pandas as pd
import numpy as np

# Define the paths to the TSV files
paths = {
    'akas': '/Users/ambrishmuniraju/Desktop/Dissertation/IMDb Data Files/title.akas.tsv',
    'basics': '/Users/ambrishmuniraju/Desktop/Dissertation/IMDb Data Files/title.basics.tsv',
    'crew': '/Users/ambrishmuniraju/Desktop/Dissertation/IMDb Data Files/title.crew.tsv',
    'ratings': '/Users/ambrishmuniraju/Desktop/Dissertation/IMDb Data Files/title.ratings.tsv'
}

# Read the TSV files into dataframes, handling missing values as NaN
akas = pd.read_csv(paths['akas'], sep='\t', na_values='\\N')
basics = pd.read_csv(paths['basics'], sep='\t', na_values='\\N')
crew = pd.read_csv(paths['crew'], sep='\t', na_values='\\N')
ratings = pd.read_csv(paths['ratings'], sep='\t', na_values='\\N')

#########################  Data Preparation ##################################
##############################################################################

# Rename the column 'titleId' to 'tconst'
akas.rename(columns={'titleId': 'tconst'}, inplace=True)

# Merge the basics and akas datasets on the 'tconst' column
merged_data = basics.merge(akas, on='tconst', how='outer')

# Filter for movies in the merged dataset
movie_data = merged_data[merged_data['titleType'] == 'movie']

# Apply filters for genre, release year, and language
filtered_data = movie_data[
    (movie_data['startYear'] >= 2016) &  # Filters for release years between 2016 and 2024
    (movie_data['startYear'] <= 2024)
]

# Get the dimensions of the dataset
print(filtered_data.shape)

# Merge the crew and ratings datasets on the 'tconst' column
merged_data_2 = crew.merge(ratings, on='tconst', how='outer')

# Final merge of filtered data with crew and ratings information
merged_data_final = filtered_data.merge(merged_data_2, on='tconst', how='inner')

# Export the final merged data to a CSV file
merged_data_final.to_csv('/Users/ambrishmuniraju/Desktop/Dissertation/IMDb Data/IMDb_final.csv', index=False)

import pandas as pd
import numpy as np
# Load the dataset ----------------------
data = pd.read_csv('/Users/ambrishmuniraju/Desktop/Dissertation/IMDb Data/IMDb_final.csv')

#Inspecting the Dataset ------------------
# Display the first few rows of the dataset
print(data.head())

# Display the data types of each column
print(data.dtypes)

# Get a concise summary of the dataframe
print(data.info())

#check for missing values
print(data.isna().sum())

import matplotlib.pyplot as plt
import seaborn as sns

# Visualization of missing values
missing_values = data.isnull().sum()
missing_values_percentage = (missing_values / len(data)) * 100

plt.figure(figsize=(12, 6))
missing_values_percentage[missing_values_percentage > 0].sort_values(ascending=False).plot(kind='bar', color='salmon')
plt.title('Percentage of Missing Values by Column')
plt.ylabel('Percentage')
plt.show()

######################### Handling Data Quality issues #######################
##############################################################################
# 1. Remove duplicate rows
data = data.drop_duplicates()

# 2. Remove unnecessary columns
data = data.drop(columns=['endYear', 'attributes'])

# 3. Rename the 'startYear' column to 'releaseYear'
data = data.rename(columns={'startYear': 'releaseYear'})

# 4. Handlling missing values

# Identify numerical and categorical columns
numerical_cols = data.select_dtypes(include=['number']).columns.tolist()
categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

# Replace missing values in numerical columns with the mean
for col in numerical_cols:
    data[col].fillna(data[col].mean(), inplace=True)

# Replace missing values in categorical columns with the mode
for col in categorical_cols:
    # Use the first mode if there are multiple modes
    mode_value = data[col].mode()[0]
    data[col].fillna(mode_value, inplace=True)
    
# Remove outliers in the 'runtime' column
data = data[data['runtimeMinutes'] <= 1500]

# Check if there are any missing values left
print(data.isna().sum())

data.to_csv('/Users/ambrishmuniraju/Desktop/Dissertation/IMDb Data/Cl_IMDb.csv', index=False)

######################### Descriptive Statistics #############################
##############################################################################

# Get descriptive statistics for numeric columns
print(data.describe())

# Get descriptive statistics for categorical columns
print(data.describe(include=['object']))

import matplotlib.pyplot as plt
import seaborn as sns

# Setting aesthetic parameters for seaborn
sns.set(style="whitegrid")

# Histograms for numeric data
data.hist(bins=30, figsize=(15, 10))
plt.show()

#########################  Filtering the dataset  ##############################
##############################################################################
data = data[
    (data['genres'].str.contains('Drama')) &  # Ensures the genre contains 'Drama'
    (data['language'] == 'en')               # Ensures the language is English
]

#########################  Feature Engineering  ##############################
##############################################################################
# Convert genres, directors, and writers columns to lists where they are separated by commas
data['genres_list'] = data['genres'].apply(lambda x: x.split(','))
data['writers_list'] = data['writers'].apply(lambda x: x.split(','))
data['directors_list'] = data['directors'].apply(lambda x: x.split(','))

# Count the number of elements in each list for genres, writers, and directors
data['genre_count'] = data['genres_list'].apply(len)
data['writer_count'] = data['writers_list'].apply(len)
data['director_count'] = data['directors_list'].apply(len)

# Calculate the age of the title (current year is 2024)
data['title_age'] = 2024 - data['releaseYear']

# Define thresholds for hit, flop, and average based on the distribution of averageRating
rating_quantiles = data['averageRating'].quantile([0.25, 0.75])
low_threshold = rating_quantiles[0.25]
high_threshold = rating_quantiles[0.75]

data['success_classification'] = pd.cut(data['averageRating'],
                                        bins=[0, low_threshold, high_threshold, 10],
                                        labels=['flop', 'average', 'hit'],
                                        right=True)

# Define short movie flag
data['is_short_movie'] = np.where(data['runtimeMinutes'] <= 90, 'Yes', 'No')

# Step 1: Assuming directors are stored in a string format and separated by commas
data['directors_list'] = data['directors'].apply(lambda x: x.split(','))

# Step 2: Explode the directors_list to count occurrences
director_frequencies = data.explode('directors_list')['directors_list'].value_counts()

# Step 3: Determine the 90th percentile cutoff for top directors
top_directors_threshold = director_frequencies.quantile(0.90)

# Function to check if any director from a movie is in the top 10% most frequent
def is_top_director(directors_list):
    return any(director in directors_list and director_frequencies[director] > top_directors_threshold for director in directors_list)

# Step 4: Apply this function to each movie's director list
data['top_directors'] = data['directors_list'].apply(is_top_director)

# Now you can see which entries have top directors
print(data[['primaryTitle', 'top_directors']].head())
print(data.columns)
# Save the enhanced dataset
data.to_csv('/Users/ambrishmuniraju/Desktop/Dissertation/IMDb Data/enhanced_dataset.csv', index=False)


import pandas as pd
import numpy as np
# Load the dataset ----------------------
data = pd.read_csv('/Users/ambrishmuniraju/Desktop/Dissertation/IMDb Data/enhanced_dataset.csv')
##########################  Feature Importance  ##############################
##############################################################################
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Select numerical and categorical features based on your dataset
numerical_features = ['numVotes', 'runtimeMinutes', 'writer_count', 'ordering', 'releaseYear', 'genre_count', 'director_count', 'title_age']
categorical_features = ['genres', 'language', 'region', 'success_classification', 'is_short_movie', 'top_directors']

# Prepare the feature matrix and target vector
X = data[numerical_features + categorical_features]
y = data['averageRating']

# Create a column transformer to apply different preprocessing to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create a pipeline that includes both preprocessing and the RandomForestRegressor
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=50, random_state=42))
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
rf_pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred_rf = rf_pipeline.predict(X_test)
print("R-squared:", r2_score(y_test, y_pred_rf))

# Get feature importances from the regressor
feature_importances = rf_pipeline.named_steps['regressor'].feature_importances_

# Get feature names from the column transformer
feature_names = numerical_features.copy()
cat_features_transformed = rf_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
feature_names.extend(cat_features_transformed)

# Create a dictionary of feature names and their importance scores
feature_importance_dict = dict(zip(feature_names, feature_importances))

# Aggregating importances for categorical features
aggregate_importances = {name: 0 for name in categorical_features}  # Initialize dict for categorical features
for name, importance in feature_importance_dict.items():
    # Check if the feature name starts with one of the categorical feature names
    for cat_feature in categorical_features:
        if name.startswith(cat_feature):
            aggregate_importances[cat_feature] += importance
            break
    else:
        # It's a numerical feature, just copy it over
        aggregate_importances[name] = importance

# Now sort the features by their aggregated importance
sorted_features = sorted(aggregate_importances.items(), key=lambda x: x[1], reverse=True)

# Print the best performing features
print("Best performing features (sorted by importance):")
for feature, importance in sorted_features:
    print(f"{feature}: {importance:.4f}")

# Plot the top 10 most important features
top_features = sorted_features[:10]
labels, values = zip(*top_features)

plt.figure(figsize=(10, 6))
plt.barh(labels, values, color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Top 10 Most Important Features')
plt.gca().invert_yaxis()  # Invert the y-axis to have the most important feature on top
plt.show()

#Creating other models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# 1. Linear Regression-------------------------------------------------
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)

# 2. Gradient Boosting -----------------------------------------------------------
gb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))
])
gb_pipeline.fit(X_train, y_train)
y_pred_gb = gb_pipeline.predict(X_test)

# 3. XGBoost----------------------------------------------------------
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(n_estimators=100, random_state=42))
])
xgb_pipeline.fit(X_train, y_train)
y_pred_xgb = xgb_pipeline.predict(X_test)

###################### Model Evaluation with Custom Metrics #######################
###################################################################################
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, max_error, explained_variance_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from math import sqrt
import numpy as np
import pandas as pd

# Custom loss functions and evaluation metrics
def huber_loss(y_true, y_pred, delta=1.0):
    residual = np.abs(y_true - y_pred)
    condition = residual <= delta
    squared_loss = 0.5 * residual**2
    linear_loss = delta * residual - 0.5 * delta**2
    return np.where(condition, squared_loss, linear_loss).mean()

def log_cosh_loss(y_true, y_pred):
    def log_cosh(x):
        return np.log((np.exp(x) + np.exp(-x)) / 2.0)
    return np.mean(log_cosh(y_pred - y_true))

# Function to print all metrics
def print_all_metrics(y_true, y_pred, model_name='Model'):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    median_ae = median_absolute_error(y_true, y_pred)
    max_err = max_error(y_true, y_pred)
    rmse = sqrt(mse)
    explained_var = explained_variance_score(y_true, y_pred)
    huber = huber_loss(y_true, y_pred)
    log_cosh = log_cosh_loss(y_true, y_pred)
    correlation_coef = np.corrcoef(y_true, y_pred)[0, 1]
    print(f"{model_name} - Predictions: {y_pred[:5]}, MSE: {mse}, R-squared: {r2}, "
          f"MAE: {mae}, Median AE: {median_ae}, Max Error: {max_err}, RMSE: {rmse}, Explained Variance: {explained_var}, "
          f"Huber Loss: {huber}, Log-cosh Loss: {log_cosh}, Correlation Coefficient: {correlation_coef}")

# Print all metrics for Random Forest
print_all_metrics(y_test, y_pred_rf, 'Random Forest before Tuning')
print_all_metrics(y_test, y_pred_lr, 'Linear Regression before Tuning')
print_all_metrics(y_test, y_pred_gb, 'Gradient Boosting before Tuning')
print_all_metrics(y_test, y_pred_xgb, 'XGBoost before Tuning')

#visualising teh results:-------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(y_true, y_pred, model_name):
    r2 = r2_score(y_true, y_pred)
    correlation = np.corrcoef(y_true, y_pred)[0, 1]

    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, label=f'{model_name}')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.title(f'{model_name}: True Values vs. Predicted Values')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Adding annotations
    plt.text(0.05, 0.95, f'R²: {r2:.2f}\nCorrelation: {correlation:.2f}', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.show()

# Plot for Linear Regression
plot_predictions(y_test, y_pred_lr, 'Linear Regression')

# Plot for Random Forest
plot_predictions(y_test, y_pred_rf, 'Random Forest')

# Plot for Gradient Boosting
plot_predictions(y_test, y_pred_gb, 'Gradient Boosting')

# Plot for XGBoost
plot_predictions(y_test, y_pred_xgb, 'XGBoost')


###################### Hyperparameter Tuning #########################
######################################################################
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, max_error, explained_variance_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from math import sqrt
import numpy as np
import pandas as pd

# Custom loss functions
def huber_loss(y_true, y_pred, delta=1.0):
    residual = np.abs(y_true - y_pred)
    condition = residual <= delta
    squared_loss = 0.5 * residual**2
    linear_loss = delta * residual - 0.5 * delta**2
    return np.where(condition, squared_loss, linear_loss).mean()

def log_cosh_loss(y_true, y_pred):
    def log_cosh(x):
        return np.log((np.exp(x) + np.exp(-x)) / 2.0)
    return np.mean(log_cosh(y_pred - y_true))

# Function to print all metrics
def print_all_metrics(y_true, y_pred, model_name='Model'):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    median_ae = median_absolute_error(y_true, y_pred)
    max_err = max_error(y_true, y_pred)
    rmse = sqrt(mse)
    explained_var = explained_variance_score(y_true, y_pred)
    huber = huber_loss(y_true, y_pred)
    log_cosh = log_cosh_loss(y_true, y_pred)
    correlation_coef = np.corrcoef(y_true, y_pred)[0, 1]
    print(f"{model_name} - Predictions: {y_pred[:5]}, MSE: {mse}, R-squared: {r2}, "
          f"MAE: {mae}, Median AE: {median_ae}, Max Error: {max_err}, RMSE: {rmse}, Explained Variance: {explained_var}, "
          f"Huber Loss: {huber}, Log-cosh Loss: {log_cosh}, Correlation Coefficient: {correlation_coef}")


# 2. Random Forest with RandomizedSearchCV --------------------------------------
from sklearn.ensemble import RandomForestRegressor

rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

rf_param_grid = {
    'regressor__n_estimators': [10, 50, 100, 200],
    'regressor__max_features': ['sqrt', 'log2'],
    'regressor__max_depth': [None, 10, 20, 30, 40],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4],
    'regressor__bootstrap': [True, False]
}

rf_random_search = RandomizedSearchCV(
    rf_pipeline,
    param_distributions=rf_param_grid,
    n_iter=10,
    cv=3,
    scoring='neg_mean_squared_error',
    verbose=1,
    n_jobs=-1,
    random_state=42
)

rf_random_search.fit(X_train, y_train)
y_pred_best_rf = rf_random_search.predict(X_test)
print("Best parameters:", rf_random_search.best_params_)
print_all_metrics(y_test, y_pred_best_rf, 'Random Forest after tuning')

import matplotlib.pyplot as plt
# Calculate performance metrics
r2_rf = r2_score(y_test, y_pred_best_rf)
correlation_rf = np.corrcoef(y_test, y_pred_best_rf)[0, 1]

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best_rf, color='blue', alpha=0.5, label='Random Forest')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Random Forest: True Values vs. Predicted Values')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.legend(loc='upper right')
plt.grid(True)

# Adding annotations
plt.text(0.05, 0.95, f'R²: {r2_rf:.2f}\nCorrelation: {correlation_rf:.2f}', transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

plt.show()

# 3. Gradient Boosting Regressor with RandomizedSearchCV -------------------------------
from sklearn.ensemble import GradientBoostingRegressor

gb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=42))
])

param_grid = {
    'regressor__n_estimators': [50, 100, 150],
    'regressor__learning_rate': [0.05, 0.1, 0.15],
    'regressor__max_depth': [3, 5, 7],
    'regressor__min_samples_split': [2, 5],
    'regressor__min_samples_leaf': [1, 3]
}

random_search = RandomizedSearchCV(
    gb_pipeline,
    param_distributions=param_grid,
    n_iter=10,
    cv=3,
    scoring='neg_mean_squared_error',
    verbose=1,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)
y_pred_gb = random_search.predict(X_test)
print("Best parameters:", random_search.best_params_)
print_all_metrics(y_test, y_pred_gb, 'Gradient Boosting after tuning')

import matplotlib.pyplot as plt
r2 = r2_score(y_test, y_pred_gb)
correlation = np.corrcoef(y_test, y_pred_gb)[0, 1]

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_gb, color='blue', alpha=0.5, label='Gradient Boosting')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Gradient Boosting: True Values vs. Predicted Values')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.legend(loc='upper right')
plt.grid(True)

# Adding annotations
plt.text(0.05, 0.95, f'R²: {r2:.2f}\nCorrelation: {correlation:.2f}', transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

plt.show()

# 4. XGBoost with RandomizedSearchCV ---------------------------------------------
from xgboost import XGBRegressor

xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(n_estimators=100, random_state=42))
])

xgb_param_grid = {
    'regressor__n_estimators': [50, 100, 200, 300],
    'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'regressor__max_depth': [3, 5, 7, 9],
    'regressor__min_child_weight': [1, 2, 3],
    'regressor__subsample': [0.6, 0.8, 1.0],
    'regressor__colsample_bytree': [0.6, 0.8, 1.0]
}

xgb_random_search = RandomizedSearchCV(
    xgb_pipeline,
    param_distributions=xgb_param_grid,
    n_iter=10,
    cv=3,
    scoring='neg_mean_squared_error',
    verbose=1,
    n_jobs=-1,
    random_state=42
)

xgb_random_search.fit(X_train, y_train)
y_pred_best_xgb = xgb_random_search.predict(X_test)
print("Best parameters:", xgb_random_search.best_params_)
print_all_metrics(y_test, y_pred_best_xgb, 'XGBoost after tuning')

import matplotlib.pyplot as plt
# Calculate performance metrics
r2_xgb = r2_score(y_test, y_pred_best_xgb)
correlation_xgb = np.corrcoef(y_test, y_pred_best_xgb)[0, 1]

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best_xgb, color='blue', alpha=0.5, label='XGBoost')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('XGBoost: True Values vs. Predicted Values')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.legend(loc='upper right')
plt.grid(True)

# Adding annotations
plt.text(0.05, 0.95, f'R²: {r2_xgb:.2f}\nCorrelation: {correlation_xgb:.2f}', transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

plt.show()

# Comparing the Models  Before Tuning---------------------------------------------------------------------------
import matplotlib.pyplot as plt

# Data for each model and metric
models = ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'XGBoost']
mse = [0.2710665687697467, 0.03834502162991697, 0.23524440093162938, 0.17645180248466186]  # Mean Squared Error
r_squared = [0.7638863997266039, 0.9665994181772687, 0.7950894398368005, 0.846300963824228]  # R-squared
mae = [0.37525006633496233, 0.0728182117330136, 0.3344735113319631, 0.28371716151019927]  # Mean Absolute Error
median_ae = [0.2850329309691064, 1.1253220577600587e-12, 0.26035236448284493, 0.20684327719936135]  # Median Absolute Error
max_error = [3.9506479341410774, 4.090452380952381, 4.044598620703611, 3.6791024208068848]  # Maximum Error
explained_variance = [0.7638865504365535, 0.9665995742259247, 0.7950897449112317, 0.846302141396944]  # Explained Variance

# Create subplots
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# Function to add value labels
def add_labels(bars, axis):
    for bar in bars:
        height = bar.get_height()
        axis.annotate(f'{height:.2f}',
                      xy=(bar.get_x() + bar.get_width() / 2, height),
                      xytext=(0, 3),  # 3 points vertical offset
                      textcoords="offset points",
                      ha='center', va='bottom')

# Plotting and adding value labels
for i, ax in enumerate(axs.flat):
    color = 'lightcoral'  
    if i == 0:
        bars = ax.bar(models, mse, color=color)
        ax.set_title('Mean Squared Error (MSE)')
        ax.set_ylabel('MSE')
    elif i == 1:
        bars = ax.bar(models, mae, color=color)
        ax.set_title('Mean Absolute Error (MAE)')
        ax.set_ylabel('MAE')
    elif i == 2:
        bars = ax.bar(models, median_ae, color=color)
        ax.set_title('Median Absolute Error')
        ax.set_ylabel('Median AE')
    elif i == 3:
        bars = ax.bar(models, max_error, color=color)
        ax.set_title('Maximum Error')
        ax.set_ylabel('Max Error')
    elif i == 4:
        bars = ax.bar(models, explained_variance, color=color)
        ax.set_title('Explained Variance')
        ax.set_ylabel('Explained Variance')
    else:
        ax.axis('off')

    add_labels(bars, ax)
    ax.set_xticklabels(models, rotation=45, ha='right')

# Add main title
fig.suptitle('Comparison of the Models Before Tuning', fontsize=16, fontweight='bold')

# Adjust layout
plt.tight_layout()

# Display the plot
plt.show()

# Comparing the Models after Tuning---------------------------------------------------------------------------
import matplotlib.pyplot as plt

# Data for each model and metric
models = ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'XGBoost']
mse = [0.31619448990581106, 0.24425033061701035, 0.2396610943316968, 0.17346053251567453]
mae = [0.44118304667609787, 0.37497992784486006, 0.38640258756305046, 0.30947176175225105]
median_ae = [0.3860402803304934, 0.3070073203981911, 0.33426152201522097, 0.2376448631286623]
max_error = [3.4954171907861777, 3.7152441568189296, 3.7219761781934677, 3.686643886566162]
explained_variance = [0.8099476350661312, 0.853190732616188, 0.8559431141681161, 0.8957326023395534]


# Create subplots
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# Function to add value labels
def add_labels(bars, axis):
    for bar in bars:
        height = bar.get_height()
        axis.annotate(f'{height:.2f}',
                      xy=(bar.get_x() + bar.get_width() / 2, height),
                      xytext=(0, 3),  # 3 points vertical offset
                      textcoords="offset points",
                      ha='center', va='bottom')

# Plotting and adding value labels
for i, ax in enumerate(axs.flat):
    if i == 0:
        bars = ax.bar(models, mse, color='red')
        ax.set_title('Mean Squared Error (MSE)')
        ax.set_ylabel('MSE')
    elif i == 1:
        bars = ax.bar(models, mae, color='red')
        ax.set_title('Mean Absolute Error (MAE)')
        ax.set_ylabel('MAE')
    elif i == 2:
        bars = ax.bar(models, median_ae, color='red')
        ax.set_title('Median Absolute Error')
        ax.set_ylabel('Median AE')
    elif i == 3:
        bars = ax.bar(models, max_error, color='red')
        ax.set_title('Maximum Error')
        ax.set_ylabel('Max Error')
    elif i == 4:
        bars = ax.bar(models, explained_variance, color='red')
        ax.set_title('Explained Variance')
        ax.set_ylabel('Explained Variance')
    else:
        ax.axis('off')

    add_labels(bars, ax)
    ax.set_xticklabels(models, rotation=45, ha='right')

# Add main title
fig.suptitle('Comparison of the Models After Tuning', fontsize=16, fontweight='bold')

# Adjust layout
plt.tight_layout()

# Display the plot
plt.show()

#Error Metrics and R² Score Comparison before Tuning -------------------------------------------------------------
import matplotlib.pyplot as plt

# Data setup
models = ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'XGBoost']
rmse = [0.5206405370020152, 0.1958188490159131, 0.48502000054804895, 0.4200616650977114]
huber_loss = [0.12136211199915771, 0.017936209819826937, 0.10640086509342019, 0.08225067424770932]
log_cosh_loss = [0.1110809587483981, 0.016433679132679648, 0.09745010332265062, 0.07554176268918415]
r_squared = [0.7638863997266039, 0.9665994181772687, 0.7950894398368005, 0.846300963824228]


# Create the plot
fig, ax1 = plt.subplots(figsize=(14, 7))

# Bar plots with light colors
width = 0.2
x = range(len(models))
bars1 = ax1.bar(x, rmse, width, label='RMSE', color='lightblue')
bars2 = ax1.bar([p + width for p in x], huber_loss, width, label='Huber Loss', color='peachpuff')
bars3 = ax1.bar([p + width * 2 for p in x], log_cosh_loss, width, label='Log-Cosh Loss', color='lightgreen')
ax1.set_ylabel('Error Metrics')
ax1.set_title('Error Metrics and R² Score Comparison of Models before Tuning')
ax1.set_xticks([p + width for p in x])
ax1.set_xticklabels(models)
ax1.legend(loc='upper right')

# Add labels on the bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

# Line plot for R-squared
ax2 = ax1.twinx()
ax2.plot(models, r_squared, 'r-o', label='R² Score')
ax2.set_ylabel('R² Score')
ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.85))

# Show plot
plt.show()

#Error Metrics and R² Score Comparison After Tuning -------------------------------------------------------------
import matplotlib.pyplot as plt

# Data setup
models = ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'XGBoost']
rmse = [0.5206405370020152, 0.3758656942080807, 0.42595153480821396, 0.3215085895316941]
huber_loss = [0.12136211199915771, 0.06629939535459012, 0.08442972032604971, 0.04909373292032188]
log_cosh_loss = [0.1110809587483981, 0.060992820100191254, 0.0775328580009213, 0.045441521677047715]
r_squared = [0.7638863997266039, 0.8769418384134443, 0.8419605826407133, 0.909961094871538]

# Create the plot
fig, ax1 = plt.subplots(figsize=(14, 7))

# Bar plots with dark colors
width = 0.2
x = range(len(models))
bars1 = ax1.bar(x, rmse, width, label='RMSE', color='navy')
bars2 = ax1.bar([p + width for p in x], huber_loss, width, label='Huber Loss', color='lightcoral')
bars3 = ax1.bar([p + width * 2 for p in x], log_cosh_loss, width, label='Log-Cosh Loss', color='green')
ax1.set_ylabel('Error Metrics')
ax1.set_title('Error Metrics and R² Score Comparison of Models after Tuning')
ax1.set_xticks([p + width for p in x])
ax1.set_xticklabels(models)
ax1.legend(loc='upper right')

# Add labels on the bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

# Line plot for R-squared
ax2 = ax1.twinx()
ax2.plot(models, r_squared, 'r-o', label='R² Score')
ax2.set_ylabel('R² Score')
ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.85))

# Show plot
plt.show()