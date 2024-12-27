import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from imblearn.over_sampling import SMOTE
from mlxtend.frequent_patterns import apriori, association_rules
from numpy.linalg import svd
from prettytable import PrettyTable
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve, confusion_matrix, recall_score, precision_score,
                             f1_score, classification_report)
from sklearn.metrics import mean_squared_error, silhouette_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pandas.plotting import scatter_matrix
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings('ignore')

np.random.seed(5805)
plt.style.use('seaborn-whitegrid')

# Set the maximum number of columns to display
pd.set_option('display.width', 600)
pd.set_option('display.max_columns', 15)
pd.set_option('display.precision', 2)
sns.set_style('whitegrid')

# ==============================================================================================================
# ===================================== Phase I: Feature Engineering & EDA =====================================
# ==============================================================================================================

# Load the dataset
stock_df = pd.read_csv('./World-Stock-Prices-Dataset.csv')

# to remove: for full dataset
stock_df = stock_df[stock_df["Ticker"].isin(['GOOGL', 'AAPL'])]

# Convert the date column to datetime format
stock_df['Date'] = pd.to_datetime(stock_df['Date'], utc=True)

# =================================================================
# ======================= Feature Engineering =====================
# =================================================================

stock_df['Volatility'] = stock_df['High'] - stock_df['Low']
stock_df['Price_Diff'] = stock_df['Close'] - stock_df['Open']
stock_df['ROC'] = stock_df.groupby('Ticker')['Close'].pct_change(5)
stock_df['ATR_14'] = stock_df.groupby('Ticker')['High'].transform(lambda x: x.diff(1).abs().rolling(window=14).mean())
stock_df['Rolling_Std_Close_5'] = stock_df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=5).std())
stock_df['Is_Quarter_End'] = (stock_df['Date'].dt.month % 3 == 0).astype('int')

num_bins = 2
bin_edges = pd.cut(stock_df['Close'], bins=num_bins, labels=False)

# Create a new column 'close_category' with the bin labels
stock_df['Close_Category'] = pd.cut(stock_df['Close'], bins=num_bins, labels=range(num_bins))

# View the first few rows of the dataset
print("\nFirst 5 rows of the World Stock Prices Dataset: ")
print(stock_df.head())


# Check the shape of the data
print("\nShape: ", stock_df.shape)

# view the data types and non-null values in the dataset
print("\nInformation about the dataset:", "\n")
print(stock_df.info())

# =============== Data Cleaning =================
# Check the missing values
print("\nMissing Values: ", stock_df.isnull().sum().sum())

# =============== Check for duplication and removal ===============
print("\nDuplicated Values: ", stock_df.duplicated().sum())

# ================ Statistical Summary ================
print("\nStatistical Summary of numerical features: ")
print(stock_df.describe())

# ==================== Data Cleaning: Fix missing data ====================

# Replace missing values for numerical features with mean
numeric_features = stock_df.select_dtypes(include=['float64', 'int64', 'int32']).columns
numeric_features = numeric_features.drop(['Is_Quarter_End'])
stock_df[numeric_features] = stock_df[numeric_features].fillna(stock_df[numeric_features].mean())

# Replace missing values for categorical features with mode
categorical_features = stock_df.select_dtypes(include=['object', 'category']).columns
categorical_features = list(categorical_features) + ['Is_Quarter_End']
stock_df[categorical_features] = stock_df[categorical_features].fillna(stock_df[categorical_features].mode().iloc[0])

# Check the missing values
print("\nMissing Values: ", stock_df.isnull().sum().sum())

# =================================== EDA =================================

# Distribution plot of continuous numerical features
plt.figure(figsize=(20, 15), facecolor='white')
plt.suptitle('Distribution Plot of Continuous Numerical Features', fontsize=20)

for i, col in enumerate(numeric_features):
    plt.subplot(5, 3, i + 1)
    sns.distplot(stock_df[col])
    plt.xlabel(col)
plt.show()

# Distribution plot of categorical features
plt.figure(figsize=(10, 10), facecolor='white')
plt.suptitle('Distribution Plot of Categorical Features', fontsize=20)

for i, col in enumerate(categorical_features):
    plt.subplot(2, 3, i + 1)
    sns.countplot(data=stock_df, x=col)
    plt.xlabel(col)
    plt.xticks(rotation=90)
plt.show()

# Outlier detection
# Boxplot of continuous/numerical features
plt.figure(figsize=(20, 15), facecolor='white')
plt.suptitle('Boxplot of Continuous Numerical Features', fontsize=20)

for i, col in enumerate(numeric_features):
    plt.subplot(5, 3, i + 1)
    sns.boxplot(data=stock_df, x=col)
    plt.xlabel(col)
plt.show()

# Relationship between continuous/numerical features and the target
plt.figure(figsize=(20, 15), facecolor='white')
plt.suptitle('Relationship between Continuous Numerical Features and the Target', fontsize=20)

target = 'Close'
for i, col in enumerate(numeric_features):
    plt.subplot(5, 3, i + 1)
    sns.scatterplot(x=stock_df[col], y=stock_df[target])
    plt.xlabel(col)
    plt.ylabel(target)
    plt.title(f'{col} vs. {target}')
plt.show()

# Outlier detection
# Relationship between categorical features and the target
plt.figure(figsize=(10, 18), facecolor='white')
plt.suptitle('Relationship between Categorical Features and the Target')

for i, col in enumerate(categorical_features):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(x=stock_df[col], y=stock_df[target])
    plt.xlabel(col)
    plt.ylabel(target)
    plt.xticks(rotation=90)
    plt.title(f'{col} vs. {target}')
plt.show()

# Line plot of stock prices over time
plt.figure(figsize=(10, 6))
sns.lineplot(x='Date', y='Close', data=stock_df, hue='Brand_Name')
plt.title('Stock Prices of popular stocks Over Time')
plt.show()

# time series decomposition
result = seasonal_decompose(stock_df['Close'], model='additive', period=30)
result.plot()
plt.show()

data_grouped = stock_df.groupby(stock_df['Date'].dt.year).mean()
plt.subplots(figsize=(20, 10))

for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
    plt.subplot(2, 2, i + 1)
    data_grouped[col].plot.bar()
plt.show()

# ========================= Correlation =========================
print("\n========================= Correlation =========================")

# Calculate the correlation matrix
correlation_matrix = stock_df.corr()

# Display the correlation matrix
print("\nCorrelation Matrix:")
print(correlation_matrix.round(2))

# Plot the correlation matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()

# ================================= Dimensionality Reduction/Feature Selection ======================

# to make sure that the collinearity does not exist in the data matrix.
stock_df.drop(['Open', 'High', 'Low'], inplace=True, axis=1)

numeric_features = stock_df.select_dtypes(include=['float64', 'int64']).columns
categorical_features = stock_df.select_dtypes(include=['object', 'category']).columns
categorical_features = categorical_features.drop(['Brand_Name', 'Industry_Tag', 'Country', 'Close_Category'])

target = 'Close_Category'

# Perform one-hot encoding
# for full dataset keep drop first true

stock_df_encoded = pd.get_dummies(stock_df, columns=categorical_features, drop_first=True)
# Display the encoded DataFrame
print("\nEncoded DataFrame:")
print(stock_df_encoded.head())

# Standardize the feature columns (exclude one-hot encoded columns)
scaler = StandardScaler()
stock_df_encoded[numeric_features] = scaler.fit_transform(stock_df_encoded[numeric_features])

X = stock_df_encoded.drop([target, 'Brand_Name', 'Industry_Tag', 'Country', 'Date', 'Close'], axis=1)
y = stock_df_encoded[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5805, shuffle=True)

# =============== Random Forest Analysis ===============
print("\n======================= Random Forest Analysis ========================")

# Initialize the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=5805, n_jobs=-1, max_depth=10)

# Train the model
rf_model.fit(X_train, y_train)

features = X_train.columns
feature_importances = rf_model.feature_importances_
indices = np.argsort(feature_importances)

plt.figure(figsize=(10, 10))
plt.title("Feature Importance's from Random Forest Analysis")
plt.barh(range(len(indices)), feature_importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.grid(True)
plt.show()

# Identify the least important features and eliminate them
threshold = 0.01

# Identify and print the features to be eliminated based on the threshold
eliminated_features = [features[i] for i in indices if feature_importances[i] < threshold]
print(f"Features to be eliminated based on importance below {threshold} in Random Forest:")
print(eliminated_features)

# Select features based on the threshold
selected_features = [features[i] for i in indices if feature_importances[i] >= threshold]
print("\nSelected features based on Random Forest:")
print(selected_features)

# ====================================== PCA =================================
print("\n================================ PCA =================================")

# Perform PCA
pca = PCA()
X_pca = pca.fit_transform(X)

# Calculate the proportion of variance explained by each component
explained_variance_ratio = pca.explained_variance_ratio_

# Calculate the cumulative explained variance
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# Find the number of features needed to explain more than 90% of the variance
n_features = np.argmax(cumulative_explained_variance > 0.9) + 1

# Print the number of features needed
print("Number of features needed to explain more than 90% of the variance:", n_features)

plt.plot(np.arange(1, len(cumulative_explained_variance) + 1, 1),
         cumulative_explained_variance)
plt.xlabel('Number of features')
plt.ylabel('Cumulative explained variance')
plt.title('Cumulative Explained Variance vs. Number of Features using PCA')
plt.grid(True)
plt.axhline(y=0.9, color='g', linestyle='--', label='90% Threshold')
# Draw a vertical line at the corresponding number of features
plt.axvline(x=n_features, color='r', linestyle='--', label=f'{n_features} Features')
plt.legend(loc='lower right')
plt.show()

# Display the names of the important features based on their associated components
important_features_names_pca = X.columns[:n_features]
print("Names of Important Features:")
print(important_features_names_pca)

print(f"Condition Number: , {np.linalg.cond(X_pca):.2f}")

# =================================== SVD =================================
print("\n================================ SVD =================================")

# Perform Singular Value Decomposition
U, Sigma, VT = svd(X, full_matrices=False)

# Variance explained by each singular value
explained_variance_ratio = (Sigma ** 2) / np.sum(Sigma ** 2)

# Calculate the cumulative explained variance
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# Find the number of features needed to explain more than 90% of the variance
n_features = np.argmax(cumulative_explained_variance > 0.9) + 1

# Print the number of features needed
print("Number of features needed to explain more than 90% of the variance:", n_features)

plt.plot(np.arange(1, len(cumulative_explained_variance) + 1, 1),
         cumulative_explained_variance)
plt.xlabel('Number of features')
plt.ylabel('Cumulative explained variance')
plt.title('Cumulative Explained Variance vs. Number of Features using SVD')
plt.grid(True)
plt.axhline(y=0.9, color='g', linestyle='--', label='90% Threshold')
# Draw a vertical line at the corresponding number of features
plt.axvline(x=n_features, color='r', linestyle='--', label=f'{n_features} Features')
plt.legend(loc='lower right')
plt.show()

# Display the names of the important features based on their associated components
important_features_names = X.columns[:n_features]
print("Names of Important Features:")
print(important_features_names)

# =================================== VIF =================================
print("\n================================ VIF =================================")

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]

vif_threshold = 5

# Identify variables with high VIF values
high_vif_variables = vif_data[vif_data["VIF"] > vif_threshold]

# Display the number of important features
num_important_features = len(high_vif_variables)
print("Number of Important Features:", num_important_features)

# Display the names of the important features
important_features_names = high_vif_variables["Feature"].tolist()
print("Names of Important Features:")
print(important_features_names)

# ========================== Covariance Matrix ========================
print("\n================================ Covariance Matrix =======================")
# Calculate the sample covariance matrix
covariance_matrix = np.cov(X, rowvar=False)

# Display the heatmap
plt.figure(figsize=(20, 20))
sns.heatmap(covariance_matrix, annot=True, cmap='RdBu', fmt=".2f", xticklabels=X.columns, yticklabels=X.columns)
plt.title('Sample Covariance Matrix Heatmap', fontsize=20)
plt.show()

# ========================== Correlation matrix =========================
# Calculate the sample Pearson correlation coefficients matrix
correlation_matrix = X.corr()

# Display the heatmap
plt.figure(figsize=(20, 20))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", xticklabels=X.columns, yticklabels=X.columns)
plt.title('Sample Pearson Correlation Coefficients Matrix Heatmap', fontsize=20)
plt.show()

# ============================= Balanced or Imbalanced Data ===============
print("\n===================== Balanced or Imbalanced Data =============================")

# Plot the class distribution
sns.countplot(x='Close_Category', data=stock_df_encoded)
plt.title('Class Distribution of Target Variable before SMOTE')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

X = stock_df_encoded.drop(['Close_Category', 'Date', 'Brand_Name', 'Industry_Tag', 'Country'], axis=1)
y = stock_df_encoded['Close_Category']
oversample = SMOTE()
X_resampled, y_resampled = oversample.fit_resample(X, y)

stock_df_encoded = pd.concat([pd.DataFrame(X_resampled, columns=X.columns),
                              pd.DataFrame({'Close_Category': y_resampled})], axis=1)

class_distribution_after_smote = stock_df_encoded['Close_Category'].value_counts()
print("\nClass distribution after SMOTE:")
print(class_distribution_after_smote)

# Plot the class distribution
sns.countplot(x='Close_Category', data=stock_df_encoded)
plt.title('Class Distribution of Target Variable before SMOTE')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# ==========================================================================================================
# ==================================== Phase II: Regression Analysis =======================================
# ==========================================================================================================
print("\n================================ Regression Analysis ==================================")

target = 'Close'

X_reg = stock_df_encoded.drop([target], axis=1)
y_reg = stock_df_encoded[target]

# add constant
X_reg = sm.add_constant(X_reg)

# split the dataset into train-test 80-20
X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, shuffle=True, random_state=5805)

table = pd.DataFrame(columns=["AIC", "BIC", "Adj. R2", "P-Value", "R2", "F-Statistic"])
removed_features = []

model_regression = sm.OLS(y_train, X_train).fit()
print("\nModel Summary before OLS:")
print(model_regression.summary())

while True:
    model_regression = sm.OLS(y_train, X_train).fit()

    # Find the feature with the maximum p-value
    max_p_value_feature = model_regression.pvalues.idxmax()

    # Check if the maximum p-value is greater than 0.01
    if model_regression.pvalues.max() > 0.01:
        # Record the statistics before dropping the feature
        table.loc[max_p_value_feature, "AIC"] = model_regression.aic.round(3)
        table.loc[max_p_value_feature, "BIC"] = model_regression.bic.round(3)
        table.loc[max_p_value_feature, "Adj. R2"] = model_regression.rsquared_adj.round(3)
        table.loc[max_p_value_feature, "P-Value"] = model_regression.pvalues[max_p_value_feature].round(3)
        table.loc[max_p_value_feature, "R2"] = model_regression.rsquared.round(3)
        table.loc[max_p_value_feature, "F-Statistic"] = model_regression.fvalue.round(3)

        # Drop the feature with the maximum p-value
        # print(model_regression.summary())
        X_train = X_train.drop(max_p_value_feature, axis=1)
        X_test = X_test.drop(max_p_value_feature, axis=1)
        removed_features.append(max_p_value_feature)
    else:
        # If the maximum p-value is not greater than 0.01, break the loop
        break

print("\nModel Summary after OLS:")
print(model_regression.summary())

selected_features_reg = X_train.columns

print("\nTable of AIC, BIC, Adj. R2, P-Value, R2, F-Statistic:")
print(table)

print("\nFinal Selected Features:", selected_features_reg)
print("\nEliminated Features are:", removed_features)

final_model_reg = sm.OLS(y_train, X_train).fit()
print("\nFinal Model OLS summary:")
print(final_model_reg.summary())

# Make predictions on the test set
y_pred_reg = final_model_reg.predict(X_test)

# Reverse transformation
scaler.fit(y_test.values.reshape(-1, 1))
y_train_original_reg = scaler.inverse_transform(y_train.values.reshape(-1, 1))
y_test_original_reg = scaler.inverse_transform(y_test.values.reshape(-1, 1))
y_pred_original_reg = scaler.inverse_transform(y_pred_reg.values.reshape(-1, 1))

# Plot original test set (Sales) vs. predicted values
plt.figure(figsize=(10, 8))
plt.plot(np.arange(1, len(y_train_original_reg) + 1, 1), y_train_original_reg, label='Train Close')
plt.plot(np.arange(1, len(y_test_original_reg) + 1, 1), y_test_original_reg, label='Actual Close')
plt.plot(np.arange(1, len(y_pred_original_reg) + 1, 1), y_pred_original_reg, label='Predicted Close')
plt.xlabel("Actual Close (Test Set)")
plt.ylabel("Predicted Close")
plt.title("Train vs. Actual vs. Predicted Close")
plt.legend(loc='lower right')
plt.show()

# Table showing the final R-squared, adjusted R-square, AIC, BIC and MSE

# Calculate R-squared, adjusted R-squared, AIC, BIC, and MSE
r_squared = final_model_reg.rsquared
adj_r_squared = final_model_reg.rsquared_adj
aic = final_model_reg.aic
bic = final_model_reg.bic
mse_reg = mean_squared_error(y_test_original_reg, y_pred_original_reg)

# Create a DataFrame to display the results
results_table = pd.DataFrame({
    'Metric': ['R-squared', 'Adjusted R-squared', 'AIC', 'BIC', 'MSE'],
    'Value': [r_squared, adj_r_squared, aic, bic, mse_reg]
})

# Display the results table
print("\nTable showing the final R-squared, adjusted R-square, AIC, BIC and MSE:")
print(results_table)

# Confidence Interval Analysis
confidence_interval = final_model_reg.conf_int()
print("Confidence Interval:\n", confidence_interval)

# ==========================================================================================================
# ===================================== Phase III: Classification Analysis =================================
# ==========================================================================================================

print("\n================================ Classification Analysis =================================")


def mean(numbers):
    if not numbers:
        raise ValueError("Input list is empty")
    return sum(numbers) / len(numbers)


# ================================== Stratified K-cross validation =================================
def kfold_cross_validation(model, X, y, k=5, verbose=True, random_state=5805):
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)

    scores = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": []
    }

    fold = 0
    for train_index, test_index in kf.split(X, y):
        fold += 1
        X_train_kf, X_test_kf = X.iloc[train_index], X.iloc[test_index]
        y_train_kf, y_test_kf = y.iloc[train_index], y.iloc[test_index]

        try:
            model.fit(X_train_kf, y_train_kf)
            y_pred_kf = model.predict(X_test_kf)

            scores["accuracy"].append(accuracy_score(y_test_kf, y_pred_kf))
            scores["precision"].append(precision_score(y_test_kf, y_pred_kf, average='macro'))
            scores["recall"].append(recall_score(y_test_kf, y_pred_kf, average='macro'))
            scores["f1"].append(f1_score(y_test_kf, y_pred_kf, average='macro'))

            if verbose:
                print(f'\nFold {fold} - Accuracy: {scores["accuracy"][-1]}, Precision: {scores["precision"][-1]}, '
                      f'Recall: {scores["recall"][-1]}, F1 Score: {scores["f1"][-1]}')
        except NotFittedError as e:
            print(f'Error in fold {fold}: {e}')

    avg_scores = {
        "average_accuracy": mean(scores["accuracy"]),
        "average_precision": mean(scores["precision"]),
        "average_recall": mean(scores["recall"]),
        "average_f1": mean(scores["f1"]),
        "all_scores": scores
    }

    # Create a DataFrame for all_scores
    scores_df = pd.DataFrame(avg_scores["all_scores"])
    avg_scores.pop("all_scores")
    # Add the average scores to the DataFrame
    avg_scores_df = pd.DataFrame.from_dict(avg_scores, orient='index', columns=['Value'])
    avg_scores_df.index.name = 'Metric'
    avg_scores_df.reset_index(inplace=True)

    # Display the DataFrame
    print("\nAverage Scores:")
    print(avg_scores_df)

    print("\nIndividual Scores:")
    print(scores_df)

    return avg_scores


# ===================================================================================================

X = pd.DataFrame(stock_df_encoded, columns=selected_features)
y = stock_df_encoded['Close_Category']

print(X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5805)

# =================================== Decision Tree Classifier ==================================
print("\n========================== Decision Tree Classifier =======================")

# develop a decision tree-based model
decision_tree = DecisionTreeClassifier(random_state=5805)
decision_tree.fit(X_train, y_train)

# make predictions on the test set
y_pred_decision_tree = decision_tree.predict(X_test)

decision_tree_accuracy = accuracy_score(y_test, y_pred_decision_tree)

# display the performance on the test set
print(f'\nTest set accuracy: {decision_tree_accuracy:.2f}')

# confusion matrix
decision_tree_confusion_matrix = confusion_matrix(y_test, y_pred_decision_tree)
# precision
decision_tree_precision = precision_score(y_test, y_pred_decision_tree)
# recall
decision_tree_recall = recall_score(y_test, y_pred_decision_tree)
# specificity
decision_tree_specificity = decision_tree_confusion_matrix[0, 0] / (decision_tree_confusion_matrix[0, 0] +
                                                                    decision_tree_confusion_matrix[0, 1])
# f-score
decision_tree_f_score = f1_score(y_test, y_pred_decision_tree)
# AUC
decision_tree_auc = roc_auc_score(y_test, y_pred_decision_tree)
# ROC
decision_tree_fpr, decision_tree_tpr, decision_tree_thresholds = roc_curve(y_test, y_pred_decision_tree)
decision_tree_roc = pd.DataFrame({'FPR': decision_tree_fpr, 'TPR': decision_tree_tpr})

# # display the important features
# feature_importance = decision_tree.feature_importances_
# feature_names = stock_df_encoded.columns
# feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
# print(f'Feature importance:\n {feature_importance_df.sort_values(by="importance", ascending=False)}')

# Plot the AUC and ROC in one plot
plt.figure(figsize=(8, 6))
plt.plot(decision_tree_fpr, decision_tree_tpr, color='darkorange', lw=2,
         label='ROC curve (area = %0.2f)' % decision_tree_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for Decision Tree')
plt.legend()
plt.grid(True)
plt.show()

# stratified k-fold cross validation
kfold_cross_validation(decision_tree, X, y)

# ================== Pre Pruned Decision Tree =================
print(f'\n======================== Pre-pruned Decision Tree ===================')

param_grid = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'splitter': ['best', 'random'],
    'max_depth': [2, 4, 6],
    'min_samples_split': [2, 4, 6],
    'max_features': [4],
    'ccp_alpha': np.linspace(0.0, 0.05, 10)
}

grid_search = GridSearchCV(decision_tree, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# display the best parameters
print(f'Best parameters: {grid_search.best_params_}')

# display the best score
print(f'Best score: {grid_search.best_score_.round(2)}')

# display the final tree
pre_pruned_tree_classifier = grid_search.best_estimator_
print(f'Final tree: {pre_pruned_tree_classifier}')

# display the accuracy on the console
pre_pruned_y_pred = pre_pruned_tree_classifier.predict(X_test)
pre_pruned_accuracy = accuracy_score(y_test, pre_pruned_y_pred)
print(f'Accuracy of the Pre Pruned Tree: {pre_pruned_accuracy:.2f}')

# display the final tree
plot_tree(pre_pruned_tree_classifier, feature_names=X.columns.tolist(), class_names=['0', '1'], filled=True)
plt.title('Pre Pruned Final tree')
plt.show()

# confusion matrix
pre_pruned_tree_confusion_matrix = confusion_matrix(y_test, pre_pruned_y_pred)
# precision
pre_pruned_tree_precision = precision_score(y_test, pre_pruned_y_pred)
# recall
pre_pruned_tree_recall = recall_score(y_test, pre_pruned_y_pred)
# specificity
pre_pruned_tree_specificity = pre_pruned_tree_confusion_matrix[0, 0] / (pre_pruned_tree_confusion_matrix[0, 0] +
                                                                        pre_pruned_tree_confusion_matrix[0, 1])
# f-score
pre_pruned_f_score = f1_score(y_test, pre_pruned_y_pred)
# AUC
pre_pruned_tree_auc = roc_auc_score(y_test, pre_pruned_y_pred)
# ROC
pre_pruned_tree_fpr, pre_pruned_tree_tpr, pre_pruned_tree_thresholds = roc_curve(y_test, pre_pruned_y_pred)
pre_pruned_tree_roc = pd.DataFrame({'FPR': pre_pruned_tree_fpr, 'TPR': pre_pruned_tree_tpr})

# Plot the AUC and ROC in one plot
plt.figure(figsize=(8, 6))
plt.plot(pre_pruned_tree_fpr, pre_pruned_tree_tpr, color='darkorange', lw=2,
         label='ROC curve (area = %0.2f)' % pre_pruned_tree_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for Pre-Pruned Decision Tree')
plt.legend()
plt.grid(True)
plt.show()

# stratified k-fold cross validation
kfold_cross_validation(pre_pruned_tree_classifier, X, y)

# ================ Post Pruned Decision Tree =================
path = grid_search.best_estimator_.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

accuracy_train, accuracy_test = [], []

for i in ccp_alphas:
    tree = DecisionTreeClassifier(random_state=5805, ccp_alpha=i)
    tree.fit(X_train, y_train)
    y_train_pred = tree.predict(X_train)
    y_test_pred = tree.predict(X_test)
    accuracy_train.append(accuracy_score(y_train, y_train_pred))
    accuracy_test.append(accuracy_score(y_test, y_test_pred))

plt.figure(figsize=(8, 8))
plt.plot(ccp_alphas, accuracy_train, marker='o', label='train', drawstyle='steps-post')
plt.plot(ccp_alphas, accuracy_test, marker='o', label='test', drawstyle='steps-post')
plt.xlabel('ccp_alpha')
plt.ylabel('Accuracy')
plt.title('Accuracy vs ccp_alpha for train and test sets')
plt.legend()
plt.grid(True)
plt.show()

print('\nOptimal ccp_alpha:', ccp_alphas[accuracy_test.index(max(accuracy_test))])

post_pruned_tree_classifier = DecisionTreeClassifier(random_state=5805,
                                                     ccp_alpha=ccp_alphas[accuracy_test.index(max(accuracy_test))])
post_pruned_tree_classifier.fit(X_train, y_train)

y_train_pred = post_pruned_tree_classifier.predict(X_train)
post_pruned_y_pred = post_pruned_tree_classifier.predict(X_test)
post_pruned_accuracy = accuracy_score(y_test, post_pruned_y_pred)

print('Accuracy on train: %.2f' % accuracy_score(y_train, y_train_pred))
print('Accuracy on test: %.2f' % post_pruned_accuracy)

plt.figure(figsize=(10, 10))
plot_tree(post_pruned_tree_classifier, filled=True, rounded=True, feature_names=X.columns.tolist(),
          class_names=['0', '1'])
plt.title(f'Decision Tree with ccp_alpha = {ccp_alphas[accuracy_test.index(max(accuracy_test))]}')
plt.show()

# confusion matrix
post_pruned_tree_confusion_matrix = confusion_matrix(y_test, post_pruned_y_pred)
# precision
post_pruned_tree_precision = precision_score(y_test, post_pruned_y_pred)
# recall
post_pruned_tree_recall = recall_score(y_test, post_pruned_y_pred)
# specificity
post_pruned_tree_specificity = post_pruned_tree_confusion_matrix[0, 0] / (post_pruned_tree_confusion_matrix[0, 0] +
                                                                          post_pruned_tree_confusion_matrix[0, 1])
# f-score
post_pruned_f_score = f1_score(y_test, post_pruned_y_pred)
# AUC
post_pruned_tree_auc = roc_auc_score(y_test, post_pruned_y_pred)
# ROC
post_pruned_tree_fpr, post_pruned_tree_tpr, post_pruned_tree_thresholds = roc_curve(y_test, post_pruned_y_pred)
post_pruned_tree_roc = pd.DataFrame({'FPR': post_pruned_tree_fpr, 'TPR': post_pruned_tree_tpr})

# Plot the AUC and ROC in one plot
plt.figure(figsize=(8, 6))
plt.plot(post_pruned_tree_fpr, post_pruned_tree_tpr, color='darkorange', lw=2,
         label='ROC curve (area = %0.2f)' % post_pruned_tree_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for Post-Pruned Decision Tree')
plt.legend()
plt.grid(True)
plt.show()

# stratified k-fold cross validation
kfold_cross_validation(post_pruned_tree_classifier, X, y)

# create a table and compare the accuracy, confusion matrix, recall, AUC, and ROC of the decision,
# pre-pruned and post pruned tree
# Create a table to compare the metrics

decision_table = {
    "Metric": ["Accuracy", "Confusion Matrix", "Recall", "AUC", "ROC", "Precision", "Specificity", "F-score"],
    "Decision Tree": [round(decision_tree_accuracy, 2), decision_tree_confusion_matrix,
                      round(decision_tree_recall, 2),
                      round(decision_tree_auc, 2), decision_tree_roc, round(decision_tree_precision, 2),
                      round(decision_tree_specificity, 2), round(decision_tree_f_score, 2)],
    "Pre-Pruned Tree": [round(pre_pruned_accuracy, 2), pre_pruned_tree_confusion_matrix,
                        round(pre_pruned_tree_recall, 2),
                        round(pre_pruned_tree_auc, 2), pre_pruned_tree_roc, round(pre_pruned_tree_precision, 2),
                        round(pre_pruned_tree_specificity, 2), round(pre_pruned_f_score, 2)],
    "Post-Pruned Tree": [round(post_pruned_accuracy, 2), post_pruned_tree_confusion_matrix,
                         round(post_pruned_tree_recall, 2),
                         round(post_pruned_tree_auc, 2), post_pruned_tree_roc, round(post_pruned_tree_precision, 2),
                         round(post_pruned_tree_specificity, 2), round(post_pruned_f_score, 2)]
}

decision_comparison_df = pd.DataFrame(decision_table)

pretty_table = PrettyTable()
pretty_table.field_names = decision_comparison_df.columns
for row in decision_comparison_df.itertuples(index=False):
    pretty_table.add_row(row)

# Display the pretty table
print("\nComparison of Decision Tree, Pre-Pruned Decision Tree and Post-Pruned Decision Tree:")
print(pretty_table)

# ================================ Logistic Regression =============================
print("\n================================ Logistic Regression ==============================")

# Create a logistic regression classifier
logistic_regression = LogisticRegression(max_iter=1000, random_state=5805)
logistic_regression.fit(X_train, y_train)

# Make predictions on the test set
y_pred_logistic_regression = logistic_regression.predict(X_test)

# Display the performance on the test set
logistic_regression_accuracy = accuracy_score(y_test, y_pred_logistic_regression)
print(f'Accuracy of the Logistic Regression Model: {logistic_regression_accuracy:.2f}')

# confusion matrix
logistic_confusion_matrix = confusion_matrix(y_test, y_pred_logistic_regression)
# precision
logistic_precision = precision_score(y_test, y_pred_logistic_regression)
# recall
logistic_recall = recall_score(y_test, y_pred_logistic_regression)
# specificity
logistic_specificity = logistic_confusion_matrix[0, 0] / (logistic_confusion_matrix[0, 0] +
                                                          logistic_confusion_matrix[0, 1])
# f-score
logistic_f_score = f1_score(y_test, y_pred_logistic_regression)
# AUC
logistic_auc = roc_auc_score(y_test, y_pred_logistic_regression)
# ROC
logistic_fpr, logistic_tpr, logistic_thresholds = roc_curve(y_test, y_pred_logistic_regression)
logistic_roc = pd.DataFrame({'FPR': logistic_fpr, 'TPR': logistic_tpr})

# Plot the AUC and ROC in one plot
plt.figure(figsize=(8, 6))
plt.plot(logistic_fpr, logistic_tpr, color='darkorange', lw=2,
         label='ROC curve (area = %0.2f)' % logistic_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for Logistic Regression')
plt.legend()
plt.grid(True)
plt.show()

# stratified k-fold cross validation
kfold_cross_validation(logistic_regression, X, y)

print("\n===================== Logistic Regression With Grid Search ====================")

# Define hyperparameters for grid search
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs', 'saga'],
}

# Perform grid search
grid_search = GridSearchCV(logistic_regression, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best parameters and the best classifier
best_params = grid_search.best_params_
best_logistic_regression = grid_search.best_estimator_

# Evaluate the performance of the logistic regression classifier on the test set
y_pred_best_log = best_logistic_regression.predict(X_test)
best_logistic_regression_accuracy = accuracy_score(y_test, y_pred_best_log)

print("\nBest Logistic Regression Classifier:")
print("Best Parameters:", best_params)
print(f"Accuracy:, {best_logistic_regression_accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred_best_log))

# confusion matrix
best_logistic_confusion_matrix = confusion_matrix(y_test, y_pred_best_log)
# precision
best_logistic_precision = precision_score(y_test, y_pred_best_log)
# recall
best_logistic_recall = recall_score(y_test, y_pred_best_log)
# specificity
best_logistic_specificity = best_logistic_confusion_matrix[0, 0] / (best_logistic_confusion_matrix[0, 0] +
                                                                    best_logistic_confusion_matrix[0, 1])
# f-score
best_logistic_f_score = f1_score(y_test, y_pred_best_log)
# AUC
best_logistic_auc = roc_auc_score(y_test, y_pred_best_log)
# ROC
best_logistic_fpr, best_logistic_tpr, best_logistic_thresholds = roc_curve(y_test, y_pred_best_log)
best_logistic_roc = pd.DataFrame({'FPR': best_logistic_fpr, 'TPR': best_logistic_tpr})

# Plot the AUC and ROC in one plot
plt.figure(figsize=(8, 6))
plt.plot(best_logistic_fpr, best_logistic_tpr, color='darkorange', lw=2,
         label='ROC curve (area = %0.2f)' % best_logistic_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for Logistic Regression With Grid Search')
plt.legend()
plt.grid(True)
plt.show()

# stratified k-fold cross validation
kfold_cross_validation(best_logistic_regression, X, y)

# ====================== Comparison Logistic =============================

logistic_table = {
    "Metric": ["Accuracy", "Confusion Matrix", "Recall", "AUC", "ROC", "Precision", "Specificity", "F-score"],
    "Logistic Regression": [round(logistic_regression_accuracy, 2), logistic_confusion_matrix,
                            round(logistic_recall, 2),
                            round(logistic_auc, 2), logistic_roc, round(logistic_precision, 2),
                            round(logistic_specificity, 2), round(logistic_f_score, 2)],
    "Logistic Regression With Grid Search": [round(best_logistic_regression_accuracy, 2),
                                             best_logistic_confusion_matrix,
                                             round(best_logistic_recall, 2),
                                             round(best_logistic_auc, 2), best_logistic_roc,
                                             round(best_logistic_precision, 2),
                                             round(best_logistic_specificity, 2), round(best_logistic_f_score, 2)],
}

logistic_comparison_df = pd.DataFrame(logistic_table)

pretty_table = PrettyTable()
pretty_table.field_names = logistic_comparison_df.columns
for row in logistic_comparison_df.itertuples(index=False):
    pretty_table.add_row(row)

# Display the pretty table
print("\nComparison of Logistic and Logistic Regression with Grid Search:")
print(pretty_table)

# ======================================= KNN ==================================================
print("\n================================ KNN =================================")

knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)
y_pred_knn = knn_classifier.predict(X_test)
knn_accuracy = accuracy_score(y_test, y_pred_knn)

print(f"Accuracy on Test Set: {knn_accuracy:.2f}")

# confusion matrix
knn_confusion_matrix = confusion_matrix(y_test, y_pred_knn)
# precision
knn_precision = precision_score(y_test, y_pred_knn)
# recall
knn_recall = recall_score(y_test, y_pred_knn)
# specificity
knn_specificity = knn_confusion_matrix[0, 0] / (knn_confusion_matrix[0, 0] +
                                                knn_confusion_matrix[0, 1])
# f-score
knn_f_score = f1_score(y_test, y_pred_knn)
# AUC
knn_auc = roc_auc_score(y_test, y_pred_knn)
# ROC
knn_fpr, knn_tpr, knn_thresholds = roc_curve(y_test, y_pred_knn)
knn_roc = pd.DataFrame({'FPR': knn_fpr, 'TPR': knn_tpr})

# Plot the AUC and ROC in one plot
plt.figure(figsize=(8, 6))
plt.plot(knn_fpr, knn_tpr, color='darkorange', lw=2,
         label='ROC curve (area = %0.2f)' % knn_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for KNN')
plt.legend()
plt.grid(True)
plt.show()

# stratified k-fold cross validation
kfold_cross_validation(knn_classifier, X, y)

print("\n======================== KNN with Grid Search =============================")

param_grid = {'n_neighbors': np.arange(1, 11)}

# Create a KNN classifier
knn_classifier = KNeighborsClassifier()

# Initialize GridSearchCV
grid_search = GridSearchCV(knn_classifier, param_grid, cv=5, scoring='accuracy')

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters and the best classifier
best_params = grid_search.best_params_
best_knn_classifier = grid_search.best_estimator_

# Make predictions on the test set using the best classifier
y_pred_best_knn = best_knn_classifier.predict(X_test)

# Evaluate the performance of the best classifier
best_knn_accuracy = accuracy_score(y_test, y_pred_best_knn)
print(f"Best K: {best_params['n_neighbors']}")
print(f"Accuracy on Test Set: {best_knn_accuracy:.2f}")

k_values = list(range(1, 11))
error_rate = []

for k in k_values:
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)
    y_pred_knn = knn_classifier.predict(X_test)
    error_rate.append(np.mean(y_pred_knn != y_test))

# Plot the accuracy scores against k values
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), error_rate, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.grid(True)
plt.show()

# confusion matrix
best_knn_confusion_matrix = confusion_matrix(y_test, y_pred_best_knn)
# precision
best_knn_precision = precision_score(y_test, y_pred_best_knn)
# recall
best_knn_recall = recall_score(y_test, y_pred_best_knn)
# specificity
best_knn_specificity = best_knn_confusion_matrix[0, 0] / (best_knn_confusion_matrix[0, 0] +
                                                          best_knn_confusion_matrix[0, 1])
# f-score
best_knn_f_score = f1_score(y_test, y_pred_best_knn)
# AUC
best_knn_auc = roc_auc_score(y_test, y_pred_best_knn)
# ROC
best_knn_fpr, best_knn_tpr, best_knn_thresholds = roc_curve(y_test, y_pred_best_knn)
best_knn_roc = pd.DataFrame({'FPR': best_knn_fpr, 'TPR': best_knn_tpr})

# Plot the AUC and ROC in one plot
plt.figure(figsize=(8, 6))
plt.plot(best_knn_fpr, best_knn_tpr, color='darkorange', lw=2,
         label='ROC curve (area = %0.2f)' % best_knn_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for KNN With Grid Search')
plt.legend()
plt.grid(True)
plt.show()

# stratified k-fold cross validation
kfold_cross_validation(best_knn_classifier, X, y)

# ====================== Comparison KNN =============================

knn_table = {
    "Metric": ["Accuracy", "Confusion Matrix", "Recall", "AUC", "ROC", "Precision", "Specificity", "F-score"],
    "KNN": [round(knn_accuracy, 2), knn_confusion_matrix,
            round(knn_recall, 2),
            round(knn_auc, 2), knn_roc, round(knn_precision, 2),
            round(knn_specificity, 2), round(knn_f_score, 2)],
    "KNN With Grid Search": [round(best_knn_accuracy, 2),
                             best_knn_confusion_matrix,
                             round(best_knn_recall, 2),
                             round(best_knn_auc, 2), best_knn_roc,
                             round(best_knn_precision, 2),
                             round(best_knn_specificity, 2), round(best_knn_f_score, 2)],
}

knn_comparison_df = pd.DataFrame(knn_table)

pretty_table = PrettyTable()
pretty_table.field_names = knn_comparison_df.columns
for row in knn_comparison_df.itertuples(index=False):
    pretty_table.add_row(row)

# Display the pretty table
print("\nComparison of KNN and KNN with Grid Search:")
print(pretty_table)

# =============================================== SVM =============================================
print("\n====================== SVM ======================")

# ===================== Linear SVM =================================
print("\n===================== Linear SVM =========================")

linear_svm = SVC(kernel='linear', C=1)
linear_svm.fit(X_train, y_train)
y_pred_linear_svm = linear_svm.predict(X_test)

linear_svm_accuracy = accuracy_score(y_test, y_pred_linear_svm)
print(f"Accuracy on Test Set: {linear_svm_accuracy:.2f}")

# confusion_matrix
linear_svm_confusion_matrix = confusion_matrix(y_test, y_pred_linear_svm)
# precision
linear_svm_precision = precision_score(y_test, y_pred_linear_svm)
# recall
linear_svm_recall = recall_score(y_test, y_pred_linear_svm)
# specificity
linear_svm_specificity = linear_svm_confusion_matrix[0, 0] / (linear_svm_confusion_matrix[0, 0] +
                                                              linear_svm_confusion_matrix[0, 1])
# f-score
linear_svm_f_score = f1_score(y_test, y_pred_linear_svm)
# AUC
linear_svm_auc = roc_auc_score(y_test, y_pred_linear_svm)
# ROC
linear_svm_fpr, linear_svm_tpr, linear_svm_thresholds = roc_curve(y_test, y_pred_linear_svm)
linear_svm_roc = pd.DataFrame({'FPR': linear_svm_fpr, 'TPR': linear_svm_tpr})

# Plot the AUC and ROC in one plot
plt.figure(figsize=(8, 6))
plt.plot(linear_svm_fpr, linear_svm_tpr, color='darkorange', lw=2,
         label='ROC curve (area = %0.2f)' % linear_svm_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for Linear SVM')
plt.legend()
plt.grid(True)
plt.show()

# stratified k-fold cross validation
kfold_cross_validation(linear_svm, X, y)

print("\n===================== Linear SVM With Grid Search =========================")

# SVM with Linear Kernel
linear_svm = SVC()
linear_params = {'C': [0.1, 1, 2], 'gamma': [1, 0.001], 'kernel': ['linear', 'poly', 'rbf']}
linear_grid = GridSearchCV(linear_svm, linear_params, cv=5)
linear_grid.fit(X_train, y_train)
best_params = linear_grid.best_params_
best_linear_svm = linear_grid.best_estimator_

print("Best Parameters:", best_params)

y_pred_best_linear_svm = best_linear_svm.predict(X_test)

best_linear_accuracy = accuracy_score(y_test, y_pred_best_linear_svm)
print(f"Accuracy on Test Set: {best_linear_accuracy:.2f}")

# confusion_matrix
best_linear_confusion_matrix = confusion_matrix(y_test, y_pred_best_linear_svm)
# precision
best_linear_precision = precision_score(y_test, y_pred_best_linear_svm)
# recall
best_linear_recall = recall_score(y_test, y_pred_best_linear_svm)
# specificity
best_linear_specificity = best_linear_confusion_matrix[0, 0] / (best_linear_confusion_matrix[0, 0] +
                                                                best_linear_confusion_matrix[0, 1])
# f-score
best_linear_f_score = f1_score(y_test, y_pred_best_linear_svm)
# AUC
best_linear_auc = roc_auc_score(y_test, y_pred_best_linear_svm)
# ROC
best_linear_fpr, best_linear_tpr, best_linear_thresholds = roc_curve(y_test, y_pred_best_linear_svm)
best_linear_roc = pd.DataFrame({'FPR': best_linear_fpr, 'TPR': best_linear_tpr})

# Plot the AUC and ROC in one plot
plt.figure(figsize=(8, 6))
plt.plot(best_linear_fpr, best_linear_tpr, color='darkorange', lw=2,
         label='ROC curve (area = %0.2f)' % best_linear_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for SVM With Linear Kernel')
plt.legend()
plt.grid(True)
plt.show()

# stratified k-fold cross validation
kfold_cross_validation(best_linear_svm, X, y)

# ====================== Comparison SVM =============================
svm_table = {
    "Metric": ["Accuracy", "Confusion Matrix", "Recall", "AUC", "ROC", "Precision", "Specificity", "F-score"],
    "Linear SVM": [round(linear_svm_accuracy, 2), linear_svm_confusion_matrix,
                   round(linear_svm_recall, 2),
                   round(linear_svm_auc, 2), linear_svm_roc, round(linear_svm_precision, 2),
                   round(linear_svm_specificity, 2), round(linear_svm_f_score, 2)],
    "Linear SVM with Grid Search": [round(best_linear_accuracy, 2), best_linear_confusion_matrix,
                                    round(best_linear_recall, 2),
                                    round(best_linear_auc, 2), best_linear_roc, round(best_linear_precision, 2),
                                    round(best_linear_specificity, 2), round(best_linear_f_score, 2)]}

svm_comparison_df = pd.DataFrame(svm_table)

pretty_table = PrettyTable()
pretty_table.field_names = svm_comparison_df.columns
for row in svm_comparison_df.itertuples(index=False):
    pretty_table.add_row(row)

# Display the pretty table
print("\nComparison of Linear SVM, Linear SVM with GridSearch, Polynomial SVM, Polynomial SVM with GridSearch, "
      "RBF SVM and RBF SVM with GridSearch:")
print(pretty_table)

# ======================================= Naive Bayes ==================================================
print("\n================================ Naive Bayes =================================")

# ===================== Gaussian Naive Bayes =================================
print("\n===================== Gaussian Naive Bayes =========================")

gaussian_nb = GaussianNB()
gaussian_nb.fit(X_train, y_train)
y_pred_gaussian_nb = gaussian_nb.predict(X_test)

gaussian_nb_accuracy = accuracy_score(y_test, y_pred_gaussian_nb)
print(f"Accuracy on Test Set: {gaussian_nb_accuracy:.2f}")

# confusion_matrix
gaussian_nb_confusion_matrix = confusion_matrix(y_test, y_pred_gaussian_nb)
# precision
gaussian_nb_precision = precision_score(y_test, y_pred_gaussian_nb)
# recall
gaussian_nb_recall = recall_score(y_test, y_pred_gaussian_nb)
# specificity
gaussian_nb_specificity = gaussian_nb_confusion_matrix[0, 0] / (gaussian_nb_confusion_matrix[0, 0] +
                                                                gaussian_nb_confusion_matrix[0, 1])
# f-score
gaussian_nb_f_score = f1_score(y_test, y_pred_gaussian_nb)
# AUC
gaussian_nb_auc = roc_auc_score(y_test, y_pred_gaussian_nb)
# ROC
gaussian_nb_fpr, gaussian_nb_tpr, gaussian_nb_thresholds = roc_curve(y_test, y_pred_gaussian_nb)
gaussian_nb_roc = pd.DataFrame({'FPR': gaussian_nb_fpr, 'TPR': gaussian_nb_tpr})

# Plot the AUC and ROC in one plot
plt.figure(figsize=(8, 6))
plt.plot(gaussian_nb_fpr, gaussian_nb_tpr, color='darkorange', lw=2,
         label='ROC curve (area = %0.2f)' % gaussian_nb_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for Gaussian Naive Bayes')
plt.legend()
plt.grid(True)
plt.show()

# stratified k-fold cross validation
kfold_cross_validation(gaussian_nb, X, y)

# ====================== Gaussian Naive Bayes With Grid Search =============================
print("\n===================== Gaussian Naive Bayes With Grid Search =========================")

gaussian_nb = GaussianNB()
gaussian_params = {'var_smoothing': np.logspace(0, -9, num=100)}
gaussian_grid = GridSearchCV(gaussian_nb, gaussian_params, cv=5)
gaussian_grid.fit(X_train, y_train)
best_params = gaussian_grid.best_params_
best_gaussian_nb = gaussian_grid.best_estimator_

print("Best Parameters:", best_params)

y_pred_best_gaussian_nb = best_gaussian_nb.predict(X_test)

best_gaussian_accuracy = accuracy_score(y_test, y_pred_best_gaussian_nb)
print(f"Accuracy on Test Set: {best_gaussian_accuracy:.2f}")

# confusion_matrix
best_gaussian_confusion_matrix = confusion_matrix(y_test, y_pred_best_gaussian_nb)
# precision
best_gaussian_precision = precision_score(y_test, y_pred_best_gaussian_nb)
# recall
best_gaussian_recall = recall_score(y_test, y_pred_best_gaussian_nb)
# specificity
best_gaussian_specificity = best_gaussian_confusion_matrix[0, 0] / (best_gaussian_confusion_matrix[0, 0] +
                                                                    best_gaussian_confusion_matrix[0, 1])
# f-score
best_gaussian_f_score = f1_score(y_test, y_pred_best_gaussian_nb)
# AUC
best_gaussian_auc = roc_auc_score(y_test, y_pred_best_gaussian_nb)
# ROC
best_gaussian_fpr, best_gaussian_tpr, best_gaussian_thresholds = roc_curve(y_test, y_pred_best_gaussian_nb)
best_gaussian_roc = pd.DataFrame({'FPR': best_gaussian_fpr, 'TPR': best_gaussian_tpr})

# Plot the AUC and ROC in one plot
plt.figure(figsize=(8, 6))
plt.plot(best_gaussian_fpr, best_gaussian_tpr, color='darkorange', lw=2,
         label='ROC curve (area = %0.2f)' % best_gaussian_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve Gaussian Naive Bayes With Grid Search')
plt.legend()
plt.grid(True)
plt.show()

# stratified k-fold cross validation
kfold_cross_validation(best_gaussian_nb, X, y)

# ====================== Comparison Naive Bayes =============================
nb_table = {
    "Metric": ["Accuracy", "Confusion Matrix", "Recall", "AUC", "ROC", "Precision", "Specificity", "F-score"],
    "Gaussian Naive Bayes": [round(gaussian_nb_accuracy, 2), gaussian_nb_confusion_matrix,
                             round(gaussian_nb_recall, 2),
                             round(gaussian_nb_auc, 2), gaussian_nb_roc, round(gaussian_nb_precision, 2),
                             round(gaussian_nb_specificity, 2), round(gaussian_nb_f_score, 2)],
    "Gaussian Naive Bayes with Grid Search": [round(best_gaussian_accuracy, 2), best_gaussian_confusion_matrix,
                                              round(best_gaussian_recall, 2),
                                              round(best_gaussian_auc, 2), best_gaussian_roc,
                                              round(best_gaussian_precision, 2),
                                              round(best_gaussian_specificity, 2), round(best_gaussian_f_score, 2)]}

nb_comparison_df = pd.DataFrame(nb_table)

pretty_table = PrettyTable()
pretty_table.field_names = nb_comparison_df.columns
for row in nb_comparison_df.itertuples(index=False):
    pretty_table.add_row(row)

# Display the pretty table
print("\nComparison of Gaussian Naive Bayes and Gaussian Naive Bayes with Grid Search:")
print(pretty_table)

# ======================================= Random Forest ==================================================
print("\n================================ Random Forest =================================")

random_forest = RandomForestClassifier(n_estimators=100, random_state=5805)
random_forest.fit(X_train, y_train)
y_pred_random_forest = random_forest.predict(X_test)

random_forest_accuracy = accuracy_score(y_test, y_pred_random_forest)
print(f"Accuracy on Test Set: {random_forest_accuracy:.2f}")

# confusion_matrix
random_forest_confusion_matrix = confusion_matrix(y_test, y_pred_random_forest)
# precision
random_forest_precision = precision_score(y_test, y_pred_random_forest)
# recall
random_forest_recall = recall_score(y_test, y_pred_random_forest)
# specificity
random_forest_specificity = random_forest_confusion_matrix[0, 0] / (random_forest_confusion_matrix[0, 0] +
                                                                    random_forest_confusion_matrix[0, 1])
# f-score
random_forest_f_score = f1_score(y_test, y_pred_random_forest)
# AUC
random_forest_auc = roc_auc_score(y_test, y_pred_random_forest)
# ROC
random_forest_fpr, random_forest_tpr, random_forest_thresholds = roc_curve(y_test, y_pred_random_forest)
random_forest_roc = pd.DataFrame({'FPR': random_forest_fpr, 'TPR': random_forest_tpr})

# Plot the AUC and ROC in one plot
plt.figure(figsize=(8, 6))
plt.plot(random_forest_fpr, random_forest_tpr, color='darkorange', lw=2,
         label='ROC curve (area = %0.2f)' % random_forest_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve Random Forest')
plt.legend()
plt.grid(True)
plt.show()

# stratified k-fold cross validation
kfold_cross_validation(random_forest, X, y)

# ====================== Random Forest With Grid Search =============================
print("\n===================== Random Forest With Grid Search =========================")

random_forest = RandomForestClassifier(random_state=5805)
random_params = {'n_estimators': [50, 100], 'max_depth': [2, 4, 8]}
random_grid = GridSearchCV(random_forest, random_params, cv=5)
random_grid.fit(X_train, y_train)
best_params = random_grid.best_params_
best_random_forest = random_grid.best_estimator_

print("Best Parameters:", best_params)

y_pred_best_random_forest = best_random_forest.predict(X_test)

best_random_forest_accuracy = accuracy_score(y_test, y_pred_best_random_forest)
print(f"Accuracy on Test Set: {best_random_forest_accuracy:.2f}")

# confusion_matrix
best_random_forest_confusion_matrix = confusion_matrix(y_test, y_pred_best_random_forest)
# precision
best_random_forest_precision = precision_score(y_test, y_pred_best_random_forest)
# recall
best_random_forest_recall = recall_score(y_test, y_pred_best_random_forest)
# specificity
best_random_forest_specificity = best_random_forest_confusion_matrix[0, 0] / (
        best_random_forest_confusion_matrix[0, 0] +
        best_random_forest_confusion_matrix[0, 1])
# f-score
best_random_forest_f_score = f1_score(y_test, y_pred_best_random_forest)
# AUC
best_random_forest_auc = roc_auc_score(y_test, y_pred_best_random_forest)
# ROC
best_random_forest_fpr, best_random_forest_tpr, best_random_forest_thresholds = roc_curve(y_test,
                                                                                          y_pred_best_random_forest)
best_random_forest_roc = pd.DataFrame({'FPR': best_random_forest_fpr, 'TPR': best_random_forest_tpr})

# Plot the AUC and ROC in one plot
plt.figure(figsize=(8, 6))
plt.plot(best_random_forest_fpr, best_random_forest_tpr, color='darkorange', lw=2,
         label='ROC curve (area = %0.2f)' % best_random_forest_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve Random Forest With Grid Search')
plt.legend()
plt.grid(True)
plt.show()

# stratified k-fold cross validation
kfold_cross_validation(best_random_forest, X, y)

# ====================== Bagging Classifier =============================
print("\n===================== Bagging Classifier =========================")

bagging_classifier = BaggingClassifier(n_estimators=100, random_state=5805)
bagging_classifier.fit(X_train, y_train)
y_pred_bagging_classifier = bagging_classifier.predict(X_test)

bagging_classifier_accuracy = accuracy_score(y_test, y_pred_bagging_classifier)
print(f"Accuracy on Test Set: {bagging_classifier_accuracy:.2f}")

# confusion_matrix
bagging_classifier_confusion_matrix = confusion_matrix(y_test, y_pred_bagging_classifier)
# precision
bagging_classifier_precision = precision_score(y_test, y_pred_bagging_classifier)
# recall
bagging_classifier_recall = recall_score(y_test, y_pred_bagging_classifier)
# specificity
bagging_classifier_specificity = bagging_classifier_confusion_matrix[0, 0] / (
        bagging_classifier_confusion_matrix[0, 0] +
        bagging_classifier_confusion_matrix[0, 1])
# f-score
bagging_classifier_f_score = f1_score(y_test, y_pred_bagging_classifier)
# AUC
bagging_classifier_auc = roc_auc_score(y_test, y_pred_bagging_classifier)
# ROC
bagging_classifier_fpr, bagging_classifier_tpr, bagging_classifier_thresholds = roc_curve(y_test,
                                                                                          y_pred_bagging_classifier)
bagging_classifier_roc = pd.DataFrame({'FPR': bagging_classifier_fpr, 'TPR': bagging_classifier_tpr})

# Plot the AUC and ROC in one plot
plt.figure(figsize=(8, 6))
plt.plot(bagging_classifier_fpr, bagging_classifier_tpr, color='darkorange', lw=2,
         label='ROC curve (area = %0.2f)' % bagging_classifier_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve Bagging Classifier')
plt.legend()
plt.grid(True)
plt.show()

# stratified k-fold cross validation
kfold_cross_validation(bagging_classifier, X, y)

# ======================= Boosting Classifier =============================
print("\n===================== Boosting Classifier =========================")

boosting_classifier = AdaBoostClassifier(n_estimators=100, random_state=5805)
boosting_classifier.fit(X_train, y_train)
y_pred_boosting_classifier = boosting_classifier.predict(X_test)

boosting_classifier_accuracy = accuracy_score(y_test, y_pred_boosting_classifier)
print(f"Accuracy on Test Set: {boosting_classifier_accuracy:.2f}")

# confusion_matrix
boosting_classifier_confusion_matrix = confusion_matrix(y_test, y_pred_boosting_classifier)
# precision
boosting_classifier_precision = precision_score(y_test, y_pred_boosting_classifier)
# recall
boosting_classifier_recall = recall_score(y_test, y_pred_boosting_classifier)
# specificity
boosting_classifier_specificity = boosting_classifier_confusion_matrix[0, 0] / (
        boosting_classifier_confusion_matrix[0, 0] +
        boosting_classifier_confusion_matrix[0, 1])
# f-score
boosting_classifier_f_score = f1_score(y_test, y_pred_boosting_classifier)
# AUC
boosting_classifier_auc = roc_auc_score(y_test, y_pred_boosting_classifier)
# ROC
boosting_classifier_fpr, boosting_classifier_tpr, boosting_classifier_thresholds \
    = roc_curve(y_test, y_pred_boosting_classifier)

boosting_classifier_roc = pd.DataFrame({'FPR': boosting_classifier_fpr, 'TPR': boosting_classifier_tpr})

# Plot the AUC and ROC in one plot
plt.figure(figsize=(8, 6))
plt.plot(boosting_classifier_fpr, boosting_classifier_tpr, color='darkorange', lw=2,
         label='ROC curve (area = %0.2f)' % boosting_classifier_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve Boosting Classifier')
plt.legend()
plt.grid(True)
plt.show()

# stratified k-fold cross validation
kfold_cross_validation(boosting_classifier, X, y)

# ========================== Stack Classifier =============================
print("\n===================== Stack Classifier =========================")

stack_classifier = StackingClassifier(estimators=[('brf', best_random_forest), ('rf', random_forest)],
                                      final_estimator=RandomForestClassifier(n_estimators=100, random_state=5805))
stack_classifier.fit(X_train, y_train)
y_pred_stack_classifier = stack_classifier.predict(X_test)

stack_classifier_accuracy = accuracy_score(y_test, y_pred_stack_classifier)
print(f"Accuracy on Test Set: {stack_classifier_accuracy:.2f}")

# confusion_matrix
stack_classifier_confusion_matrix = confusion_matrix(y_test, y_pred_stack_classifier)
# precision
stack_classifier_precision = precision_score(y_test, y_pred_stack_classifier)
# recall
stack_classifier_recall = recall_score(y_test, y_pred_stack_classifier)
# specificity
stack_classifier_specificity = stack_classifier_confusion_matrix[0, 0] / (stack_classifier_confusion_matrix[0, 0] +
                                                                          stack_classifier_confusion_matrix[0, 1])
# f-score
stack_classifier_f_score = f1_score(y_test, y_pred_stack_classifier)
# AUC
stack_classifier_auc = roc_auc_score(y_test, y_pred_stack_classifier)
# ROC
stack_classifier_fpr, stack_classifier_tpr, stack_classifier_thresholds \
    = roc_curve(y_test, y_pred_stack_classifier)
stack_classifier_roc = pd.DataFrame({'FPR': stack_classifier_fpr, 'TPR': stack_classifier_tpr})

# Plot the AUC and ROC in one plot
plt.figure(figsize=(8, 6))
plt.plot(stack_classifier_fpr, stack_classifier_tpr, color='darkorange', lw=2,
         label='ROC curve (area = %0.2f)' % stack_classifier_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve Stack Classifier')
plt.legend()
plt.grid(True)
plt.show()

# stratified k-fold cross validation
kfold_cross_validation(stack_classifier, X, y)

# ====================== Comparison Random Forest, Random Forest With GridSearch, Bagging Classifier,
# Boosting Classifier and Stack Classifier =============================

ensemble_table = {'Metric': ['Accuracy', 'Confusion Matrix', 'Recall', 'AUC', 'ROC', 'Precision', 'Specificity',
                             'F-score'],
                  'Random Forest': [round(random_forest_accuracy, 2), random_forest_confusion_matrix,
                                    round(random_forest_recall, 2),
                                    round(random_forest_auc, 2), random_forest_roc, round(random_forest_precision, 2),
                                    round(random_forest_specificity, 2), round(random_forest_f_score, 2)],
                  'Random Forest with Grid Search': [round(best_random_forest_accuracy, 2),
                                                     best_random_forest_confusion_matrix,
                                                     round(best_random_forest_recall, 2),
                                                     round(best_random_forest_auc, 2), best_random_forest_roc,
                                                     round(best_random_forest_precision, 2),
                                                     round(best_random_forest_specificity, 2),
                                                     round(best_random_forest_f_score, 2)],
                  'Bagging Classifier': [round(bagging_classifier_accuracy, 2), bagging_classifier_confusion_matrix,
                                         round(bagging_classifier_recall, 2),
                                         round(bagging_classifier_auc, 2), bagging_classifier_roc,
                                         round(bagging_classifier_precision, 2),
                                         round(bagging_classifier_specificity, 2),
                                         round(bagging_classifier_f_score, 2)],
                  'Boosting Classifier': [round(boosting_classifier_accuracy, 2), boosting_classifier_confusion_matrix,
                                          round(boosting_classifier_recall, 2),
                                          round(boosting_classifier_auc, 2), boosting_classifier_roc,
                                          round(boosting_classifier_precision, 2),
                                          round(boosting_classifier_specificity, 2),
                                          round(boosting_classifier_f_score, 2)],
                  'Stack Classifier': [round(stack_classifier_accuracy, 2), stack_classifier_confusion_matrix,
                                       round(stack_classifier_recall, 2),
                                       round(stack_classifier_auc, 2), stack_classifier_roc,
                                       round(stack_classifier_precision, 2),
                                       round(stack_classifier_specificity, 2),
                                       round(stack_classifier_f_score, 2)]}

ensemble_comparison_df = pd.DataFrame(ensemble_table)

pretty_table = PrettyTable()
pretty_table.field_names = ensemble_comparison_df.columns
for row in ensemble_comparison_df.itertuples(index=False):
    pretty_table.add_row(row)

# Display the pretty table
print("\nComparison of Random Forest, Random Forest with Grid Search, Bagging Classifier, Boosting Classifier and "
      "Stack Classifier:")
print(pretty_table)

# ======================================= Multi-layered Perceptron ==================================================
print("\n================================ Multi-layered Perceptron =================================")

mlp = MLPClassifier(hidden_layer_sizes=(10, 10), random_state=5805)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)

mlp_accuracy = accuracy_score(y_test, y_pred_mlp)
print(f"Accuracy on Test Set: {mlp_accuracy:.2f}")

# confusion_matrix
mlp_confusion_matrix = confusion_matrix(y_test, y_pred_mlp)
# precision
mlp_precision = precision_score(y_test, y_pred_mlp)
# recall
mlp_recall = recall_score(y_test, y_pred_mlp)
# specificity
mlp_specificity = mlp_confusion_matrix[0, 0] / (mlp_confusion_matrix[0, 0] +
                                                mlp_confusion_matrix[0, 1])
# f-score
mlp_f_score = f1_score(y_test, y_pred_mlp)
# AUC
mlp_auc = roc_auc_score(y_test, y_pred_mlp)
# ROC
mlp_fpr, mlp_tpr, mlp_thresholds = roc_curve(y_test, y_pred_mlp)
mlp_roc = pd.DataFrame({'FPR': mlp_fpr, 'TPR': mlp_tpr})

# Plot the AUC and ROC in one plot
plt.figure(figsize=(8, 6))
plt.plot(mlp_fpr, mlp_tpr, color='darkorange', lw=2,
         label='ROC curve (area = %0.2f)' % mlp_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve Multi-layered Perceptron')
plt.legend()
plt.grid(True)
plt.show()

# stratified k-fold cross validation
kfold_cross_validation(mlp, X, y)

# ====================== Multi-layered Perceptron With Grid Search =============================
print("\n===================== Multi-layered Perceptron With Grid Search =========================")

mlp = MLPClassifier(random_state=5805)

mlp_params = {'hidden_layer_sizes': [(10, 10)],
              'activation': ['relu', 'tanh', 'logistic'],
              'solver': ['sgd', 'adam'],
              'learning_rate': ['constant', 'adaptive']}

mlp_grid = GridSearchCV(mlp, mlp_params, cv=5)
mlp_grid.fit(X_train, y_train)
best_params = mlp_grid.best_params_
best_mlp = mlp_grid.best_estimator_

print("Best Parameters:", best_params)

y_pred_best_mlp = best_mlp.predict(X_test)

best_mlp_accuracy = accuracy_score(y_test, y_pred_best_mlp)
print(f"Accuracy on Test Set: {best_mlp_accuracy:.2f}")

# confusion_matrix
best_mlp_confusion_matrix = confusion_matrix(y_test, y_pred_best_mlp)
# precision
best_mlp_precision = precision_score(y_test, y_pred_best_mlp)
# recall
best_mlp_recall = recall_score(y_test, y_pred_best_mlp)
# specificity
best_mlp_specificity = best_mlp_confusion_matrix[0, 0] / (best_mlp_confusion_matrix[0, 0] +
                                                          best_mlp_confusion_matrix[0, 1])
# f-score
best_mlp_f_score = f1_score(y_test, y_pred_best_mlp)
# AUC
best_mlp_auc = roc_auc_score(y_test, y_pred_best_mlp)
# ROC
best_mlp_fpr, best_mlp_tpr, best_mlp_thresholds = roc_curve(y_test, y_pred_best_mlp)
best_mlp_roc = pd.DataFrame({'FPR': best_mlp_fpr, 'TPR': best_mlp_tpr})

# Plot the AUC and ROC in one plot
plt.figure(figsize=(8, 6))
plt.plot(best_mlp_fpr, best_mlp_tpr, color='darkorange', lw=2,
         label='ROC curve (area = %0.2f)' % best_mlp_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve Multi-layered Perceptron With Grid Search')
plt.legend()
plt.grid(True)
plt.show()

# stratified k-fold cross validation
kfold_cross_validation(best_mlp, X, y)

# ====================== Comparison Multi-layered Perceptron =============================

mlp_table = {'Metric': ['Accuracy', 'Confusion Matrix', 'Recall', 'AUC', 'ROC', 'Precision', 'Specificity',
                        'F-score'],
             'Multi-layered Perceptron': [round(mlp_accuracy, 2), mlp_confusion_matrix,
                                          round(mlp_recall, 2),
                                          round(mlp_auc, 2), mlp_roc, round(mlp_precision, 2),
                                          round(mlp_specificity, 2),
                                          round(mlp_f_score, 2)],
             'Multi-layered Perceptron with Grid Search': [round(best_mlp_accuracy, 2),
                                                           best_mlp_confusion_matrix,
                                                           round(best_mlp_recall, 2),
                                                           round(best_mlp_auc, 2), best_mlp_roc,
                                                           round(best_mlp_precision, 2),
                                                           round(best_mlp_specificity, 2),
                                                           round(best_mlp_f_score, 2)]}

mlp_comparison_df = pd.DataFrame(mlp_table)

pretty_table = PrettyTable()
pretty_table.field_names = mlp_comparison_df.columns
for row in mlp_comparison_df.itertuples(index=False):
    pretty_table.add_row(row)

# Display the pretty table
print("\nComparison of Multi-layered Perceptron and Multi-layered Perceptron with Grid Search:")
print(pretty_table)

# ==================== Comparison of All Models ==============================

# # Combine multiple tables into one
# combined_table = pd.concat(
#     [decision_comparison_df, logistic_comparison_df.iloc[:, 1:], knn_comparison_df.iloc[:, 1:],
#      svm_comparison_df.iloc[:, 1:],
#      nb_comparison_df.iloc[:, 1:], ensemble_comparison_df.iloc[:, 1:], mlp_comparison_df], axis=1)
#
#
# pretty_table_combined = PrettyTable()
# pretty_table_combined.title = "Comparison of All Classification Models"
# pretty_table_combined.field_names = combined_table.columns
# for row in combined_table.itertuples(index=False):
#     pretty_table_combined.add_row(row)
#
# # Display the pretty table
# print(pretty_table_combined)

# ===============================================================================================================
# ======================================= Phase IV: Clustering and Association ==================================
# ===============================================================================================================

# ==================================== KMean ======================================
print("\n================================ KMean =================================")

scatter_matrix(stock_df_encoded, alpha=0.8, figsize=(20, 20), diagonal='hist',
               marker='o', edgecolors='b', facecolor='yellow', s=50)

plt.suptitle('Scatter Plot Matrix of Features', y=0.95)
plt.show()


def calculate_WSS(points, kmax):
    sse = []
    for k in range(1, kmax + 1):
        kmeans = KMeans(n_clusters=k).fit(points)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0
        # calculate square of Euclidean distance of each point
        # from its cluster center and add to current WSS

        for i in range(len(stock_df_encoded)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2
        sse.append(curr_sse)
    return sse


k = 10
sse = calculate_WSS(stock_df_encoded.values, k)
plt.figure()
plt.plot(np.arange(1, k + 1, 1), np.array(sse), 'bx-')
plt.xticks(np.arange(1, k + 1, 1))
plt.grid()
plt.xlabel('k')
plt.ylabel('WSS')
plt.title('k selection in k-mean Elbow Algorithm')
plt.show()

# ========================================================
# Silhouette Method for selection of K
# ========================================================
sil = []
kmax = 15

for k in range(2, kmax + 1):
    kmeans = KMeans(n_clusters=k).fit(X)
    labels = kmeans.labels_
    sil.append(silhouette_score(X, labels, metric='euclidean'))
plt.figure()
plt.plot(np.arange(2, k + 1, 1), sil, 'bx-')
plt.xticks(np.arange(2, k + 1, 1))
plt.grid()
plt.xlabel('k')
plt.ylabel('Silhouette score')
plt.title('Silhouette Method')
plt.show()

# ==================================== Apriori algorithm ======================================
print("\n================================ Apriori algorithm =================================\n")

stock_df_encoded_binary = (stock_df_encoded > 0).astype(int)

apriori_df = apriori(stock_df_encoded_binary, min_support=0.2, use_colnames=True, verbose=1)
print("\nApriori Dataframe:")
print(apriori_df)
df_ar = association_rules(apriori_df, metric='confidence', min_threshold=0.6)
df_ar = df_ar.sort_values(['confidence', 'lift'], ascending=[False, False])
print("\nAssociation Rules:")
print(df_ar.to_string())
