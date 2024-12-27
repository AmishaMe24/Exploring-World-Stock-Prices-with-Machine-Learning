# Exploring World Stock Prices with Machine Learning

## Project Overview
The *Exploring World Stock Prices with Machine Learning* project aims to predict and classify stock price movements using historical data. This project leverages advanced machine learning techniques and statistical analysis to provide actionable insights into global stock price trends.

## Objectives
- Develop predictive models to forecast stock prices.
- Analyze and visualize trends and patterns in historical stock data.
- Perform feature engineering and dimensionality reduction to optimize model performance.
- Compare and evaluate multiple machine learning models for classification and regression tasks.

## Key Features
- **Data Preprocessing:** Handled missing values, normalized data, and removed outliers using statistical methods.
- **Feature Engineering:** Created features like daily price volatility, rate of change, and rolling statistics to capture temporal patterns.
- **Dimensionality Reduction:** Applied PCA and SVD to reduce multicollinearity and improve computational efficiency.
- **Model Development:** Implemented and optimized models such as Decision Trees, Random Forest, SVM, KNN, Naive Bayes, and MLP.
- **Evaluation Metrics:** Used AUC, R-squared, MSE, and accuracy to assess model performance.
- **Clustering and Association Analysis:** Applied K-Means clustering and Apriori algorithm to uncover patterns in stock movements.

## Dataset
- **Source:** Kaggle's *World Stock Prices (Daily Updated)* dataset
- **Features:**
  - Date, Open, High, Low, Close, Volume, Dividends, Stock Splits, Brand Name, Ticker, Industry, and Country.
  - Derived features include daily volatility, price differences, and rolling statistics.

## Tools and Technologies
- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Plotly
- **Techniques:** Feature Engineering, Dimensionality Reduction (PCA, SVD), Hyperparameter Tuning (Grid Search)
- **Deployment:** Jupyter Notebook

## Methodology
1. **Data Preprocessing:**
   - Cleaned and prepared data by handling missing values and removing outliers.
   - Created additional features to capture stock trends and patterns.

2. **Exploratory Data Analysis (EDA):**
   - Visualized relationships and distributions using heatmaps, scatter plots, and histograms.

3. **Model Training:**
   - Regression: Predicted stock closing prices using models like Random Forest and Linear Regression. Also did OLS regression analysis.
   - Classification: Categorized stock price movements using Decision Trees, SVM, KNN, and MLP.

4. **Evaluation:**
   - Compared models using AUC, accuracy, and error metrics.

5. **Clustering and Association:**
   - Applied K-Means clustering to group stocks based on performance.
   - Used Apriori algorithm for association rule mining.

## Results
- Achieved **98% accuracy** using Multi-layer Perceptron (MLP) for classification tasks.
- PCA and SVD effectively reduced dimensionality while retaining over 90% variance.
- Discovered actionable insights into stock trends and performance using clustering and association rules.

## Visualizations
- Feature importance plots to highlight key predictors.
- Correlation heatmaps to identify relationships between variables.
- ROC curves for evaluating classification model performance.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/AmishaMe24/Exploring-World-Stock-Prices-with-Machine-Learning.git
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

## Usage
1. Load the dataset by following the notebook instructions.
2. Execute cells step-by-step for preprocessing, visualization, and modeling.
3. Customize parameters for feature engineering and model optimization.

## Future Improvements
- Incorporate additional datasets to enhance prediction robustness.
- Explore deep learning techniques for more accurate stock price forecasting.
- Deploy the solution as an interactive web application.

## Contributions
Contributions are welcome! Please fork the repository and submit a pull request with your suggestions or improvements.

## License
This project is licensed under the MIT License.

## Acknowledgments
- Thanks to Kaggle for providing the dataset.
- Special thanks to Dr. Reza Jafari for guidance and support throughout the project.

## Contact
For any queries, please reach out to:
- **Author:** Amisha Mehta
