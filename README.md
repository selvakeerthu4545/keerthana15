
# CROP YIELD PREDICTION USING MACHINE LEARNING

Predicting crop yields can be a complex task that involves various factors like weather conditions, soil quality, and more. While a simple linear regression, as shown in the previous response, is a basic approach, more advanced machine learning techniques can yield better results. 

This code uses scikit-learn's LinearRegression model to predict crop yields, splitting the data into training and testing sets for evaluation. It also calculates the Mean Squared Error to assess the model's performance and includes a simple plot to visualize the regression line. You can modify the dataset and extend this code for more complex prediction scenarios.



1. Import necessary libraries:
   - numpy (as np) for numerical operations.
   - matplotlib.pyplot (as plt) for data visualization.
   - LinearRegression from sklearn.linear_model for creating a linear regression model.
   - train_test_split from sklearn.model_selection for splitting the data into training and testing sets.
   - mean_squared_error from sklearn.metrics for calculating the Mean Squared Error.

2. Create sample data for training:
   - years represents the years (in this case, from 2010 to 2015) as input features. It's reshaped to a column vector using .reshape(-1, 1).
   - crop_yield represents the corresponding crop yield values.

3. Split the data into training and testing sets:
   - train_test_split is used to randomly split the data into training and testing subsets. In this case, 80% of the data is used for training, and 20% for testing, with a fixed random seed for reproducibility (random_state=42).

4. Create and train a linear regression model:
   - LinearRegression is used to create a linear regression model instance.
   - model.fit(X_train, y_train) trains the model using the training data.

5. Make predictions on the test set:
   - model.predict(X_test) predicts crop yields for the test set.

6. Calculate the Mean Squared Error (MSE) to evaluate the model:
   - mean_squared_error(y_test, y_pred) computes the MSE between the actual test set values (y_test) and the predicted values (y_pred).

7. Predict crop yield for a given year:
   - A single year, 2016, is selected for prediction and reshaped into a column vector.
   - model.predict(year_to_predict) predicts the crop yield for the given year.

8. Plot the training data and regression line:
   - Scatter plot: plt.scatter(years, crop_yield, color='blue') displays the training data points in blue.
   - Regression line: plt.plot(years, model.predict(years), color='red') plots the linear regression line in red.
   - Labels and title: plt.xlabel, plt.ylabel, and plt.title set labels and the title for the plot.
   - plt.show() displays the plot.



