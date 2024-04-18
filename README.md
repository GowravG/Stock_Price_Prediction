**Stock Price Prediction using LSTM** 

This project develops a machine learning model to predict stock price movements (up or down) based on historical price data using a Long Short-Term Memory (LSTM) network. The model is trained to understand patterns in stock prices and make daily predictions for the coming week.

**Features**

Predicts whether the stock price will be up or down the next day.
Utilizes a sequential model for time series data to capture trends and patterns over time.
The model outputs predictions for the next week and compares them to actual movements.

**Libraries Used**

pandas: For data manipulation and analysis.
numpy: For numerical operations.
sklearn.preprocessing.MinMaxScaler: For normalizing features.
sklearn.model_selection.train_test_split: For splitting the data into training and test sets.
tensorflow.keras: For building and training the LSTM model.

**Data**

The dataset used in this project is based on the SPY ETF, which tracks the S&P 500 index. The relevant features include Open, High, Low, Close prices, and Volume of trades. The target variable is a binary representation indicating whether the next day's close is higher than the current day's.

**Model**

This project uses the Long Short-Term Memory neural network (LSTM) for times series forecasting:

--> LSTM layers to process time series data.

--> Dropout layers to prevent overfitting.

--> Dense layers for prediction output.

--> It predicts binary outcomes (1 for price up, 0 for down).


**Results**

The model will print the predictions for the next week and display them alongside the actual price movements from the last available week in the dataset for comparison.
Unfortunately this model's accuracy is only around 53%. Accuracy could be increased using a different model or a combination of models.
**
Future Enhancements**
Implement more sophisticated feature engineering.
Test different architectures and hyperparameters for the LSTM.
Incorporate more granular data, such as minute-by-minute prices for more data points to train on.
