import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load the dataset
spy_data = pd.read_csv('spy.csv')

# Create target variable (1 if Close price is up from the previous day, else 0)
spy_data['Target'] = (spy_data['Close'].shift(-1) > spy_data['Close']).astype(int)
spy_data['Target'].fillna(0, inplace=True)  # Handle the last entry

# Normalize features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(spy_data[['Open', 'High', 'Low', 'Close', 'Volume']])

# Define a function to create sequences
def create_sequences(data, target, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(target[i + sequence_length])
    return np.array(X), np.array(y)

# Sequence length
sequence_length = 10

# Create sequences
X, y = create_sequences(scaled_features, spy_data['Target'].values, sequence_length)

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(50, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Predict the next week's movements
last_sequence = X[-1:]  # Get the last sequence from the data
next_week_predictions = []

for _ in range(7):  # Predict for the next 7 days
    next_day_pred = model.predict(last_sequence)
    next_week_predictions.append((next_day_pred > 0.5).astype(int)[0][0])
    
    # Roll the sequence to remove the oldest day
    new_sequence = last_sequence[:, 1:, :]
    
    # Create a new day which is a copy of the last day in the sequence with updated 'Close' price
    new_day = new_sequence[:, -1, :].copy()
    new_day[:, -1] = next_day_pred  # Update the 'Close' price with the predicted value
    
    # Append the new day to the sequence
    new_sequence = np.append(new_sequence, [new_day], axis=1)
    last_sequence = new_sequence.reshape(1, sequence_length, 5)
    
#  Extract actual last week movements
actual_last_week = spy_data['Target'][-7:].values

# Print predictions and actual results for comparison
print("Predictions for the next week (1 for Up, 0 for Down):")
print(next_week_predictions)
print("Actual last week movements (1 for Up, 0 for Down):")
print(actual_last_week.tolist())