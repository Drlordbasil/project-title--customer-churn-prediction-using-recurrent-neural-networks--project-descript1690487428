import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam


# Read the dataset
data = pd.read_csv('customer_data.csv')

# Perform data preprocessing
data = data.fillna(0)

# Encode categorical variables
cat_cols = ['gender', 'location']
for col in cat_cols:
    encoder = LabelEncoder()
    data[col] = encoder.fit_transform(data[col])

# Perform feature scaling
num_cols = ['age', 'transaction_amount']
scaler = StandardScaler()
data[num_cols] = scaler.fit_transform(data[num_cols])

# Split the data into training and testing sets
X = data.drop(['churn'], axis=1)
y = data['churn']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create time series generators for the training and testing sets
train_data_gen = TimeseriesGenerator(
    X_train.values, y_train.values, length=10, batch_size=32)
test_data_gen = TimeseriesGenerator(
    X_test.values, y_test.values, length=10, batch_size=32)

# Build the RNN model
model = Sequential()
model.add(LSTM(units=32, return_sequences=True,
          input_shape=(10, X_train.shape[1])))
model.add(Dropout(0.2))
model.add(LSTM(units=32))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
optimizer = Adam(lr=0.001)  # Use 'lr' instead of 'learning_rate'
model.compile(optimizer=optimizer, loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_data_gen, epochs=10)

# Evaluate the model
loss, accuracy = model.evaluate(test_data_gen)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')

# Make predictions on new data
new_data = pd.read_csv('new_customer_data.csv')
new_data = new_data.fillna(0)
new_cat_cols = ['gender', 'location']
for col in new_cat_cols:
    new_data[col] = encoder.transform(new_data[col])
new_data[num_cols] = scaler.transform(new_data[num_cols])

new_data_gen = TimeseriesGenerator(
    new_data.values, np.zeros(len(new_data)), length=10, batch_size=1)
predictions = model.predict(new_data_gen)

# Print predictions
for i, pred in enumerate(predictions):
    print(f"Customer {i+1} is predicted to churn: {pred > 0.5}")
