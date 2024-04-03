import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from keras.utils import to_categorical


# Load dataset
data = pd.read_csv("cleaned_songs_data_test_1.csv")

# Split data into features and target
X = data.drop(columns=['song_emotion'])
y = data['song_emotion']

# Shuffle and split data into train and test sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=450, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape input for CNN
X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

# One-hot encode target labels
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

# Define CNN model
cnn_model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_scaled.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.01),
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.10),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.25),
    Dense(5, activation='softmax')  # Assuming 5 emotions
])

# Compile model
optimizer = Adam(learning_rate=0.001)  # Specify learning rate directly
cnn_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Define learning rate reduction callback
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)

# Train model
history = cnn_model.fit(X_train_reshaped, y_train_encoded, epochs=200, batch_size=64, validation_split=0.1, callbacks=[reduce_lr])

# Evaluate model
cnn_loss, cnn_accuracy = cnn_model.evaluate(X_test_reshaped, y_test_encoded, verbose=0)
print("CNN Accuracy:",cnn_accuracy)

# Save model
cnn_model.save("mon_emotion_classifier_12.keras")
