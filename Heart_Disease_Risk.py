#this project i applied my learning for ML through college 
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def generate_heart_disease_data(num_samples=1000):
    np.random.seed(42)
    age = np.random.normal(55, 15, num_samples)
    cholesterol = np.random.normal(200, 50, num_samples)
    blood_pressure = np.random.normal(130, 20, num_samples)
    
    risk = (age > 60) | (cholesterol > 240) | (blood_pressure > 140)
    
    X = np.column_stack([age, cholesterol, blood_pressure])
    y = risk.astype(int)
    
    return X, y

X, y = generate_heart_disease_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential([
    Dense(64, activation='relu', input_shape=(3,)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train_scaled, y_train, 
                    epochs=100, 
                    batch_size=32, 
                    validation_split=0.2,
                    verbose=0)

loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f'Test Accuracy: {accuracy*100:.2f}%')

def predict_heart_disease_risk(age, cholesterol, blood_pressure):
    input_data = np.array([[age, cholesterol, blood_pressure]])
    input_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_scaled)
    risk_probability = prediction[0][0]
    
    return f"Heart Disease Risk: {risk_probability*100:.2f}%"

print(predict_heart_disease_risk(65, 250, 150))
