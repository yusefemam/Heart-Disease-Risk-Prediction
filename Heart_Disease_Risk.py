#A special code for people who suffer from heart diseases 
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizer, TFBertModel


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

def generate_heart_disease_data(num_samples=1000):
    np.random.seed(42)
    age = np.random.normal(55, 15, num_samples)
    cholesterol = np.random.normal(200, 50, num_samples)
    blood_pressure = np.random.normal(130, 20, num_samples)
    description = ["Patient has a history of smoking and high cholesterol." if i % 2 == 0 else "Patient has no significant medical history." for i in range(num_samples)]
    
    risk = (age > 60) | (cholesterol > 240) | (blood_pressure > 140)
    
    X = np.column_stack([age, cholesterol, blood_pressure])
    y = risk.astype(int)
    
    return X, y, description

def extract_bert_embeddings(descriptions):
    inputs = tokenizer(descriptions, return_tensors='tf', padding=True, truncation=True, max_length=128)
    outputs = bert_model(inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]  # Use the [CLS] token representation
    return embeddings.numpy()

X, y, descriptions = generate_heart_disease_data()


nlp_features = extract_bert_embeddings(descriptions)

X_combined = np.hstack((X, nlp_features))

X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

input_numeric = Input(shape=(3,))
input_nlp = Input(shape=(nlp_features.shape[1],))

x_numeric = Dense(64, activation='relu')(input_numeric)
x_numeric = BatchNormalization()(x_numeric)
x_numeric = Dropout(0.5)(x_numeric)

x_nlp = Dense(64, activation='relu')(input_nlp)
x_nlp = BatchNormalization()(x_nlp)
x_nlp = Dropout(0.5)(x_nlp)

merged = Concatenate()([x_numeric, x_nlp])
x = Dense(64, activation='relu')(merged)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(32, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=[input_numeric, input_nlp], outputs=output)

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])


early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')

history = model.fit([X_train_scaled[:, :3], X_train_scaled[:, 3:]], y_train, 
                    epochs=100, 
                    batch_size=32, 
                    validation_split=0.2,
                    callbacks=[early_stopping, model_checkpoint],
                    verbose=1)

loss, accuracy = model.evaluate([X_test_scaled[:, :3], X_test_scaled[:, 3:]], y_test)
print(f'Test Accuracy: {accuracy*100:.2f}%')

def predict_heart_disease_risk(age, cholesterol, blood_pressure, description):
    input_numeric_data = np.array([[age, cholesterol, blood_pressure]])
    input_nlp_data = extract_bert_embeddings([description])
    
    input_numeric_scaled = scaler.transform(input_numeric_data)
    input_combined_scaled = np.hstack((input_numeric_scaled, input_nlp_data))
    
    prediction = model.predict([input_combined_scaled[:, :3], input_combined_scaled[:, 3:]])
    risk_probability = prediction[0][0]
    
    return f"Heart Disease Risk: {risk_probability*100:.2f}%"

print(predict_heart_disease_risk(65, 250, 150, "Patient has a history of smoking and high cholesterol."))
