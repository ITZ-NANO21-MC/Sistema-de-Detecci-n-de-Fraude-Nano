import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import dump, load
import os
import json
import logging

# Cofigura logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudDetectionModel:
    def __init__(self, model_version='v1.0'):
        self.model_version = model_version
        self.model = None
        self.preprocessor = None
        self.threshold = 0.7  # Umbral para clasificación de fraude
        self.model_path = f'fraud_model_{model_version}.h5'
        self.preprocessor_path = f'preprocessor_{model_version}.joblib'

    def generate_synthetic_data(self, n_samples=10000):
        """Genera datos sintéticos para entrenamiento"""
        logger.info(f"Generando {n_samples} muestras sintéticas...")
        
        np.random.seed(42)
        data = {
            'amount': np.random.exponential(500, n_samples),
            'hour_of_day': np.random.randint(0, 24, n_samples),
            'merchant_type': np.random.choice(['retail', 'online', 'travel', 'services'], n_samples),
            'user_history': np.random.normal(0, 1, n_samples),
            'device_type': np.random.choice(['mobile', 'desktop', 'tablet'], n_samples),
            'location_match': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        }
        
        # Crear características de riesgo
        risk_factors = (
            0.3 * (data['amount'] > 1000) +
            0.4 * (data['hour_of_day'] < 6) +
            0.2 * (data['merchant_type'] == 'online') +
            0.1 * (data['location_match'] == 0)
        )
        
        # Crear etiquetas de fraude con ruido
        fraud_prob = 1 / (1 + np.exp(-(risk_factors - 1.5 + np.random.normal(0, 0.5, n_samples))))
        data['is_fraud'] = (fraud_prob > 0.5).astype(int)
        
        logger.info(f"Datos generados. Fraudes: {data['is_fraud'].sum()/n_samples:.2%}")
        return pd.DataFrame(data)

    def create_preprocessor(self):
        """Crea pipeline de preprocesamiento"""
        numeric_features = ['amount', 'hour_of_day','user_history']
        categorical_features = ['merchant_type', 'device_type']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough'
        )
        
        return preprocessor

    def build_model(self, input_shape):
        """Construye la arquitectura del modelo"""
        model = Sequential([
            Dense(64, activation='relu', input_shape=(input_shape,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), 
                     tf.keras.metrics.Recall(name='recall')]
        )
        
        model.summary()
        return model

    def train(self, n_samples=10000, epochs=20, batch_size=32):
        """Entrena el modelo con datos sintéticos"""
        # Generar y preparar datos
        df = self.generate_synthetic_data(n_samples)
        X = df.drop('is_fraud', axis=1)
        y = df['is_fraud']
        
        # Crear y ajustar preprocesador
        self.preprocessor = self.create_preprocessor()
        X_processed = self.preprocessor.fit_transform(X)
        
        # Dividir datos
        X_train, X_val, y_train, y_val = train_test_split(
            X_processed, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Construir y entrenar modelo
        self.model = self.build_model(X_train.shape[1])
        
        early_stopping = EarlyStopping(
            monitor='val_recall', 
            patience=5, 
            mode='max', 
            restore_best_weights=True
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            class_weight={0: 1, 1: 10}  # Mayor peso a la clase minoritaria (fraude)
        )
        
        # Evaluar modelo
        val_loss, val_acc, val_precision, val_recall = self.model.evaluate(X_val, y_val)
        logger.info(f"Resultados validación: Acc={val_acc:.4f}, Prec={val_precision:.4f}, Rec={val_recall:.4f}")
        
        # Guardar modelo y preprocesador
        self.save_model()
        
        return history

    def save_model(self):
        """Guarda el modelo y preprocesador en disco"""
        if self.model:
            self.model.save(self.model_path)
            logger.info(f"Modelo guardado en {self.model_path}")
        
        if self.preprocessor:
            dump(self.preprocessor, self.preprocessor_path)
            logger.info(f"Preprocesador guardado en {self.preprocessor_path}")
    
    def load_model(self):
        """Carga modelo y preprocesador desde disco"""
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
            logger.info(f"Modelo cargado desde {self.model_path}")
        
        if os.path.exists(self.preprocessor_path):
            self.preprocessor = load(self.preprocessor_path)
            logger.info(f"Preprocesador cargado desde {self.preprocessor_path}")
        
        return self.model is not None and self.preprocessor is not None
    
    def predict(self, transaction_data):
        """Realiza predicción de fraude para una transacción"""
        if not self.model or not self.preprocessor:
            if not self.load_model():
                logger.error("Modelo no cargado y no disponible para predicción")
                return {
                    'error': 'Model not loaded',
                    'probability': 0.0,
                    'is_fraud': False
                }
        
        try:
            # Registrar características disponibles
            logger.debug("Características disponibles: %s", list(transaction_data.keys()))
            
            # Convertir datos a DataFrame para preprocesamiento
            df = pd.DataFrame([transaction_data])
            
            # Verificar características faltantes
            missing_features = [f for f in self.preprocessor.feature_names_in_ 
                            if f not in df.columns]
            
            if missing_features:
                logger.error(f"Características faltantes: {missing_features}")
                raise ValueError(f"Columnas faltantes: {missing_features}")
            
            # Aplicar preprocesamiento
            processed_data = self.preprocessor.transform(df)
            
            # Realizar predicción
            probability = self.model.predict(processed_data, verbose=0)[0][0]
            is_fraud = probability > self.threshold
            
            return {
                'probability': float(probability),
                'is_fraud': bool(is_fraud)
            }
        except Exception as e:
            logger.error(f"Error en predicción: {str(e)}", exc_info=True)
            return {
                'error': str(e),
                'probability': 0.0,
                'is_fraud': False
            }

    def predict_from_kafka(self, kafka_message):
        """Procesa mensaje de Kafka y realiza predicción"""
        try:
            # Definir valores por defecto para todos los campos requeridos
            default_values = {
                'amount': 0.0,
                'hour_of_day': 12,  # Valor por defecto: mediodía
                'merchant_type': 'unknown',
                'device_type': 'desktop',
                'user_history': 0.0,
                'location_match': 1
            }
            
            # Crear transaction_data con valores por defecto
            transaction_data = {**default_values, **kafka_message}
            
            # Asegurar tipos correctos
            transaction_data['amount'] = float(transaction_data['amount'])
            transaction_data['hour_of_day'] = int(transaction_data['hour_of_day'])
            transaction_data['user_history'] = float(transaction_data['user_history'])
            transaction_data['location_match'] = int(transaction_data['location_match'])
            
            # Extraer campos de tiempo si es necesario
            if 'timestamp' in kafka_message and 'hour_of_day' not in kafka_message:
                try:
                    dt = pd.to_datetime(kafka_message['timestamp'])
                    transaction_data['hour_of_day'] = dt.hour
                    logger.info(f"Extraído hour_of_day de timestamp: {dt.hour}")
                except:
                    pass
            
            logger.debug(f"Datos para predicción: {transaction_data}")
            
            # Realizar predicción
            prediction = self.predict(transaction_data)
            
            # Añadir metadata
            prediction.update({
                'transaction_id': kafka_message['transaction_id'],
                'model_version': self.model_version,
                'amount': transaction_data['amount']
            })
            
            return prediction
            
        except KeyError as e:
            logger.error(f"Falta campo requerido en mensaje Kafka: {str(e)}")
            return {
                'error': f'Missing field: {str(e)}',
                'transaction_id': kafka_message.get('transaction_id', 'unknown')
            }
