import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from fraud_model import FraudDetectionModel
import joblib
import os
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar preprocesador si existe
PREPROCESSOR_PATH = 'preprocessor.joblib'

# Campos requeridos y sus valores por defecto
REQUIRED_FIELDS = {
    'amount': 0.0,
    'hour_of_day': 12,  # Valor por defecto: mediodía
    'merchant_type': 'unknown',
    'device_type': 'desktop',
    'user_history': 0.0,
    'location_match': 1  # Asumir coincidencia de ubicación
}

def load_preprocessor():
    """Carga el preprocesador desde el modelo de detección de fraude"""
    try:
        model = FraudDetectionModel(model_version='v1.0')
        if model.load_model():
            logger.info("Preprocesador cargado desde fraud_model")
            return model.preprocessor
    except Exception as e:
        logger.error(f"No se pudo cargar preprocesador desde modelo: {str(e)}")
    
    return None
    
def create_preprocessor():
    """Crea un nuevo preprocesador"""
    # Definir columnas numéricas y categóricas
    numeric_features = ['amount', 'hour_of_day', 'user_history']
    categorical_features = ['merchant_type', 'device_type']
    
    # Crear transformador de columnas
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop'  # Ignorar campos no relevantes
    )
    
    return preprocessor

def save_preprocessor(preprocessor):
    """Guarda el preprocesador en disco"""
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    logger.info(f"Preprocesador guardado en {PREPROCESSOR_PATH}")

def ensure_required_fields(data):
    """Asegura que todos los campos requeridos estén presentes con valores por defecto"""
    for field, default in REQUIRED_FIELDS.items():
        if field not in data:
            logger.warning(f"Campo faltante: {field}. Usando valor por defecto: {default}")
            data[field] = default
    return data

def extract_timestamp_fields(data):
    """Extrae campos de tiempo del timestamp si está presente"""
    if 'timestamp' in data:
        try:
            dt = pd.to_datetime(data['timestamp'])
            data['hour_of_day'] = dt.hour
            data['day_of_week'] = dt.dayofweek
            logger.info(f"Extraídos campos de tiempo: hora={dt.hour}, día={dt.dayofweek}")
        except Exception as e:
            logger.error(f"Error procesando timestamp: {str(e)}")
            data['hour_of_day'] = 12  # Valor por defecto
    return data

def preprocess(transaction):
    """Preprocesa una transacción para el modelo"""
    try:
        # Validar y completar campos requeridos
        transaction = ensure_required_fields(transaction)
        
        # Extraer campos de tiempo si es necesario
        transaction = extract_timestamp_fields(transaction)
        
        # Convertir a DataFrame
        df = pd.DataFrame([transaction])
        
        # Cargar o crear preprocesador
        preprocessor = load_preprocessor()
        if preprocessor is None:
            logger.info("Creando nuevo preprocesador")
            preprocessor = create_preprocessor()
            
            # Ajustar con datos iniciales
            sample_data = {field: [default] for field, default in REQUIRED_FIELDS.items()}
            sample_df = pd.DataFrame(sample_data)
            preprocessor.fit(sample_df)
            save_preprocessor(preprocessor)
        
        # Aplicar preprocesamiento
        processed_data = preprocessor.transform(df)
        
        # Convertir a lista y agregar a la transacción
        transaction['features'] = processed_data.tolist()[0]
        
        # Mantener todos los campos originales
        return transaction
        
    except Exception as e:
        logger.error(f"Error en preprocesamiento: {str(e)}")
        # Devolver transacción con campos requeridos como mínimo
        return {**REQUIRED_FIELDS, **transaction}