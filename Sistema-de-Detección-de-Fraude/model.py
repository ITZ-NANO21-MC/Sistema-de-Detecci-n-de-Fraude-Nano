from fraud_model import FraudDetectionModel
import logging

# Configurar logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Inicializar el modelo
fraud_model_ = FraudDetectionModel(model_version='v1.0')
logger.info("Preprocesador del modelo: %s", fraud_model_.preprocessor)

# Cargar o entrenar el modelo al importar
if not fraud_model_.load_model():
    logger.info("Entrenando modelo inicial...")
    fraud_model_.train(n_samples=10000, epochs=20)
    logger.info("Modelo entrenado y listo para predicciones")

# Verificar características esperadas
if fraud_model_.preprocessor:
    expected_features = ['amount', 'hour_of_day', 'user_history',
                         'merchant_type', 'device_type']
    for feat in expected_features:
        if feat not in fraud_model_.preprocessor.feature_names_in_:
            logger.warning(f"Característica faltante en preprocesador: {feat}")

def predict_fraud(transaction):
    """Predice fraude para una transacción preprocesada"""
    # Usar el método predict_from_kafka que maneja mensajes de Kafka
    return fraud_model_.predict_from_kafka(transaction)