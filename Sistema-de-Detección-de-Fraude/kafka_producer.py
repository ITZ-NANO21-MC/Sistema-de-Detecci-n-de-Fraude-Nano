from confluent_kafka import Producer
import json
import os
import logging

# Configurar logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

conf = {
    'bootstrap.servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
    'client.id': 'fraud-detection-producer'
}

producer = Producer(conf)

def send_to_kafka(topic, data):
    """Envía datos a un topic de Kafka con manejo de errores"""
    if data is None:
        logger.error("Intento de enviar datos None a Kafka")
        return False
    
    try:
        # Verificar que tenemos transaction_id
        transaction_id = data.get('transaction_id', 'unknown')
        
        producer.produce(
            topic=topic,
            key=transaction_id,
            value=json.dumps(data).encode('utf-8')
        )
        producer.flush()
        logger.info(f"Enviado a {topic}: {transaction_id}")
        return True
    except Exception as e:
        logger.error(f"Error enviando a Kafka: {str(e)}")
        return False