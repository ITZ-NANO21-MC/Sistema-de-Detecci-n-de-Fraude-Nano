from confluent_kafka import Consumer, KafkaError
from datetime import datetime
from kafka_producer import send_to_kafka
import json
import preprocessing
import model
import database
import os
import logging

# Configuración de logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configuración de Kafka
conf = {
    'bootstrap.servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
    'group.id': 'fraud-detection-group',
    'auto.offset.reset': 'earliest',
    'enable.auto.commit': False,
    'session.timeout.ms': 6000,
    'heartbeat.interval.ms': 2000
}

def start_consumer():
    """Inicia el consumidor de Kafka para procesar transacciones"""
    consumer = Consumer(conf)
    topics = ['transactions', 'preprocessed-transactions', 'fraud-results']
    consumer.subscribe(topics)
    
    logger.info(f"Consumidor iniciado. Suscrito a: {', '.join(topics)}")
    
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            
            # Manejo de errores
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    logger.info(f"Fin de partición alcanzado en {msg.topic()} [{msg.partition()}]")
                    continue
                else:
                    logger.error(f"Error en el consumidor: {msg.error()}")
                    continue
            
            # Decodificar mensaje
            try:
                data = json.loads(msg.value().decode('utf-8'))
            except Exception as e:
                logger.error(f"Error decodificando mensaje: {str(e)}")
                continue
            
            topic = msg.topic()
            logger.info(f"Mensaje recibido en {topic}: {data.get('transaction_id', 'unknown')}")
            
            # Procesar según el topic
            if topic == 'transactions':
                process_raw_transaction(data)
            elif topic == 'preprocessed-transactions':
                process_for_model(data)
            elif topic == 'fraud-results':
                database.save_result(data)
                
    except KeyboardInterrupt:
        logger.info("Consumidor detenido por el usuario")
    except Exception as e:
        logger.error(f"Error crítico en consumidor: {str(e)}")
    finally:
        consumer.close()
        logger.info("Consumidor cerrado correctamente")

def process_raw_transaction(data):
    """Preprocesa la transacción y envía al siguiente topic"""
    try:
        logger.info(f"Preprocesando transacción: {data['transaction_id']}")
        
        processed_data = preprocessing.preprocess(data)
        
        if processed_data is None:
            logger.error("Preprocesamiento fallido. Enviando a DLQ")
            raise ValueError("Preprocesamiento devolvió None")
        
        send_to_kafka('preprocessed-transactions', processed_data)
    except Exception as e:
        logger.error(f"Error en preprocesamiento: {str(e)}")
        send_to_kafka('fraud-dlq', {
            'transaction_id': data.get('transaction_id', 'unknown'),
            'original_data': data,
            'error': str(e),
            'stage': 'preprocessing',
            'timestamp': datetime.utcnow().isoformat()
        })

def process_for_model(data):
    """Aplica el modelo de detección de fraude"""
    try:
        logger.info(f"Procesando para modelo: {data.get('transaction_id', 'unknown')}")
        
        # Registrar datos completos para diagnóstico
        logger.debug("Datos completos recibidos: %s", json.dumps(data, indent=2))
        
        prediction = model.predict_fraud(data)
        
        logger.debug(f"Predicción: {prediction}")
        send_to_kafka('fraud-results', prediction)
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}", exc_info=True)
        send_to_kafka('fraud-dlq', {
            'transaction_id': data.get('transaction_id', 'unknown'),
            'original_data': data,
            'error': str(e),
            'stage': 'prediction',
            'timestamp': datetime.utcnow().isoformat()
        })

if __name__ == '__main__':
    start_consumer()