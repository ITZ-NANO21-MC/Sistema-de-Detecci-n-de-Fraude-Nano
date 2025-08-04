from flask import Flask, request, jsonify
from kafka_producer import send_to_kafka
from database import get_transaction_details
from datetime import datetime
import uuid

app = Flask(__name__)

@app.route('/transaction', methods=['POST'])
def new_transaction():
    """Endpoint para recibir nuevas transacciones"""
    data = request.json
    
    # Validar campos mínimos
    if 'amount' not in data or 'user_id' not in data:
        return jsonify({"error": "Faltan campos requeridos: amount y user_id"}), 400
    
    # Generar ID único para la transacción
    transaction_id = str(uuid.uuid4())
    data['transaction_id'] = transaction_id
    
    # Añadir timestamp si no está presente
    if 'timestamp' not in data:
        data['timestamp'] = datetime.utcnow().isoformat()
    
    # Enviar a Kafka para procesamiento
    send_to_kafka('transactions', data)
    
    return jsonify({
        'status': 'Transaction received',
        'transaction_id': transaction_id
    }), 202

@app.route('/transaction/<transaction_id>', methods=['GET'])
def get_transaction(transaction_id):
    """Endpoint para obtener resultados de fraude"""
    result = get_transaction_details(transaction_id)
    if not result: 
        return jsonify({'error': 'Transaction not found'}), 404
    return jsonify({
        'transaction_id': result['transaction_id'],
        'amount': result['amount'],
        'is_fraud': result['is_fraud'],
        'probability': result['probability'],
        'model_version': result['model_version']
    }), 200

if __name__ == '__main__':
    app.run(port=5000, debug=True)
