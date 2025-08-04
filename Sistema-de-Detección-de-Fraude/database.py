import sqlite3
from contextlib import closing
import json

def init_db():
    """Inicializa la base de datos"""
    with closing(sqlite3.connect('fraud_detection.db')) as conn:
        conn.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
        id TEXT PRIMARY KEY,
        amount REAL,
        user_id TEXT,
        merchant TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        is_fraud BOOLEAN,
        probability REAL,
        model_version TEXT,
        raw_data TEXT  
        )
        ''')
        conn.commit()

def save_result(data):
    """Guarda (o ignora si ya existe) los resultados en la base de datos"""
    with closing(sqlite3.connect('fraud_detection.db')) as conn:
        conn.execute('''
        INSERT INTO transactions
        (id, amount, user_id, merchant, is_fraud, probability, model_version, raw_data)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO NOTHING
        ''', (
            data['transaction_id'],
            data.get('amount', 0.0),
            data.get('user_id'),
            data.get('merchant'),
            data.get('is_fraud', False),
            data.get('probability', 0.0),
            data.get('model_version', 'unknown'),
            json.dumps(data)
        ))
        conn.commit()
        
def get_transaction_details(transaction_id):
    """Obtiene detalles de una transacción"""
    with closing(sqlite3.connect('fraud_detection.db')) as conn:
        cursor = conn.execute('''
        SELECT * FROM transactions WHERE id = ?
        ''', (transaction_id,))
        row = cursor.fetchone()
        
        if row:
            return {
                'transaction_id': row[0],
                'amount': row[1],
                'is_fraud': bool(row[5]),
                'probability': row[6],
                'model_version': row[7]
            }
    return None

# Inicializar BD al importar
init_db()

