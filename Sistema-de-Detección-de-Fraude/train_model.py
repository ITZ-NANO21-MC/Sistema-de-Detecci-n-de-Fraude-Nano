from fraud_model import FraudDetectionModel
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entrenar modelo de detección de fraude')
    parser.add_argument('--version', type=str, default='v1.0', help='Versión del modelo')
    parser.add_argument('--samples', type=int, default=10000, help='Número de muestras sintéticas')
    parser.add_argument('--epochs', type=int, default=20, help='Épocas de entrenamiento')
    
    args = parser.parse_args()
    
    model = FraudDetectionModel(model_version=args.version)
    model.train(n_samples=args.samples, epochs=args.epochs)
    
    print(f"Modelo {args.version} entrenado y guardado exitosamente")