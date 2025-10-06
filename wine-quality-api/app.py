from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("✅ Modelo y scaler cargados exitosamente")
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    model = None
    scaler = None

FEATURE_NAMES = [
    'fixed_acidity', 'volatile_acidity', 'citric_acid',
    'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
    'total_sulfur_dioxide', 'density', 'pH',
    'sulphates', 'alcohol'
]

VALID_RANGES = {
    'fixed_acidity': (4.0, 16.0),
    'volatile_acidity': (0.1, 2.0),
    'citric_acid': (0.0, 1.0),
    'residual_sugar': (0.5, 16.0),
    'chlorides': (0.01, 0.2),
    'free_sulfur_dioxide': (1.0, 80.0),
    'total_sulfur_dioxide': (5.0, 300.0),
    'density': (0.98, 1.01),
    'pH': (2.5, 4.5),
    'sulphates': (0.3, 2.0),
    'alcohol': (8.0, 15.0)
}

@app.route('/')
def home():
    return jsonify({
        'message': '🍷 Wine Quality Prediction API',
        'version': '1.0',
        'author': 'Tu Nombre',
        'description': 'API para predecir calidad de vinos usando Machine Learning',
        'model': 'Random Forest Classifier',
        'endpoints': {
            '/': 'Información de la API',
            '/health': 'Health check',
            '/predict': 'Predicción de calidad (POST)',
            '/example': 'Ejemplo de datos',
            '/stats': 'Estadísticas del modelo',
            '/features': 'Características requeridas'
        }
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy' if model and scaler else 'unhealthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'service': 'Wine Quality API'
    })

@app.route('/example')
def example():
    return jsonify({
        'examples': {
            'low_quality_example': {
                'fixed_acidity': 7.4,
                'volatile_acidity': 0.7,
                'citric_acid': 0.0,
                'residual_sugar': 1.9,
                'chlorides': 0.076,
                'free_sulfur_dioxide': 11.0,
                'total_sulfur_dioxide': 34.0,
                'density': 0.9978,
                'pH': 3.51,
                'sulphates': 0.56,
                'alcohol': 9.4
            },
            'high_quality_example': {
                'fixed_acidity': 8.5,
                'volatile_acidity': 0.28,
                'citric_acid': 0.56,
                'residual_sugar': 1.8,
                'chlorides': 0.092,
                'free_sulfur_dioxide': 35.0,
                'total_sulfur_dioxide': 103.0,
                'density': 0.9969,
                'pH': 3.26,
                'sulphates': 0.75,
                'alcohol': 10.5
            }
        }
    })

@app.route('/features')
def features():
    return jsonify({
        'features_required': FEATURE_NAMES,
        'total_features': len(FEATURE_NAMES)
    })

@app.route('/stats')
def stats():
    return jsonify({
        'model_info': {
            'algorithm': 'Random Forest Classifier',
            'n_features': len(FEATURE_NAMES),
            'accuracy': 0.85
        },
        'dataset_info': {
            'name': 'Wine Quality Dataset',
            'source': 'UCI Machine Learning Repository',
            'total_samples': 1599
        }
    })

def validate_feature_ranges(data):
    errors = []
    for feature, value in data.items():
        if feature in VALID_RANGES:
            min_val, max_val = VALID_RANGES[feature]
            if not (min_val <= float(value) <= max_val):
                errors.append(f"{feature}: {value} fuera de rango [{min_val}, {max_val}]")
    return errors

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None:
            return jsonify({'error': 'Modelo no disponible'}), 503
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Datos no proporcionados'}), 400
        
        missing_fields = [field for field in FEATURE_NAMES if field not in data]
        if missing_fields:
            return jsonify({
                'error': 'Campos faltantes',
                'missing_fields': missing_fields
            }), 400
        
        try:
            features = [float(data[field]) for field in FEATURE_NAMES]
        except (ValueError, TypeError) as e:
            return jsonify({'error': 'Valores deben ser números'}), 400
        
        range_errors = validate_feature_ranges(data)
        if range_errors:
            return jsonify({'error': 'Valores fuera de rango', 'range_errors': range_errors}), 400
        
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        quality_label = 'high' if prediction == 1 else 'low'
        confidence = float(max(probabilities))
        
        return jsonify({
            'prediction': {
                'quality': quality_label,
                'confidence': confidence,
                'probabilities': {
                    'low_quality': float(probabilities[0]),
                    'high_quality': float(probabilities[1])
                }
            },
            'interpretation': f"El vino tiene {confidence:.1%} de probabilidad de ser de {quality_label} calidad"
        })
        
    except Exception as e:
        return jsonify({'error': 'Error interno del servidor'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint no encontrado'}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)