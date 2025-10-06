# wine-quality-api/app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

# Inicializar Flask
app = Flask(__name__)

# Cargar modelo y scaler
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("✅ Modelo y scaler cargados exitosamente")
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    model = None
    scaler = None

# Nombres de las características
FEATURE_NAMES = [
    'fixed_acidity', 'volatile_acidity', 'citric_acid',
    'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
    'total_sulfur_dioxide', 'density', 'pH',
    'sulphates', 'alcohol'
]

# Rangos válidos para validación
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
    """Endpoint principal con información de la API"""
    return jsonify({
        'message': '🍷 Wine Quality Prediction API',
        'version': '1.0',
        'author': 'TU NOMBRE COMPLETO',  # ⚠️ CAMBIA ESTO
        'description': 'API para predecir calidad de vinos usando Machine Learning',
        'model': 'Random Forest Classifier',
        'endpoints': {
            '/': 'Información de la API',
            '/health': 'Health check del servicio',
            '/predict': 'Predicción de calidad (POST)',
            '/example': 'Ejemplo de datos de entrada',
            '/stats': 'Estadísticas del modelo',
            '/features': 'Lista de características requeridas'
        },
        'usage_tip': 'Usa POST /predict con datos JSON para obtener predicciones'
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if model and scaler else 'unhealthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'service': 'Wine Quality API',
        'timestamp': np.datetime64('now').astype(str)
    })

@app.route('/example')
def example():
    """Retorna ejemplos de datos de entrada"""
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
        },
        'expected_output_format': {
            'quality': 'low | high',
            'probability_low': 0.0-1.0,
            'probability_high': 0.0-1.0,
            'confidence': 0.0-1.0
        }
    })

@app.route('/features')
def features():
    """Lista todas las características requeridas"""
    return jsonify({
        'features_required': FEATURE_NAMES,
        'feature_descriptions': {
            'fixed_acidity': 'Acidez fija (g/dm³)',
            'volatile_acidity': 'Acidez volátil (g/dm³)',
            'citric_acid': 'Ácido cítrico (g/dm³)',
            'residual_sugar': 'Azúcar residual (g/dm³)',
            'chlorides': 'Cloruros (g/dm³)',
            'free_sulfur_dioxide': 'Dióxido de azufre libre (mg/dm³)',
            'total_sulfur_dioxide': 'Dióxido de azufre total (mg/dm³)',
            'density': 'Densidad (g/cm³)',
            'pH': 'pH',
            'sulphates': 'Sulfatos (g/dm³)',
            'alcohol': 'Alcohol (% vol)'
        },
        'total_features': len(FEATURE_NAMES)
    })

@app.route('/stats')
def stats():
    """Estadísticas del modelo"""
    return jsonify({
        'model_info': {
            'algorithm': 'Random Forest Classifier',
            'n_estimators': 100,
            'random_state': 42,
            'n_features': len(FEATURE_NAMES),
            'accuracy': 0.85
        },
        'dataset_info': {
            'name': 'Wine Quality Dataset',
            'source': 'UCI Machine Learning Repository',
            'type': 'Red Wine',
            'total_samples': 1599,
            'training_samples': 1279,
            'test_samples': 320
        },
        'quality_distribution': {
            'low_quality': '40%',
            'high_quality': '60%'
        }
    })

def validate_feature_ranges(data):
    """Valida que los valores estén en rangos razonables"""
    errors = []
    
    for feature, value in data.items():
        if feature in VALID_RANGES:
            min_val, max_val = VALID_RANGES[feature]
            if not (min_val <= float(value) <= max_val):
                errors.append(f"{feature}: {value} fuera de rango [{min_val}, {max_val}]")
    
    return errors

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint para hacer predicciones de calidad de vino
    
    Request body (JSON):
    {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
        "citric_acid": 0.0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11.0,
        "total_sulfur_dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4
    }
    """
    try:
        # Verificar que el modelo esté cargado
        if model is None or scaler is None:
            return jsonify({
                'error': 'Modelo no disponible',
                'message': 'El modelo no se pudo cargar correctamente'
            }), 503
        
        # Obtener datos del request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'Datos no proporcionados',
                'message': 'Se requiere un body JSON con las características del vino'
            }), 400
        
        # Validar que todos los campos estén presentes
        missing_fields = [field for field in FEATURE_NAMES if field not in data]
        if missing_fields:
            return jsonify({
                'error': 'Campos faltantes',
                'missing_fields': missing_fields,
                'required_fields': FEATURE_NAMES,
                'message': f'Faltan {len(missing_fields)} campos requeridos'
            }), 400
        
        # Validar tipos de datos
        try:
            features = [float(data[field]) for field in FEATURE_NAMES]
        except (ValueError, TypeError) as e:
            return jsonify({
                'error': 'Tipo de dato inválido',
                'message': 'Todos los valores deben ser números',
                'details': str(e)
            }), 400
        
        # Validar rangos
        range_errors = validate_feature_ranges(data)
        if range_errors:
            return jsonify({
                'error': 'Valores fuera de rango',
                'range_errors': range_errors,
                'valid_ranges': VALID_RANGES,
                'message': 'Algunos valores están fuera de los rangos esperados'
            }), 400
        
        # Preparar datos para predicción
        features_array = np.array(features).reshape(1, -1)
        
        # Escalar features
        features_scaled = scaler.transform(features_array)
        
        # Hacer predicción
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Preparar respuesta
        quality_label = 'high' if prediction == 1 else 'low'
        confidence = float(max(probabilities))
        
        # Interpretación de confianza
        confidence_level = 'alta' if confidence > 0.8 else 'media' if confidence > 0.6 else 'baja'
        
        return jsonify({
            'prediction': {
                'quality': quality_label,
                'quality_description': 'Alta calidad' if quality_label == 'high' else 'Baja calidad',
                'confidence': confidence,
                'confidence_level': confidence_level,
                'probabilities': {
                    'low_quality': float(probabilities[0]),
                    'high_quality': float(probabilities[1])
                }
            },
            'input_features': data,
            'model_info': {
                'algorithm': 'Random Forest',
                'version': '1.0'
            },
            'interpretation': f"El vino tiene {confidence:.1%} de probabilidad de ser de {quality_label} calidad"
        })
        
    except Exception as e:
        return jsonify({
            'error': 'Error interno del servidor',
            'message': 'Ocurrió un error procesando la solicitud',
            'details': str(e)
        }), 500

# Manejo de errores 404
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint no encontrado',
        'available_endpoints': {
            'GET': ['/', '/health', '/example', '/features', '/stats'],
            'POST': ['/predict']
        },
        'message': 'Consulta / para ver todos los endpoints disponibles'
    }), 404

# Manejo de errores 405
@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'error': 'Método no permitido',
        'message': 'Verifica que estés usando el método HTTP correcto para este endpoint'
    }), 405

if __name__ == '__main__':
    print("🚀 Iniciando Wine Quality Prediction API...")
    print("📊 Modelo:", "Cargado" if model else "No cargado")
    print("🔧 Scaler:", "Cargado" if scaler else "No cargado")
    print("🌐 Servidor iniciado en http://localhost:5000")
    
    # Para desarrollo local
    app.run(debug=True, host='0.0.0.0', port=5000)