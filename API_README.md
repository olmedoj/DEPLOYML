# Diabetes Prediction API

API REST profesional para realizar inferencias con el modelo de predicción de diabetes usando FastAPI.

## Descripción

Esta API proporciona endpoints para predecir el riesgo de diabetes en pacientes basándose en mediciones médicas diagnósticas. Utiliza un modelo XGBoost entrenado con el dataset Pima Indians Diabetes.

## Estructura del Proyecto

```
DEPLOY/
├── api.py                    # Aplicación FastAPI principal
├── model_pipeline.py        # Script de entrenamiento del modelo
├── diabetes_pipeline.pkl    # Pipeline entrenado (generado)
├── label_encoder.pkl        # Codificador de etiquetas (generado)
├── requirements.txt         # Dependencias del proyecto
└── API_README.md           # Este archivo
```

## Requisitos Previos

1. Python 3.8 o superior
2. Entorno virtual activado
3. Modelo entrenado (archivos .pkl generados)

## Instalación

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Entrenar el modelo (si no existe)

```bash
python model_pipeline.py
```

Este comando generará los archivos:
- `diabetes_pipeline.pkl`
- `label_encoder.pkl`

## Uso de la API

### Iniciar el servidor

```bash
python api.py
```

El servidor se iniciará en: `http://localhost:8000`

### Documentación interactiva

Una vez iniciado el servidor, accede a:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Endpoints Disponibles

### 1. Root Endpoint

**GET** `/`

Información básica de la API.

```bash
curl http://localhost:8000/
```

### 2. Health Check

**GET** `/health`

Verifica el estado de la API y si los modelos están cargados.

```bash
curl http://localhost:8000/health
```

### 3. Predicción Individual

**POST** `/predict`

Realiza una predicción para un solo paciente.

**Request Body:**
```json
{
  "preg": 6,
  "plas": 148,
  "pres": 72,
  "skin": 35,
  "insu": 0,
  "mass": 33.6,
  "pedi": 0.627,
  "age": 50
}
```

**Ejemplo con curl:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "preg": 6,
    "plas": 148,
    "pres": 72,
    "skin": 35,
    "insu": 0,
    "mass": 33.6,
    "pedi": 0.627,
    "age": 50
  }'
```

**Response:**
```json
{
  "prediction": "tested_positive",
  "probability_negative": 0.2345,
  "probability_positive": 0.7655,
  "confidence": 0.7655,
  "input_features": {
    "preg": 6,
    "plas": 148,
    "pres": 72,
    "skin": 35,
    "insu": 0,
    "mass": 33.6,
    "pedi": 0.627,
    "age": 50
  }
}
```

### 4. Predicción por Lotes

**POST** `/predict/batch`

Realiza predicciones para múltiples pacientes en una sola petición.

**Request Body:**
```json
[
  {
    "preg": 6,
    "plas": 148,
    "pres": 72,
    "skin": 35,
    "insu": 0,
    "mass": 33.6,
    "pedi": 0.627,
    "age": 50
  },
  {
    "preg": 1,
    "plas": 85,
    "pres": 66,
    "skin": 29,
    "insu": 0,
    "mass": 26.6,
    "pedi": 0.351,
    "age": 31
  }
]
```

**Límite:** Máximo 100 pacientes por petición.

## Descripción de Features

| Feature | Descripción | Rango Válido |
|---------|-------------|--------------|
| `preg` | Número de embarazos | 0 - 20 |
| `plas` | Concentración de glucosa en plasma (mg/dL) | 0 - 300 |
| `pres` | Presión arterial diastólica (mm Hg) | 0 - 200 |
| `skin` | Grosor del pliegue cutáneo del tríceps (mm) | 0 - 100 |
| `insu` | Insulina sérica a las 2 horas (mu U/ml) | 0 - 900 |
| `mass` | Índice de masa corporal (kg/m²) | 0 - 70 |
| `pedi` | Función de pedigrí de diabetes | 0 - 3 |
| `age` | Edad en años | 21 - 120 |

## Validación de Datos

La API valida automáticamente:
- Tipos de datos correctos
- Rangos válidos para cada feature
- Campos requeridos presentes

Si los datos no son válidos, retorna un error 422 con detalles específicos.

## Pruebas Manuales

### Con Python y requests

```python
import requests

# Predicción individual
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "preg": 6,
        "plas": 148,
        "pres": 72,
        "skin": 35,
        "insu": 0,
        "mass": 33.6,
        "pedi": 0.627,
        "age": 50
    }
)

print(response.json())
```

### Con la documentación interactiva

Accede a `http://localhost:8000/docs` y utiliza la interfaz Swagger UI para probar los endpoints directamente desde el navegador.

## Códigos de Estado HTTP

- **200 OK**: Predicción exitosa
- **422 Unprocessable Entity**: Datos de entrada inválidos
- **500 Internal Server Error**: Error en el servidor
- **503 Service Unavailable**: Modelo no cargado

## Logging

La API registra automáticamente:
- Inicio y carga de modelos
- Predicciones realizadas
- Errores y excepciones

Los logs se muestran en la consola con formato:
```
2025-10-01 13:45:00 - api - INFO - Making prediction for input: [...]
```

## Consideraciones de Producción

Para desplegar en producción, considere:

1. **Configuración de Workers**
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
   ```

2. **Variables de Entorno**
   - Configurar rutas de modelos
   - Configurar nivel de logging
   - Configurar límites de rate limiting

3. **Seguridad**
   - Implementar autenticación (API keys, OAuth2)
   - Usar HTTPS
   - Configurar CORS apropiadamente
   - Implementar rate limiting

4. **Monitoreo**
   - Implementar métricas de rendimiento
   - Monitorear latencia de predicciones
   - Alertas para errores de modelo

5. **Escalabilidad**
   - Usar load balancer
   - Considerar contenedores (Docker)
   - Implementar caché para predicciones frecuentes

## Troubleshooting

### Error: Model not loaded

**Problema:** Los archivos .pkl no existen o no se pueden cargar.

**Solución:**
1. Ejecutar `python model_pipeline.py` para generar los modelos
2. Verificar que los archivos existen en el directorio
3. Verificar permisos de lectura

### Error: Connection refused

**Problema:** El servidor no está corriendo.

**Solución:**
1. Iniciar el servidor con `python api.py`
2. Verificar que el puerto 8000 no esté en uso
3. Revisar logs de inicio del servidor

### Error: Validation error

**Problema:** Datos de entrada fuera de rango o tipo incorrecto.

**Solución:**
1. Verificar que todos los campos estén presentes
2. Verificar que los valores estén en los rangos válidos
3. Revisar el mensaje de error detallado en la respuesta

## Contacto y Soporte

Para reportar problemas o solicitar nuevas funcionalidades, contacte al equipo de ML.

## Licencia

Este proyecto es de uso interno. Todos los derechos reservados.
