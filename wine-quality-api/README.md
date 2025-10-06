# 🍷 Wine Quality Prediction API

API para predecir la calidad de vinos usando Machine Learning.

## 👨‍💻 Autor
**Tu Nombre**

## 📊 Dataset
- **Fuente:** UCI Machine Learning Repository
- **Muestras:** 1,599 vinos tintos
- **Características:** 11 propiedades químicas

## 🤖 Modelo
- **Algoritmo:** Random Forest Classifier
- **Precisión:** 85%
- **Target:** Calidad (Alta/Baja)

## 🌐 Endpoints

### GET `/`
Información general de la API

### GET `/health`  
Health check del servicio

### GET `/example`
Ejemplos de datos de entrada

### GET `/features`
Lista de características requeridas

### GET `/stats`
Estadísticas del modelo

### POST `/predict`
Realiza una predicción de calidad

## 📝 Uso

### Predicción
```bash
curl -X POST https://TU-USUARIO.pythonanywhere.com/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'