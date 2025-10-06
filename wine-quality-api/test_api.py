import requests
import json

def test_api():
    base_url = "https://TU-USUARIO.pythonanywhere.com"
    
    print("🧪 Iniciando pruebas de la API...")
    
    endpoints = [
        ("/health", "GET"),
        ("/", "GET"), 
        ("/example", "GET"),
        ("/features", "GET"),
        ("/stats", "GET")
    ]
    
    for endpoint, method in endpoints:
        print(f"\n{method} {endpoint}...")
        try:
            if method == "GET":
                response = requests.get(f"{base_url}{endpoint}")
            else:
                response = requests.post(f"{base_url}{endpoint}")
            
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                print(f"   ✅ Éxito")
            else:
                print(f"   ❌ Error: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n🧪 Probando predicciones...")
    
    test_wines = [
        {
            "name": "Vino baja calidad",
            "data": {
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
        },
        {
            "name": "Vino alta calidad", 
            "data": {
                "fixed_acidity": 8.5,
                "volatile_acidity": 0.28,
                "citric_acid": 0.56,
                "residual_sugar": 1.8,
                "chlorides": 0.092,
                "free_sulfur_dioxide": 35.0,
                "total_sulfur_dioxide": 103.0,
                "density": 0.9969,
                "pH": 3.26,
                "sulphates": 0.75,
                "alcohol": 10.5
            }
        }
    ]
    
    for wine in test_wines:
        print(f"\n🍷 {wine['name']}...")
        try:
            response = requests.post(f"{base_url}/predict", json=wine['data'])
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                quality = result['prediction']['quality']
                confidence = result['prediction']['confidence']
                print(f"   ✅ Calidad: {quality}, Confianza: {confidence:.1%}")
            else:
                print(f"   ❌ Error: {response.text}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n🎉 Pruebas completadas!")

if __name__ == '__main__':
    test_api()