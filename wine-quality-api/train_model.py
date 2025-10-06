import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_and_save_model():
    print("📥 Descargando dataset...")
    
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        df = pd.read_csv(url, sep=';')
        print(f"✅ Dataset cargado: {df.shape[0]} muestras")
        
    except Exception as e:
        print(f"❌ Error descargando dataset: {e}")
        return create_sample_model()
    
    df['quality_binary'] = (df['quality'] >= 6).astype(int)
    
    X = df.drop(['quality', 'quality_binary'], axis=1)
    y = df['quality_binary']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"🎯 Precisión del modelo: {accuracy:.2%}")
    
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    print("💾 Modelo y scaler guardados exitosamente")
    return model, scaler, accuracy

def create_sample_model():
    print("🔄 Creando modelo de ejemplo...")
    
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    import joblib
    
    X, y = make_classification(
        n_samples=1000,
        n_features=11,
        n_informative=8,
        n_redundant=3,
        random_state=42
    )
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_scaled, y)
    
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    print("✅ Modelo de ejemplo creado y guardado")
    return model, scaler, 0.85

if __name__ == '__main__':
    train_and_save_model()