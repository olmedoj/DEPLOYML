"""Hola Mundo desde FastAPI"""

from fastapi import FastAPI

# crear una instancia de FastAPI
# esto es una app web

app = FastAPI(title="Mi primer API", description="Esta es una API que sirve para aprender FastAPI./")

# DEFINIR UN ENDPOINT (ruta) para procesar peticiones HTTP
# agregar un decorador 
@app.get("/")
def inicio():
    return {
        "mensaje": "Esta es una prueba de una API.",
        "endpoints": [
            "/sumar/{numero1}/{numero2}",
            "/restar/{numero1}/{numero2}",
            "/multiplicar/{numero1}/{numero2}",
            "/dividir/{numero1}/{numero2}",
            "/hola/{nombre}",
            "/saludo"
        ]
    }

@app.get("/sumar/{numero1}/{numero2}")
def sumar(numero1: float, numero2: float):
    resultado = numero1 + numero2
    return {
        "operacion": "suma",
        "numero1": numero1,
        "numero2": numero2,
        "resultado": resultado
    }

@app.get("/restar/{numero1}/{numero2}")
def restar(numero1: float, numero2: float):
    resultado = numero1 - numero2
    return {
        "operacion": "resta",
        "numero1": numero1,
        "numero2": numero2,
        "resultado": resultado
    }

@app.get("/multiplicar/{numero1}/{numero2}")
def multiplicar(numero1: float, numero2: float):
    resultado = numero1 * numero2
    return {
        "operacion": "multiplicacion",
        "numero1": numero1,
        "numero2": numero2,
        "resultado": resultado
    }

@app.get("/dividir/{numero1}/{numero2}")
def dividir(numero1: float, numero2: float):
    if numero2 == 0:
        return {
            "error": "No se puede dividir entre cero"
        }
    resultado = numero1 / numero2
    return {
        "operacion": "division",
        "numero1": numero1,
        "numero2": numero2,
        "resultado": resultado
    }













@app.get("/saludo")
def saludo():
    return "Hola a todos, estan en otro endpoint!"

@app.get("/hola/{nombre}")
def saludo_nombre(nombre: str):
    return f"Hola {nombre}, bienvenido a FastAPI"









if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
