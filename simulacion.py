from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Estudiante(BaseModel):
    horas_estudio: float
    asistencia: float
    tareas_entregadas: int

class Flor(BaseModel):
    longitud_petalo: float
    ancho_petalo: float


@app.get("/")
def inicio():
    return {"mensaje": "Bienvenido a la simulacion",
    "modelos": ["predictor_notas", "clasificador_flores"],
    "endpoints": ["/predecir/nota", "/clasificar/flor"]}

@app.post("/predecir/nota")
def predecir_nota(estudiante: Estudiante):
    nota_base = 0
    print(nota_base)
    nota_base += min(estudiante.horas_estudio * 2, 40)
    print(nota_base)
    nota_base += min(estudiante.asistencia /100, 20) * 30
    print(nota_base)
    nota_base += min(estudiante.tareas_entregadas*3, 30)
    print(nota_base)
    nota_final = round(nota_base, 2)

    if nota_final >= 70:
        estado = "aprobado"
    elif nota_final >= 60:
        estado = "aprobado con recuperacion"
    else:
        estado = "reprobado"
    
    return {
        "estudiante": estudiante,
        "nota_predicha": nota_final,
        "estado": estado
    }

@app.post("/clasificar/flor")
def clasificar_flor(flor: Flor):
    if flor.longitud_petalo > 3 and flor.ancho_petalo > 2:
        return {"flor": flor, "clase": "roja"}
    elif flor.longitud_petalo < 2 and flor.ancho_petalo < 1:
        return {"flor": flor, "clase": "blanca"}
    else:
        return {"flor": flor, "clase": "amarilla"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)


    