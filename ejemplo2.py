from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="Mi primer API con POST", description="Esta es una API que sirve para aprender FastAPI.")

class Persona(BaseModel):
    nombre: str
    edad: int
    ciudad: str

class Operacion(BaseModel):
    numero1: float
    numero2: float

personas = []

@app.get("/")
def inicio():
    return {
        "mensaje": "API con GET y POST",
        "endpoints": { 
            "GET": ["/personas", "/personas/count"], 
            "POST": ["/personas", "/calcular/suma"]
            }
    }

@app.get("/personas")
def get_personas():
    return {"personas": personas, "total": len(personas)}

@app.post("/personas")
def post_personas(persona: Persona):
    personas.append(persona.dict())
    return {
        "mensaje": "Persona agregada correctamente",
        "persona": persona,
        "total_personas": len(personas)
    }



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)