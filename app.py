from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/hello")
def hello():
    return {"message": "Hello, World!"}

@app.get("/greet/{name}")
def greet(name: str):
    return {"message": f"Hello, {name}!"}   
