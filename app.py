import numpy as np

from fastapi import FastAPI


app = FastAPI()

@app.get("/")
def root():
    return {"message": "Welcome to my FastAPI server!"}

@app.get("/hello")
def hello():
    return {"message": "Hello, World!"}

@app.get("/greet/{name}")
def greet(name: str):
    return {"message": f"Hello, {name}!"}   

@app.get("/vector/{dim}")
def vector(dim: int):
    return {"vector": np.random.random(dim).tolist()}