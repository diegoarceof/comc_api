from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/hello")
def hello():
    return {"message": "Hello, World!"}

@app.get("/greet/{name}")
def greet(name: str):
    return {"message": f"Hello, {name}!"}   

@app.post("/add")
def add(data: dict):
    return {"result": data["num1"] + data["num2"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)