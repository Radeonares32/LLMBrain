from fastapi import FastAPI

app = FastAPI(title="Sample API")


@app.get("/health")
def health_check():
    return {"status": "ok"}
