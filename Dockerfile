FROM python:3.11-slim

WORKDIR /apps

RUN pip install --no-cache-dir fastapi uvicorn pandas joblib pydantic

COPY app/ .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
