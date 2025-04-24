FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY ["xgboost_model.pkl", "app.py", "./"]

EXPOSE 8080

ENV PORT=8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
