FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
COPY flask_app/ /app/flask_app/

COPY models/vectorizer.pkl /app/models/vectorizer.pkl

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["gunicorn","-b","0.0.0.0:5000","flask_app.app:app"]