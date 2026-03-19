# Stage 1: Build
FROM python:3.10 as build
WORKDIR /app

# 1. Install dependencies to a specific prefix
COPY flask_app/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# 2. Copy the app code and model into the build area
COPY flask_app/ /app/flask_app/
COPY models/vectorizer.pkl /app/models/vectorizer.pkl

# Stage 2: Final (The actual image that runs)
FROM python:3.10-slim as final
WORKDIR /app

# 3. Copy the installed libraries from the build stage to the system path
COPY --from=build /install /usr/local

# 4. Copy the application files from the build stage
COPY --from=build /app/flask_app /app/flask_app
COPY --from=build /app/models /app/models

# 5. Pre-download NLTK data (essential for your preprocessing_utility.py)
RUN python -m nltk.downloader wordnet omw-1.4 punkt stopwords

EXPOSE 5000

# 6. Set the correct entry point
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "flask_app.app:app"]