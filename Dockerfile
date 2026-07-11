FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK resources
RUN python -m nltk.downloader \
    stopwords \
    punkt \
    punkt_tab \
    averaged_perceptron_tagger \
    averaged_perceptron_tagger_eng \
    wordnet \
    omw-1.4

COPY . .

RUN pip install -e .

EXPOSE 5000

CMD ["python", "app.py"]