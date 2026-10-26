FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY models/ models/

EXPOSE 5000
CMD ["gunicorn", "--chdir", "app", "-b", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "app:app"]
