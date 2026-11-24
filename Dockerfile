FROM python:3.12-slim

WORKDIR /app

# Install libraries first, so Docker can reuse this layer when only code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Then copy the code and the champion model
COPY app/ app/
COPY src/ src/
COPY models/ models/
COPY params.yaml .

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
