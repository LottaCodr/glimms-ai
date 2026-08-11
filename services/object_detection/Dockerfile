FROM python:3.11-slim
WORKDIR /app
RUN addgroup --system app && adduser --system --group app
COPY services/object_detection/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shared/ ./shared/
COPY services/object_detection/app/ ./app/
# Model weights are optional; the service uses a deterministic development
# fallback until a model is mounted or supplied in this directory.
COPY services/object_detection/models/ ./models/
USER app
EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
