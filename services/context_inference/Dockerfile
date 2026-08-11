FROM python:3.11-slim
WORKDIR /app
RUN addgroup --system app && adduser --system --group app
COPY services/context_inference/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shared/ ./shared/
COPY services/context_inference/app/ ./app/
USER app
EXPOSE 8008
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8008"]
