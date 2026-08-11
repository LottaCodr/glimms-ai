FROM python:3.11-slim
WORKDIR /app
RUN addgroup --system app && adduser --system --group app
COPY services/llm_reasoning/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY services/llm_reasoning/app/ ./app/
USER app
EXPOSE 8005
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8005"]
