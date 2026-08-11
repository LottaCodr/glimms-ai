FROM python:3.11-slim
WORKDIR /app
RUN addgroup --system app && adduser --system --group app
COPY services/mockup_compositor/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shared/ ./shared/
COPY services/mockup_compositor/app/ ./app/
USER app
EXPOSE 8006
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8006"]
