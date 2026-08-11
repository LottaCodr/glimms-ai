# glimms-ai

> All 8 Python/FastAPI AI microservices for the Glimms platform. Each service is independently deployable and runs on its own port.

## Services

| Service | Port | Responsibility |
|---|---|---|
| `object-detection` | 8001 | YOLOv8 — detects clothing, furniture, plants from images |
| `attribute-extractor` | 8002 | CLIP embeddings, dominant color (k-means), texture analysis |
| `embedding-engine` | 8003 | Upserts style vectors to Pinecone; ANN similarity search |
| `permutation-engine` | 8004 | Generates outfit/room combinations with color theory + cultural filters |
| `llm-reasoning` | 8005 | GPT-4o / Claude — generates titles, explanations, styling tips |
| `mockup-compositor` | 8006 | Assembles visual mockups using Pillow, uploads to S3 |
| `quality-guard` | 8007 | Detects blur/low-light; generates re-capture guidance |
| `context-inference` | 8008 | Maps climate + culture → style constraints |

## Start all services (local dev)

```bash
cp .env.example .env
docker-compose up
```

## Start a single service

```bash
cd services/llm_reasoning
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8005
```

## API Docs

Each service exposes Swagger UI at `http://localhost:<PORT>/docs`

## Adding a model (object detection)

1. Download a YOLOv8 ONNX model: `yolo export model=yolov8n.pt format=onnx`
2. Place it at `services/object_detection/models/yolov8n.onnx`
3. Set `MODEL_PATH=models/yolov8n.onnx` in `.env`
