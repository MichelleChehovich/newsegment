# YOLOv8 Segmentation API

REST API для сегментации изображений на базе YOLOv8 и FastAPI.

## Запуск через Docker Compose

```bash
docker compose up --build
```

Фоновый режим:

```bash
docker compose up --build -d
```

Сервис будет доступен на `http://localhost:8000`.

## API

### `GET /` — проверка работы

```bash
curl http://localhost:8000/
```

### `GET /health` — health check

```bash
curl http://localhost:8000/health
```

### `POST /analyze` — сегментация изображения

Принимает изображение (`multipart/form-data`), возвращает результат сегментации в base64.

```bash
curl -X POST http://localhost:8000/analyze \
  -F "image=@photo.jpg"
```

Ответ:

```json
{
  "success": true,
  "image": "<base64-строка PNG>"
}
```

## Локальный запуск (без Docker)

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Модель `yolov8n-seg.pt` скачивается автоматически при первом запросе.
