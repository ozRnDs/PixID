# PixID
Part of the PixSel project - Face Recognition server

# Build command
```bash
docker build -f deployment/Dockerfile -t ozrnds/PixID:$(cz version -p) .
```

# Run container
```
docker run -v ./data:/models -p 8010:8000 my-fastapi-app
```