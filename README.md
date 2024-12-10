# PixID
Part of the PixSel project - Face Recognition server

# Build command
```bash
docker build -f deployment/Dockerfile -t ozrnds/PixID:$(cz version -p) .
```

# Run container
```bash
docker run --rm --name face-detection -d -v ./data:/models -p 8010:8000 ozrnds/pixid:0.1.0
```