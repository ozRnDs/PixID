import asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException
# from deepface import DeepFace
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from tempfile import NamedTemporaryFile
from enum import Enum
import shutil
import time
from loguru import logger
import cv2
import os
import numpy as np
# from .face_detector import BatchFaceDetector

# from .redis import RedisInterface

app = FastAPI()
# redis_interface = RedisInterface()


class FaceRecognitionModels(Enum):
    VGG_FACE = "VGG-Face"
    FACENET = "Facenet"
    FACENET512 = "Facenet512"
    OPENFACE = "OpenFace"
    DEEPFACE = "DeepFace"
    DEEPID = "DeepID"
    ARCFACE = "ArcFace"
    DLIB = "Dlib"
    SFACE = "SFace"
    GHOSTFACENET = "GhostFaceNet"

class FaceDetectionModel(Enum):
    OPENCV = "opencv"
    SSD = "ssd"
    DLIB = "dlib"
    MTCNN = "mtcnn"
    FASTMTCNN = "fastmtcnn"
    RETINAFACE = "retinaface"
    MEDIAPIPE = "mediapipe"
    YOLOV8 = "yolov8"
    YUNET = "yunet"
    CENTERFACE = "centerface"

# @app.post("/face/recognize")
# async def recognize_face(file: UploadFile = File(...)):
#     """
#     Recognize a face in the uploaded image using DeepFace.
#     """
#     if file.content_type not in ["image/jpeg", "image/png"]:
#         raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG are supported.")
    
#     with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
#         shutil.copyfileobj(file.file, temp_file)
#         temp_file_path = temp_file.name
    
#     try:
#         start_time = time.perf_counter()
#         embeddings = DeepFace.represent(img_path=temp_file_path, model_name=FaceRecognitionModels.SFACE.value) #, detector_backend=FaceDetectionModel.YOLOV8.value)
#         run_time = time.perf_counter() - start_time
#         return {"status": "success", "embeddings": embeddings, "run_time": run_time, "number_of_faces": len(embeddings)}
#     except Exception as e:
#         logger.exception(e)
#         raise HTTPException(status_code=500, detail=f"Face recognition failed: {str(e)}")
#     finally:
#         os.remove(temp_file_path)

@app.post("/face/insight-recognize")
async def insightface_recognize(file: UploadFile = File(...)):
    """
    Recognize a face in the uploaded image using InsightFace.
    """
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG are supported.")
    
    with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name

    try:
        start_time = time.perf_counter()
        
        # Initialize InsightFace model
        app_insight = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition'], root="/models")
        app_insight.prepare(ctx_id=0, det_size=(640, 640))
        img = cv2.imread(temp_file_path)
        # Analyze the image
        faces = app_insight.get(img)
        run_time = time.perf_counter() - start_time

        insights = []
        for face in faces:
            bbox = face.bbox.tolist()
            embedding = face.embedding.tolist()
            image_name = file.filename

            # Save to Redis
            # redis_key = redis_interface.add_embedding(image_name, bbox, embedding)
            # insights.append({"bbox": bbox, "embedding": embedding, "redis_key": redis_key})
            insights.append({"bbox": bbox, "embedding": embedding})

        return {"status": "success", "insights": insights, "run_time": run_time, "number_of_faces": len(insights)}
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=f"InsightFace recognition failed: {str(e)}")
    finally:
        os.remove(temp_file_path)

# @app.post("/face/vector-search")
# async def vector_search(query: dict):
#     """
#     Perform vector similarity search for a list of query embeddings.
#     """
#     embeddings = query.get("embeddings")  # Accept a list of embeddings
#     k = query.get("k", 5)  # Default to top-5 results for each embedding

#     if not embeddings or not isinstance(embeddings, list):
#         raise HTTPException(
#             status_code=400, 
#             detail="A list of embeddings is required for vector search."
#         )

#     try:
#         all_results = []
#         results = redis_interface.vector_search(embeddings, k=k)
        
#         return {"status": "success", "results": results}
#     except Exception as e:
#         logger.exception(e)
#         raise HTTPException(status_code=500, detail=f"Vector search failed: {str(e)}")

@app.post("/generate-cli-command")
async def generate_redis_cli_command(embedding: dict):
    """
    Generate a Redis CLI command for vector search based on input embeddings.
    """
    embeddings = embedding.get("embeddings")
    print(f"embeddings size: {len(embeddings)}")
    k = embedding.get("k", 5)  # Default to top-5 results

    if not embeddings or not isinstance(embeddings, list):
        raise HTTPException(status_code=400, detail="A valid 'embeddings' list is required.")

    try:
        # Convert embeddings to a byte array
        query_vector = np.array(embeddings, dtype=np.float32)
        logger.info(f"np size: {query_vector.shape}")
        # Convert byte array to hex string for CLI
        query_vector_hex = query_vector.tobytes().hex()

        # Generate Redis CLI command
        # cli_command = (
        #     f"FT.SEARCH embedding_index "
        #     f"\"*=>[KNN {k} @embedding $query_vec AS score]\" "
        #     f"PARAMS 2 query_vec \"\\x{query_vector_hex}\" "
        #     f"SORTBY score ASC RETURN 3 image_name face_bbox score"
        # )
        cli_command = (
            f"FT.SEARCH embedding_index "
            f"\"*=>[KNN {k} @embedding $query_vec]\" "
            f"PARAMS 2 query_vec \"\\x{query_vector_hex}\" "
            f"DIALECT 2"
        )

        return {"status": "success", "cli_command": cli_command}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate CLI command: {str(e)}")


# face_detector = BatchFaceDetector()

# @app.post("/face/detect")
# async def load_image(file: UploadFile = File(...)):
#     """
#     Recognize a face in the uploaded image using InsightFace.
#     """
#     if file.content_type not in ["image/jpeg", "image/png"]:
#         raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG are supported.")
    
#     with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
#         shutil.copyfileobj(file.file, temp_file)
#         temp_file_path = temp_file.name

#     img = cv2.imread(temp_file_path)
#     img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

#     await face_detector.add_image(image_name=temp_file_path, image=img)

#     return "Image Loaded"

async def start_uvicorn():
    """
    Run Uvicorn in an asyncio-compatible way.
    """
    import uvicorn
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, loop="asyncio")
    server = uvicorn.Server(config)
    await server.serve()

async def run_main():
    import uvicorn

    # process_images_task = asyncio.create_task(face_detector.process_images())
    try:
        await start_uvicorn()
    finally:
        pass
        # await face_detector.aclose()
        # await asyncio.wait([process_images_task])

if __name__ == "__main__":
    asyncio.run(run_main())
    
