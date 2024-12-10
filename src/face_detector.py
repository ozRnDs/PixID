import asyncio
import numpy as np
from typing import List, Dict
from loguru import logger
from batch_face import RetinaFace

class BatchFaceDetector:
    def __init__(self):
        self._images_dict:Dict[str, List[np.ndarray]] = {}
        self._images_names: Dict[str,List[str]] = {}
        self._terminate = False
        self._processing = False

        self._detector = RetinaFace(gpu_id=0, fp16=True)

        self._max_size = 720
        self._resize = 1
        self._threshold=0.95

    async def add_image(self, image_name: str, image: np.ndarray):
        while self._processing:
            await asyncio.sleep(0.2)
        image_size = f"{image.shape[0]}x{image.shape[1]}"
        if not image_size in self._images_dict:
            self._images_dict[image_size] = []
            self._images_names[image_size] = []
        self._images_dict[image_size].append(image)
        self._images_names[image_size].append(image_name)
        logger.info(f"Added image {image_name} to the process queue")

    def detect_faces(self, images: np.ndarray, images_names: List[str]):
        logger.info(f"Processing batch size: {len(images)}")
        all_faces = self._detector(images, threshold=self._threshold, resize=self._resize, max_size=self._max_size, batch_size=len(images_names))
        return all_faces

    async def process_images(self):
        logger.info("Waiting for images")
        while True:
            if self._terminate:
                break
            self._processing=False
            await asyncio.sleep(1)
            self._processing = True
            max_key = None
            for key in self._images_dict:
                if max_key is None:
                    max_key=key
                    continue
                if len(self._images_dict[max_key])<len(self._images_dict[key]):
                    max_key=key
            if max_key is None:
                continue
            logger.info("Processing Batch")
            try:
                images_to_process = self._images_dict.pop(max_key)
                images_names = self._images_names.pop(max_key)
                faces = await asyncio.to_thread(self.detect_faces,images=images_to_process,images_names=images_names)
            except Exception as err:
                logger.exception(err)
            logger.info("Finished Processing Batch")

    async def aclose(self):
        logger.info("Closing Batch Face Detector")
        self._terminate = True
