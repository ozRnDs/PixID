# Faster alignment function - to replace detection.py of the deepface

def align_img_wrt_eyes(img, left_eye, right_eye):
    if left_eye is None or right_eye is None or img.shape[0] == 0 or img.shape[1] == 0:
        return img, 0

    angle = float(np.degrees(np.arctan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0])))

    # Get rotation matrix
    center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1)

    # Perform affine transformation (rotation)
    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)
    return img, angle