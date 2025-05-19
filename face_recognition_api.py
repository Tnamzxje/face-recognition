import imageio
import dlib
import numpy as np
import cv2
import os

face_detector = dlib.get_frontal_face_detector()

predictor_model = './models/shape_predictor_68_face_landmarks.dat'
pose_predictor = dlib.shape_predictor(predictor_model)

face_recognition_model = './models/dlib_face_recognition_resnet_model_v1.dat'
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)


def _ensure_rgb_uint8(img):
    import numpy as np
    # If grayscale, stack to 3 channels
    if img.ndim == 2:
        img = np.stack((img,) * 3, axis=-1)
    # If has alpha channel, drop alpha
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    # Clip and convert dtype
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    # Đảm bảo mảng là contiguous
    img = np.ascontiguousarray(img)
    # Final check
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise RuntimeError(f"Image is not valid RGB uint8! dtype={img.dtype}, shape={img.shape}, min={img.min()}, max={img.max()}")
    return img


def _rect_to_tuple(rect):
    return rect.top(), rect.right(), rect.bottom(), rect.left()

def _tuple_to_rect(rect):
    return dlib.rectangle(rect[3], rect[0], rect[1], rect[2])


def _trim_rect_tuple_to_bounds(rect, image_shape):
    return max(rect[0], 0), min(rect[1], image_shape[1]), min(rect[2], image_shape[0]), max(rect[3], 0)


def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    return np.linalg.norm(face_encodings - face_to_compare, axis=1)


def _raw_face_locations(img, number_of_times_to_upsample=1):
    img = _ensure_rgb_uint8(img)
    # Debug: kiểm tra kiểu dữ liệu và giá trị pixel
    # print(f"[DEBUG] img dtype: {img.dtype}, shape: {img.shape}, min: {img.min()}, max: {img.max()}")
    return face_detector(img, number_of_times_to_upsample)

def face_locations(img, number_of_times_to_upsample=1):
    return [_trim_rect_tuple_to_bounds(_rect_to_tuple(face), img.shape)
            for face in _raw_face_locations(img, number_of_times_to_upsample)]


def _raw_face_landmarks(face_image, face_locations=None):
    if face_locations is None:
        face_locations = _raw_face_locations(face_image)
    else:
        face_locations = [_tuple_to_rect(loc) for loc in face_locations]
    return [pose_predictor(face_image, loc) for loc in face_locations]


def face_landmarks(face_image, face_locations=None):
    landmarks = _raw_face_landmarks(face_image, face_locations)
    landmarks_as_tuples = [[(p.x, p.y) for p in lm.parts()] for lm in landmarks]
    return [{
        "chin": pts[0:17],
        "left_eyebrow": pts[17:22],
        "right_eyebrow": pts[22:27],
        "nose_bridge": pts[27:31],
        "nose_tip": pts[31:36],
        "left_eye": pts[36:42],
        "right_eye": pts[42:48],
        "top_lip": pts[48:55] + [pts[64], pts[63], pts[62], pts[61], pts[60]],
        "bottom_lip": pts[54:60] + [pts[48], pts[60], pts[67], pts[66], pts[65], pts[64]]
    } for pts in landmarks_as_tuples]


def face_encodings(face_image, known_face_locations=None, num_jitters=1):
    face_image = _ensure_rgb_uint8(face_image)
    raw_landmarks = _raw_face_landmarks(face_image, known_face_locations)
    return [np.array(face_encoder.compute_face_descriptor(face_image, lm, num_jitters))
            for lm in raw_landmarks]


def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    return list(face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)


def load_image_file(filename, mode='RGB'):
    img = imageio.imread(filename)
    img = _ensure_rgb_uint8(img)
    return img
