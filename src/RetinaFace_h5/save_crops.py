import cv2
from tqdm import tqdm
from pathlib import Path
from src.RetinaFace_h5.utils import find_face
from src.utils import PARENT_DIR

padding = 20
new_img_path = PARENT_DIR / 'temp' / f'faces_{padding}px'
new_img_path.mkdir(exist_ok=True, parents=True)

DATASET_DIR = Path('/home/psv/file/project/recognition_dataset/office/')

if __name__ == '__main__':
    imgs = list(DATASET_DIR.glob('*.jpg'))
    p_bar = tqdm(imgs)
    for idx, img_path in enumerate(p_bar):
        p_bar.set_description(f'{img_path}')
        img, face_landmarks = find_face(img_path, threshold=0.8)

        detected_bboxes = [face_landmarks[key]["facial_area"] for key in face_landmarks.keys()]
        detected_bboxes = [[0, xmin, ymin, xmax, ymax] for (xmin, ymin, xmax, ymax) in detected_bboxes]

        h, w, c = img.shape
        for pred in detected_bboxes:
            _, xmin, ymin, xmax, ymax = pred
            crop_face = img[max(ymin - padding, 0):ymax + padding, xmin - padding:min(xmax + padding, w * h)]
            face_crop_path = f'{new_img_path / f"{Path(img_path).stem}_{idx}.jpg"}'
            cv2.imwrite(face_crop_path, crop_face)
