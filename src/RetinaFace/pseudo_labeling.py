import cv2
from tqdm import tqdm
from pathlib import Path

from src.RetinaFace.utils import find_face
from src.utils import PARENT_DIR, draw_face, bbox2yolobbox, write_label

new_img_dir_path = PARENT_DIR / 'temp' / 'RetinaFace29'
new_img_dir_path.mkdir(exist_ok=True, parents=True)

threshold = 0.90
DATASET_DIR = Path('/home/psv/file/project/179-Москва-Бессмертный_полк/screenshots/')

if __name__ == '__main__':
    imgs = list(DATASET_DIR.glob('*.jpg'))
    p_bar = tqdm(imgs)
    for idx, img_path in enumerate(p_bar):
        p_bar.set_description(f'{img_path}')

        new_img_path = new_img_dir_path / img_path.name
        img, PRED = find_face(str(img_path), threshold=threshold)
        if isinstance(PRED, dict):  # if faces in PRED
            marked_img = draw_face(img, PRED, threshold=threshold, show=False)
            cv2.imwrite(str(new_img_path), marked_img)

            PRED_yolo = list(map(lambda bbox: bbox2yolobbox(img, bbox), PRED.values())) if PRED else []
        else:
            cv2.imwrite(str(new_img_path), img)  # just make a copy

            PRED_yolo = []

        write_label(new_img_path.with_suffix('.txt'), PRED_yolo)
