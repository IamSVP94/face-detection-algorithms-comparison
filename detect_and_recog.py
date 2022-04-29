import time

import cv2
from tqdm import tqdm
from pathlib import Path
from retinaface import RetinaFace
from src.RetinaFace_h5.utils import find_face
from src.utils import PARENT_DIR, draw_face
from src.IResNet100.utils import ONNXRecognator, Person

new_img_dir_path = PARENT_DIR / 'temp' / 'office'
new_img_dir_path.mkdir(exist_ok=True, parents=True)

recog_model = ONNXRecognator()

DIR = PARENT_DIR / 'temp/LABELED'
person1 = Person(path=DIR / 'person1/mpv-shot0399_11.jpg', label='Sergey', color=(0, 255, 0))
person2 = Person(path=DIR / 'person2/mpv-shot0268_4.jpg', label='Farid', color=(255, 0, 0))

if __name__ == '__main__':
    DATASET_DIR = Path('/home/psv/file/project/recognition_dataset2/office/')
    imgs = list(DATASET_DIR.glob('*.jpg'))
    txts = [txt.with_suffix('.jpg') for txt in DATASET_DIR.glob('*.txt')]
    not_ready_img = [img for img in imgs if img not in txts]
    p_bar = tqdm(not_ready_img)
    for idx, img_path in enumerate(p_bar):
        p_bar.set_description(f'{img_path}')
        img, PREDS = find_face(str(img_path))
        if isinstance(PREDS, dict):  # if faces in PRED
            whoes, scores, colores = [], [], []
            for pred in PREDS.values():
                xmin, ymin, xmax, ymax = pred['facial_area']

                padding = 20
                crop_face = img[ymin - padding:ymax + padding, xmin - padding:xmax + padding]

                unknown = Person(img=crop_face)
                score = recog_model.cos_dist(unknown, [person1, person2], img_path, show=False)
                whoes.append(unknown.label)
                scores.append(score)
                colores.append(unknown.color)

            marked_img = draw_face(img, PREDS, colores=colores, labels=whoes, scores=scores)

            # cv2.imshow("marked_img", marked_img)
            # cv2.waitKey()
            cv2.imwrite(str(new_img_dir_path / img_path.name), marked_img)
        else:  # if haven't faces
            cv2.imwrite(str(new_img_dir_path / img_path.name), img)
