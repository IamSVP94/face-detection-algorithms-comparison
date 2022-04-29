import time

import cv2
import numpy as np
from pathlib import Path
import onnxruntime as ort
from src.utils import PARENT_DIR, draw_face
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist


class ONNXRecognator:
    def __init__(self, onnx_path=PARENT_DIR / 'src/IResNet/model/R100_Glint360K.onnx'):
        print(f'Inference "{onnx_path}" with {ort.get_device()}')
        self.path = onnx_path
        self.net = ort.InferenceSession(self.path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.size = tuple(self.net.get_inputs()[0].shape[-2:])
        self.img_save_dir = PARENT_DIR / 'temp/'
        self.durations = []

    def embedding(self, blob):
        return self.net.run(None, {self.net.get_inputs()[0].name: blob})[0]

    def cos_dist(self, unknown, knowns, img_path=None, mode='mean', metric='euclidean', treshhold=17, show=False):
        # metric = 'cosine', treshhold = 0.9
        # metric = 'euclidean', treshhold = 17

        distances1, distances2, distances3, distances4 = [], [], [], []
        for person in knowns:
            start = time.perf_counter_ns()
            distances1.append(cdist(unknown.embeding0, person.embeding0, metric=metric)[0][0])
            self.durations.append((time.perf_counter_ns() - start) / (10 ** 6))

            start = time.perf_counter_ns()
            distances2.append(cdist(unknown.embeding0, person.embeding1, metric=metric)[0][0])
            self.durations.append((time.perf_counter_ns() - start) / (10 ** 6))

            start = time.perf_counter_ns()
            distances3.append(cdist(unknown.embeding1, person.embeding0, metric=metric)[0][0])
            self.durations.append((time.perf_counter_ns() - start) / (10 ** 6))

            start = time.perf_counter_ns()
            distances4.append(cdist(unknown.embeding1, person.embeding1, metric=metric)[0][0])
            self.durations.append((time.perf_counter_ns() - start) / (10 ** 6))

        # distances1 = [cdist(unknown.embeding0, person.embeding0, metric=metric)[0][0] for person in knowns]
        # distances2 = [cdist(unknown.embeding0, person.embeding1, metric=metric)[0][0] for person in knowns]
        # distances3 = [cdist(unknown.embeding1, person.embeding0, metric=metric)[0][0] for person in knowns]
        # distances4 = [cdist(unknown.embeding1, person.embeding1, metric=metric)[0][0] for person in knowns]

        distances = []
        for idx, (dist1, dist2, dist3, dist4) in enumerate(zip(distances1, distances2, distances3, distances4)):
            if mode == 'sum':
                distances.append(dist1 + dist2 + dist3 + dist4)
            elif mode == 'mean':
                distances.append((dist1 + dist2 + dist3 + dist4) / 4)
        who_near = np.argmin(distances)
        label = knowns[who_near].label if distances[who_near] <= treshhold else 'unknown'

        if show:
            fig, ax = plt.subplots(nrows=2, ncols=len(knowns) + 1)
            fig.suptitle(f'metric={metric}\nthreshhhold={treshhold}\n')
            for idx, person in enumerate(knowns):
                ax[0][idx].imshow(person.img[:, :, ::-1])
                ax[0][idx].set_title(f'{person.label}')

                ax[1][idx].imshow(person.img_profile[:, :, ::-1])
                ax[1][idx].set_title(f'{round(distances[idx], 7)}')
            else:
                ax[0][len(knowns)].imshow(unknown.img[:, :, ::-1])
                ax[0][len(knowns)].set_title(f'"{label}"')

                ax[1][len(knowns)].imshow(unknown.img[:, :, ::-1])
                ax[1][len(knowns)].set_title(f'{who_near + 1} person\n{round(distances[who_near], 7)}')

            fig.show()
            fig.savefig(f'{self.img_save_dir / img_path.stem}_{idx}_{label}.jpg')
        return label


recog_model = ONNXRecognator()


class Person:
    def __init__(self, path: [str, Path] = None, img=None, net=None, label='unknown', path_profile=None):
        self.path = str(path) if path else None
        if img is None:
            self.img = cv2.imread(self.path)
        else:
            self.img = img

        self.label = label
        self.path_profile = path_profile
        if net:
            self.blob_size = net.size
        else:
            self.blob_size = (112, 112)

        self.embeding0 = net.embedding(self._get_blob(self.img))
        self.embeding1 = net.embedding(self._get_blob(cv2.flip(self.img, 1)))

        if self.path_profile:
            self.img_n = 2
            self.img_profile = cv2.imread(str(self.path_profile))
            self.embeding_profile0 = net.embedding(self._get_blob(self.img_profile))
            self.embeding_profile1 = net.embedding(self._get_blob(cv2.flip(self.img_profile, 1)))
        else:
            self.img_n = 1
            self.img_profile = None
            self.embeding_profile0, self.embeding_profile1 = None, None

    def _get_blob(self, img, scalefactor=1.0 / 128.0, mean=(127.5, 127.5, 127.5)):
        return cv2.dnn.blobFromImage(
            image=img, swapRB=True, scalefactor=scalefactor, size=self.blob_size, mean=mean,
        )
