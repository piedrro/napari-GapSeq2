import traceback
import numpy as np
import cv2
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
import multiprocessing
from multiprocessing import Pool, cpu_count, shared_memory
import os
from functools import partial
from napari_gapseq2._widget_utils_worker import Worker
from qtpy.QtWidgets import QFileDialog
import math
import json
import matplotlib.pyplot as plt
import json
from datetime import datetime


def transform_image(img, transform_matrix, progress_callback=None):

    w, h = img.shape[-2:]

    n_frames = img.shape[0]
    n_segments = math.ceil(n_frames / 100)
    image_splits = np.array_split(img, n_segments)

    transformed_image = []

    iter = 0

    for index, image in enumerate(image_splits):
        image = np.moveaxis(image, 0, -1)
        image = cv2.warpPerspective(image, transform_matrix, (h, w), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        # image = np.moveaxis(image, -1, 0)

        transformed_image.append(image)
        iter += 250
        progress = int((iter / n_frames) * 100)

        if progress_callback is not None:
            progress_callback(progress)

    transformed_image = np.dstack(transformed_image)
    transformed_image = np.moveaxis(transformed_image, -1, 0)

    return transformed_image


class _tranform_utils:

    def normalize_image(self, img, norm_method="minmax"):

        if norm_method == "minmax":
            img = img - np.min(img)
            img = img / np.max(img)
        elif norm_method == "mean":
            img = img - np.mean(img)
            img = img / np.std(img)

        return img


    def compute_transform_matrix(self):

        try:
            if self.dataset_dict != {}:

                dataset_name = self.tform_compute_dataset.currentText()
                target_channel = self.tform_compute_target_channel.currentText()
                reference_channel = self.tform_compute_ref_channel.currentText()

                target_locs = None
                reference_locs = None

                if dataset_name in self.localisation_dict["fiducials"].keys():

                    fiducial_dict = self.localisation_dict["fiducials"][dataset_name]

                    if target_channel.lower() in fiducial_dict.keys():
                        target_locs = fiducial_dict[target_channel.lower()]["localisations"]

                    if reference_channel.lower() in fiducial_dict.keys():
                        reference_locs = fiducial_dict[reference_channel.lower()]["localisations"]

                if len(reference_locs) > 0 and len(target_locs) > 0:

                    reference_points = [[loc.x, loc.y] for loc in reference_locs]
                    target_points = [[loc.x, loc.y] for loc in target_locs]

                    reference_points, target_points = self.compute_registration_keypoints(reference_points, target_points)

                    reference_points = np.array(reference_points)
                    target_points = np.array(target_points)

                    self.transform_matrix, _ = cv2.findHomography(target_points, reference_points, cv2.RANSAC)

                    print(f"Transform matrix: {self.transform_matrix}")

                    if self.save_tform.isChecked():
                        self.save_transform_matrix()

        except:
            print(traceback.format_exc())
            pass


    def save_transform_matrix(self):

        try:

            if self.transform_matrix is not None:

                # get save file name and path
                date = datetime.now().strftime("%y%m%d")
                file_name = f'gapseq_transform_matrix-{date}.txt'

                dataset_name = self.tform_compute_dataset.currentText()
                channel_name = self.tform_compute_target_channel.currentText()

                path = self.dataset_dict[dataset_name][channel_name.lower()]["path"]
                path_directory = os.path.dirname(path)

                tform_path = os.path.join(path_directory, file_name)

                tform_path = QFileDialog.getSaveFileName(self, 'Save transform matrix', tform_path, 'Text files (*.txt)')[0]

                if tform_path != "":
                    print(f"Saving transform matrix to {tform_path}")

                    with open(tform_path, 'w') as filehandle:
                        json.dump(self.transform_matrix.tolist(), filehandle)

        except:
            print(traceback.format_exc())
            pass


    def _apply_transform_matrix_cleanup(self):

        self.tform_apply_progressbar.setValue(0)
        self.update_active_image(channel=self.active_channel)

    def _apply_transform_matrix(self, progress_callback=None):

        print("Applying transform matrix...")

        try:

            if self.dataset_dict != {}:

                from qtpy.QtWidgets import QComboBox
                self.tform_apply_target = self.findChild(QComboBox, 'tform_apply_target')

                apply_channel = self.tform_apply_target.currentText()

                if "donor" in apply_channel.lower():
                    ref_emission = "d"
                else:
                    ref_emission = "a"

                target_images = []
                total_frames = 0
                iter = 0

                for dataset_name, dataset_dict in self.dataset_dict.items():
                    for channel_name, channel_dict in dataset_dict.items():
                        channel_ref = channel_dict["channel_ref"]
                        channel_emission = channel_ref[-1].lower()
                        if channel_emission == ref_emission:
                            n_frames = channel_dict["data"].shape[0]
                            total_frames += n_frames
                            target_images.append({"dataset_name": dataset_name,"channel_name": channel_name})

                for i in range(len(target_images)):

                    dataset_name = target_images[i]["dataset_name"]
                    channel_name = target_images[i]["channel_name"]

                    img = self.dataset_dict[dataset_name][channel_name.lower()]["data"].copy()

                    def transform_progress(progress):
                        nonlocal iter
                        iter += progress
                        progress = int((iter / total_frames) * 100)
                        progress_callback.emit(progress)

                    img = transform_image(img, self.transform_matrix,progress_callback=transform_progress)
                    self.dataset_dict[dataset_name][channel_name.lower()]["data"] = img

        except:
            print(traceback.format_exc())
            pass


    def apply_transform_matrix(self):

        try:

            if self.dataset_dict != {}:

                if hasattr(self, "transform_matrix") == False:

                    print("No transform matrix loaded.")

                else:

                    if self.transform_matrix is None:

                        print("No transform matrix loaded.")

                    else:
                        worker = Worker(self._apply_transform_matrix)
                        worker.signals.progress.connect(partial(self.gapseq_progress, progress_bar=self.tform_apply_progressbar))
                        worker.signals.finished.connect(self._apply_transform_matrix_cleanup)
                        self.threadpool.start(worker)

        except:
            print(traceback.format_exc())
            pass


    def import_transform_matrix(self):

        try:

            desktop = os.path.expanduser("~/Desktop")
            path, filter = QFileDialog.getOpenFileName(self, "Open Files", desktop, "Files (*.txt *.mat)")

            self.transform_matrix = None

            if path != "":
                if os.path.isfile(path) == True:
                    if path.endswith(".txt"):
                        with open(path, 'r') as f:
                            transform_matrix = json.load(f)

                    if path.endswith(".mat"):
                        from pymatreader import read_mat
                        transform_matrix = read_mat(path)["TFORM"]["tdata"]["T"].T

                    transform_matrix = np.array(transform_matrix, dtype=np.float64)

                    if transform_matrix.shape == (3, 3):
                        self.transform_matrix = transform_matrix

                        print(f"Loaded transformation matrix:\n{transform_matrix}")

                    else:
                        print("Transformation matrix is wrong shape, should be (3,3)")

        except:
            print(traceback.format_exc())
            pass




