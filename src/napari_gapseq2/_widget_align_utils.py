import traceback
import numpy as np
import cv2
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
import multiprocessing
from multiprocessing import Pool, cpu_count, shared_memory
import os
from functools import partial
from napari_gapseq2._widget_utils_compute import Worker
from qtpy.QtWidgets import QFileDialog
import math
import json
import matplotlib.pyplot as plt
import json
from datetime import datetime
from napari_gapseq2._widget_transform_utils import transform_image
from scipy.optimize import least_squares

class _align_utils:

    def update_align_reference_channel(self):

        try:

            datast_name = self.gapseq_dataset_selector.currentText()

            if datast_name in self.dataset_dict.keys():

                fret_modes = [self.dataset_dict[datast_name][channel]["FRET"] for channel in self.dataset_dict[datast_name].keys()]
                channel_refs = [self.dataset_dict[datast_name][channel]["channel_ref"] for channel in self.dataset_dict[datast_name].keys()]

                channel_refs = list(set(channel_refs))
                fret_mode = list(set(fret_modes))[0]

                self.align_reference_channel.clear()

                if fret_mode == True:
                    if "dd" in channel_refs:
                        self.align_reference_channel.addItem("Donor")
                    else:
                        self.align_reference_channel.addItem("Acceptor")
                else:
                    for channel in channel_refs:
                        self.align_reference_channel.addItem(channel.upper())

        except:
            pass

    def affine_transform_matrix(self, points_src, points_dst):
        # Function to optimize
        def min_func(params):
            a, b, c, d, e, f = params
            transformed = np.dot(points_src, np.array([[a, b], [c, d]])) + np.array([e, f])
            return np.ravel(transformed - points_dst)

        # Initial guess
        x0 = np.array([1, 0, 0, 1, 0, 0])

        # Solve using least squares
        result = least_squares(min_func, x0)

        # Construct the transformation matrix
        a, b, c, d, e, f = result.x
        matrix = np.array([[a, b, e], [c, d, f], [0, 0, 1]])

        return matrix


    def _align_datasets_cleanup(self):

        self.update_active_image()
        self.align_progressbar.setValue(0)
        self.gapseq_align_datasets.setEnabled(True)

    def _align_datasets(self, progress_callback):

        try:

            reference_dataset = self.align_reference_dataset.currentText()
            reference_channel = self.align_reference_channel.currentText()

            dataset_list = list(self.dataset_dict.keys())
            dataset_list.remove(reference_dataset)

            total_frames = 0
            for dataset in dataset_list:
                for channel_name, channel_dict in self.dataset_dict[dataset].items():
                    total_frames += channel_dict["data"].shape[0]


            dst_locs = self.localisation_dict["fiducials"][reference_dataset][reference_channel.lower()]["localisations"].copy()

            iter = 0

            for dataset in dataset_list:

                src_locs = self.localisation_dict["fiducials"][dataset][reference_channel.lower()]["localisations"].copy()

                dst_pts = [[loc.x, loc.y] for loc in dst_locs]
                src_pts = [[loc.x, loc.y] for loc in src_locs]

                dst_pts = np.array(dst_pts).astype(np.float32)
                src_pts = np.array(src_pts).astype(np.float32)

                bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

                matches = bf.match(dst_pts, src_pts)
                matches = sorted(matches, key=lambda x: x.distance)

                dst_pts = np.float32([dst_pts[m.queryIdx] for m in matches]).reshape(-1, 2)
                src_pts = np.float32([src_pts[m.trainIdx] for m in matches]).reshape(-1, 2)

                transform_matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if transform_matrix.shape == (3,3):

                    for channel_name, channel_dict in self.dataset_dict[dataset].items():

                        print(f"Aligning {dataset} {channel_name}...")

                        img = channel_dict["data"].copy()

                        def transform_progress(progress):
                            nonlocal iter
                            iter += progress
                            progress = int((iter / total_frames) * 100)
                            progress_callback.emit(progress)

                        img = transform_image(img, transform_matrix, progress_callback=transform_progress)

                        self.dataset_dict[dataset][channel_name.lower()]["data"] = img.copy()

        except:
            print(traceback.format_exc())
            self.align_progressbar.setValue(0)
            self.gapseq_align_datasets.setEnabled(True)
            pass



    def align_datasets(self):

        try:

            if self.dataset_dict != {}:

                reference_dataset = self.align_reference_dataset.currentText()
                reference_channel = self.align_reference_channel.currentText()

                missing_fiducial_list = []

                for dataset_name in self.dataset_dict.keys():
                    if dataset_name not in self.localisation_dict["fiducials"].keys():
                        missing_fiducial_list.append(dataset_name)
                    else:
                        if reference_channel.lower() not in self.localisation_dict["fiducials"][dataset_name].keys():
                            missing_fiducial_list.append(dataset_name)
                        else:
                            localisation_dict =  self.localisation_dict["fiducials"][dataset_name][reference_channel.lower()]

                            if "fitted" not in localisation_dict.keys():
                                missing_fiducial_list.append(dataset_name)
                            else:
                                if localisation_dict["fitted"] == False:
                                    missing_fiducial_list.append(dataset_name)

                if len(missing_fiducial_list) > 0:
                    missing_fiducial_list = ", ".join(missing_fiducial_list)
                    print(f"Missing fitted {reference_channel} fiducials for {missing_fiducial_list}")
                else:

                    align_dict = {}
                    for dataset_name in self.dataset_dict.keys():
                        localisations = self.localisation_dict["fiducials"][dataset_name][reference_channel.lower()]["localisations"]
                        align_dict[dataset_name] = localisations

                    self.align_progressbar.setValue(0)
                    self.gapseq_align_datasets.setEnabled(False)

                    worker = Worker(self._align_datasets)
                    worker.signals.progress.connect(partial(self.gapseq_progress, progress_bar=self.align_progressbar))
                    worker.signals.finished.connect(self._align_datasets_cleanup)
                    self.threadpool.start(worker)

        except:
            print(traceback.format_exc())
            self.align_progressbar.setValue(0)
            self.gapseq_align_datasets.setEnabled(True)
            pass