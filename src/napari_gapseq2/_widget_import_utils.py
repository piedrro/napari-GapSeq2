import traceback
import numpy as np
import os
from PIL import Image
from qtpy.QtWidgets import QFileDialog
import multiprocessing
from multiprocessing import Pool, cpu_count, shared_memory
from napari_gapseq2._widget_utils_worker import Worker
from functools import partial
from PyQt5.QtWidgets import QApplication, QPushButton, QWidget, QVBoxLayout, QShortcut
from PyQt5.QtGui import QKeySequence
import json
import copy
import time
import scipy.ndimage
import multiprocessing
from multiprocessing import Process, shared_memory, Pool
import copy
from functools import partial
import cv2
import math

def import_image_frame(import_job):

    frame = None

    try:

        frame_index = import_job["frame_index"]
        path = import_job["path"]

        with Image.open(path) as img:
            img.seek(frame_index)
            frame = img.copy()

        frame = np.array(frame)

    except:
        print(traceback.format_exc())

    return frame_index, frame

def transform_image_frame(dat):

    try:

        shared_mem = shared_memory.SharedMemory(name=dat["shared_memory_name"])
        np_array = np.ndarray(dat["shape"], dtype=dat["dtype"], buffer=shared_mem.buf)

        # Perform preprocessing steps and overwrite original image
        img = np_array[dat["frame_index"]]
        transform_matrix = dat["transform_matrix"]

        h, w = img.shape[:2]

        img = cv2.warpPerspective(img, transform_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

        # overwrite the shared memory block
        np_array[dat["frame_index"]] = img

        # Ensure to close shared memory in child processes
        shared_mem.close()

    except:
        print(traceback.format_exc())
        pass



def transform_image(img, transform_matrix, progress_callback=None):

    print("Applying transform matrix...")

    h, w = img.shape[:2]

    n_frames = img.shape[0]
    n_segments = math.ceil(n_frames / 100)
    image_splits = np.array_split(img, n_segments)

    transformed_image = []

    iter = 0

    for index, image in enumerate(image_splits):
        image = np.moveaxis(image, 0, -1)
        image = cv2.warpPerspective(image, transform_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

        transformed_image.append(image)
        iter += 250
        progress = int((iter / n_frames) * 100)

        if progress_callback is not None:
            progress_callback(progress)

    transformed_image = np.dstack(transformed_image)
    transformed_image = np.moveaxis(transformed_image, -1, 0)

    return transformed_image




class _import_utils:



    def _apply_transform_matrix_cleanup(self):

        self.tform_progressbar.setValue(0)

        self.update_active_image(channel=self.active_channel)


    def _apply_transform_matrix(self, progress_callback=None):

        print("Applying transform matrix...")

        try:

            if self.dataset_dict != {}:

                target_images = []
                total_frames = 0
                iter = 0
                for dataset_name, dataset_dict in self.dataset_dict.items():
                    for channel_name, channel_dict in dataset_dict.items():
                        channel_ref = channel_dict["channel_ref"]
                        if channel_ref[-1].lower() == "a":
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
                        worker.signals.progress.connect(partial(self.gapseq_progress, progress_bar=self.tform_progressbar))
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







    def _gapseq_import_data_cleanup(self):

        try:

            dataset_names = list(self.dataset_dict.keys())

            self.gapseq_dataset_selector.clear()
            self.gapseq_dataset_selector.addItems(dataset_names)
            self.picasso_dataset.clear()
            self.picasso_dataset.addItems(dataset_names)
            self.export_dataset.clear()
            self.export_dataset.addItems(dataset_names)
            self.cluster_dataset.clear()
            self.cluster_dataset.addItems(dataset_names)

            self.update_channel_select_buttons()
            self.update_active_image()

            self.gapseq_import.setEnabled(True)
            self.update_export_options()

        except:
            print(traceback.format_exc())
            self.gapseq_import.setEnabled(True)
            pass


    def _gapseq_import_data(self, progress_callback=None, path=None):

        try:

            with Image.open(path) as img:
                num_frames = img.n_frames

            n_cpu = multiprocessing.cpu_count()//2

            import_jobs = []

            for frame_index in range(num_frames):
                import_jobs.append({"frame_index": frame_index,"path": path,})


            def callback(*args, offset=0):
                iter.append(1)
                progress = int((len(iter) / num_frames) * 100)
                if progress_callback != None:
                    progress_callback.emit(progress - offset)
                return

            iter = []

            with Pool(n_cpu) as p:

                imported_data = [p.apply_async(import_image_frame, args=(i,), callback=callback) for i in import_jobs]
                imported_data = [r.get() for r in imported_data]

                imported_data = sorted(imported_data, key=lambda x: x[0])

                frame_index, image_list = zip(*imported_data)

            self.postprocess_image_list(image_list, path)

        except:
            print(traceback.format_exc())
            pass


    def register_keyboard_shortcuts(self, key, func, func_kwargs=None, overwrite=True):

        if func_kwargs is None:
            func_kwargs = {}

        # Creating a partial function with the provided keyword arguments
        wrapped_func = partial(func, **func_kwargs)

        # Setting a name for the wrapped function
        if isinstance(func, partial):
            base_name = func.func.__name__
        else:
            base_name = func.__name__
        wrapped_func.__name__ = base_name + "_" + key

        # Binding the key with the wrapped function
        self.viewer.bind_key(key, wrapped_func, overwrite=overwrite)



    def postprocess_image_list(self, image_list, path):

        image_dict = {}
        fiducials_dict = {}

        try:

            image_list = list(image_list)

            import_mode = self.gapseq_import_mode.currentText()
            channel_layout = self.gapseq_channel_layout.currentText()
            alex_first_frame = self.gapseq_alex_first_frame.currentText()
            dataset_name = self.gapseq_dataset_name.text()

            for image_index, image in enumerate(image_list):

                if import_mode in ["Donor", "Acceptor", "DA", "DD", "AA", "AD"]:

                    if import_mode.lower() not in image_dict.keys():
                        image_dict[import_mode.lower()] = {"data":[]}

                    image_dict[import_mode.lower()]["data"].append(image)

                elif import_mode == "FRET":

                    if "donor" not in image_dict.keys():
                        image_dict["donor"] = {"data":[]}
                    if "acceptor" not in image_dict.keys():
                        image_dict["acceptor"] = {"data":[]}

                    if channel_layout == "Donor-Acceptor":
                        imageD, imageA = self.split_img(image)
                    else:
                        imageA, imageD = self.split_img(image)

                    image_dict["donor"]["data"].append(imageD)
                    image_dict["acceptor"]["data"].append(imageA)

                elif import_mode == "ALEX":

                    if "ad" not in image_dict.keys():
                        image_dict["ad"] = {"data":[]}
                    if "aa" not in image_dict.keys():
                        image_dict["aa"] = {"data":[]}
                    if "dd" not in image_dict.keys():
                        image_dict["dd"] = {"data":[]}
                    if "da" not in image_dict.keys():
                        image_dict["da"] = {"data":[]}

                    if image_index % 2 == 0:
                        # Even frames [0, 2, 4, ...]
                        if alex_first_frame == "Donor":
                            excitation = "donor"
                        else:
                            excitation = "acceptor"
                    else:
                        # Odd frames [1, 3, 5, ...]
                        if alex_first_frame == "Donor":
                            excitation = "acceptor"
                        else:
                            excitation = "donor"

                    if channel_layout == "Donor-Acceptor":
                        imgD, imgA = self.split_img(image)
                        image_dict[excitation[0]+"d"]["data"].append(imgD)
                        image_dict[excitation[0]+"a"]["data"].append(imgA)
                    else:
                        imgA, imgD = self.split_img(image)
                        image_dict[excitation[0]+"a"]["data"].append(imgA)
                        image_dict[excitation[0]+"d"]["data"].append(imgD)

                image_list[image_index] = None

            for channel in image_dict.keys():
                image_dict[channel]["data"] = np.stack(image_dict[channel]["data"])


            if import_mode in ["Donor", "Acceptor", "DA", "DD", "AA", "AD"]:

                if import_mode == "Donor":
                    image_dict[import_mode.lower()]["excitation"] = "donor"
                    image_dict[import_mode.lower()]["emission"] = "donor"
                    image_dict[import_mode.lower()]["channel_ref"] = "dd"
                    image_dict[import_mode.lower()]["FRET"] = True
                    func = partial(self.update_active_image, channel=import_mode.lower())
                    self.gapseq_show_dd.clicked.connect(func)

                elif import_mode == "Acceptor":
                    image_dict[import_mode.lower()]["excitation"] = "donor"
                    image_dict[import_mode.lower()]["emission"] = "acceptor"
                    image_dict[import_mode.lower()]["channel_ref"] = "da"
                    image_dict[import_mode.lower()]["FRET"] = True
                    func = partial(self.update_active_image, channel=import_mode.lower())
                    self.gapseq_show_dd.clicked.connect(func)

                elif import_mode == "DD":
                    image_dict[import_mode.lower()]["excitation"] = "donor"
                    image_dict[import_mode.lower()]["emission"] = "donor"
                    image_dict[import_mode.lower()]["channel_ref"] = "dd"
                    image_dict[import_mode.lower()]["FRET"] = False
                    func = partial(self.update_active_image, channel=import_mode.lower())
                    self.gapseq_show_dd.clicked.connect(func)

                elif import_mode == "DA":
                    image_dict[import_mode.lower()]["excitation"] = "donor"
                    image_dict[import_mode.lower()]["emission"] = "acceptor"
                    image_dict[import_mode.lower()]["channel_ref"] = "da"
                    image_dict[import_mode.lower()]["FRET"] = False
                    func = partial(self.update_active_image, channel=import_mode.lower())

                    self.gapseq_show_dd.clicked.connect(func)

                elif import_mode == "AD":
                    image_dict[import_mode.lower()]["excitation"] = "acceptor"
                    image_dict[import_mode.lower()]["emission"] = "donor"
                    image_dict[import_mode.lower()]["channel_ref"] = "ad"
                    image_dict[import_mode.lower()]["FRET"] = False
                    func = partial(self.update_active_image, channel=import_mode.lower())
                    self.gapseq_show_dd.clicked.connect(func)

                elif import_mode == "AA":
                    image_dict[import_mode.lower()]["excitation"] = "acceptor"
                    image_dict[import_mode.lower()]["emission"] = "acceptor"
                    image_dict[import_mode.lower()]["channel_ref"] = "aa"
                    image_dict[import_mode.lower()]["FRET"] = False
                    func = partial(self.update_active_image, channel=import_mode.lower())
                    self.gapseq_show_dd.clicked.connect(func)

            elif import_mode == "FRET":

                image_dict["donor"]["excitation"] = "donor"
                image_dict["donor"]["emission"] = "donor"
                image_dict["donor"]["channel_ref"] = "dd"
                image_dict["donor"]["FRET"] = True

                self.gapseq_show_dd.clicked.connect(partial(self.update_active_image, channel="donor"))


                image_dict["acceptor"]["excitation"] = "donor"
                image_dict["acceptor"]["emission"] = "acceptor"
                image_dict["acceptor"]["channel_ref"] = "da"
                image_dict["acceptor"]["FRET"] = True

                self.gapseq_show_da.clicked.connect(partial(self.update_active_image, channel="acceptor"))

            elif import_mode == "ALEX":

                image_dict["dd"]["excitation"] = "donor"
                image_dict["dd"]["emission"] = "donor"
                image_dict["da"]["excitation"] = "donor"
                image_dict["da"]["emission"] = "acceptor"
                image_dict["aa"]["excitation"] = "acceptor"
                image_dict["aa"]["emission"] = "acceptor"
                image_dict["ad"]["excitation"] = "acceptor"
                image_dict["ad"]["emission"] = "donor"

                image_dict["dd"]["channel_ref"] = "dd"
                image_dict["da"]["channel_ref"] = "da"
                image_dict["aa"]["channel_ref"] = "aa"
                image_dict["ad"]["channel_ref"] = "ad"

                image_dict["dd"]["FRET"] = False
                image_dict["da"]["FRET"] = False
                image_dict["aa"]["FRET"] = False
                image_dict["ad"]["FRET"] = False

                funct_aa = partial(self.update_active_image, channel="aa")
                funct_ad = partial(self.update_active_image, channel="ad")
                funct_da = partial(self.update_active_image, channel="da")
                funct_dd = partial(self.update_active_image, channel="dd")

                self.gapseq_show_dd.clicked.connect(funct_dd)
                self.gapseq_show_da.clicked.connect(funct_da)
                self.gapseq_show_aa.clicked.connect(funct_aa)
                self.gapseq_show_ad.clicked.connect(funct_ad)

            for channel in image_dict.keys():

                image_dict[channel]["path"] = path
                image_dict[channel]["import_mode"] = import_mode

                if import_mode in ["Donor", "Acceptor", "DA", "DD", "AA", "AD"]:
                    image_dict[channel]["channel_layout"] = None
                else:
                    image_dict[channel]["channel_layout"] = channel_layout
                if import_mode == "ALEX":
                    image_dict[channel]["alex_first_frame"] = alex_first_frame
                else:
                    image_dict[channel]["alex_first_frame"] = None

            for channel in image_dict.keys():
                if channel not in fiducials_dict.keys():
                    fiducials_dict[channel] = {}

            self.dataset_dict[dataset_name] = image_dict

            if dataset_name not in self.localisation_dict["fiducials"].keys():
                self.localisation_dict["fiducials"][dataset_name] = fiducials_dict

        except:
            print(traceback.format_exc())


    def gapseq_import_data(self):

        try:

            desktop = os.path.expanduser("~/Desktop")
            path = QFileDialog.getOpenFileName(self, 'Open file', desktop, "Image files (*.tif *.tiff)")[0]

            if path != "":

                self.gapseq_import.setEnabled(False)

                worker = Worker(self._gapseq_import_data, path=path)
                worker.signals.progress.connect(partial(self.gapseq_progress,
                    progress_bar=self.gapseq_import_progressbar))
                worker.signals.finished.connect(self._gapseq_import_data_cleanup)
                self.threadpool.start(worker)

        except:
            self.gapseq_import.setEnabled(True)
            pass

    def split_img(self, img):

        imgL = img[:, : img.shape[-1] // 2]
        imgR = img[:, img.shape[-1] // 2:]

        return imgL, imgR