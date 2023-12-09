import traceback
import numpy as np
import os
from PIL import Image
from qtpy.QtWidgets import QFileDialog
import multiprocessing
from multiprocessing import Pool, cpu_count, shared_memory
from napari_gapseq2._widget_utils_compute import Worker
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
import tifffile
import concurrent.futures
import matplotlib.pyplot as plt

def import_image_data(dat):

    try:

        path = dat["path"]
        frame_index = dat["frame_index"]
        channels = dat["channels"]
        channel_frame = dat["channel_frame"]
        channel_images = dat["channel_images"]
        image_shape = dat["image_shape"]

        with Image.open(path) as img:
            img.seek(frame_index)
            img_frame = img.copy()

        img_frame = np.array(img_frame)

        if len(channels) == 1:
            img_frames = [img_frame]
        else:
            img_frames = np.array_split(img_frame, 2, axis=-1)

        for channel, channel_img in zip(channels, img_frames):

            shared_mem = channel_images[channel]
            np_array = np.ndarray(image_shape, dtype=dat["dtype"], buffer=shared_mem.buf)
            np_array[channel_frame] = channel_img

    except:
        print(traceback.format_exc())
        pass








class _import_utils:

    def create_shared_image(self, image_shape, image_dtype):

        shared_mem = None

        try:

            image_shape = tuple(np.array(image_shape).astype(int))

            image_size = np.prod(image_shape) * (image_dtype.itemsize)
            image_size = int(image_size)

            image_size_gb = image_size / 1000000000

            shared_mem = shared_memory.SharedMemory(create=True, size=image_size)
            shared_memory_name = shared_mem.name
            shared_image = np.ndarray(image_shape, dtype=image_dtype, buffer=shared_mem.buf)

        except:
            print(traceback.format_exc())
            pass

        return shared_mem

    def get_image_info(self, path):

        with Image.open(path) as img:
            n_frames = img.n_frames
            page_shape = img.size

            img.seek(0)
            np_img = np.array(img)
            dtype = np_img.dtype

        image_shape = (n_frames, page_shape[1], page_shape[0])

        return n_frames, image_shape, dtype

    def populate_import_lists(self, progress_callback=None, paths=[]):

        image_list = []
        import_dict = {}
        shared_images = {}

        try:

            import_mode = self.gapseq_import_mode.currentText()
            import_limit = self.gapseq_import_limt.currentText()
            channel_layout = self.gapseq_channel_layout.currentText()
            alex_first_frame = self.gapseq_alex_first_frame.currentText()

            for path_index, path in enumerate(paths):

                file_name = os.path.basename(path)

                dataset_name = file_name

                if dataset_name not in shared_images.keys():
                    shared_images[dataset_name] = {}

                n_frames, image_shape, dtype = self.get_image_info(path)

                if import_mode.lower() in ["donor", "acceptor", "dd", "da", "ad", "aa"]:

                    if import_limit != "None":
                        import_limit = int(self.gapseq_import_limt.currentText())
                    else:
                        import_limit = n_frames

                    image_shape = (import_limit, image_shape[1], image_shape[2])

                    frame_list = list(range(n_frames))[:import_limit]

                    unique_frames = np.unique(frame_list)
                    n_frames = len(unique_frames)

                    channel_names = [import_mode.lower()]
                    channel_list = [channel_names] * n_frames

                    channel_images = {}
                    for channel in channel_names:
                        shared_image = self.create_shared_image(image_shape, dtype)
                        channel_images[channel] = shared_image
                        shared_images[dataset_name][channel] = shared_image

                    image_dict = {"path": path,
                                  "dataset_name": dataset_name,
                                  "n_frames": n_frames,
                                  "channel_names": channel_names,
                                  "channel_list": channel_list,
                                  "frame_list": frame_list,
                                  "channel_frame_list": frame_list,
                                  "channel_images": channel_images,
                                  "image_shape": image_shape,
                                  "channel_layout": channel_layout,
                                  "alex_first_frame": alex_first_frame,
                                  "dtype": dtype,
                                  "import_mode": import_mode.lower()}

                    image_list.append(image_dict)

                elif import_mode.lower() == "fret":

                    if import_limit != "None":
                        import_limit = int(self.gapseq_import_limt.currentText())
                    else:
                        import_limit = n_frames

                    frame_list = list(range(n_frames))[:import_limit]

                    if channel_layout.lower() == "donor-acceptor":
                        channel_names = ["donor", "acceptor"]
                    else:
                        channel_names = ["acceptor", "donor"]

                    image_shape = (import_limit, image_shape[1], image_shape[2]//2)

                    unique_frames = np.unique(frame_list)
                    n_frames = len(unique_frames)

                    channel_list = [channel_names] * n_frames

                    channel_images = {}
                    for channel in channel_names:
                        shared_image = self.create_shared_image(image_shape, dtype)
                        channel_images[channel] = shared_image
                        shared_images[dataset_name][channel] = shared_image

                    image_dict = {"path": path,
                                  "dataset_name": dataset_name,
                                  "n_frames": n_frames,
                                  "channel_names": channel_names,
                                  "channel_list": channel_list,
                                  "frame_list": frame_list,
                                  "channel_frame_list": frame_list,
                                  "channel_images": channel_images,
                                  "image_shape": image_shape,
                                  "channel_layout": channel_layout,
                                  "alex_first_frame": alex_first_frame,
                                  "dtype": dtype,
                                  "import_mode": import_mode.lower()}

                    image_list.append(image_dict)

                elif import_mode.lower() == "alex":

                    if import_limit != "None":
                        import_limit = int(self.gapseq_import_limt.currentText())
                    else:
                        import_limit = n_frames

                    frame_list = list(range(n_frames))

                    even_frames = frame_list[::2][:import_limit]
                    odd_frames = frame_list[1::2][:import_limit]

                    frame_list = np.unique(np.concatenate([even_frames, odd_frames]))
                    frame_list = np.sort(frame_list).tolist()
                    channel_frame_list = np.repeat(np.arange(len(frame_list)//2), 2)

                    n_frames = len(frame_list)
                    image_shape = (n_frames//2, image_shape[1], image_shape[2]//2)

                    channel_list = []

                    for frame in frame_list:
                        if frame % 2 == 0:
                            if alex_first_frame.lower() == "donor":
                                channel_ex = "d"
                            else:
                                channel_ex = "a"
                        else:
                            if alex_first_frame.lower() == "donor":
                                channel_ex = "a"
                            else:
                                channel_ex = "d"

                        if channel_layout.lower() == "donor-acceptor":
                            channel_names = [f"{channel_ex}d", f"{channel_ex}a"]
                        else:
                            channel_names = [f"{channel_ex}a", f"{channel_ex}d"]

                        channel_list.append(channel_names)

                    channel_names = np.unique(channel_list)

                    channel_images = {}
                    for channel in channel_names:
                        shared_image = self.create_shared_image(image_shape, dtype)
                        channel_images[channel] = shared_image
                        shared_images[dataset_name][channel] = shared_image

                    image_dict = {"path": path,
                                  "dataset_name": dataset_name,
                                  "n_frames": n_frames,
                                  "channel_names": channel_names,
                                  "channel_list": channel_list,
                                  "frame_list": frame_list,
                                  "channel_frame_list": channel_frame_list,
                                  "channel_images": channel_images,
                                  "image_shape": image_shape,
                                  "channel_layout": channel_layout,
                                  "alex_first_frame": alex_first_frame,
                                  "dtype": dtype,
                                  "import_mode": import_mode.lower()}

                    image_list.append(image_dict)

                channel_layout = self.gapseq_channel_layout.currentText()
                alex_first_frame = self.gapseq_alex_first_frame.currentText()

                if dataset_name not in import_dict.keys():
                    import_dict[dataset_name] = {"path":path,
                                                 "import_mode": import_mode.lower(),
                                                 "import_limit": import_limit,
                                                 "channel_layout": channel_layout,
                                                 "alex_first_frame": alex_first_frame,
                                                 "image_shape": image_shape,
                                                 "dtype": dtype,}

        except:
            print(traceback.format_exc())

        return image_list, shared_images, import_dict

    def populate_comute_jobs(self, image_list):

        compute_jobs = []

        for image_dict in image_list:

            frame_list = np.unique(image_dict["frame_list"])
            channel_list = image_dict["channel_list"]
            channel_frame_list = image_dict["channel_frame_list"]

            for (frame_index, channels, channel_frame) in zip(frame_list,channel_list, channel_frame_list):

                compute_job = {"frame_index": frame_index,
                               "channels":channels,
                               "channel_frame": channel_frame}

                compute_job = {**compute_job, **image_dict}
                compute_jobs.append(compute_job)

        return compute_jobs

    def process_compute_jobs(self, compute_jobs, progress_callback=None):

        cpu_count = int(multiprocessing.cpu_count() * 0.75)
        timeout_duration = 10  # Timeout in seconds

        with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
            # Submit all jobs and store the future objects
            futures = {executor.submit(import_image_data, job): job for job in compute_jobs}

            iter = 0
            for future in concurrent.futures.as_completed(futures):
                job = futures[future]
                try:
                    result = future.result(timeout=timeout_duration)  # Process result here
                except concurrent.futures.TimeoutError:
                    # print(f"Task {job} timed out after {timeout_duration} seconds.")
                    pass
                except Exception as e:
                    # print(f"Error occurred in task {job}: {e}")  # Handle other exceptions
                    pass

                # Update progress
                iter += 1
                progress = int((iter / len(compute_jobs)) * 100)
                progress_callback.emit(progress)  # Emit the signal

    def populate_dataset_dict(self, import_dict):

        try:

            for dataset_name, dataset_dict in import_dict.items():

                image_dict = {}

                path = dataset_dict["path"]
                image_shape = dataset_dict["image_shape"]
                dtype = dataset_dict["dtype"]
                import_mode = dataset_dict["import_mode"]
                channel_layout = dataset_dict["channel_layout"]
                alex_first_frame = dataset_dict["alex_first_frame"]

                dataset_images = self.shared_images[dataset_name]

                channel_names = dataset_images.keys()

                for channel_name, shared_mem in dataset_images.items():

                    image = np.ndarray(image_shape, dtype=dtype, buffer=shared_mem.buf).copy()

                    shared_mem.close()
                    shared_mem.unlink()

                    if channel_name not in image_dict.keys():
                        image_dict[channel_name] = {"data": []}

                    if channel_name.lower() in ["donor", "acceptor"]:
                        channel_display_name = channel_name.capitalize()
                    else:
                        channel_display_name = channel_name.upper()

                    if channel_name in ["donor", "acceptor", "da", "dd"]:
                        excitation = "d"
                    else:
                        excitation = "a"

                    if channel_name in ["donor", "ad", "dd"]:
                        emission = "d"
                    else:
                        emission = "a"

                    if import_mode.lower() == "fret":
                        fret = True
                    else:
                        fret = False

                    channel_ref = f"{excitation}{emission}"

                    image_dict[channel_name]["data"] = image
                    image_dict[channel_name]["path"] = path
                    image_dict[channel_name]["channel_ref"] = channel_ref
                    image_dict[channel_name]["excitation"] = excitation
                    image_dict[channel_name]["emission"] = emission
                    image_dict[channel_name]["channel_layout"] = channel_layout
                    image_dict[channel_name]["alex_first_frame"] = alex_first_frame
                    image_dict[channel_name]["FRET"] = fret
                    image_dict[channel_name]["import_mode"] = import_mode

                self.dataset_dict[dataset_name] = image_dict

        except:
            pass

    def closed_shared_images(self):

        if hasattr(self, "shared_images"):
            for dataset_name, dataset_dict in self.shared_images.items():
                for channel_name, shared_mem in dataset_dict.items():
                    shared_mem.close()
                    shared_mem.unlink()

    def _gapseq_import_data(self, progress_callback=None, paths=[]):


        try:

            image_list, self.shared_images, import_dict = self.populate_import_lists(paths=paths)

            compute_jobs = self.populate_comute_jobs(image_list)

            self.process_compute_jobs(compute_jobs, progress_callback=progress_callback)

            self.populate_dataset_dict(import_dict)

            self.closed_shared_images()

        except:
            print(traceback.format_exc())
            pass

    def populate_dataset_combos(self):

        try:

            dataset_names = list(self.dataset_dict.keys())

            self.gapseq_dataset_selector.blockSignals(True)
            self.gapseq_dataset_selector.clear()
            self.gapseq_dataset_selector.addItems(dataset_names)
            self.gapseq_dataset_selector.blockSignals(False)

            self.picasso_dataset.blockSignals(True)
            self.picasso_dataset.clear()
            self.picasso_dataset.addItems(dataset_names)
            self.picasso_dataset.blockSignals(False)

            self.export_dataset.blockSignals(True)
            self.export_dataset.clear()
            self.export_dataset.addItems(dataset_names)
            self.export_dataset.blockSignals(False)

            self.cluster_dataset.blockSignals(True)
            self.cluster_dataset.clear()
            self.cluster_dataset.addItems(dataset_names)
            self.cluster_dataset.blockSignals(False)

            self.tform_compute_dataset.blockSignals(True)
            self.tform_compute_dataset.clear()
            self.tform_compute_dataset.addItems(dataset_names)
            self.tform_compute_dataset.blockSignals(False)

            self.gapseq_old_dataset_name.blockSignals(True)
            self.gapseq_old_dataset_name.clear()
            self.gapseq_old_dataset_name.addItems(dataset_names)
            self.gapseq_old_dataset_name.blockSignals(False)

            self.align_reference_dataset.blockSignals(True)
            self.align_reference_dataset.clear()
            self.align_reference_dataset.addItems(dataset_names)
            self.align_reference_dataset.blockSignals(False)


            self.undrift_dataset_selector.blockSignals(True)
            self.undrift_dataset_selector.clear()
            self.undrift_dataset_selector.addItems(dataset_names)
            self.undrift_dataset_selector.blockSignals(False)

            self.colo_dataset.blockSignals(True)
            self.colo_dataset.clear()
            self.colo_dataset.addItems(dataset_names)
            self.colo_dataset.blockSignals(False)

            if len(dataset_names) > 1:
                dataset_names.insert(0, "All Datasets")

            self.traces_export_dataset.blockSignals(True)
            self.traces_export_dataset.clear()
            self.traces_export_dataset.addItems(dataset_names)
            self.traces_export_dataset.blockSignals(False)

            self.filtering_datasets.blockSignals(True)
            self.filtering_datasets.clear()
            self.filtering_datasets.addItems(dataset_names)
            self.filtering_datasets.blockSignals(False)


        except:
            print(traceback.format_exc())

    def initialise_localisation_dict(self):

        if hasattr(self, "localisation_dict"):
            self.localisation_dict = {"bounding_boxes": {}, "fiducials": {}}

        if hasattr(self, "dataset_dict"):
            for dataset_name, dataset_dict in self.dataset_dict.items():

                if dataset_name not in self.localisation_dict.keys():
                    self.localisation_dict["fiducials"][dataset_name] = {}

                fiducial_dict = self.localisation_dict["fiducials"][dataset_name]

                for channel_name, channel_dict in dataset_dict.items():
                    if channel_name not in fiducial_dict.keys():
                        fiducial_dict[channel_name.lower()] = {}

                self.localisation_dict["fiducials"][dataset_name] = fiducial_dict

    def _gapseq_import_data_cleanup(self):

        self.initialise_localisation_dict()
        self.populate_dataset_combos()

        self.update_channel_select_buttons()
        self.update_active_image()
        self.update_export_options()
        self.populate_export_combos()
        self.update_filtering_channels()

        self.update_align_reference_channel()

        self.gapseq_import.setEnabled(True)
        self.gapseq_import_progressbar.setValue(0)

    def gapseq_import_data(self):

        try:

            desktop = os.path.expanduser("~/Desktop")
            paths = QFileDialog.getOpenFileNames(self, 'Open file', desktop, "Image files (*.tif *.tiff)")[0]

            paths = [path for path in paths if path != ""]

            if paths != []:

                self.gapseq_import.setEnabled(False)

                worker = Worker(self._gapseq_import_data, paths=paths)
                worker.signals.progress.connect(partial(self.gapseq_progress,
                    progress_bar=self.gapseq_import_progressbar))
                worker.signals.finished.connect(self._gapseq_import_data_cleanup)
                self.threadpool.start(worker)

        except:
            self.gapseq_import.setEnabled(True)
            pass