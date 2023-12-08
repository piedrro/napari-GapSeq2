import copy
import numpy as np
import traceback
from napari_gapseq2._widget_utils_worker import Worker
from functools import partial
import matplotlib.pyplot as plt
from qtpy.QtWidgets import QFileDialog,QComboBox, QCheckBox
from multiprocessing import Process, shared_memory, Pool
import multiprocessing
from picasso.gaussmle import gaussmle
from picasso import gausslq, lib, localize
import warnings
from numba.core.errors import NumbaPendingDeprecationWarning
import time
import concurrent.futures
import pandas as pd

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=NumbaPendingDeprecationWarning)
np.seterr(divide='ignore', invalid='ignore')

LOCS_DTYPE = [
    ("frame", "u4"),
    ("x", "f4"),
    ("y", "f4"),
    ("photons", "f4"),
    ("sx", "f4"),
    ("sy", "f4"),
    ("bg", "f4"),
    ("lpx", "f4"),
    ("lpy", "f4"),
    ("net_gradient", "f4"),
    ("likelihood", "f4"),
    ("iterations", "i4"),
    # ("loc_index", "u4")
]

def locs_from_fits(identifications, theta, CRLBs, likelihoods, iterations, box):

    box_offset = int(box / 2)
    y = theta[:, 0] + identifications.y - box_offset
    x = theta[:, 1] + identifications.x - box_offset
    lpy = np.sqrt(CRLBs[:, 0])
    lpx = np.sqrt(CRLBs[:, 1])
    locs = np.rec.array(
        (
            identifications.frame,
            x,
            y,
            theta[:, 2],
            theta[:, 5],
            theta[:, 4],
            theta[:, 3],
            lpx,
            lpy,
            identifications.net_gradient,
            likelihoods,
            iterations,
            # identifications.loc_index,
        ),
        dtype=LOCS_DTYPE,
    )
    locs.sort(kind="mergesort", order="frame")
    return locs

def get_loc_from_fit(loc, theta, CRLBs, likelihoods, iterations, box):

    box_offset = int(box / 2)
    y = (theta[:, 0]
         + loc.y - box_offset)
    x = (theta[:, 1] +
         loc.x - box_offset)
    lpy = np.sqrt(CRLBs[:, 0])
    lpx = np.sqrt(CRLBs[:, 1])

    loc.x = x
    loc.y = y
    loc.photons = theta[:, 2]
    loc.sx = theta[:, 5]
    loc.sy = theta[:, 4]
    loc.bg = theta[:, 3]
    loc.lpx = lpx
    loc.lpy = lpy
    loc.net_gradient = loc.net_gradient
    loc.likelihood = likelihoods
    loc.iterations = iterations

    return loc








def create_frame_locs(loc, n_frames):

    frame_locs = []
    for frame_index in range(n_frames):
        frame_loc = copy.deepcopy(loc)
        frame_loc.frame = frame_index
        frame_locs.append(frame_loc)

    frame_locs = np.array(frame_locs, dtype=loc).view(np.recarray)

    return frame_locs


def extract_picasso_spot_metrics(dat):

    spot_metrics = {}

    try:

        # Load data from shared memory
        shared_mem = dat["shared_mem"]
        np_array = np.ndarray(dat["shape"], dtype=dat["dtype"], buffer=shared_mem.buf)

        [x1,x2,y1,y2] = dat["loc_bound"]
        spot_data = [np_array[dat["frame_index"], y1:y2, x1:x2].copy()]
        spot_loc = dat["spot_loc"]
        spot_x = spot_loc[0].x
        spot_y = spot_loc[0].y

        try:
            thetas, CRLBs, likelihoods, iterations = gaussmle(spot_data, eps=0.001, max_it=100, method="sigma")
            loc = locs_from_fits(spot_loc, thetas, CRLBs, likelihoods, iterations, dat["box_size"])
        except:
            loc = None

        # metadata
        spot_metrics["dataset"] = dat["dataset"]
        spot_metrics["channel"] = dat["channel"]
        spot_metrics["frame_index"] = dat["frame_index"]
        spot_metrics["spot_index"] = dat["loc_index"]
        spot_metrics["spot_x"] = spot_x
        spot_metrics["spot_y"] = spot_y
        spot_metrics["box_size"] = dat["box_size"]

        # picasso metrics
        if type(loc) != type(None):

            spot_metrics["spot_photons"] = loc[0].photons
            spot_metrics["spot_bg"] = loc[0].bg
            spot_metrics["spot_sx"] = loc[0].sx
            spot_metrics["spot_sy"] = loc[0].sy
            spot_metrics["spot_lpx"] = loc[0].lpx
            spot_metrics["spot_lpy"] = loc[0].lpy
            spot_metrics["spot_net_gradient"] = loc[0].net_gradient
            spot_metrics["spot_likelihood"] = loc[0].likelihood
            spot_metrics["spot_iterations"] = loc[0].iterations
        else:
            spot_metrics["spot_photons"] = np.nan
            spot_metrics["spot_bg"] = np.nan
            spot_metrics["spot_sx"] = np.nan
            spot_metrics["spot_sy"] = np.nan
            spot_metrics["spot_lpx"] = np.nan
            spot_metrics["spot_lpy"] = np.nan
            spot_metrics["spot_net_gradient"] = np.nan
            spot_metrics["spot_likelihood"] = np.nan
            spot_metrics["spot_iterations"] = np.nan

        spot_metrics = pd.Series(spot_metrics)

    except:
        print(traceback.format_exc())
        spot_metrics = None
        pass

    return spot_metrics


def extract_spot_metrics(dat):

    spot_metrics = {}
    frame = None

    try:

        # Load data from shared memory
        shared_mem = dat["shared_mem"]
        np_array = np.ndarray(dat["shape"], dtype=dat["dtype"], buffer=shared_mem.buf)
        spot_size = dat["spot_size"]

        n_pixels = spot_size**2

        [x1,x2,y1,y2] = dat["spot_bound"]  #
        spot_mask = dat["spot_mask"]
        spot_mask = spot_mask.astype(np.uint8)

        spot_background_mask = dat["spot_background_mask"]

        spot_overlap = dat["background_overlap_mask"][y1:y2, x1:x2]

        if spot_overlap.shape == spot_background_mask.shape:
            spot_background_mask = spot_background_mask & spot_overlap

        spot_mask = np.logical_not(spot_mask).astype(int)
        spot_background_mask = np.logical_not(spot_background_mask).astype(int)

        spot_loc = dat["spot_loc"]
        spot_x = spot_loc.x
        spot_y = spot_loc.y
        coords = [spot_loc.x, spot_loc.y]

        # Perform preprocessing steps and overwrite original image
        spot_values = np_array[:, y1:y2, x1:x2].copy()
        spot_background = np_array[:, y1:y2, x1:x2].copy()

        spot_mask = np.repeat(spot_mask[np.newaxis, :, :], len(spot_values), axis=0)
        spot_background_mask = np.repeat(spot_background_mask[np.newaxis, :, :], len(spot_background), axis=0)

        spot_values = np.ma.array(spot_values, mask=spot_mask)
        spot_background = np.ma.array(spot_background, mask=spot_background_mask)

        # metadata
        spot_metrics["dataset"] = [dat["dataset"]]*len(spot_values)
        spot_metrics["channel"] = [dat["channel"]]*len(spot_values)
        spot_metrics["frame_index"] = np.arange(len(spot_values)).tolist()
        spot_metrics["spot_index"] = [dat["spot_index"]]*len(spot_values)
        spot_metrics["spot_x"] = [spot_x]*len(spot_values)
        spot_metrics["spot_y"] = [spot_y]*len(spot_values)
        spot_metrics["spot_size"] = [dat["spot_size"]]*len(spot_values)

        # metrics
        spot_mean = np.nanmean(spot_values,axis=(1,2)).data
        spot_median = np.nanmedian(spot_values,axis=(1,2)).data
        spot_sum = np.nansum(spot_values,axis=(1,2)).data
        spot_max = np.nanmax(spot_values,axis=(1,2)).data
        spot_std = np.nanstd(spot_values,axis=(1,2)).data

        spot_mean_bg = np.nanmean(spot_background,axis=(1,2)).data
        spot_median_bg = np.nanmedian(spot_background,axis=(1,2)).data
        spot_sum_bg = spot_mean_bg*n_pixels
        spot_max_bg = np.nanmax(spot_background,axis=(1,2)).data
        spot_std_bg = np.nanstd(spot_background,axis=(1,2)).data

        # populate spot metrics dict
        spot_metrics["spot_mean"] = spot_mean
        spot_metrics["spot_median"] = spot_median
        spot_metrics["spot_sum"] = spot_sum
        spot_metrics["spot_max"] = spot_max
        spot_metrics["spot_std"] = spot_std

        spot_metrics["spot_mean_bg"] = spot_mean_bg
        spot_metrics["spot_median_bg"] = spot_median_bg
        spot_metrics["spot_sum_bg"] = spot_sum_bg
        spot_metrics["spot_max_bg"] = spot_max_bg
        spot_metrics["spot_std_bg"] = spot_std_bg

        n_frames = len(spot_values)

        # populate spot metrics dataframe
        reshaped_spot_metrics = []
        for i in range(n_frames):
            new_dict = {key: spot_metrics[key][i] for key in spot_metrics}
            reshaped_spot_metrics.append(new_dict)

        spot_metrics = pd.DataFrame.from_dict(reshaped_spot_metrics)

    except:
        print(traceback.format_exc())
        spot_metrics = None
        pass

    return spot_metrics


class _trace_compute_utils:

    def generate_spot_bounds(self, locs, box_size):

        spot_bounds = []

        for loc_index, loc in enumerate(locs):

            x,y = loc.x, loc.y

            if box_size % 2 == 0:
                x += 0.5
                y += 0.5
                x, y = round(x), round(y)
                x1 = x - (box_size // 2)
                x2 = x + (box_size // 2)
                y1 = y - (box_size // 2)
                y2 = y + (box_size // 2)
            else:
                # Odd spot width
                x, y = round(x), round(y)
                x1 = x - (box_size // 2)
                x2 = x + (box_size // 2)+1
                y1 = y - (box_size // 2)
                y2 = y + (box_size // 2)+1

            spot_bounds.append([x1,x2,y1,y2])

        return spot_bounds

    def _gapseq_compute_traces_cleanup(self):

        print("Finished computing traces.")
        self.compute_traces.setEnabled(True)

        self.populate_plot_combos()
        self.populate_export_combos()
        self.initialize_plot()

    def _get_bbox_localisations(self, n_frames):

        bbox_locs = None

        try:

            localisation_dict = copy.deepcopy(self.localisation_dict["bounding_boxes"])

            locs = copy.deepcopy(localisation_dict["localisations"])

            # Define new dtype including the new field
            new_dtype = np.dtype(locs.dtype.descr + [('loc_index', "<u4")])

            # Create a new array with the new dtype
            extended_locs = np.zeros(locs.shape, dtype=new_dtype)

            for field in locs.dtype.names:
                extended_locs[field] = locs[field]

            extended_locs['loc_index'] = 1

            extended_locs = np.array(extended_locs, dtype=new_dtype).view(np.recarray)

            for loc_index, loc in enumerate(extended_locs):
                loc.loc_index = loc_index

            bbox_locs = []
            for frame_index in range(n_frames):
                frame_locs = copy.deepcopy(extended_locs)
                for loc_index, loc in enumerate(frame_locs):
                    loc.frame = frame_index
                    loc.loc_index = loc.loc_index
                bbox_locs.extend(frame_locs)

            bbox_locs = np.array(bbox_locs, dtype=new_dtype).view(np.recarray)

        except:
            self.compute_traces.setEnabled(True)
            print(traceback.format_exc())
            pass

        return bbox_locs

    def generate_localisation_mask(self, spot_size, spot_shape="square", buffer_size=0, bg_width=1, plot=False):

        box_size = spot_size + (bg_width * 2) + (buffer_size * 2)

        # Create a grid of coordinates
        y, x = np.ogrid[:box_size, :box_size]

        # Adjust center based on box size
        if box_size % 2 == 0:
            center = (box_size / 2 - 0.5, box_size / 2 - 0.5)
        else:
            center = (box_size // 2, box_size // 2)

        if spot_shape.lower() == "circle":
            # Calculate distance from the center for circular mask
            distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

            # Central spot mask
            inner_radius = spot_size // 2
            mask = distance <= inner_radius

            # Buffer mask
            buffer_outer_radius = inner_radius + buffer_size
            buffer_mask = (distance > inner_radius) & (distance <= buffer_outer_radius)

            # Background mask (outside the buffer zone)
            background_outer_radius = buffer_outer_radius + bg_width
            background_mask = (distance > buffer_outer_radius) & (distance <= background_outer_radius)

        elif spot_shape.lower() == "square":
            # Create square mask
            half_size = spot_size // 2
            mask = (abs(x - center[0]) <= half_size) & (abs(y - center[1]) <= half_size)

            # Create square background mask (one pixel larger on each side)
            buffer_mask = (abs(x - center[0]) <= half_size + buffer_size) & (abs(y - center[1]) <= half_size + buffer_size)
            background_mask = (abs(x - center[0]) <= half_size + buffer_size + bg_width) & (abs(y - center[1]) <= half_size + buffer_size + bg_width)
            background_mask = background_mask & ~buffer_mask

        if plot == True:
            plt.figure(figsize=(6, 6))
            plt.imshow(mask, cmap='gray', interpolation='none')
            plt.xticks(np.arange(-0.5, box_size, 1), [])
            plt.yticks(np.arange(-0.5, box_size, 1), [])
            plt.grid(color='blue', linestyle='-', linewidth=2)
            plt.title(f"{box_size}x{box_size} Spot Mask")
            plt.show()

            plt.figure(figsize=(6, 6))
            plt.imshow(background_mask, cmap='gray', interpolation='none')
            plt.xticks(np.arange(-0.5, box_size, 1), [])
            plt.yticks(np.arange(-0.5, box_size, 1), [])
            plt.grid(color='blue', linestyle='-', linewidth=2)
            plt.title(f"{box_size}x{box_size} Background Mask")
            plt.show()

        return mask, buffer_mask, background_mask

    def generate_background_overlap_mask(self, locs, spot_mask, spot_background_mask, image_mask_shape):

        global_spot_mask = np.zeros(image_mask_shape, dtype=np.uint8)
        global_background_mask = np.zeros(image_mask_shape, dtype=np.uint8)

        spot_mask = spot_mask.astype(np.uint16)
        spot_background_mask = spot_background_mask

        spot_bounds = self.generate_spot_bounds(locs,  len(spot_mask[0]))

        for loc_index, [x1,x2,y1,y2] in enumerate(spot_bounds):

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(global_spot_mask.shape[1], x2), min(global_spot_mask.shape[0], y2)

            global_spot_mask[y1:y2, x1:x2] += spot_mask[:y2-y1, :x2-x1]
            global_background_mask[y1:y2, x1:x2] += spot_background_mask[:y2-y1, :x2-x1]

        global_spot_mask[global_spot_mask > 0] = 1
        global_background_mask[global_background_mask > 0] = 1

        intersection_mask = global_spot_mask & global_background_mask

        global_background_mask = global_background_mask - intersection_mask

        return global_background_mask

    def create_shared_images(self):

        self.shared_images = []

        for dataset_name, dataset_dict in self.dataset_dict.items():
            for channel_name, channel_dict in dataset_dict.items():
                image = channel_dict.pop("data")

                shared_mem = shared_memory.SharedMemory(create=True, size=image.nbytes)
                shared_memory_name = shared_mem.name
                shared_image = np.ndarray(image.shape, dtype=image.dtype, buffer=shared_mem.buf)
                shared_image[:] = image[:]

                n_frames = image.shape[0]

                self.shared_images.append({"dataset": dataset_name,
                                      "channel": channel_name,
                                      "n_frames": n_frames,
                                      "shape": image.shape,
                                      "dtype": image.dtype,
                                      "shared_mem": shared_mem,
                                      "shared_memory_name": shared_memory_name})

        return self.shared_images

    def restore_shared_images(self):

        if hasattr(self, "shared_images"):

            for dat in self.shared_images:

                try:

                    shared_mem = dat["shared_mem"]

                    np_array = np.ndarray(dat["shape"], dtype=dat["dtype"], buffer=shared_mem.buf)

                    self.dataset_dict[dat["dataset"]][dat["channel"]]["data"] = np_array.copy()

                    shared_mem.close()
                    shared_mem.unlink()

                except:
                    print(traceback.format_exc())
                    pass

    def extract_spot_metrics_wrapper(self, progress_callback):

        spot_metrics = []

        try:

            self.traces_spot_size = self.findChild(QComboBox, "traces_spot_size")

            spot_size = int(self.traces_spot_size.currentText())
            spot_shape = self.traces_spot_shape.currentText()
            buffer_size = int(self.traces_background_buffer.currentText())
            bg_width = int(self.traces_background_width.currentText())

            localisation_dict = copy.deepcopy(self.localisation_dict["bounding_boxes"])
            locs = localisation_dict["localisations"].copy()

            spot_mask, buffer_mask, spot_background_mask = self.generate_localisation_mask(spot_size, spot_shape,
                buffer_size, bg_width, plot=False)

            spot_bounds = self.generate_spot_bounds(locs, len(spot_mask[0]))

            compute_jobs = []

            for image_dict in self.shared_images:
                mask_shape = image_dict["shape"][1:]

                background_overlap_mask = self.generate_background_overlap_mask(locs, spot_mask, spot_background_mask, mask_shape)

                for spot_index, (spot_loc, spot_bound) in enumerate(zip(locs, spot_bounds)):
                    compute_task = {"spot_index": spot_index,
                                    "spot_size": spot_size,
                                    "spot_mask": spot_mask,
                                    "spot_background_mask": spot_background_mask,
                                    "background_overlap_mask": background_overlap_mask,
                                    "spot_loc": spot_loc,
                                    "spot_bound": spot_bound, }
                    compute_task = {**compute_task, **image_dict}
                    compute_jobs.append(compute_task)

            # print(f"Computing {len(compute_jobs)} spot metrics")
            #
            # extract_spot_metrics(compute_jobs[0])
            # self.compute_traces.setEnabled(True)

            cpu_count = int(multiprocessing.cpu_count() * 0.75)
            timeout_duration = 10  # Timeout in seconds

            with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
                # Submit all jobs and store the future objects
                futures = {executor.submit(extract_spot_metrics, job): job for job in compute_jobs}

                iter = 0
                for future in concurrent.futures.as_completed(futures):
                    job = futures[future]
                    try:
                        result = future.result(timeout=timeout_duration)  # Process result here
                        spot_metrics.append(result)
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

        except:
            self.compute_traces.setEnabled(True)
            self.restore_shared_images()
            print(traceback.format_exc())

        return spot_metrics

    def extract_picasso_spot_metrics_wrapper(self, progress_callback):

        spot_metrics = []

        try:
            localisation_dict = copy.deepcopy(self.localisation_dict["bounding_boxes"])

            locs = localisation_dict["localisations"].copy()
            box_size = localisation_dict["box_size"]

            loc_bounds = self.generate_spot_bounds(localisation_dict["localisations"], box_size)

            spot_locs = []
            for loc in locs:
                spot_loc = [loc.copy()]
                spot_loc = np.rec.fromrecords(spot_loc, dtype=loc.dtype)
                spot_locs.append(spot_loc)


            compute_jobs = []

            for image_dict in self.shared_images:
                n_frames = image_dict["n_frames"]
                for frame_index in range(n_frames):
                    for loc_index, (spot_loc, loc_bound) in enumerate(zip(spot_locs, loc_bounds)):

                        compute_task = {"image_dict": image_dict,
                                        "frame_index": frame_index,
                                        "loc_index": loc_index,
                                        "spot_loc": spot_loc,
                                        "loc_bound": loc_bound,
                                        "box_size": box_size,
                                        }
                        compute_task = {**compute_task, **image_dict}
                        compute_jobs.append(compute_task)

            cpu_count = int(multiprocessing.cpu_count() * 0.75)
            timeout_duration = 10  # Timeout in seconds

            with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
                # Submit all jobs and store the future objects
                futures = {executor.submit(extract_picasso_spot_metrics, job): job for job in compute_jobs}

                iter = 0
                for future in concurrent.futures.as_completed(futures):
                    job = futures[future]
                    try:
                        result = future.result(timeout=timeout_duration)  # Process result here
                        spot_metrics.append(result)
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


        except:
            self.compute_traces.setEnabled(True)
            self.restore_shared_images()
            print(traceback.format_exc())
            pass

        return spot_metrics

    def populatate_traces_dict(self, spot_metrics=None, picasso_spot_metrics=None):

        try:

            self.traces_dict = {}

            # format spot metrics into dataframe
            if spot_metrics is not None:

                spot_metrics = pd.concat(spot_metrics)
                spot_metrics.sort_values(by=["dataset", "channel", "spot_index", "frame_index"], inplace=True)

            # format picasso spot metrics into dataframe and merge with spot metrics
            if picasso_spot_metrics is not None:

                picasso_spot_metrics = pd.DataFrame(picasso_spot_metrics)
                picasso_spot_metrics.sort_values(by=["dataset", "channel", "spot_index", "frame_index"], inplace=True)

                merge_keys = ["dataset", "channel", "spot_index","frame_index","spot_x", "spot_y"]
                spot_metrics = pd.merge(spot_metrics, picasso_spot_metrics, on=merge_keys, how='left')

            # populate traces dict
            if spot_metrics is not None:

                for names, data in spot_metrics.groupby(["dataset", "channel", "spot_index"]):

                    dataset, channel, spot_index = names

                    if dataset not in self.traces_dict.keys():
                        self.traces_dict[dataset] = {}
                    if channel not in self.traces_dict[dataset].keys():
                        self.traces_dict[dataset][channel] = {}
                    if spot_index not in self.traces_dict[dataset][channel].keys():
                        self.traces_dict[dataset][channel][spot_index] = {}

                    for column in data.columns:
                        if column not in ["dataset", "channel", "spot_index", "frame_index"]:
                            self.traces_dict[dataset][channel][spot_index][column] = data[column].values

        except:
            print(traceback.format_exc())
            pass

    def _gapseq_compute_traces(self, progress_callback=None, picasso=False):

        try:

            self.shared_images = self.create_shared_images()

            self.spot_metrics = self.extract_spot_metrics_wrapper(progress_callback)

            if self.compute_with_picasso.isChecked():
                self.picasso_spot_metrics = self.extract_picasso_spot_metrics_wrapper(progress_callback)
            else:
                self.picasso_spot_metrics = None

            self.restore_shared_images()

            self.populatate_traces_dict(self.spot_metrics, self.picasso_spot_metrics)

        except:
            self.compute_traces.setEnabled(True)
            self.restore_shared_images()
            print(traceback.format_exc())
            pass


    def gapseq_compute_traces(self):

        try:

            if self.localisation_dict != {}:

                layer_names = [layer.name for layer in self.viewer.layers]

                if "bounding_boxes" in self.localisation_dict.keys():
                    if "fitted" in self.localisation_dict["bounding_boxes"].keys():

                        self.compute_traces.setEnabled(False)

                        worker = Worker(self._gapseq_compute_traces)
                        worker.signals.progress.connect(partial(self.gapseq_progress, progress_bar=self.compute_traces_progressbar))
                        worker.signals.finished.connect(self._gapseq_compute_traces_cleanup)
                        worker.signals.error.connect(self._gapseq_compute_traces_cleanup)
                        self.threadpool.start(worker)

        except:
            self.compute_traces.setEnabled(True)
            self.restore_shared_images()
            print(traceback.format_exc())


    def visualise_background_masks(self):

        try:

            import cv2

            if "bounding_boxes" in self.localisation_dict.keys():
                if "fitted" in self.localisation_dict["bounding_boxes"].keys():

                    spot_size = int(self.traces_spot_size.currentText())
                    spot_shape = self.traces_spot_shape.currentText()
                    buffer_size = int(self.traces_background_buffer.currentText())
                    bg_width = int(self.traces_background_width.currentText())

                    localisation_dict = copy.deepcopy(self.localisation_dict["bounding_boxes"])
                    locs = localisation_dict["localisations"]

                    spot_mask, buffer_mask, bg_mask = self.generate_localisation_mask(spot_size, spot_shape, buffer_size, bg_width)

                    mask_shape = self.dataset_dict[self.active_dataset][self.active_channel]["data"].shape[-2:]

                    background_overlap_mask = self.generate_background_overlap_mask(locs, buffer_mask, bg_mask, mask_shape)

                    spot_bounds = self.generate_spot_bounds(locs, len(bg_mask[0]))

                    bg_mask = bg_mask.astype(np.uint8)

                    mask_shape = self.dataset_dict[self.active_dataset][self.active_channel]["data"].shape[-2:]

                    global_spot_mask = np.zeros(mask_shape, dtype=np.uint16)

                    for loc_index, [x1, x2, y1, y2] in enumerate(spot_bounds):
                        temp_mask = np.zeros(mask_shape, dtype=np.uint8)

                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(global_spot_mask.shape[1], x2), min(global_spot_mask.shape[0], y2)

                        temp_mask[y1:y2, x1:x2] += bg_mask
                        global_spot_mask[temp_mask > 0] = loc_index + 1


                    binary_spot_mask = global_spot_mask > 0
                    overlap_mask = binary_spot_mask & background_overlap_mask
                    inverse_overlap_mask = np.logical_not(overlap_mask)

                    global_spot_mask[inverse_overlap_mask] = 0

                    if "Background Mask" in self.viewer.layers:
                        self.viewer.layers.remove("Background Mask")
                    self.viewer.add_labels(global_spot_mask,
                        opacity=0.4, name="Background Mask")

        except:
            print(traceback.format_exc())

    def visualise_spot_masks(self):

        try:
            import cv2

            if "bounding_boxes" in self.localisation_dict.keys():
                if "fitted" in self.localisation_dict["bounding_boxes"].keys():

                    spot_size = int(self.traces_spot_size.currentText())
                    spot_shape = self.traces_spot_shape.currentText()
                    buffer_size = int(self.traces_background_buffer.currentText())
                    bg_width = int(self.traces_background_width.currentText())

                    localisation_dict = copy.deepcopy(self.localisation_dict["bounding_boxes"])
                    locs = localisation_dict["localisations"]

                    spot_mask, buffer_mask, spot_background_mask = self.generate_localisation_mask(spot_size, spot_shape, buffer_size, bg_width)

                    spot_bounds = self.generate_spot_bounds(locs, len(spot_mask[0]))

                    spot_mask = spot_mask.astype(np.uint8)

                    mask_shape = self.dataset_dict[self.active_dataset][self.active_channel]["data"].shape[-2:]

                    global_spot_mask = np.zeros(mask_shape, dtype=np.uint16)

                    for loc_index, [x1,x2,y1,y2] in enumerate(spot_bounds):

                        temp_mask = np.zeros(mask_shape, dtype=np.uint8)

                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(global_spot_mask.shape[1], x2), min(global_spot_mask.shape[0], y2)

                        temp_mask[y1:y2, x1:x2] += spot_mask
                        global_spot_mask[temp_mask > 0] = loc_index + 1

                    if "Spot Mask" in self.viewer.layers:
                        self.viewer.layers.remove("Spot Mask")
                    self.viewer.add_labels(
                        global_spot_mask,
                        opacity=0.8,
                        name="Spot Mask")

        except:
            print(traceback.format_exc())
            pass




camera_info = {"baseline": 100.0, "gain": 1, "sensitivity": 1.0, "qe": 0.9, }

# method = "lq"
# gain = 1
#
# total_locs = 0
# progress_dict = {}
# for image_index, image_dict in enumerate(self.shared_images):
#     total_locs += image_dict["n_frames"] * len(locs)
#     if image_index not in progress_dict:
#         progress_dict[image_index] = 0
#
#
# for image_index, image_dict in enumerate(self.shared_images):
#
#     n_frames = image_dict["n_frames"]
#
#     image_data = np.ndarray(image_dict["shape"],
#         dtype=image_dict["dtype"], buffer=image_dict["shared_mem"].buf)
#
#     image_locs = []
#
#     for frame_index in range(n_frames):
#         locs_copy = copy.deepcopy(locs)
#         for loc in locs_copy:
#             loc.frame = frame_index
#             image_locs.append(loc)
#
#     image_locs = np.rec.fromrecords(image_locs, dtype=locs.dtype)
#
#     detected_loc_spots = localize.get_spots(image_data,
#         image_locs, box_size, camera_info)
#
#     fs = gausslq.fit_spots_parallel(detected_loc_spots, asynch=True)
#
#     n_tasks = len(fs)
#     while lib.n_futures_done(fs) < n_tasks:
#         progress = (lib.n_futures_done(fs) / n_tasks) * 100
#         progress_dict[image_index] = progress
#         total_progress = int(np.sum(list(progress_dict.values())) / total_locs)
#         progress_callback.emit(total_progress)
#         time.sleep(0.1)
#
#     theta = gausslq.fits_from_futures(fs)
#     em = gain > 1
#
#     fitted_locs = gausslq.locs_from_fits(image_locs, theta, box_size, em)