import copy
import numpy as np
import traceback
from napari_gapseq2._widget_utils_worker import Worker
from functools import partial
import matplotlib.pyplot as plt
from qtpy.QtWidgets import QFileDialog,QComboBox
from multiprocessing import Process, shared_memory, Pool
import multiprocessing
from picasso.gaussmle import gaussmle
from picasso import gausslq, lib, localize


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
    ("loc_index", "u4")
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
            identifications.loc_index,
        ),
        dtype=LOCS_DTYPE,
    )
    locs.sort(kind="mergesort", order="frame")
    return locs






def extract_spot_metrics(dat):

    spot_metrics = {}
    frame = None

    try:

        # Load data from shared memory
        shared_mem = dat["shared_mem"]
        np_array = np.ndarray(dat["shape"], dtype=dat["dtype"], buffer=shared_mem.buf)

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

        # Perform preprocessing steps and overwrite original image
        spot_values = np_array[:, y1:y2, x1:x2].copy()
        spot_background = np_array[:, y1:y2, x1:x2].copy()

        spot_mask = np.repeat(spot_mask[np.newaxis, :, :], len(spot_values), axis=0)
        spot_background_mask = np.repeat(spot_background_mask[np.newaxis, :, :], len(spot_values), axis=0)

        spot_values = np.ma.array(spot_values, mask=spot_mask)
        spot_background = np.ma.array(spot_background, mask=spot_background)

        mean = np.ma.mean(spot_values,axis=(1,2))

        # metadata
        spot_metrics["dataset"] = dat["dataset"]
        spot_metrics["channel"] = dat["channel"]
        spot_metrics["spot_index"] = dat["spot_index"]
        spot_metrics["spot_size"] = dat["spot_size"]

        # metrics
        spot_metrics["spot_mean"] = np.ma.mean(spot_values,axis=(1,2))
        spot_metrics["spot_sum"] = np.ma.sum(spot_values,axis=(1,2))
        spot_metrics["spot_max"] = np.ma.max(spot_values,axis=(1,2))
        spot_metrics["spot_std"] = np.ma.std(spot_values,axis=(1,2))
        spot_metrics["bg_mean"] = np.ma.mean(spot_background,axis=(1,2))
        spot_metrics["bg_sum"] = np.ma.sum(spot_background,axis=(1,2))
        spot_metrics["bg_max"] = np.ma.max(spot_background,axis=(1,2))
        spot_metrics["bg_std"] = np.ma.std(spot_background,axis=(1,2))
        spot_metrics["snr_mean"] = spot_metrics["spot_mean"] / spot_metrics["bg_mean"]
        spot_metrics["snr_sum"] = spot_metrics["spot_sum"] / spot_metrics["bg_sum"]
        spot_metrics["snr_max"] = spot_metrics["spot_max"] / spot_metrics["bg_max"]
        spot_metrics["snr_std"] = spot_metrics["spot_std"] / spot_metrics["bg_std"]


        # thetas, CRLBs, likelihoods, iterations = gaussmle(spot_list, eps=0.001, max_it=100, method="sigma")
        # locs = locs_from_fits(spot_locs, thetas, CRLBs, likelihoods, iterations, len(spot_list[0]))
        #
        # for loc in locs:
        #     if loc.loc_index in spot_metrics.keys():
        #         loc_index = loc.loc_index
        #         spot_metrics[loc_index]["picasso_photons"] = loc.photons
        #         spot_metrics[loc_index]["picasso_sx"] = loc.sx
        #         spot_metrics[loc_index]["picasso_sy"] = loc.sy
        #         spot_metrics[loc_index]["picasso_lpx"] = loc.lpx
        #         spot_metrics[loc_index]["picasso_lpy"] = loc.lpy
        #         spot_metrics[loc_index]["picasso_bg"] = loc.bg

    except:
        print(traceback.format_exc())
        spot_metrics = None
        pass

    return spot_metrics

class _trace_compute_utils:


    def generate_spot_bounds(self, locs, spot_mask):

        spot_mask_width = len(spot_mask[0])

        spot_bounds = []

        for loc_index, loc in enumerate(locs):

            x,y = loc.x, loc.y

            if spot_mask_width % 2 == 0:
                x += 0.5
                y += 0.5
                x, y = round(x), round(y)
                x1 = x - (spot_mask_width // 2)
                x2 = x + (spot_mask_width // 2)
                y1 = y - (spot_mask_width // 2)
                y2 = y + (spot_mask_width // 2)
            else:
                # Odd spot width
                x, y = round(x), round(y)
                x1 = x - (spot_mask_width // 2)
                x2 = x + (spot_mask_width // 2)+1
                y1 = y - (spot_mask_width // 2)
                y2 = y + (spot_mask_width // 2)+1

            spot_bounds.append([x1,x2,y1,y2])

        return spot_bounds



    def _gapseq_compute_traces_cleanup(self):

        print("Finished computing traces.")
        self.compute_traces.setEnabled(True)


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


    def generate_localisation_mask(self, spot_size, spot_shape = "square", plot=False):

        box_size = spot_size + 2

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

            # Slightly larger radius for the ring (background mask)
            outer_radius = inner_radius + 1
            background_mask = (distance > inner_radius) & (distance <= outer_radius)
        elif spot_shape.lower() == "square":
            # Create square mask
            half_size = spot_size // 2
            mask = (abs(x - center[0]) <= half_size) & (abs(y - center[1]) <= half_size)

            # Create square background mask (one pixel larger on each side)
            background_mask = (abs(x - center[0]) <= half_size + 1) & (abs(y - center[1]) <= half_size + 1)
            background_mask = background_mask & ~mask

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

        return mask, background_mask



    def generate_background_overlap_mask(self, locs, spot_mask, spot_background_mask, image_mask_shape):

        global_spot_mask = np.zeros(image_mask_shape, dtype=np.uint8)
        global_background_mask = np.zeros(image_mask_shape, dtype=np.uint8)

        spot_mask = spot_mask.astype(np.uint16)
        spot_background_mask = spot_background_mask

        spot_bounds = self.generate_spot_bounds(locs, spot_mask)

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
                    pass


    def _gapseq_compute_traces(self, progress_callback=None):

        try:

            self.traces_spot_size = self.findChild(QComboBox, "traces_spot_size")

            spot_size = int(self.traces_spot_size.currentText())
            spot_shape = self.traces_spot_shape.currentText()

            localisation_dict = copy.deepcopy(self.localisation_dict["bounding_boxes"])
            locs = localisation_dict["localisations"].copy()

            spot_mask, spot_background_mask = self.generate_localisation_mask(spot_size,spot_shape, plot=False)

            spot_bounds = self.generate_spot_bounds(locs, spot_mask)

            self.shared_images = self.create_shared_images()

            compute_jobs = []

            for image_dict in self.shared_images:

                mask_shape = image_dict["shape"][1:]

                background_overlap_mask = self.generate_background_overlap_mask(locs,
                    spot_mask, spot_background_mask, mask_shape)

                for spot_index, (spot_loc,spot_bound) in enumerate(zip(locs,spot_bounds)):
                    compute_task = {"spot_size": spot_size,
                                    "spot_mask": spot_mask,
                                    "spot_background_mask": spot_background_mask,
                                    "background_overlap_mask": background_overlap_mask,
                                    "spot_loc": spot_loc,
                                    "spot_bound": spot_bound,
                                    "spot_index": spot_index,}
                    compute_task = {**compute_task, **image_dict}
                    compute_jobs.append(compute_task)

            iter = 0
            def callback(*args, offset=0):
                nonlocal iter
                iter += 1
                progress = int((iter / len(compute_jobs)) * 100)
                if progress_callback != None:
                    progress_callback.emit(progress - offset)
                return

            cpu_count = int(multiprocessing.cpu_count() * 0.75)

            with Pool(cpu_count) as p:
                spot_metrics = [p.apply_async(extract_spot_metrics, args=(i,), callback=callback) for i in compute_jobs]
                spot_metrics = [p.get() for p in spot_metrics]
                p.close()
                p.join()

            self.restore_shared_images()

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
            print(traceback.format_exc())


    def visualise_spot_masks(self):

        try:
            import cv2

            if "bounding_boxes" in self.localisation_dict.keys():
                if "fitted" in self.localisation_dict["bounding_boxes"].keys():

                    spot_size = int(self.traces_spot_size.currentText())
                    spot_shape = self.traces_spot_shape.currentText()

                    localisation_dict = copy.deepcopy(self.localisation_dict["bounding_boxes"])
                    locs = localisation_dict["localisations"]

                    spot_mask, spot_background_mask = self.generate_localisation_mask(spot_size, spot_shape, plot=False)

                    spot_bounds = self.generate_spot_bounds(locs, spot_mask)

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
                        opacity=0.3,
                        name="Spot Mask")

        except:
            print(traceback.format_exc())
            pass