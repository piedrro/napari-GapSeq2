import copy
import numpy as np
import traceback
from napari_gapseq2._widget_utils_worker import Worker
from functools import partial
import matplotlib.pyplot as plt
from qtpy.QtWidgets import QFileDialog,QComboBox
from multiprocessing import Process, shared_memory, Pool
import multiprocessing


def extract_spot_metrics(dat):
    try:
        # Access the shared memory
        shared_mem = shared_memory.SharedMemory(name=dat["shared_memory_name"])
        np_array = np.ndarray(dat["shape"], dtype=dat["dtype"], buffer=shared_mem.buf)

        # Perform preprocessing steps and overwrite original image
        frame = np_array[dat["frame_index"]]

        spot_mask = dat["spot_mask"]
        spot_background_mask = dat["spot_background_mask"]

        spot_width = len(spot_mask[0])

        background_overlap_mask = dat["background_overlap_mask"]

        for loc_index, loc in enumerate(dat["spot_locs"]):

            x,y = loc.x, loc.y

            if spot_width % 2 == 0:
                x += 0.5
                y += 0.5
                x, y = round(x), round(y)
                x1 = x - (spot_width // 2)
                x2 = x + (spot_width // 2)
                y1 = y - (spot_width // 2)
                y2 = y + (spot_width // 2)
            else:
                # Odd spot width
                x, y = round(x), round(y)
                x1 = x - (spot_width // 2)
                x2 = x + (spot_width // 2)+1
                y1 = y - (spot_width // 2)
                y2 = y + (spot_width // 2)+1

            roi = frame[y1:y2, x1:x2]

            spot_overlap = background_overlap_mask[y1:y2, x1:x2].copy()
            spot_background_mask = spot_background_mask & spot_overlap

            spot_values = roi[spot_mask].copy()
            background_values = roi[spot_background_mask].copy()

            print(np.mean(spot_values), np.mean(background_values))

    except:
        print(traceback.format_exc())
        pass



class _trace_compute_utils:

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


    def generate_localisation_mask(self, spot_size, plot=False):

        box_size = spot_size + 2

        # Radius for the central spot
        inner_radius = (box_size - 2) // 2

        # Slightly larger radius for the ring
        outer_radius = inner_radius + 1

        # Create a grid of coordinates
        y, x = np.ogrid[:box_size, :box_size]

        # Adjust center based on box size
        if box_size % 2 == 0:
            center = (box_size / 2 - 0.5, box_size / 2 - 0.5)
        else:
            center = (box_size // 2, box_size // 2)

        # Calculate distance from the center
        distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

        # Central spot mask
        mask = distance <= inner_radius

        # Ring mask
        background_mask = (distance > inner_radius) & (distance <= outer_radius)

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

        spot_width = len(spot_mask[0])

        for loc_index, loc in enumerate(locs):

            x, y = loc.x, loc.y

            if spot_width % 2 == 0:
                x += 0.5
                y += 0.5
                x, y = round(x), round(y)
                x1 = x - (spot_width // 2)
                x2 = x + (spot_width // 2)
                y1 = y - (spot_width // 2)
                y2 = y + (spot_width // 2)
            else:
                # Odd spot width
                x, y = round(x), round(y)
                x1 = x - (spot_width // 2)
                x2 = x + (spot_width // 2) + 1
                y1 = y - (spot_width // 2)
                y2 = y + (spot_width // 2) + 1

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(global_spot_mask.shape[1], x2), min(global_spot_mask.shape[0], y2)

            global_spot_mask[y1:y2, x1:x2] += spot_mask[:y2-y1, :x2-x1]
            global_background_mask[y1:y2, x1:x2] += spot_background_mask[:y2-y1, :x2-x1]

        global_spot_mask[global_spot_mask > 0] = 1
        global_background_mask[global_background_mask > 0] = 1

        intersection_mask = global_spot_mask & global_background_mask

        global_background_mask = global_background_mask - intersection_mask

        return global_background_mask




    def _gapseq_compute_traces(self, progress_callback=None):


        try:

            from picasso import gausslq, lib, localize

            self.traces_spot_size = self.findChild(QComboBox, "traces_spot_size")

            spot_size = int(self.traces_spot_size.currentText())
            box_size = int(self.picasso_box_size.currentText())

            localisation_dict = copy.deepcopy(self.localisation_dict["bounding_boxes"])

            loc_centres = self.get_localisation_centres(localisation_dict["localisations"])

            spot_mask, spot_background_mask = self.generate_localisation_mask(spot_size, plot=False)

            target_images = []
            total_frames = 0

            for dataset_name, dataset_dict in self.dataset_dict.items():
                for channel_name, channel_dict in dataset_dict.items():
                    n_frames = channel_dict["data"].shape[0]
                    total_frames += n_frames
                    target_images.append({"dataset_name": dataset_name,
                                          "channel_name": channel_name,
                                          "n_frames": n_frames})

            iter = 0

            for i in range(len(target_images)):

                dataset_name = target_images[i]["dataset_name"]
                channel_name = target_images[i]["channel_name"]
                n_frames = target_images[i]["n_frames"]

                if channel_name.lower() == "aa":

                    bbox_locs = self._get_bbox_localisations(n_frames)

                    spot_locs = copy.deepcopy(bbox_locs)
                    spot_locs = [loc for loc in bbox_locs if loc.frame == 100]

                    image = self.dataset_dict[dataset_name][channel_name.lower()]["data"].copy()
                    image = copy.deepcopy(image)

                    mask_shape = image.shape[1:]

                    background_overlap_mask = self.generate_background_overlap_mask(spot_locs,
                        spot_mask, spot_background_mask, mask_shape)


                    shared_mem = shared_memory.SharedMemory(create=True, size=image.nbytes)
                    shared_memory_name = shared_mem.name
                    shared_image = np.ndarray(image.shape, dtype=image.dtype, buffer=shared_mem.buf)
                    shared_image[:] = image[:]

                    compute_jobs = []

                    for frame_index in range(1):

                        compute_jobs.append({"frame_index": frame_index,
                                             "shape": image.shape,
                                             "dtype": image.dtype,
                                             "shared_memory_name": shared_memory_name,
                                             "spot_size": spot_size,
                                             "spot_mask": spot_mask,
                                             "spot_background_mask": spot_background_mask,
                                             "background_overlap_mask": background_overlap_mask,
                                             "spot_locs": spot_locs})

                    spot_metrics = extract_spot_metrics(compute_jobs[0])

                    break


                    # def callback(*args, offset=0):
                    #     nonlocal iter
                    #     iter += 1
                    #     progress = int((iter / total_frames) * 100)
                    #     if progress_callback != None:
                    #         progress_callback.emit(progress - offset)
                    #     return
                    #
                    # cpu_count = int(multiprocessing.cpu_count() * 0.75)
                    #
                    # with Pool(cpu_count) as p:
                    #     imported_data = [p.apply_async(extract_spot_metrics, args=(i,), callback=callback) for i in compute_jobs]
                    #     imported_data = [p.get() for p in imported_data]
                    #     p.close()
                    #
                    # shared_mem.close()

                    break

        except:
            self.compute_traces.setEnabled(True)
            print(traceback.format_exc())
            pass



    def gapseq_compute_traces(self):

        try:

            if self.localisation_dict != {}:

                layer_names = [layer.name for layer in self.viewer.layers]

                if "bounding_boxes" in self.localisation_dict.keys():

                    fitted = self.localisation_dict["bounding_boxes"]["fitted"]

                    self.compute_traces.setEnabled(False)

                    worker = Worker(self._gapseq_compute_traces)
                    worker.signals.progress.connect(partial(self.gapseq_progress, progress_bar=self.compute_traces_progressbar))
                    worker.signals.finished.connect(self._gapseq_compute_traces_cleanup)
                    worker.signals.error.connect(self._gapseq_compute_traces_cleanup)
                    self.threadpool.start(worker)

        except:
            self.compute_traces.setEnabled(True)
            print(traceback.format_exc())