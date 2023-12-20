import copy
import numpy as np
import traceback
from napari_gapseq2._widget_utils_compute import Worker
import time
from functools import partial
from sklearn.cluster import DBSCAN
import pandas as pd
from picasso.clusterer import extract_valid_labels
import os
from multiprocessing import Process, shared_memory, Pool
from picasso import gausslq, lib, localize
from picasso.localize import get_spots, identify_frame
from picasso.gaussmle import gaussmle
from functools import partial
import concurrent.futures
import multiprocessing
from itertools import chain

def picasso_detect(dat):

    result = None

    try:

        frame_index = dat["frame_index"]
        min_net_gradient = dat["min_net_gradient"]
        box_size = dat["box_size"]
        roi = dat["roi"]
        dataset = dat["dataset"]
        channel = dat["channel"]
        detect = dat["detect"]
        fit = dat["fit"]

        # Access the shared memory
        shared_mem = shared_memory.SharedMemory(name=dat["shared_memory_name"])
        np_array = np.ndarray(dat["shape"], dtype=dat["dtype"], buffer=shared_mem.buf)

        # Perform preprocessing steps and overwrite original image
        frame = np_array[frame_index].copy()

        if detect:
            locs = identify_frame(frame, min_net_gradient, box_size, 0, roi=roi)
        else:
            locs = dat["frame_locs"]

        expected_loc_length = 4

        if fit:
            expected_loc_length = 12
            try:
                image = np.expand_dims(frame, axis=0)
                camera_info = {"baseline": 100.0, "gain": 1, "sensitivity": 1.0, "qe": 0.9, }
                spot_data = get_spots(image, locs, box_size, camera_info)
                theta, CRLBs, likelihoods, iterations = gaussmle(spot_data, eps=0.001, max_it=100, method="sigma")
                locs = localize.locs_from_fits(locs.copy(), theta, CRLBs, likelihoods, iterations, box_size)
            except:
                pass

        for loc in locs:
            loc.frame = frame_index

        render_locs= {}
        render_locs[frame_index] = np.vstack((locs.y, locs.x)).T.tolist()

        locs = [loc for loc in locs if len(loc) == expected_loc_length]
        locs = np.array(locs).view(np.recarray)

        result = {"dataset": dataset, "channel": channel, "frame_index": frame_index,
                  "locs": locs,"render_locs": render_locs}

    except:
        print(traceback.format_exc())
        pass

    return result


class _picasso_detect_utils:

    def populate_localisation_dict(self, loc_dict, render_loc_dict, detect_mode, image_channel, box_size, fitted=False):

        detect_mode = detect_mode.lower()

        try:

            for dataset_name, locs in loc_dict.items():

                render_locs = render_loc_dict[dataset_name]

                if detect_mode == "fiducials":

                    loc_centres = self.get_localisation_centres(locs)

                    fiducial_dict = {"localisations": [], "localisation_centres": [], "render_locs": {}}

                    fiducial_dict["localisations"] = locs.copy()
                    fiducial_dict["localisation_centres"] = loc_centres.copy()
                    fiducial_dict["render_locs"] = render_locs

                    fiducial_dict["fitted"] = fitted
                    fiducial_dict["box_size"] = box_size

                    if dataset_name not in self.localisation_dict["fiducials"].keys():
                        self.localisation_dict["fiducials"][dataset_name] = {}
                    if image_channel not in self.localisation_dict["fiducials"][dataset_name].keys():
                        self.localisation_dict["fiducials"][dataset_name][image_channel.lower()] = {}

                    self.localisation_dict["fiducials"][dataset_name][image_channel.lower()] = fiducial_dict.copy()

                else:

                    loc_centres = self.get_localisation_centres(locs)

                    self.localisation_dict["bounding_boxes"]["localisations"] = locs.copy()
                    self.localisation_dict["bounding_boxes"]["localisation_centres"] = loc_centres.copy()
                    self.localisation_dict["bounding_boxes"]["fitted"] = fitted
                    self.localisation_dict["bounding_boxes"]["box_size"] = box_size


        except:
            print(traceback.format_exc())
            self.picasso_progressbar.setValue(0)
            self.picasso_detect.setEnabled(True)
            self.picasso_fit.setEnabled(True)
            self.picasso_detectfit.setEnabled(False)

    def _picasso_wrapper_result(self, result):

        try:
            fitted, loc_dict, render_loc_dict, total_locs = result

            detect_mode = self.picasso_detect_mode.currentText()
            image_channel = self.picasso_channel.currentText()
            box_size = int(self.picasso_box_size.currentText())

            dataset_names = list(loc_dict.keys())

            if len(dataset_names) > 0:

                self.populate_localisation_dict(loc_dict, render_loc_dict, detect_mode,
                    image_channel, box_size, fitted)

                if fitted:
                    print("Fitted {} localisations".format(total_locs))
                else:
                    print("Detected {} localisations".format(total_locs))

                self.update_active_image(channel=image_channel.lower(), dataset=self.active_dataset)
                self.draw_fiducials(update_vis=True)
                self.draw_bounding_boxes()

                self.gapseq_progress(100, self.picasso_progressbar)
                self.picasso_detect.setEnabled(True)
                self.picasso_fit.setEnabled(True)
                self.picasso_detectfit.setEnabled(False)

        except:
            print(traceback.format_exc())


    def get_frame_locs(self, dataset_name, image_channel, frame_index):

        try:

            loc_dict, n_locs = self.get_loc_dict(dataset_name,
                image_channel.lower(), type = "fiducials")

            if "localisations" not in loc_dict.keys():
                return None
            else:
                locs = loc_dict["localisations"]
                locs = [loc for loc in locs if loc.frame == frame_index]
                locs = np.array(locs).view(np.recarray)

                return locs

        except:
            print(traceback.format_exc())
            return None



    def _picasso_wrapper(self, progress_callback, detect, fit, min_net_gradient, roi, box_size, dataset_name, image_channel, frame_mode):

        loc_dict = {}
        render_loc_dict = {}
        total_locs = 0

        try:

            if dataset_name == "All Datasets":
                dataset_list = list(self.dataset_dict.keys())
            else:
                dataset_list = [dataset_name]

            channel_list = [image_channel.lower()]

            self.shared_images = self.create_shared_images(dataset_list=dataset_list, channel_list=channel_list)

            compute_jobs = []

            for image_dict in self.shared_images:

                if frame_mode.lower() == "active":
                    frame_list = [self.viewer.dims.current_step[0]]
                else:
                    n_frames = image_dict['shape'][0]
                    frame_list = list(range(n_frames))

                for frame_index in frame_list:

                    frame_locs = self.get_frame_locs( image_dict["dataset"],
                        image_channel, frame_index)

                    if detect == False and frame_locs is None:
                        continue
                    else:
                        compute_job = {"dataset":image_dict["dataset"],
                                       "channel":image_dict["channel"],
                                       "frame_index": frame_index,
                                       "shared_memory_name": image_dict['shared_memory_name'],
                                       "shape": image_dict['shape'],
                                       "dtype": image_dict['dtype'],
                                       "detect": detect,
                                       "fit": fit,
                                       "min_net_gradient": int(min_net_gradient),
                                       "box_size": int(box_size),
                                       "roi": roi,
                                       "frame_locs": frame_locs,
                                       }

                    compute_jobs.append(compute_job)

            if len(compute_jobs) > 0:

                timeout_duration = 10  # Timeout in seconds

                loc_dict = {}
                render_loc_dict = {}

                if frame_mode.lower() == "active":
                    executor_class = concurrent.futures.ThreadPoolExecutor
                    cpu_count = 1
                else:
                    executor_class = concurrent.futures.ProcessPoolExecutor
                    cpu_count = int(multiprocessing.cpu_count() * 0.9)

                with executor_class(max_workers=cpu_count) as executor:
                    futures = {executor.submit(picasso_detect, job): job for job in compute_jobs}

                iter = 0
                for future in concurrent.futures.as_completed(futures):
                    job = futures[future]
                    try:
                        result = future.result(timeout=timeout_duration)  # Process result here

                        if result is not None:
                            dataset_name = result["dataset"]

                            if dataset_name not in loc_dict:
                                loc_dict[dataset_name] = []
                                render_loc_dict[dataset_name] = {}

                            locs = result["locs"]
                            render_locs = result["render_locs"]

                            loc_dict[dataset_name].extend(locs)
                            render_loc_dict[dataset_name] = {**render_loc_dict[dataset_name], **render_locs}

                        iter += 1
                        progress = int((iter / len(compute_jobs)) * 100)
                        progress_callback.emit(progress)  # Emit the signal

                    except concurrent.futures.TimeoutError:
                        # print(f"Task {job} timed out after {timeout_duration} seconds.")
                        pass
                    except Exception as e:
                        print(f"Error occurred in task {job}: {e}")  # Handle other exceptions
                        pass

                total_locs = 0
                for dataset, locs in loc_dict.items():
                    locs = np.hstack(locs).view(np.recarray).copy()
                    locs.sort(kind="mergesort", order="frame")
                    locs = np.array(locs).view(np.recarray)
                    loc_dict[dataset] = locs
                    total_locs += len(locs)

            self.restore_shared_images()

            self.picasso_progressbar.setValue(0)
            self.picasso_detect.setEnabled(True)
            self.picasso_fit.setEnabled(True)
            self.picasso_detectfit.setEnabled(False)

        except:
            print(traceback.format_exc())
            self.restore_shared_images()

            self.picasso_progressbar.setValue(0)
            self.picasso_detect.setEnabled(True)
            self.picasso_fit.setEnabled(True)
            self.picasso_detectfit.setEnabled(False)

            loc_dict = {}
            render_loc_dict = {}
            total_locs = 0

        return fit, loc_dict, render_loc_dict, total_locs

    def gapseq_picasso(self, detect = False, fit = False):

        try:
            if self.dataset_dict != {}:

                self.picasso_progressbar.setValue(0)
                self.picasso_detect.setEnabled(False)
                self.picasso_fit.setEnabled(False)
                self.picasso_detectfit.setEnabled(False)

                min_net_gradient = self.picasso_min_net_gradient.text()
                box_size = int(self.picasso_box_size.currentText())
                dataset_name = self.picasso_dataset.currentText()
                image_channel = self.picasso_channel.currentText()
                frame_mode = self.picasso_frame_mode.currentText()
                detect_mode = self.picasso_detect_mode.currentText()

                roi = self.generate_roi()

                if min_net_gradient.isdigit() and image_channel != "":
                    worker = Worker(self._picasso_wrapper,
                        detect=detect,
                        fit=fit,
                        min_net_gradient=min_net_gradient,
                        roi = roi,
                        box_size=box_size,
                        dataset_name=dataset_name,
                        image_channel=image_channel,
                        frame_mode=frame_mode)
                    worker.signals.progress.connect(partial(self.gapseq_progress, progress_bar=self.picasso_progressbar))
                    worker.signals.result.connect(self._picasso_wrapper_result)
                    self.threadpool.start(worker)

        except:
            print(traceback.format_exc())
            self.picasso_progressbar.setValue(0)
            self.picasso_detect.setEnabled(True)
            self.picasso_fit.setEnabled(True)
            self.picasso_detectfit.setEnabled(True)
            pass

    def generate_roi(self):

        border_width = self.picasso_roi_border_width.text()
        window_cropping = self.picasso_window_cropping.isChecked()

        roi = None

        try:

            generate_roi = False

            if window_cropping:
                layers_names = [layer.name for layer in self.viewer.layers if layer.name not in ["bounding_boxes", "fiducials"]]

                crop = self.viewer.layers[layers_names[0]].corner_pixels[:, -2:]
                [[y1, x1], [y2, x2]] = crop

                generate_roi = True

            else:

                if type(border_width) == str:
                    border_width = int(border_width)
                    if border_width > 0:
                        generate_roi = True
                elif type(border_width) == int:
                    if border_width > 0:
                        generate_roi = True

            if generate_roi:

                dataset = self.picasso_dataset.currentText()
                channel = self.picasso_channel.currentText()

                if dataset == "All Datasets":
                    dataset = list(self.dataset_dict.keys())[0]

                image_shape = self.dataset_dict[dataset][channel.lower()]["data"].shape

                frame_shape = image_shape[1:]

                if window_cropping:

                    border_width = int(border_width)

                    if x1 < border_width:
                        x1 = border_width
                    if y1 < border_width:
                        y1 = border_width
                    if x2 > frame_shape[1] - border_width:
                        x2 = frame_shape[1] - border_width
                    if y2 > frame_shape[0] - border_width:
                        y2 = frame_shape[0] - border_width

                    roi = [[y1, x1], [y2, x2]]

                else:

                    roi = [[int(border_width), int(border_width)],
                           [int(frame_shape[0] - border_width), int(frame_shape[1] - border_width)]]

        except:
            print(traceback.format_exc())
            pass

        return roi
    def gapseq_cluster_localisations(self):

        try:

            # mode = self.cluster_mode.currentText()
            dataset = self.cluster_dataset.currentText()
            channel = self.cluster_channel.currentText()

            locs = self.localisation_dict["fiducials"][dataset][channel.lower()]["localisations"]

            n_frames = len(np.unique([loc.frame for loc in locs]))

            cluster_dataset = np.vstack((locs.x, locs.y)).T

            # Applying DBSCAN
            dbscan = DBSCAN(eps=0.5, min_samples=5, n_jobs=-1)
            dbscan.fit(cluster_dataset)

            # Extracting labels
            labels = dbscan.labels_

            # Finding unique clusters
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            unique_labels = set(labels)

            # Filtering out noise (-1 label)
            filtered_data = cluster_dataset[labels != -1]

            # Corresponding labels after filtering out noise
            filtered_labels = labels[labels != -1]

            # Finding cluster centers
            cluster_centers = np.array([filtered_data[filtered_labels == i].mean(axis=0) for i in range(n_clusters)])

            clustered_locs = []
            render_locs = {}

            for cluster_index in range(len(cluster_centers)):
                for frame_index in range(n_frames):
                    [locX, locY] = cluster_centers[cluster_index]
                    new_loc = (int(frame_index), float(locX), float(locY))
                    clustered_locs.append(new_loc)

                    if frame_index not in render_locs.keys():
                        render_locs[frame_index] = []

                    render_locs[frame_index].append([locY, locX])

            # Convert list to recarray
            clustered_locs = np.array(clustered_locs, dtype=[('frame', '<u4'), ('x', '<f4'), ('y', '<f4')]).view(np.recarray)

            cluster_loc_centers = self.get_localisation_centres(clustered_locs)

            self.localisation_dict["fiducials"][dataset][channel.lower()]["localisations"] = clustered_locs
            self.localisation_dict["fiducials"][dataset][channel.lower()]["localisation_centres"] = cluster_loc_centers
            self.localisation_dict["fiducials"][dataset][channel.lower()]["render_locs"] = render_locs

            self.draw_fiducials(update_vis=True)

        except:
            print(traceback.format_exc())
            self.picasso_progressbar.setValue(0)
            self.picasso_detect.setEnabled(True)
            self.picasso_fit.setEnabled(True)
            self.picasso_detectfit.setEnabled(False)
            pass

    def export_picasso_locs(self, locs):

        try:

            dataset_name = self.picasso_dataset.currentText()
            image_channel = self.picasso_channel.currentText()
            min_net_gradient = int(self.picasso_min_net_gradient.text())
            box_size = int(self.picasso_box_size.currentText())

            path = self.dataset_dict[dataset_name][image_channel.lower()]["path"]
            image_shape = self.dataset_dict[dataset_name][image_channel.lower()]["data"].shape

            base, ext = os.path.splitext(path)
            path = base + f"_{image_channel}_picasso_locs.hdf5"

            info = [{"Byte Order": "<", "Data Type": "uint16", "File": path,
                     "Frames": image_shape[0], "Height": image_shape[1],
                     "Micro-Manager Acquisiton Comments": "", "Width": image_shape[2], },
                    {"Box Size": box_size, "Fit method": "LQ, Gaussian",
                     "Generated by": "Picasso Localize",
                     "Min. Net Gradient": min_net_gradient, "Pixelsize": 130, "ROI": None, }]

            from picasso.io import save_locs
            # save_locs(path, locs, info)

        except:
            print(traceback.format_exc())
            pass
