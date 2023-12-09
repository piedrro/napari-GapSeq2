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












class _picasso_detect_utils:


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
            pass



    def populate_localisation_dict(self, locs, detect_mode, dataset_name, image_channel, box_size, fitted=False):

        detect_mode = detect_mode.lower()

        try:

            if detect_mode == "fiducials":

                loc_centres = self.get_localisation_centres(locs)

                fiducial_dict = {"localisations": [], "localisation_centres": [], "render_locs": {}}

                fiducial_dict["localisations"] = locs.copy()
                fiducial_dict["localisation_centres"] = loc_centres.copy()
                if fitted:
                    fiducial_dict["render_locs"] = self.fitted_render_locs.copy()
                else:
                    fiducial_dict["render_locs"] = self.detected_render_locs.copy()

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


    def filter_localisations(self, locs):

        if self.picasso_window_cropping.isChecked():

            layers_names = [layer.name for layer in self.viewer.layers if layer.name not in ["bounding_boxes", "fiducials"]]

            crop = self.viewer.layers[layers_names[0]].corner_pixels[:, -2:]

            [[y1, x1], [y2, x2]] = crop

            x_range = [x1, x2]
            y_range = [y1, y2]

            filtered_locs = []

            for loc in locs:
                locX = loc.x
                locY = loc.y
                if locX > x_range[0] and locX < x_range[1] and locY > y_range[0] and locY < y_range[1]:
                    filtered_locs.append(copy.deepcopy(loc))

            filtered_locs = np.array(filtered_locs)
            filtered_locs = np.rec.fromrecords(filtered_locs, dtype=locs.dtype)

            return filtered_locs

        else:
            return locs

    def _detect_localisations_cleanup(self):

        try:

            detect_mode = self.picasso_detect_mode.currentText()
            dataset_name = self.picasso_dataset.currentText()
            image_channel = self.picasso_channel.currentText()
            box_size = int(self.picasso_box_size.currentText())

            self.populate_localisation_dict(self.detected_locs, detect_mode,
                dataset_name, image_channel, box_size, fitted=False)

            n_frames = len(np.unique([loc[0] for loc in self.detected_locs]))
            print("detected {} localisations from {} frame(s)".format(len(self.detected_locs), n_frames))

            self.gapseq_progress(100, self.picasso_progressbar)
            self.picasso_detect.setEnabled(True)
            self.picasso_fit.setEnabled(True)

            self.draw_fiducials(update_vis=True)
            self.draw_bounding_boxes()

            self.gapseq_dataset_selector.blockSignals(True)
            self.gapseq_dataset_selector.setCurrentIndex(self.gapseq_dataset_selector.findText(dataset_name))
            self.gapseq_dataset_selector.blockSignals(False)
            self.update_active_image(channel=image_channel.lower(), dataset=dataset_name)

        except:
            print(traceback.format_exc())
            self.picasso_progressbar.setValue(0)
            self.picasso_detect.setEnabled(True)
            self.picasso_fit.setEnabled(True)
            pass

    def _fit_localisations_cleanup(self):

        try:
            detect_mode = self.picasso_detect_mode.currentText()
            dataset_name = self.picasso_dataset.currentText()
            image_channel = self.picasso_channel.currentText()
            box_size = int(self.picasso_box_size.currentText())

            if hasattr(self, "fitted_locs"):
                self.populate_localisation_dict(self.fitted_locs, detect_mode,
                    dataset_name, image_channel, box_size, fitted=True)
                n_frames = len(np.unique([loc[0] for loc in self.fitted_locs]))
                print("Fitted {} localisations from {} frame(s)".format(len(self.fitted_locs), n_frames))
            else:
                print("No localisations fitted")

            self.gapseq_progress(100, self.picasso_progressbar)
            self.picasso_detect.setEnabled(True)
            self.picasso_fit.setEnabled(True)

            self.draw_fiducials()
            self.draw_bounding_boxes()

            self.gapseq_dataset_selector.blockSignals(True)
            self.gapseq_dataset_selector.setCurrentIndex(self.gapseq_dataset_selector.findText(dataset_name))
            self.gapseq_dataset_selector.blockSignals(False)
            self.update_active_image(channel=image_channel.lower(), dataset=dataset_name)

            self.export_picasso_locs(self.fitted_locs)

        except:
            print(traceback.format_exc())
            self.picasso_progressbar.setValue(0)
            self.picasso_detect.setEnabled(True)
            self.picasso_fit.setEnabled(True)
            pass

    def _fit_localisations(self, progress_callback, detected_locs, min_net_gradient, box_size, camera_info, dataset_name, image_channel, frame_mode, detect_mode):

        try:
            from picasso import gausslq, lib, localize

            method = "lq"
            gain = 1

            localisation_centres = self.get_localisation_centres(detected_locs)

            if frame_mode.lower() == "active":
                image_data = self.dataset_dict[dataset_name][image_channel.lower()]["data"][0]
                image_data = np.expand_dims(image_data, axis=0)
            else:
                image_data = self.dataset_dict[dataset_name][image_channel.lower()]["data"]

            n_detected_frames = len(np.unique([loc[0] for loc in localisation_centres]))

            if n_detected_frames != image_data.shape[0]:
                print("Picasso can only Detect AND Fit localisations with same image frame mode")
            else:
                detected_loc_spots = localize.get_spots(image_data, detected_locs, box_size, camera_info)

                print(f"Picasso fitting {len(detected_locs)} spots...")

                if method == "lq":

                    fs = gausslq.fit_spots_parallel(detected_loc_spots, asynch=True)

                    n_tasks = len(fs)
                    while lib.n_futures_done(fs) < n_tasks:
                        progress = (lib.n_futures_done(fs)/ n_tasks) * 100
                        progress_callback.emit(progress)
                        time.sleep(0.1)

                    theta = gausslq.fits_from_futures(fs)
                    em = gain > 1

                    self.fitted_locs = gausslq.locs_from_fits(detected_locs, theta, box_size, em)

                    if frame_mode.lower() == "active":
                        for loc in self.fitted_locs:
                            loc.frame = self.viewer.dims.current_step[0]

                    n_frames = np.unique(self.fitted_locs.frame)
                    self.fitted_render_locs = {}

                    for frame in n_frames:
                        frame_locs = self.fitted_locs[self.fitted_locs['frame'] == frame].copy()
                        frame_coords = np.vstack((frame_locs.y, frame_locs.x)).T.tolist()

                        if frame not in self.fitted_render_locs.keys():
                            self.fitted_render_locs[frame] = []

                        self.fitted_render_locs[frame] = frame_coords

                        progress = (frame / n_frames) * 100
                        progress_callback.emit(progress)

        except:
            print(traceback.format_exc())
            self.picasso_progressbar.setValue(0)
            self.picasso_detect.setEnabled(True)
            self.picasso_fit.setEnabled(True)
            pass

    def _detect_localisations(self, progress_callback, min_net_gradient, roi, box_size, camera_info, dataset_name, image_channel, frame_mode, detect_mode):

        try:
            from picasso import localize

            min_net_gradient = int(min_net_gradient)

            if frame_mode.lower() == "active":
                image_data = self.dataset_dict[dataset_name][image_channel.lower()]["data"][0]
                image_data = np.expand_dims(image_data, axis=0)
            else:
                image_data = self.dataset_dict[dataset_name][image_channel.lower()]["data"]

            n_frames = image_data.shape[0]

            curr, futures = localize.identify_async(image_data, min_net_gradient, box_size, roi=roi)

            while curr[0] < n_frames:
                progress = (curr[0]/ n_frames) * 100
                progress_callback.emit(progress)
                time.sleep(0.1)

            self.detected_locs = localize.identifications_from_futures(futures)

            if frame_mode.lower() == "active":
                for loc in self.detected_locs:
                    loc.frame = self.viewer.dims.current_step[0]


            n_frames = np.unique(self.detected_locs.frame)
            self.detected_render_locs = {}

            for frame in n_frames:
                frame_locs = self.detected_locs[self.detected_locs['frame'] == frame].copy()
                frame_coords = np.vstack((frame_locs.y, frame_locs.x)).T.tolist()

                if frame not in self.detected_render_locs.keys():
                    self.detected_render_locs[frame] = []

                self.detected_render_locs[frame] = frame_coords

                progress = (frame / n_frames) * 100
                progress_callback.emit(progress)

        except:
            print(traceback.format_exc())
            self.picasso_progressbar.setValue(0)
            self.picasso_detect.setEnabled(True)
            self.picasso_fit.setEnabled(True)
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

    def gapseq_picasso_detect(self):

        try:
            if self.dataset_dict != {}:

                self.picasso_progressbar.setValue(0)
                self.picasso_detect.setEnabled(False)
                self.picasso_fit.setEnabled(False)

                min_net_gradient = self.picasso_min_net_gradient.text()
                box_size = int(self.picasso_box_size.currentText())
                dataset_name = self.picasso_dataset.currentText()
                image_channel = self.picasso_channel.currentText()
                frame_mode = self.picasso_frame_mode.currentText()
                detect_mode = self.picasso_detect_mode.currentText()

                roi = self.generate_roi()

                camera_info = {"baseline": 100.0, "gain": 1, "sensitivity": 1.0, "qe": 0.9, }

                if min_net_gradient.isdigit() and image_channel != "":
                    worker = Worker(self._detect_localisations,
                        min_net_gradient=min_net_gradient,
                        roi = roi,
                        box_size=box_size,
                        camera_info=camera_info,
                        dataset_name=dataset_name,
                        image_channel=image_channel,
                        frame_mode=frame_mode,
                        detect_mode=detect_mode)
                    worker.signals.progress.connect(partial(self.gapseq_progress, progress_bar=self.picasso_progressbar))
                    worker.signals.finished.connect(self._detect_localisations_cleanup)
                    self.threadpool.start(worker)

        except:
            print(traceback.format_exc())
            self.picasso_progressbar.setValue(0)
            self.picasso_detect.setEnabled(True)
            self.picasso_fit.setEnabled(True)
            pass

    def gapseq_picasso_fit(self):

        try:

            min_net_gradient = self.picasso_min_net_gradient.text()
            box_size = int(self.picasso_box_size.currentText())
            dataset_name = self.picasso_dataset.currentText()
            image_channel = self.picasso_channel.currentText()
            frame_mode = self.picasso_frame_mode.currentText()
            detect_mode = self.picasso_detect_mode.currentText()

            if detect_mode.lower() == "fiducials":
                localisation_dict = self.localisation_dict["fiducials"][dataset_name][image_channel.lower()]
            else:
                localisation_dict = self.localisation_dict["bounding_boxes"]

            if "localisations" in localisation_dict.keys():

                self.picasso_progressbar.setValue(0)
                self.picasso_detect.setEnabled(False)
                self.picasso_fit.setEnabled(False)

                detected_locs = localisation_dict["localisations"]

                camera_info = {"baseline": 100.0, "gain": 1, "sensitivity": 1.0, "qe": 0.9, }

                if min_net_gradient.isdigit() and image_channel != "":
                    worker = Worker(self._fit_localisations,
                        detected_locs=detected_locs,
                        min_net_gradient=min_net_gradient,
                        box_size=box_size,
                        camera_info=camera_info,
                        dataset_name=dataset_name,
                        image_channel=image_channel,
                        frame_mode=frame_mode,
                        detect_mode=detect_mode)

                    worker.signals.progress.connect(partial(self.gapseq_progress, progress_bar=self.picasso_progressbar))
                    worker.signals.finished.connect(self._fit_localisations_cleanup)
                    self.threadpool.start(worker)

        except:
            print(traceback.format_exc())
            self.picasso_progressbar.setValue(0)
            self.picasso_detect.setEnabled(True)
            self.picasso_fit.setEnabled(True)
            pass

