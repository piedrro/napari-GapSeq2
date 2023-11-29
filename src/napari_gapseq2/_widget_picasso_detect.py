import copy
import numpy as np
import traceback
from napari_gapseq2._widget_utils_worker import Worker
import time

class _picasso_detect_utils:

    def populate_localisation_dict(self, locs, detect_mode, dataset_name, image_channel):

        detect_mode = detect_mode.lower()

        try:

            if detect_mode == "fiducials":

                loc_centres = self.get_localisation_centres(locs)

                self.localisation_dict["fiducials"][dataset_name][image_channel.lower()] = {"localisations": [], "localisation_centres": []}

                self.localisation_dict["fiducials"][dataset_name][image_channel.lower()]["localisations"] = locs.copy()
                self.localisation_dict["fiducials"][dataset_name][image_channel.lower()]["localisation_centres"] = loc_centres.copy()

            else:

                loc_centres = self.get_localisation_centres(locs)

                self.localisation_dict["bounding_boxes"]["localisations"] = locs.copy()
                self.localisation_dict["bounding_boxes"]["localisation_centres"] = loc_centres.copy()

        except:
            print(traceback.format_exc())


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
            localisation_centres = self.get_localisation_centres(self.detected_locs)

            detect_mode = self.picasso_detect_mode.currentText()
            dataset_name = self.picasso_dataset.currentText()
            image_channel = self.picasso_channel.currentText()

            self.populate_localisation_dict(self.detected_locs, detect_mode, dataset_name, image_channel)

            n_frames = len(np.unique([loc[0] for loc in self.detected_locs]))
            print("detected {} localisations from {} frame(s)".format(len(localisation_centres), n_frames))

            self.draw_fiducials()
            self.draw_bounding_boxes()

            self.gapseq_dataset_selector.setCurrentIndex(self.gapseq_dataset_selector.findText(dataset_name))


        except:
            print(traceback.format_exc())
            pass


    def _fit_localisations_cleanup(self):

        try:
            localisation_centres = self.get_localisation_centres(self.fitted_locs)

            detect_mode = self.picasso_detect_mode.currentText()
            dataset_name = self.picasso_dataset.currentText()
            image_channel = self.picasso_channel.currentText()

            self.populate_localisation_dict(self.fitted_locs, detect_mode, dataset_name, image_channel)

            n_frames = len(np.unique([loc[0] for loc in self.fitted_locs]))
            print("Fitted {} localisations from {} frame(s)".format(len(localisation_centres), n_frames))

            self.draw_fiducials()
            self.draw_bounding_boxes()

            self.gapseq_dataset_selector.setCurrentIndex(self.gapseq_dataset_selector.findText(dataset_name))
        except:
            print(traceback.format_exc())
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
                        progress = (lib.n_futures_done(fs) / n_tasks) * 100
                        progress_callback.emit(progress)
                        time.sleep(0.1)

                    theta = gausslq.fits_from_futures(fs)
                    em = gain > 1
                    self.fitted_locs = gausslq.locs_from_fits(detected_locs, theta, box_size, em)

                    self.fitted_locs = self.filter_localisations(self.fitted_locs)

                print(f"Picasso fitted {len(self.fitted_locs)} spots")

        except:
            print(traceback.format_exc())
            pass

    def _detect_localisations(self, progress_callback, min_net_gradient, box_size, camera_info, dataset_name, image_channel, frame_mode, detect_mode):

        try:
            from picasso import localize

            min_net_gradient = int(min_net_gradient)

            if frame_mode.lower() == "active":
                image_data = self.dataset_dict[dataset_name][image_channel.lower()]["data"][0]
                image_data = np.expand_dims(image_data, axis=0)
            else:
                image_data = self.dataset_dict[dataset_name][image_channel.lower()]["data"]

            curr, futures = localize.identify_async(image_data, min_net_gradient, box_size, roi=None)
            self.detected_locs = localize.identifications_from_futures(futures)
            self.detected_locs = self.filter_localisations(self.detected_locs)

            if frame_mode.lower() == "active":
                for loc in self.detected_locs:
                    loc.frame = self.viewer.dims.current_step[0]

        except:
            print(traceback.format_exc())
            pass


    def gapseq_picasso_detect(self):

        try:
            if self.dataset_dict != {}:

                min_net_gradient = self.picasso_min_net_gradient.text()
                dataset_name = self.picasso_dataset.currentText()
                image_channel = self.picasso_channel.currentText()
                frame_mode = self.picasso_frame_mode.currentText()
                detect_mode = self.picasso_detect_mode.currentText()

                if detect_mode.lower() == "undrift fiducials":
                    self.undrift_channel = image_channel

                camera_info = {"baseline": 100.0, "gain": 1, "sensitivity": 1.0, "qe": 0.9, }

                if min_net_gradient.isdigit() and image_channel != "":
                    worker = Worker(self._detect_localisations,
                        min_net_gradient=min_net_gradient,
                        box_size=5,
                        camera_info=camera_info,
                        dataset_name=dataset_name,
                        image_channel=image_channel,
                        frame_mode=frame_mode,
                        detect_mode=detect_mode)

                    worker.signals.finished.connect(self._detect_localisations_cleanup)
                    self.threadpool.start(worker)

        except:
            print(traceback.format_exc())
            pass

    def gapseq_picasso_fit(self):

        try:

            min_net_gradient = self.picasso_min_net_gradient.text()
            dataset_name = self.picasso_dataset.currentText()
            image_channel = self.picasso_channel.currentText()
            frame_mode = self.picasso_frame_mode.currentText()
            detect_mode = self.picasso_detect_mode.currentText()

            if detect_mode.lower() == "fiducials":
                localisation_dict = self.localisation_dict["fiducials"][dataset_name][image_channel.lower()]
            else:
                localisation_dict = self.localisation_dict["bounding_boxes"]

            if "localisations" in localisation_dict.keys():
                detected_locs = localisation_dict["localisations"]

                camera_info = {"baseline": 100.0, "gain": 1, "sensitivity": 1.0, "qe": 0.9, }

                if min_net_gradient.isdigit() and image_channel != "":
                    worker = Worker(self._fit_localisations,
                        detected_locs=detected_locs,
                        min_net_gradient=min_net_gradient,
                        box_size=5,
                        camera_info=camera_info,
                        dataset_name=dataset_name,
                        image_channel=image_channel,
                        frame_mode=frame_mode,
                        detect_mode=detect_mode)

                    worker.signals.finished.connect(self._fit_localisations_cleanup)
                    self.threadpool.start(worker)

        except:
            print(traceback.format_exc())
            pass

