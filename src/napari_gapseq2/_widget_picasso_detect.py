import copy
import numpy as np
import traceback
from napari_gapseq2._widget_utils_worker import Worker
import time

class _picasso_detect_utils:

    def populate_localisation_dict(self, locs, detect_mode, dataset_name, image_channel):

        detect_mode = detect_mode.lower()


        try:
            detect_mode = detect_mode.lower()
            excitation, emission = image_channel

            if detect_mode == "fiducials":

                loc_centres = self.get_localisation_centres(locs)

                self.localisation_dict["fiducials"][dataset_name][image_channel.lower()] = {"localisations": [], "localisation_centres": []}

                self.localisation_dict["fiducials"][dataset_name][image_channel.lower()]["localisations"] = locs.copy()
                self.localisation_dict["fiducials"][dataset_name][image_channel.lower()]["localisation_centres"] = loc_centres.copy()


        #     else:
        #
        #         print("Detect mode: ", detect_mode)
        #
        #         if self.transform_matrix is None:
        #
        #             loc_centres = self.get_localisation_centres(locs)
        #
        #             for channel in self.image_dict.keys():
        #
        #                 if detect_mode not in self.image_dict[channel].keys():
        #                     self.image_dict[channel][detect_mode] = {"localisations": [], "localisation_centres": []}
        #
        #                 self.image_dict[channel][detect_mode]["localisations"] = locs.copy()
        #                 self.image_dict[channel][detect_mode]["localisation_centres"] = loc_centres.copy()
        #
        #         else:
        #
        #             if emission.lower() == "a":
        #                 donor_locs = copy.deepcopy(locs)
        #                 acceptor_locs = copy.deepcopy(locs)
        #                 acceptor_locs = self.apply_transform(acceptor_locs, inverse = False)
        #
        #                 donor_loc_centres = self.get_localisation_centres(donor_locs)
        #                 acceptor_loc_centres = self.get_localisation_centres(acceptor_locs)
        #             else:
        #                 acceptor_locs = copy.deepcopy(locs)
        #                 donor_locs = copy.deepcopy(locs)
        #                 donor_locs = self.apply_transform(donor_locs, inverse = True)
        #
        #                 acceptor_loc_centres = self.get_localisation_centres(acceptor_locs)
        #                 donor_loc_centres = self.get_localisation_centres(donor_locs)
        #
        #             for channel in self.image_dict.keys():
        #
        #                 channel_ex, channel_em = channel
        #
        #                 if detect_mode not in self.image_dict[channel].keys():
        #                     self.image_dict[channel][detect_mode] = {"localisations": [], "localisation_centres": []}
        #
        #                 if channel_em.lower() == "d":
        #                     self.image_dict[channel][detect_mode]["localisations"] = copy.deepcopy(acceptor_locs)
        #                     self.image_dict[channel][detect_mode]["localisation_centres"] = copy.deepcopy(acceptor_loc_centres)
        #
        #                 else:
        #                     self.image_dict[channel][detect_mode]["localisations"] = copy.deepcopy(donor_locs)
        #                     self.image_dict[channel][detect_mode]["localisation_centres"] = copy.deepcopy(donor_loc_centres)

        except:
            print(traceback.format_exc())




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
            # self.draw_localisations()

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
            # self.draw_localisations()

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

