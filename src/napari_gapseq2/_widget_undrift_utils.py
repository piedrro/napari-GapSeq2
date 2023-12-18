import traceback
import numpy as np
import copy
from napari_gapseq2._widget_utils_compute import Worker
import scipy.ndimage
import multiprocessing
from multiprocessing import Process, shared_memory, Pool
import copy
from functools import partial
import concurrent.futures




def undrift_image(dat):

    undrifted_data = [None, None]

    try:

        drift = dat["drift"]
        frame_index = dat["frame_index"]

        # Access the shared memory
        shared_mem = shared_memory.SharedMemory(name=dat["shared_memory_name"])
        np_array = np.ndarray(dat["shape"], dtype=dat["dtype"], buffer=shared_mem.buf)

        # Perform preprocessing steps and overwrite original image
        img = np_array[frame_index]

        drift = [-drift[1], -drift[0]]

        img = scipy.ndimage.shift(img, drift, mode='constant', cval=0.0)

        # overwrite the shared memory block
        np_array[frame_index] = img

        # Ensure to close shared memory in child processes
        shared_mem.close()

    except:
        print(traceback.format_exc())
        pass

    return frame_index


class _undrift_utils:

    def _gapseq_undrift_images(self, progress_callback=None, drift=None):

        try:

            dataset_name = self.gapseq_dataset_selector.currentText()
            channel_names = list(self.dataset_dict[dataset_name].keys())

            self.shared_images = self.create_shared_images(channel_list = channel_names)

            compute_jobs = []

            for image_dict in self.shared_images:

                n_frames = image_dict['shape'][0]

                for frame_index in range(n_frames):

                    compute_job = {"frame_index": frame_index,
                                   "shared_memory_name": image_dict['shared_memory_name'],
                                   "shape": image_dict['shape'],
                                   "dtype": image_dict['dtype'],
                                   "drift_index": frame_index,
                                   "drift": drift[frame_index]}

                    compute_jobs.append(compute_job)

            cpu_count = int(multiprocessing.cpu_count() * 0.9)
            timeout_duration = 10  # Timeout in seconds

            with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
                # Submit all jobs and store the future objects
                futures = {executor.submit(undrift_image, job): job for job in compute_jobs}

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

            self.restore_shared_images()

        except:
            self.restore_shared_images()
            self.apply_undrift.setEnabled(True)
            self.undrift_progressbar.setValue(0)
            pass


    def _gapseq_undrift_images_cleanup(self):

        try:

            layer_names = [layer.name for layer in self.viewer.layers]

            dataset_name = self.gapseq_dataset_selector.currentText()
            active_channel = self.active_channel

            if dataset_name in layer_names:
                self.viewer.layers.data = self.dataset_dict[dataset_name][active_channel]["data"]

            for layer in self.viewer.layers:
                layer.refresh()

            self.undrift_localisations()

            self.draw_fiducials(update_vis=True)

            self.apply_undrift.setEnabled(True)
            self.undrift_progressbar.setValue(0)

            self.image_layer.data = self.dataset_dict[self.active_dataset][self.active_channel]["data"]

        except:
            print(traceback.format_exc())
            self.apply_undrift.setEnabled(True)
            self.undrift_progressbar.setValue(0)
            pass

    def gapseq_undrift_images(self):

        try:

            dataset_name = self.undrift_dataset_selector.currentText()
            undrift_channel = self.undrift_channel_selector.currentText()

            channel_dict = self.dataset_dict[dataset_name][undrift_channel.lower()]

            if "drift" in channel_dict.keys():
                drift = channel_dict["drift"]

                self.apply_undrift.setEnabled(False)
                self.undrift_progressbar.setValue(0)

                worker = Worker(self._gapseq_undrift_images, drift=drift)
                worker.signals.progress.connect(partial(self.gapseq_progress, progress_bar=self.undrift_progressbar))
                worker.signals.finished.connect(self._gapseq_undrift_images_cleanup)
                self.threadpool.start(worker)

        except:
            print(traceback.format_exc())
            self.apply_undrift.setEnabled(True)
            self.undrift_progressbar.setValue(0)
            pass

    def undrift_localisations(self):

        try:

            dataset_name = self.gapseq_dataset_selector.currentText()

            for channel_name, channel_data in self.dataset_dict[dataset_name].items():

                fiducial_dict = self.localisation_dict["fiducials"][dataset_name][channel_name.lower()]

                if "drift" in channel_data.keys() and "localisations" in fiducial_dict.keys():

                    locs = fiducial_dict["localisations"]
                    n_detected_frames = len(np.unique([loc.frame for loc in locs]))
                    n_image_frames = len(channel_data["data"])

                    drift = channel_data["drift"]

                    render_locs = {}

                    for loc in locs:
                        frame = loc.frame
                        loc.x = loc.x - drift[loc.frame][1]
                        loc.y = loc.y - drift[loc.frame][0]

                        if frame not in render_locs.keys():
                            render_locs[frame] = []

                        render_locs[frame].append([loc.y, loc.x])

                    localisation_centres = self.get_localisation_centres(locs)

                    self.localisation_dict["fiducials"][dataset_name][channel_name.lower()]["localisations"] = locs
                    self.localisation_dict["fiducials"][dataset_name][channel_name.lower()]["localisation_centres"] = localisation_centres
                    self.localisation_dict["fiducials"][dataset_name][channel_name.lower()]["render_locs"] = render_locs

        except:
            print(traceback.format_exc())
            pass

    def _picasso_undrift_cleanup(self):

        self.undrift_progressbar.setValue(0)
        self.detect_undrift.setEnabled(True)
        self.apply_undrift.setEnabled(True)
        self.undrift_channel_selector.setEnabled(True)

    def _picasso_undrift(self, progress_callback, undrift_locs, picasso_info, segmentation=20):

        try:
            from picasso.postprocess import undrift as picasso_undrift, segment
            from picasso.imageprocess import rcc

            n_frames = picasso_info[0]["Frames"]
            len_segments = n_frames // segmentation
            n_pairs = int(len_segments * (len_segments - 1) / 2)

            def segmentation_callback(progress, start=0,end=50):
                progress = start + (progress/len_segments) * end
                progress_callback.emit(progress)

            def undrift_callback(progress, start=50,end=100):
                progress = start + (progress/n_pairs) * end
                progress_callback.emit(progress)

            drift, _ = picasso_undrift(
                undrift_locs,
                picasso_info,
                segmentation=segmentation,
                display=False,
                segmentation_callback=segmentation_callback,
                rcc_callback=undrift_callback,
            )

            dataset_name = self.gapseq_dataset_selector.currentText()

            for channel_name, channel_data in self.dataset_dict[dataset_name].items():
                channel_data["drift"] = drift

        except:
            print(traceback.format_exc())
            self.undrift_progressbar.setValue(0)
            self.detect_undrift.setEnabled(True)
            self.apply_undrift.setEnabled(True)
            self.undrift_channel_selector.setEnabled(True)
            pass

    def gapseq_picasso_undrift(self):

        try:

            if self.dataset_dict != {}:

                dataset_name = self.gapseq_dataset_selector.currentText()
                undrift_channel = self.undrift_channel_selector.currentText()

                fiducial_dict = self.localisation_dict["fiducials"][dataset_name][undrift_channel.lower()]

                if "localisations" not in fiducial_dict.keys():

                    print(f"Fiducials not detected in dataset: '{dataset_name}' channel: '{undrift_channel}'")

                else:
                    fitted = fiducial_dict["fitted"]

                    undrift_locs = fiducial_dict["localisations"].copy()
                    n_detected_frames = len(np.unique([loc.frame for loc in undrift_locs]))

                    n_frames, height, width = self.dataset_dict[dataset_name][undrift_channel.lower()]["data"].shape

                    if n_detected_frames != n_frames or fitted is False:

                        print("Fitted fiducials in all frames required for undrifting.")

                    else:

                        self.undrift_progressbar.setValue(0)
                        self.detect_undrift.setEnabled(False)
                        self.apply_undrift.setEnabled(False)
                        self.undrift_channel_selector.setEnabled(False)

                        picasso_info = [{'Frames': n_frames, 'Height': height, 'Width': width}, {}]

                        worker = Worker(self._picasso_undrift, undrift_locs=undrift_locs, picasso_info=picasso_info)
                        worker.signals.progress.connect(partial(self.gapseq_progress, progress_bar=self.undrift_progressbar))
                        worker.signals.finished.connect(self._picasso_undrift_cleanup)
                        self.threadpool.start(worker)

        except:
            print(traceback.format_exc())
            self.undrift_progressbar.setValue(0)
            self.detect_undrift.setEnabled(True)
            self.apply_undrift.setEnabled(True)
            self.undrift_channel_selector.setEnabled(True)
            pass


