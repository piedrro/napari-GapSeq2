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
from picasso.postprocess import undrift as picasso_undrift, segment
from picasso.imageprocess import rcc



def undrift_image(dat):

    undrifted_data = [None, None]

    try:

        drift = dat["drift"]
        frame_index = dat["frame_index"]
        stop_event = dat["stop_event"]

        if not stop_event.is_set():

            # Access the shared memory
            shared_mem = shared_memory.SharedMemory(name=dat["shared_memory_name"])
            np_array = np.ndarray(dat["shape"], dtype=dat["dtype"], buffer=shared_mem.buf)

            # Perform preprocessing steps and overwrite original image
            img = np_array[frame_index]

            drift = [-drift[1], -drift[0]]

            img = scipy.ndimage.shift(img, drift, mode='constant', cval=0.0)

            # overwrite the shared memory block
            np_array[frame_index] = img

    except:
        print(traceback.format_exc())
        pass

    return frame_index




class _undrift_utils:

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


    def _undrift_images_finished(self):

        try:

            self.image_layer.data = self.dataset_dict[self.active_dataset][self.active_channel]["data"]

            self.undrift_localisations()
            self.draw_fiducials(update_vis=True)

            for layer in self.viewer.layers:
                layer.refresh()

            self.update_ui()

        except:
            print(traceback.format_exc())

            self.update_ui()

    def _compute_undrift(self, progress_callback, undrift_dict, segmentation=20):

        total_datasets = len(undrift_dict)
        progress_per_dataset = 50 / total_datasets

        for dataset_index, (dataset, dataset_dict) in enumerate(undrift_dict.items()):

            n_locs = dataset_dict["n_locs"]
            loc_dict = dataset_dict["loc_dict"]
            undrift_locs = loc_dict["localisations"]
            picasso_info = dataset_dict["picasso_info"]

            n_frames = picasso_info[0]["Frames"]
            len_segments = n_frames // segmentation
            n_pairs = int(len_segments * (len_segments - 1))

            dataset_start = dataset_index * progress_per_dataset
            seg_start = dataset_start
            seg_end = dataset_start + progress_per_dataset / 2
            undrift_start = seg_end
            undrift_end = dataset_start + progress_per_dataset

            def segmentation_callback(progress, start=seg_start, end=seg_end):
                progress = start + (progress / len_segments) * end
                progress_callback.emit(progress)

            def undrift_callback(progress, start=undrift_start, end=undrift_end):
                progress = start + (progress / n_pairs) * end
                progress_callback.emit(progress)

            drift, _ = picasso_undrift(undrift_locs,
                picasso_info, segmentation=segmentation,
                display=False, segmentation_callback=segmentation_callback,
                rcc_callback=undrift_callback, )

            undrift_dict[dataset]["drift"] = drift

        return undrift_dict


    def _undrift_images(self, progress_callback, undrift_dict, segmentation=20):

        try:

            undrift_dict = self._compute_undrift(progress_callback, undrift_dict, segmentation=segmentation)

            dataset_list = list(undrift_dict.keys())
            channel_list = [undrift_dict[dataset_list[0]]["channel"]]

            self.shared_images = self.create_shared_images()

            compute_jobs = []

            for image_dict in self.shared_images:

                dataset = image_dict["dataset"]
                n_frames = image_dict['shape'][0]

                frame_index_list = list(range(n_frames))
                image_drift = undrift_dict[dataset]["drift"]

                for frame_index, frame_drift in zip(frame_index_list, image_drift):

                    compute_jobs.append({"shared_memory_name": image_dict["shared_memory_name"],
                                         "shape": image_dict["shape"],
                                         "dtype": image_dict["dtype"],
                                         "frame_index": frame_index,
                                         "drift": frame_drift,
                                         "stop_event": self.stop_event,
                                         })

            cpu_count = int(multiprocessing.cpu_count() * 0.9)
            timeout_duration = 10  # Timeout in seconds

            with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
                # Submit all jobs and store the future objects
                futures = {executor.submit(undrift_image, job): job for job in compute_jobs}

                iter = 0
                for future in concurrent.futures.as_completed(futures):

                    if self.stop_event.is_set():
                        future.cancel()
                    else:
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
                        progress = 50 + int((iter / len(compute_jobs)) * 50)
                        progress_callback.emit(progress)  # Emit the signal

            self.restore_shared_images()

        except:
            self.restore_shared_images()

            self.update_ui()

            print(traceback.format_exc())
            pass


    def undrift_images(self):

        try:

            dataset = self.undrift_dataset_selector.currentText()
            channel = self.undrift_channel_selector.currentText()

            if dataset == "All Datasets":
                dataset_list = list(self.dataset_dict.keys())
            else:
                dataset_list = [dataset]

            undrift_dict = {}

            for dataset in dataset_list:
                loc_dict, n_locs, _ = self.get_loc_dict(dataset, channel.lower())
                if n_locs > 0 and loc_dict["fitted"] == True:

                    n_frames,height,width = self.dataset_dict[dataset][channel.lower()]["data"].shape
                    picasso_info = [{'Frames': n_frames, 'Height': height, 'Width': width}, {}]

                    undrift_dict[dataset] = {"loc_dict": loc_dict, "n_locs": n_locs,
                                             "picasso_info": picasso_info, "channel": channel.lower()}
                else:
                    print("No fitted localizations found for dataset: " + dataset)

            if undrift_dict != {}:

                self.update_ui(init=True)

                self.worker = Worker(self._undrift_images, undrift_dict=undrift_dict)
                self.worker.signals.progress.connect(partial(self.gapseq_progress, progress_bar=self.undrift_progressbar))
                self.worker.signals.finished.connect(self._undrift_images_finished)
                self.worker.signals.error.connect(self.update_ui)
                self.threadpool.start(self.worker)

        except:
            self.update_ui()

            print(traceback.format_exc())
            pass

