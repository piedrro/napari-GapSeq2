import traceback
import numpy as np
import copy
from napari_gapseq2._widget_utils_worker import Worker
import scipy.ndimage
import multiprocessing
from multiprocessing import Process, shared_memory, Pool
import copy
from functools import partial

def undrift_image(dat):

    undrifted_data = [None, None]

    try:
        # Access the shared memory
        shared_mem = shared_memory.SharedMemory(name=dat["shared_memory_name"])
        np_array = np.ndarray(dat["shape"], dtype=dat["dtype"], buffer=shared_mem.buf)

        # Perform preprocessing steps and overwrite original image
        img = np_array[dat["drift_index"]]

        drift = dat["drift"]
        drift = [-drift[1], -drift[0]]

        img = scipy.ndimage.shift(img, drift, mode='constant', cval=0.0)

        # overwrite the shared memory block
        np_array[dat["drift_index"]] = img

        # Ensure to close shared memory in child processes
        shared_mem.close()

        undrifted_data = [dat["drift_index"], img]

    except:
        print(traceback.format_exc())
        pass

    return dat["drift_index"]


class _undrift_utils:

    def _gapseq_undrift_images(self, progress_callback=None, drift=None):

        dataset_name = self.gapseq_dataset_selector.currentText()

        total_frames = [len(self.dataset_dict[dataset_name][channel]["data"]) for channel in self.dataset_dict[dataset_name].keys()]
        total_frames = np.sum(total_frames)

        iter = []

        for channel_name, channel_data in self.dataset_dict[dataset_name].items():

            try:

                drift_list = channel_data["drift"]

                image = channel_data["data"]
                image = copy.deepcopy(image)

                shared_mem = shared_memory.SharedMemory(create=True, size=image.nbytes)
                shared_memory_name = shared_mem.name
                shared_image = np.ndarray(image.shape, dtype=image.dtype, buffer=shared_mem.buf)
                shared_image[:] = image[:]

                undrift_jobs = []

                for drift_index, drift in enumerate(drift_list):

                    drift = list(drift)

                    undrift_jobs.append({"drift_index": drift_index,
                                         "drift": drift,
                                         "shared_memory_name": shared_memory_name,
                                         "shape": image.shape,
                                         "dtype": image.dtype})

                cpu_count = int(multiprocessing.cpu_count()*0.75)

                def callback(*args, offset=0):
                    iter.append(1)
                    progress = int((len(iter) / total_frames) * 100)
                    if progress_callback != None:
                        progress_callback.emit(progress - offset)
                    return

                with Pool(cpu_count) as p:
                    imported_data = [p.apply_async(undrift_image, args=(i,), callback=callback) for i in undrift_jobs]
                    imported_data = [p.get() for p in imported_data]
                    p.close()

                self.dataset_dict[dataset_name][channel_name]["data"] = shared_image.copy()

                shared_mem.close()
                shared_mem.unlink()

            except:
                print(traceback.format_exc())
                self.apply_undrift.setEnabled(True)
                self.undrift_progressbar.setValue(0)
                self.dataset_dict[dataset_name][channel_name]["data"] = image


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


