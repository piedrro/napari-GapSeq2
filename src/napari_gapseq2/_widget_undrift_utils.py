import traceback
import numpy as np
import copy
from napari_gapseq2._widget_utils_worker import Worker
import scipy.ndimage
import multiprocessing
from multiprocessing import Process, shared_memory, Pool
import copy


def undrift_image(dat):

    undrifted_data = [None, None]

    try:
        # Access the shared memory
        shared_mem = shared_memory.SharedMemory(name=dat["shared_memory_name"])
        np_array = np.ndarray(dat["shape"], dtype=dat["dtype"], buffer=shared_mem.buf)

        # Perform preprocessing steps and overwrite original image
        img = np_array[dat["drift_index"]]

        img = scipy.ndimage.shift(img, dat["drift"], mode='constant', cval=0.0)

        # overwrite the shared memory block
        np_array[dat["drift_index"]] = img

        # Ensure to close shared memory in child processes
        shared_mem.close()

        undrifted_data = [dat["drift_index"], img]

    except:
        print(traceback.format_exc())
        pass

    return undrifted_data


class _undrift_utils:


    def _gapseq_undrift_images(self, progress_callback=None):

        for channel_name, channel_data in self.image_dict.items():

            try:

                image = channel_data.pop("data")
                image = copy.deepcopy(image)

                shared_mem = shared_memory.SharedMemory(create=True, size=image.nbytes)
                shared_memory_name = shared_mem.name
                shared_image = np.ndarray(image.shape, dtype=image.dtype, buffer=shared_mem.buf)
                shared_image[:] = image[:]

                undrift_jobs = []

                for drift_index, drift in enumerate(self.drift):

                    drift = list(drift)

                    undrift_jobs.append({"drift_index": drift_index,
                                         "drift": drift,
                                         "shared_memory_name": shared_memory_name,
                                         "shape": image.shape,
                                         "dtype": image.dtype})

                cpu_count = multiprocessing.cpu_count() // 2

                with Pool(cpu_count) as p:
                    _ = p.map(undrift_image, undrift_jobs)

                self.image_dict[channel_name]["data"] = shared_image.copy()

                shared_mem.close()
                shared_mem.unlink()

            except:
                self.image_dict[channel_name]["data"] = image


    def _gapseq_undrift_images_cleanup(self):

        print("Undrifted images")

    def gapseq_undrift_images(self):

        try:

            if self.drift is not None:
                worker = Worker(self._gapseq_undrift_images)
                worker.signals.finished.connect(self._gapseq_undrift_images_cleanup)
                self.threadpool.start(worker)

        except:
            print(traceback.format_exc())
            pass

    def undrift_localisations(self, undrift_channel, undrift_mode):

        try:

            if self.drift is not None:

                localisations = self.image_dict[undrift_channel][undrift_mode]["localisations"].copy()

                n_frames = len(np.unique([loc.frame for loc in localisations]))

                new_localisations = []

                frame_0_locs = [loc for loc in localisations if loc.frame == 0]

                for frame in range(n_frames):

                    frame_locs = copy.deepcopy(frame_0_locs)

                    for loc in frame_locs:

                        loc.frame = frame
                        loc.x = loc.x + self.drift[frame][0]
                        loc.y = loc.y + self.drift[frame][1]

                    new_localisations.extend(frame_locs)

                new_localisations = np.rec.fromrecords(new_localisations, dtype=localisations.dtype)

                new_localisation_centres = self.get_localisation_centres(new_localisations)

                self.image_dict[undrift_channel][undrift_mode]["localisations"] = new_localisations
                self.image_dict[undrift_channel][undrift_mode]["localisation_centres"] = new_localisation_centres

        except:
            print(traceback.format_exc())
            pass


    def _picasso_undrift_cleanup(self):

        undrift_mode = "undrift fiducials"
        undrift_channel = self.undrift_channel

        self.undrift_localisations(undrift_channel, undrift_mode)
        self.draw_localisations()

    def _picasso_undrift(self, progress_callback, undrift_locs, picasso_info):

        self.drift = None

        try:
            from picasso.postprocess import undrift as picasso_undrift

            print("Picasso Undrifting...")

            drift, _ = picasso_undrift(
                undrift_locs,
                picasso_info,
                segmentation=20,
                display=False,)

            self.drift = drift

        except:
            print(traceback.format_exc())
            pass

    def gapseq_picasso_undrift(self):

        try:

            if self.image_dict != {}:

                undrift_mode = "undrift fiducials"
                undrift_channel = self.undrift_channel

                if undrift_channel == None:
                    print("No Undrift Fiducials detected")
                else:
                    if undrift_mode in self.image_dict[undrift_channel].keys():
                        if "localisations" in self.image_dict[undrift_channel][undrift_mode].keys():

                            undrift_locs = self.image_dict[undrift_channel][undrift_mode]["localisations"].copy()
                            localisation_centres = self.image_dict[undrift_channel][undrift_mode]["localisation_centres"].copy()

                            n_detected_frames = len(np.unique([loc[0] for loc in localisation_centres]))

                            n_frames, height, width = self.image_dict[undrift_channel]["data"].shape

                            if n_detected_frames != n_frames:
                                print("Undrift Fidicuals not detected in all frames")
                            else:

                                picasso_info = [{'Frames': n_frames, 'Height': height, 'Width': width}, {}]

                                worker = Worker(self._picasso_undrift, undrift_locs=undrift_locs, picasso_info=picasso_info)
                                worker.signals.finished.connect(self._picasso_undrift_cleanup)
                                self.threadpool.start(worker)

        except:
            print(traceback.format_exc())
            pass


