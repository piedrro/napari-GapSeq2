import numpy as np
import traceback
import matplotlib.pyplot as plt
import tifffile
import os
import psutil
from qtpy.QtWidgets import QFileDialog

class _export_utils:

    def update_export_options(self):

        if self.dataset_dict != {}:

            dataset_name = self.export_dataset.currentText()

            export_channel_list = []

            if dataset_name in self.dataset_dict.keys():

                import_mode_list = []

                for channel_name, channel_data in self.dataset_dict[dataset_name].items():

                    import_mode_list.append(channel_data["import_mode"])

                    export_channel_list.append(channel_name.upper())

                import_mode_list = list(set(import_mode_list))

                export_channel_list = import_mode_list + export_channel_list

                self.export_channel.clear()
                self.export_channel.addItems(export_channel_list)



    def get_free_RAM(self):

            try:

                memory_info = psutil.virtual_memory()
                available_memory = memory_info.available

                return available_memory

            except:
                print(traceback.format_exc())
                pass

    def export_data(self):

        try:


            if self.dataset_dict != {}:

                dataset_name = self.export_dataset.currentText()
                export_channel = self.export_channel.currentText()

                if export_channel == "ALEX":
                    self.export_alex_data(dataset_name)
                elif export_channel == "FRET":
                    self.export_fret_data(dataset_name)
                else:
                    self.export_channel_data(dataset_name, export_channel)

        except:
            print(traceback.format_exc())
            pass

    def export_fret_data(self, dataset_name):

        print("Exporting FRET data")

    def export_channel_data(self, dataset_name, export_channel):

        try:

            channel_dict = self.dataset_dict[dataset_name][export_channel.lower()]
            image_path = channel_dict["path"]
            import_mode = channel_dict["import_mode"]

            export_path = os.path.normpath(image_path)
            export_dir = os.path.dirname(export_path)
            file_name = os.path.basename(export_path)
            file_name = os.path.splitext(file_name)[0]
            if import_mode in ["ALEX", "FRET"]:
                export_path = os.path.join(export_dir,file_name + f"_{export_channel}_gapseq_processed.tif")
            else:
                export_path = os.path.join(export_dir,file_name + "_gapseq_processed.tif")

            image = channel_dict["data"]

            tifffile.imwrite(export_path, image)

            print(f"Exported {export_channel} data to {export_path}")

        except:
            print(traceback.format_exc())
            pass


    def export_alex_data(self, dataset_name):

        try:

            dataset_dict = self.dataset_dict[dataset_name]

            image_shapes = [channel_data["data"].shape for channel_name, channel_data in dataset_dict.items()]
            image_disk_sizes = [abs(int(channel_data["data"].nbytes)) for channel_name, channel_data in dataset_dict.items()]
            image_paths = [channel_data["path"] for channel_name, channel_data in dataset_dict.items()]

            export_path = os.path.normpath(image_paths[0])
            export_dir = os.path.dirname(export_path)
            file_name = os.path.basename(export_path)
            file_name = os.path.splitext(file_name)[0]
            export_path = os.path.join(export_dir,file_name + "_gapseq_processed.tif")

            export_path = QFileDialog.getSaveFileName(self, 'Save ALEX data', export_path, 'Text files (*.tif)')[0]

            if export_path == "":

                image_disk_size = 0
                for size in image_disk_sizes:
                    image_disk_size += size

                free_RAM = self.get_free_RAM()

                if free_RAM > image_disk_size:
                    disk_export = False
                else:
                    disk_export = True

                n_frames = image_shapes[0][0]
                height = image_shapes[0][1]
                width = image_shapes[0][2]

                export_shape = (n_frames*2,height,width*2)


                if disk_export == True:
                    disk_array_path = 'mmapped_array.dat'
                    export_array = np.memmap(disk_array_path, dtype='uint16', mode='w+', shape=export_shape)
                else:
                    disk_array_path = None
                    export_array = np.zeros(export_shape, dtype="uint16")

                iter = 0

                for channel_name, channel_data in dataset_dict.items():

                    channel_layout = channel_data["channel_layout"]
                    alex_first_frame = channel_data["alex_first_frame"]
                    channel_ref = channel_data["channel_ref"]

                    for frame_index in range(len(channel_data["data"])):

                        frame = channel_data["data"][frame_index]

                        left_image = False
                        if channel_layout == "Donor-Acceptor":
                            if channel_ref[-1] == "d":
                                left_image = True
                        else:
                            if channel_ref[-1] == "a":
                                left_image = True

                        if alex_first_frame == "Donor":
                            if channel_ref[0] == "d":
                                mapped_frame_index = frame_index * 2
                            else:
                                mapped_frame_index = frame_index * 2 + 1
                        else:
                            if channel_ref[0] == "a":
                                mapped_frame_index = frame_index * 2
                            else:
                                mapped_frame_index = frame_index * 2 + 1

                        if left_image == True:
                            export_array[mapped_frame_index][0:height,0:width] = frame
                        else:
                            export_array[mapped_frame_index][0:height,width:width*2] = frame

                        iter += 1

                if disk_export:
                    # Make sure to flush changes to disk
                    export_array.flush()

                    with tifffile.TiffWriter(export_path) as tiff:
                        for idx in range(export_array.shape[0]):
                            tiff.write(export_array[idx])

                    if os.path.exists(disk_array_path):
                        os.remove(disk_array_path)

                else:
                    tifffile.imwrite(export_path, export_array)
                    del export_array

                print(f"Exported data to {export_path}")

        except:
            print(traceback.format_exc())
            return None