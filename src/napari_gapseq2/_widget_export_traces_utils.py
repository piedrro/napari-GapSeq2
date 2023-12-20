import traceback
import os
from qtpy.QtWidgets import QFileDialog
from napari_gapseq2._widget_utils_compute import Worker
from functools import partial
import numpy as np
import json


class npEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int32):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


class _export_traces_utils:


    def get_export_traces_path(self, dialog=True):

        if self.traces_dict != {}:

            dataset_name = self.traces_export_dataset.currentText()
            export_channel = self.traces_export_channel.currentText()
            export_mode = self.traces_export_mode.currentText()

            if dataset_name == "All Datasets":
                dataset_name = list(self.traces_dict.keys())[0]
            if export_channel == "All Channels":
                export_channel = list(self.traces_dict[dataset_name].keys())[0]

            import_path = self.dataset_dict[dataset_name][export_channel.lower()]["path"]

            import_directory = os.path.dirname(import_path)
            file_name = os.path.basename(import_path)
            file_name, file_extension = os.path.splitext(file_name)

            if export_mode == "JSON Dataset":
                file_extension = ".json"
            elif export_mode == "DAT":
                file_extension = ".dat"
            elif export_mode == "Excel":
                file_extension = ".xlsx"
            else:
                print(f"Export mode {export_mode} not recognized.")

            export_path = os.path.join(import_directory, file_name + file_extension)
            export_path = os.path.normpath(export_path)

            if dialog:
                export_path = QFileDialog.getSaveFileName(self, "Save File", export_path, "All Files (*)")[0]

            export_directory = os.path.dirname(export_path)

            return export_path, export_directory

    def export_traces_json(self, progress_callback=None, export_path=""):

        json_dict = self.populate_json_dict(progress_callback)

        with open(export_path, "w") as f:
            json.dump(json_dict, f, cls=npEncoder)

        self.traces_export_status = True

    def export_traces_dat(self, progress_callback=None, export_path=""):

        try:
            print("Exporting traces to DAT...")
        except:
            print(traceback.format_exc())
            pass

    def export_traces_excel(self, progress_callback=None, export_path=""):

        try:

            print("Exporting traces to Excel...")
        except:
            print(traceback.format_exc())

            pass

    def populate_json_dict(self, progress_callback=None):

        dataset_name = self.traces_export_dataset.currentText()
        channel_name = self.traces_export_channel.currentText()
        metric_name = self.traces_export_metric.currentText()
        background_mode = self.traces_export_background.currentText()

        metric_key = self.get_dict_key(self.metric_dict, metric_name)

        if background_mode != "None":
            if "local" in background_mode.lower():
                background_metric_key = metric_key + "_local_bg"
            else:
                background_metric_key = metric_key + "_global_bg"
        else:
            background_metric_key = None

        if dataset_name == "All Datasets":
            dataset_list = list(self.traces_dict.keys())
        else:
            dataset_list = [dataset_name]

        if channel_name == "All Channels":

            channel_list = [channel for dataset_dict in self.traces_dict.values() for channel in dataset_dict.keys()]
            channel_list = list(set(channel_list))
            channel_list = [channel for channel in channel_list if "efficiency" not in channel.lower()]
            iteration_channel = channel_list[0]
            if set(["dd", "da"]).issubset(channel_list):
                channel_list.append("alex_efficiency")
            if set(["donor", "acceptor"]).issubset(channel_list):
                channel_list.append("fret_efficiency")

        elif channel_name.lower() == "fret":
            channel_list = ["donor", "acceptor"]
        elif channel_name.lower() == "alex":
            channel_list = ["dd", "da", "ad", "aa"]
        else:
            channel_list = [channel_name]

        json_dict = {"metadata": {}, "data": {}}

        n_traces = 0
        for dataset in dataset_list:
            if dataset in self.traces_dict:
                for channel in channel_list:
                    if channel in self.traces_dict[dataset]:
                        n_traces += len(self.traces_dict[dataset][channel].copy())

        iter = 0
        for dataset in dataset_list:

            if dataset not in json_dict["data"]:
                json_dict["data"][dataset] = []

            if channel_name == "All Channels" or "efficiency" in channel_name.lower():
                dataset_channels = self.traces_dict[dataset].keys()
                if set(["dd", "da"]).issubset(dataset_channels):
                    self.compute_alex_efficiency(dataset, metric_key, background_metric_key,
                        progress_callback, clip_data=False)
                elif set(["Donor", "Acceptor"]).issubset(dataset_channels):
                    self.compute_fret_efficiency(dataset, metric_key, background_metric_key,
                        progress_callback, clip_data=False)

            for channel in channel_list:

                if channel in self.traces_dict[dataset].keys():

                    channel_dict = self.traces_dict[dataset][channel].copy()

                    n_traces = len(channel_dict.keys())

                    if json_dict["data"][dataset] == []:
                        json_dict["data"][dataset] = [{} for _ in range(n_traces)]

                    for trace_index, trace_dict in channel_dict.items():

                        if channel.lower() in ["dd", "da", "ad", "aa"]:
                            channel_name = channel.upper()
                        elif channel.lower() == "alex_efficiency":
                            channel_name = "efficiency"
                        elif channel.lower() == "fret_efficiency":
                            channel_name = "efficiency"
                        else:
                            channel_name = channel.capitalize()

                        if channel_name not in json_dict["data"][dataset][trace_index]:
                            json_dict["data"][dataset][trace_index][channel_name] = []

                        data = trace_dict[metric_key].copy()

                        if "efficiency" not in channel and background_mode != "None":
                            background = np.array(trace_dict[background_metric_key].copy())
                            data = data - background

                        data = np.array(data).astype(float).tolist()

                        json_dict["data"][dataset][trace_index][channel_name] = data

                        iter += 1

                        if progress_callback is not None:
                            progress_callback.emit(iter / n_traces * 100)

        return json_dict

    def populate_export_dict(self):

        try:

            dataset_name = self.traces_export_dataset.currentText()
            channel_name = self.traces_export_channel.currentText()
            metric_name = self.traces_export_metric.currentText()
            background_mode = self.traces_export_background.currentText()

            metric_key = self.get_dict_key(self.metric_dict, metric_name)
            background_metric_key = metric_key + "_bg"

            if dataset_name == "All Datasets":
                dataset_list = list(self.traces_dict.keys())
            else:
                dataset_list = [dataset_name]

            if channel_name == "All Channels":
                channel_list = list(self.traces_dict[dataset_list[0]].keys())
            elif channel_name.lower() == "fret":
                channel_list = ["donor", "acceptor"]
            elif channel_name.lower() == "alex":
                channel_list = ["dd", "da", "ad", "aa"]
            else:
                channel_list = [channel_name]

            channel_list = [chan for chan in channel_list if "efficiency" not in chan.lower()]

            export_dict = {}

            for dataset in dataset_list:
                for channel in channel_list:

                    channel_dict = self.traces_dict[dataset_name][channel].copy()

                    for trace_index, trace_dict in channel_dict.items():

                        loc_dict = {}

                        if channel.lower() in ["dd", "da", "ad", "aa"]:
                            channel_name = channel.upper()
                        else:
                            channel_name = channel.capitalize()

                        if dataset not in export_dict.keys():
                            export_dict[dataset] = []


                        data = np.array(trace_dict[metric_key].copy())

                        if "efficiency" not in channel and background_mode != "None":
                            background = np.array(trace_dict[background_metric_key].copy())
                            data = data - background

                        data = data.astype(float).tolist()

                        loc_dict[channel_name] = data

                        export_dict[dataset].append(loc_dict)

        except:
            print(traceback.format_exc())
            pass

        return export_dict

    def export_traces_cleanup(self, export_path):

        if self.traces_export_status == True:
            print("Traces exported to: {}".format(export_path))

        self.export_progressbar.setValue(0)
        self.gapseq_export_traces.setEnabled(True)

    def export_traces_error(self, error_message):

        self.export_progressbar.setValue(0)
        self.gapseq_export_traces.setEnabled(True)

    def export_traces(self):

        try:

            self.gapseq_export_traces.setEnabled(False)

            export_path, export_directory = self.get_export_traces_path(dialog=True)

            if export_path != "" and os.path.isdir(export_directory):

                self.export_progressbar.setValue(0)
                self.gapseq_export_traces.setEnabled(False)
                self.traces_export_status = False

                export_mode = self.traces_export_mode.currentText()

                if export_mode == "JSON Dataset":

                    worker = Worker(self.export_traces_json, export_path=export_path)
                    worker.signals.progress.connect(partial(self.gapseq_progress,
                        progress_bar=self.export_progressbar))
                    worker.signals.finished.connect(partial(self.export_traces_cleanup,
                        export_path=export_path))
                    worker.signals.error.connect(self.export_traces_error)
                    self.threadpool.start(worker)

                if export_mode == "DAT":

                    worker = Worker(self.export_traces_dat, export_path=export_path)
                    worker.signals.progress.connect(partial(self.gapseq_progress,
                        progress_bar=self.export_progressbar))
                    worker.signals.finished.connect(partial(self.export_traces_cleanup,
                        export_path=export_path))
                    worker.signals.error.connect(self.export_traces_error)
                    self.threadpool.start(worker)

                if export_mode == "Excel":

                    worker = Worker(self.export_traces_excel, export_path=export_path)
                    worker.signals.progress.connect(partial(self.gapseq_progress,
                        progress_bar=self.export_progressbar))
                    worker.signals.finished.connect(partial(self.export_traces_cleanup,
                        export_path=export_path))
                    worker.signals.error.connect(self.export_traces_error)
                    self.threadpool.start(worker)

            else:
                self.gapseq_export_traces.setEnabled(True)

        except:
            print(traceback.format_exc())
            self.export_progressbar.setValue(0)
            self.gapseq_export_traces.setEnabled(False)
            pass

    def populate_export_combos(self):

        try:
            export_channel_list = []

            export_dataset = self.traces_export_dataset.currentText()

            print(export_dataset)

            if export_dataset == "All Datasets":

                for dataset_name, dataset_dict in self.dataset_dict.items():
                    for channel_name in dataset_dict.keys():

                        if channel_name.lower() in ["donor", "acceptor"]:
                            channel_name = channel_name.capitalize()
                        else:
                            channel_name = channel_name.upper()

                        export_channel_list.append(channel_name)
                        export_channel_list = list(set(export_channel_list))

                        print(export_channel_list)

            else:

                for channel_name in self.dataset_dict[export_dataset].keys():

                    if channel_name.lower() in ["donor", "acceptor"]:
                        channel_name = channel_name.capitalize()
                    else:
                        channel_name = channel_name.upper()

                    export_channel_list.append(channel_name)

            if set(["Donor", "Acceptor"]).issubset(set(export_channel_list)):
                export_channel_list.insert(0, "FRET")
            if set(["DD","DA","AA","AD"]).issubset(set(export_channel_list)):
                export_channel_list.insert(0, "ALEX")

            export_channel_list.insert(0, "All Channels")

            self.traces_export_channel.blockSignals(True)
            self.traces_export_channel.clear()
            self.traces_export_channel.addItems(export_channel_list)
            self.traces_export_channel.blockSignals(True)

            if hasattr(self, "traces_dict"):

                dataset_name = self.traces_export_dataset.currentText()
                channel_name = self.traces_export_channel.currentText()

                if dataset_name in self.traces_dict.keys():

                    if dataset_name == "All Datasets":
                        dataset_names = list(self.traces_dict.keys())
                        dataset_name = dataset_names[0]

                    if channel_name.lower() == "fret":
                        channel_name = "donor"
                    if channel_name.lower() == "alex":
                        channel_name = "dd"
                    if channel_name.lower() == "all channels":
                        channel_names = list(self.traces_dict[dataset_name].keys())
                        channel_names = [chan for chan in channel_names if "efficiency" not in chan.lower()]
                        channel_names = [chan for chan in channel_names if chan.lower() not in ["fret", "alex"]]
                        channel_name = channel_names[0]

                    if dataset_name in self.traces_dict.keys():
                        if channel_name.lower() in self.traces_dict[dataset_name].keys():

                            traces_channel_dict = self.traces_dict[dataset_name][channel_name.lower()]
                            metric_names = traces_channel_dict[0].keys()

                            self.traces_export_metric.blockSignals(True)

                            self.traces_export_metric.clear()


                            for metric in metric_names:
                                if metric in self.metric_dict.keys():
                                    self.traces_export_metric.addItem(self.metric_dict[metric])

                            self.traces_export_metric.blockSignals(False)

        except:
            print(traceback.format_exc())
            pass
