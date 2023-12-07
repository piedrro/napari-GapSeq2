from PyQt5.QtGui import QCursor
from PyQt5.QtCore import Qt
from qtpy.QtWidgets import QSlider
from PyQt5.QtWidgets import QVBoxLayout, QWidget
import numpy as np
import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import traceback
from functools import partial
import uuid
import copy
from napari_gapseq2._widget_utils_worker import Worker

class _plot_utils:

    def update_plot_combos(self, combo=""):

        if combo == "plot_data":
            self.update_plot_channel_combo()
            self.update_plot_metrics_combos()
        elif combo == "plot_channel":
            self.update_plot_metrics_combos()

    def populate_plot_combos(self):
        self.populate_plot_data_combo()
        self.update_plot_channel_combo()
        self.update_plot_metrics_combos()

    def populate_plot_data_combo(self):

        try:

            if hasattr(self, "traces_dict"):

                if self.traces_dict != {}:

                    self.updating_plot_combos = True

                    self.plot_data.blockSignals(True)
                    self.plot_data.clear()
                    self.plot_data.addItems(self.traces_dict.keys())
                    self.plot_data.blockSignals(False)

                    self.updating_plot_combos = False

        except:
            print(traceback.format_exc())

    def update_plot_channel_combo(self):

        try:

            if hasattr(self, "traces_dict"):

                if self.traces_dict != {}:

                    dataset_name = self.plot_data.currentText()

                    if dataset_name != "":

                        self.updating_plot_combos = True

                        self.plot_channel.blockSignals(True)
                        self.plot_channel.clear()

                        channel_names = list(self.traces_dict[dataset_name].keys())

                        if set(channel_names).issubset(["da", "dd", "aa", "ad"]):
                            self.plot_channel.addItem("ALEX Data")
                            self.plot_channel.addItem("ALEX Efficiency")
                        if set(channel_names).issubset(["donor", "acceptor"]):
                            self.plot_channel.addItem("FRET Data")
                            self.plot_channel.addItem("FRET Efficiency")

                        for channel in channel_names:
                            if channel in ["da", "dd", "aa", "ad"]:
                                self.plot_channel.addItem(channel.upper())
                            else:
                                self.plot_channel.addItem(channel.capitalize())

                        self.plot_channel.blockSignals(False)

                        self.updating_plot_combos = False



        except:
            print(traceback.format_exc())

    def update_plot_metrics_combos(self):

        try:

            if hasattr(self, "traces_dict"):

                if self.traces_dict != {}:

                    dataset_name = self.plot_data.currentText()
                    channel_name = self.plot_channel.currentText()

                    if channel_name in ["ALEX Data","ALEX Efficiency"]:
                        channel_key = "dd"
                    elif channel_name in ["FRET Data","FRET Efficiency"]:
                        channel_key = "donor"
                    else:
                        channel_key = channel_name.lower()

                    if dataset_name != "" and channel_name != "":

                        channel_dict = self.traces_dict[dataset_name][channel_key]

                        n_traces = len(channel_dict)

                        if n_traces > 0:

                            self.updating_plot_combos = True

                            self.plot_metric.blockSignals(True)
                            self.plot_background_metric.blockSignals(True)

                            self.plot_metric.clear()
                            self.plot_background_metric.clear()

                            self.plot_background_metric.clear()
                            self.plot_background_metric.addItem("None")

                            metric_names = channel_dict[0].keys()

                            self.metric_dict = {
                                "spot_mean": "Mean",
                                "spot_sum": "Sum",
                                "spot_max": "Maximum",
                                "spot_std": "std",
                                "snr_mean": "Mean SNR",
                                "snr_std": "std SNR",
                                "snr_max": "Maximum SNR",
                                "snr_sum": "Sum SNR",
                                "spot_photons": "Picasso Photons",
                            }

                            self.background_metric_dict = {
                                "bg_mean": "Local Mean",
                                "bg_sum": "Local Sum",
                                "bg_std": "Local std",
                                "bg_max": "Local Maximum",
                                "spot_bg": "Picasso Background",
                            }

                            for metric in metric_names:
                                if metric in self.metric_dict.keys():
                                    self.plot_metric.addItem(self.metric_dict[metric])
                                if metric in self.background_metric_dict.keys():
                                    self.plot_background_metric.addItem(self.background_metric_dict[metric])

                            self.plot_metric.blockSignals(False)
                            self.plot_background_metric.blockSignals(False)

                            self.updating_plot_combos = False

        except:
            print(traceback.format_exc())

    def get_dict_key(self, dict, target_value):

        dict_key = None

        if target_value not in ["None", None]:

            for key, value in dict.items():
                if value == target_value:
                    dict_key = key
                    break

        return dict_key

    def compute_fret_efficiency(self,  dataset_name, metric_key,background_metric_key,n_pixels,
            progress_callback=None, gamma_correction=1):

        try:

            dataset_dict = self.traces_dict[dataset_name].copy()

            n_traces = len(dataset_dict["donor"])

            for trace_index in range(n_traces):

                donor = dataset_dict["donor"][trace_index][metric_key]
                acceptor = dataset_dict["acceptor"][trace_index][metric_key]

                if background_metric_key in self.background_metric_dict.keys():
                    donor_bg = dataset_dict["donor"][trace_index][background_metric_key].copy()
                    acceptor_bg = dataset_dict["acceptor"][trace_index][background_metric_key].copy()
                    donor = donor - donor_bg
                    acceptor = acceptor - acceptor_bg

                efficiency = acceptor / ((gamma_correction * donor) + acceptor)
                efficiency = efficiency.tolist()

                dataset_dict["fret_efficiency"][trace_index][metric_key] = efficiency

                if progress_callback is not None:
                    progress = int(100 * trace_index / n_traces)
                    progress_callback.emit(progress)

            self.traces_dict[dataset_name] = dataset_dict

        except:
            print(traceback.format_exc())
            pass

    def compute_alex_efficiency(self, dataset_name, metric_key,background_metric_key,n_pixels,
            progress_callback=None, gamma_correction=1):

        try:

            dataset_dict = self.traces_dict[dataset_name].copy()

            n_traces = len(dataset_dict["dd"])

            dataset_dict["alex_efficiency"] = {}
            for trace_index in range(n_traces):
                if trace_index not in dataset_dict["alex_efficiency"]:
                    dataset_dict["alex_efficiency"][trace_index] = {metric_key: []}

            for trace_index in range(n_traces):

                dd = copy.deepcopy(dataset_dict["dd"][trace_index][metric_key])
                da = copy.deepcopy(dataset_dict["da"][trace_index][metric_key])

                # if trace_index == 0:
                #     print("dd",metric_key, dd[:5])
                #     print("da",metric_key, da[:5])
                #     eff = da / ((gamma_correction * dd) + da)
                #     print("efficiency",metric_key, eff[:5])

                gamma_correction = 1

                if background_metric_key == None:

                    efficiency = da / ((gamma_correction * dd) + da)
                    efficiency = np.array(efficiency)

                elif "sum" not in background_metric_key.lower():

                    dd_bg = dataset_dict["dd"][trace_index][background_metric_key].copy()
                    da_bg = dataset_dict["da"][trace_index][background_metric_key].copy()
                    dd = dd - dd_bg
                    da = da - da_bg

                    efficiency = da / ((gamma_correction * dd) + da)
                    efficiency = np.array(efficiency)

                else:

                    dd_bg = dataset_dict["dd"][trace_index]["bg_mean"].copy()
                    da_bg = dataset_dict["da"][trace_index]["bg_mean"].copy()

                    dd_bg = dd_bg * n_pixels
                    da_bg = da_bg * n_pixels

                    dd = dd - dd_bg
                    da = da - da_bg

                    efficiency = da / ((gamma_correction * dd) + da)
                    efficiency = np.array(efficiency)

                dataset_dict["alex_efficiency"][trace_index][metric_key] = efficiency

            self.traces_dict[dataset_name] = dataset_dict

        except:
            print(traceback.format_exc())
            pass


    def populate_plot_dict(self, progress_callback=None):

        try:

            dataset_name = self.plot_data.currentText()
            channel_name = self.plot_channel.currentText()
            metric_name = self.plot_metric.currentText()
            background_metric_name = self.plot_background_metric.currentText()

            metric_key = self.get_dict_key(self.metric_dict, metric_name)
            background_metric_key = self.get_dict_key(self.background_metric_dict, background_metric_name)

            if dataset_name == "All Datasets":
                plot_datasets = self.traces_dict.keys()
            else:
                plot_datasets = [dataset_name]

            if channel_name == "ALEX Data":
                plot_channels = ["dd", "da", "ad", "aa"]
                iteration_channel = "aa"
            elif channel_name == "ALEX Efficiency":
                plot_channels = ["alex_efficiency"]
                iteration_channel = "aa"
            elif channel_name == "FRET Data":
                plot_channels = ["donor", "acceptor"]
                iteration_channel = "donor"
            elif channel_name == "FRET Efficiency":
                plot_channels = ["fret_efficiency"]
                iteration_channel = "donor"
            else:
                plot_channels = [channel_name.lower()]
                iteration_channel = channel_name.lower()


            n_traces = len(self.traces_dict[dataset_name][iteration_channel])
            n_iterations = len(plot_datasets) * len(plot_channels) * n_traces
            spot_size = self.traces_dict[dataset_name][iteration_channel][0]["spot_size"][0].copy()
            n_pixels = spot_size**2

            if channel_name == "ALEX Efficiency":
                self.compute_alex_efficiency(dataset_name, metric_key,
                    background_metric_key, n_pixels, progress_callback)
            elif channel_name == "FRET Efficiency":
                self.compute_fret_efficiency(dataset_name, metric_key,
                    background_metric_key, n_pixels, progress_callback)

            iter = 0

            plot_dict = {}

            for dataset_name in plot_datasets:

                for channel in plot_channels:
                    channel_dict = self.traces_dict[dataset_name][channel].copy()
                    for trace_index, trace_dict in channel_dict.items():

                        data = np.array(trace_dict[metric_key].copy())

                        if "efficiency" not in channel:
                            if background_metric_name != "None":
                                background = np.array(trace_dict[background_metric_key].copy())
                                data = data - background

                        # if trace_index == 0:
                        #     print(metric_key, background_metric_key, data[:5])

                        if channel in ["dd", "da", "ad", "aa"]:
                            label = f"{channel.upper()} [{metric_name}]"
                        elif channel == "alex_efficiency":
                            label = f"Alex Efficiency [{metric_name}]"
                        elif channel == "fret_efficiency":
                            label = f"FRET Efficiency [{metric_name}]"
                        else:
                            label = f"{channel.capitalize()} [{metric_name}]"

                        if dataset_name not in plot_dict.keys():
                            plot_dict[dataset_name] = {}
                        if trace_index not in plot_dict[dataset_name].keys():
                            plot_dict[dataset_name][trace_index] = {"labels": [], "data": []}

                        plot_dict[dataset_name][trace_index]["labels"].append(label)
                        plot_dict[dataset_name][trace_index]["data"].append(data)

                        iter += 1

                        if progress_callback is not None:
                            progress = int((iter/n_iterations) * 100)
                            progress_callback.emit(progress)

            # print(f"plot_dict keys: {plot_dict.keys()}")
            #
            # for dataset_name in plot_dict.keys():
            #     print(dataset_name,
            #         len(plot_dict[dataset_name]),
            #         plot_dict[dataset_name][0]["labels"],
            #         len(plot_dict[dataset_name][0]["data"][0]))

            self.plot_dict = plot_dict

        except:
            print(channel)
            print(traceback.format_exc())
            pass

    def initialize_plot(self):

        try:
            if hasattr(self, "traces_dict"):

                if self.traces_dict != {}:

                    dataset_name = self.plot_data.currentText()
                    channel_name = self.plot_channel.currentText()
                    metric_name = self.plot_metric.currentText()
                    background_metric_name = self.plot_background_metric.currentText()

                    if dataset_name != "" and channel_name != "" and metric_name != "" and background_metric_name != "":

                        if self.updating_plot_combos == False:

                            self.plot_localisation_number.setEnabled(False)
                            self.plot_data.setEnabled(False)
                            self.plot_channel.setEnabled(False)
                            self.plot_metric.setEnabled(False)
                            self.plot_background_metric.setEnabled(False)
                            self.split_plots.setEnabled(False)
                            self.normalise_plots.setEnabled(False)

                            self.populate_plot_dict()
                            self.update_plot_layout()
                            self.plot_traces()

                            self.plot_localisation_number.setEnabled(True)
                            self.plot_data.setEnabled(True)
                            self.plot_channel.setEnabled(True)
                            self.plot_metric.setEnabled(True)
                            self.plot_background_metric.setEnabled(True)
                            self.split_plots.setEnabled(True)
                            self.normalise_plots.setEnabled(True)

        except:
            print(traceback.format_exc())
            pass

    def update_plot_layout(self):

        try:

            self.plot_grid = {}

            self.graph_canvas.clear()

            split = self.split_plots.isChecked()
            plot_mode = self.plot_channel.currentText()

            efficiency_plot = False

            n_traces = []

            for plot_index, (dataset_name, dataset_dict) in enumerate(self.plot_dict.items()):

                plot_labels = dataset_dict[0]["labels"]
                n_plot_lines = len(plot_labels)
                n_traces.append(len(dataset_dict))

                sub_plots = []

                if plot_mode == "FRET Data + FRET Efficiency" and split==False:

                    layout = pg.GraphicsLayout()
                    self.graph_canvas.addItem(layout, row=plot_index, col=0)

                    for line_index in range(2):
                        p = CustomPlot()

                        layout.addItem(p, row=line_index, col=0)

                        if self.plot_settings.plot_showy.isChecked() == False:
                            p.hideAxis('left')

                        if self.plot_settings.plot_showx.isChecked() == False:
                            p.hideAxis('bottom')
                        elif line_index != 1:
                            p.hideAxis('bottom')

                        sub_plots.append(p)

                    for j in range(1, len(sub_plots)):
                        sub_plots[j].setXLink(sub_plots[0])

                    efficiency_plot = True

                elif split == True and n_plot_lines > 1:

                    layout = pg.GraphicsLayout()
                    self.graph_canvas.addItem(layout, row=plot_index, col=0)

                    for line_index in range(n_plot_lines):
                        p = CustomPlot()

                        layout.addItem(p, row=line_index, col=0)

                        if line_index != n_plot_lines - 1:
                            p.hideAxis('bottom')

                        sub_plots.append(p)

                    for j in range(1, len(sub_plots)):
                        sub_plots[j].setXLink(sub_plots[0])

                else:
                    layout = self.graph_canvas

                    p = CustomPlot()

                    p.hideAxis('top')
                    p.hideAxis('right')

                    layout.addItem(p, row=plot_index, col=0)

                    for line_index in enumerate(plot_labels):
                        sub_plots.append(p)

                plot_lines = []
                plot_lines_labels = []

                for axes_index, plot in enumerate(sub_plots):

                    line_label = plot_labels[axes_index]
                    line_format = pg.mkPen(color=100 + axes_index * 100, width=2)
                    plot_line = plot.plot(np.zeros(10), pen=line_format, name=line_label)
                    plot.enableAutoRange()
                    plot.autoRange()

                    legend = plot.legend
                    legend.anchor(itemPos=(1, 0), parentPos=(1, 0), offset=(-10, 10))

                    plot_details = f"{dataset_name}"

                    if axes_index == 0:
                        plot.setTitle(plot_details)
                        title_plot = plot

                    plotmeta = plot.metadata
                    plotmeta[axes_index] = {"plot_dataset": dataset_name, "line_label": line_label}

                    plot_lines.append(plot_line)
                    plot_lines_labels.append(line_label)

                    self.plot_grid[plot_index] = {
                        "sub_axes": sub_plots,
                        "title_plot": title_plot,
                        "plot_lines": plot_lines,
                        "plot_dataset": dataset_name,
                        "plot_index": plot_index,
                        "n_plot_lines": len(plot_lines),
                        "split": split,
                        "plot_lines_labels": plot_lines_labels,
                        "efficiency_plot": efficiency_plot,
                        }

            n_traces = max(n_traces)
            self.plot_localisation_number = self.findChild(QSlider, 'plot_localisation_number')
            self.plot_localisation_number.setMaximum(n_traces-1)

            plot_list = []
            for plot_index, grid in enumerate(self.plot_grid.values()):
                sub_axes = grid["sub_axes"]
                sub_plots = []
                for plot in sub_axes:
                    sub_plots.append(plot)
                    plot_list.append(plot)
            for i in range(1, len(plot_list)):
                plot_list[i].setXLink(plot_list[0])
            plot.getViewBox().sigXRangeChanged.connect(lambda: auto_scale_y(plot_list))

        except:
            print(traceback.format_exc())
            pass

        return self.plot_grid

    def plot_traces(self, update=False):

        try:

            if self.plot_grid != {}:

                localisation_number = self.plot_localisation_number.value()

                for plot_index, grid in enumerate(self.plot_grid.values()):

                    plot_dataset = grid["plot_dataset"]
                    sub_axes = grid["sub_axes"]
                    plot_lines = grid["plot_lines"]
                    plot_lines_labels = grid["plot_lines_labels"]
                    title_plot = grid["title_plot"]

                    plot_details = f"{plot_dataset} [#:{localisation_number}]"

                    plot_ranges = {"xRange": [0, 100], "yRange": [0, 100]}
                    for line_index, (plot, line, plot_label) in enumerate(zip(sub_axes, plot_lines, plot_lines_labels)):

                        legend = plot.legend
                        data = self.plot_dict[plot_dataset][localisation_number]["data"][line_index]

                        if self.normalise_plots.isChecked() and "efficiency" not in plot_label.lower():
                            data = (data - np.min(data)) / (np.max(data) - np.min(data))

                        # print(np.min(data), np.max(data), np.max(data) - np.min(data))

                        plot_line = plot_lines[line_index]
                        plot_line.setData(data)

                        if plot_ranges["xRange"][1] < len(data):
                            plot_ranges["xRange"][1] = len(data)
                        if plot_ranges["yRange"][1] < np.max(data):
                            plot_ranges["yRange"][1] = np.max(data)
                        if plot_ranges["yRange"][0] > np.min(data):
                            plot_ranges["yRange"][0] = np.min(data)
                        if plot_ranges["xRange"][0] > 0:
                            plot_ranges["xRange"][0] = 0


                    for line_index, (plot, line, plot_label) in enumerate(zip(sub_axes, plot_lines, plot_lines_labels)):
                        plot.setRange(
                            xRange=plot_ranges["xRange"],
                            yRange=plot_ranges["yRange"],
                            padding=0.0,
                            disableAutoRange=True,
                            )

        except:
            # print(traceback.format_exc())
            pass













def auto_scale_y(sub_plots):

    try:

        for p in sub_plots:
            data_items = p.listDataItems()

            if not data_items:
                return

            y_min = np.inf
            y_max = -np.inf

            # Get the current x-range of the plot
            plot_x_min, plot_x_max = p.getViewBox().viewRange()[0]

            for index, item in enumerate(data_items):
                if item.name() != "hmm_mean":

                    y_data = item.yData
                    x_data = item.xData

                    # Get the indices of y_data that lies within the current x-range
                    idx = np.where((x_data >= plot_x_min) & (x_data <= plot_x_max))

                    if len(idx[0]) > 0:  # If there's any data within the x-range
                        y_min = min(y_min, y_data[idx].min())
                        y_max = max(y_max, y_data[idx].max())

                    if plot_x_min < 0:
                        x_min = 0
                    else:
                        x_min = plot_x_min

                    if plot_x_max > x_data.max():
                        x_max = x_data.max()
                    else:
                        x_max = plot_x_max

            p.getViewBox().setYRange(y_min, y_max, padding=0)
            p.getViewBox().setXRange(x_min, x_max, padding=0)

    except:
        pass


class CustomPlot(pg.PlotItem):

    def __init__(self, title="", colour="", *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.metadata = {}

        self.setMenuEnabled(False)
        self.symbolSize = 100

        legend = self.addLegend(offset=(10, 10))
        legend.setBrush('w')
        legend.setLabelTextSize("8pt")
        self.hideAxis('top')
        self.hideAxis('right')
        self.getAxis('left').setWidth(30)

        self.title = title
        self.colour = colour

        if self.title != "":
            self.setLabel('top', text=title, size="3pt", color=colour)

    def setMetadata(self, metadata_dict):
        self.metadata = metadata_dict

    def getMetadata(self):
        return self.metadata


class CustomPyQTGraphWidget(pg.GraphicsLayoutWidget):

    def __init__(self, parent):
        super().__init__()

        self.parent = parent
        self.frame_position_memory = {}
        self.frame_position = None

def mousePressEvent(self, event):

    if hasattr(self.parent, "plot_grid"):

        if event.modifiers() & Qt.ControlModifier:

            xpos = self.get_event_x_postion(event, mode="click")

        elif event.modifiers() & Qt.AltModifier:

            xpos = self.get_event_x_postion(event, mode="click")

        super().mousePressEvent(event)  # Process the event further

def keyPressEvent(self, event):

    if hasattr(self.parent, "plot_grid"):

        pass

        super().keyPressEvent(event)  # Process the event further

def get_event_x_postion(self, event,  mode="click"):

    self.xpos = None

    if hasattr(self.parent, "plot_grid"):

        if mode == "click":
            pos = event.pos()
            self.scene_pos = self.mapToScene(pos)
        else:
            pos = QCursor.pos()
            self.scene_pos = self.mapFromGlobal(pos)

        # Iterate over all plots
        plot_grid = self.parent.plot_grid

        for plot_index, grid in enumerate(plot_grid.values()):
            sub_axes = grid["sub_axes"]

            for axes_index in range(len(sub_axes)):
                plot = sub_axes[axes_index]

                viewbox = plot.vb
                plot_coords = viewbox.mapSceneToView(self.scene_pos)

        self.xpos = plot_coords.x()

    return self.xpos
