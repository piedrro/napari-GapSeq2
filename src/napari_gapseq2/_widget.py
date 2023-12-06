"""
This module contains four napari widgets declared in
different ways:

- a pure Python function flagged with `autogenerate: true`
    in the plugin manifest. Type annotations are used by
    magicgui to generate widgets for each parameter. Best
    suited for simple processing tasks - usually taking
    in and/or returning a layer.
- a `magic_factory` decorated function. The `magic_factory`
    decorator allows us to customize aspects of the resulting
    GUI, including the widgets associated with each parameter.
    Best used when you have a very simple processing task,
    but want some control over the autogenerated widgets. If you
    find yourself needing to define lots of nested functions to achieve
    your functionality, maybe look at the `Container` widget!
- a `magicgui.widgets.Container` subclass. This provides lots
    of flexibility and customization options while still supporting
    `magicgui` widgets and convenience methods for creating widgets
    from type annotations. If you want to customize your widgets and
    connect callbacks, this is the best widget option for you.
- a `QWidget` subclass. This provides maximal flexibility but requires
    full specification of widget layouts, callbacks, events, etc.

References:
- Widget specification: https://napari.org/stable/plugins/guides.html?#widgets
- magicgui docs: https://pyapp-kit.github.io/magicgui/

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

from magicgui import magic_factory
from magicgui.widgets import CheckBox, Container, create_widget
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget
from skimage.util import img_as_float
from qtpy.QtCore import QObject, QRunnable, QThreadPool
from qtpy.QtWidgets import (QWidget,QVBoxLayout,QTabWidget,QFrame, QSizePolicy, QSlider, QComboBox,QLineEdit, QProgressBar, QLabel, QCheckBox)
from PIL import Image
from tqdm import tqdm
import numpy as np
import tifffile
from qtpy.QtCore import QObject
from qtpy.QtCore import QRunnable
from PyQt5.QtCore import pyqtSignal, pyqtSlot
import sys
import traceback
import time
import json
import copy
from scipy.spatial import procrustes
from scipy.spatial import distance
import cv2

from napari_gapseq2._widget_utils_worker import Worker, WorkerSignals
from napari_gapseq2._widget_undrift_utils import _undrift_utils
from napari_gapseq2._widget_picasso_detect import _picasso_detect_utils
from napari_gapseq2._widget_loc_utils import _loc_utils, picasso_loc_utils
from napari_gapseq2._widget_import_utils import _import_utils
from napari_gapseq2._widget_events import _events_utils
from napari_gapseq2._widget_export_utils import _export_utils
from napari_gapseq2._widget_transform_utils import _tranform_utils
from napari_gapseq2._widget_trace_compute_utils import _trace_compute_utils
from napari_gapseq2._widget_plot_utils import _plot_utils, CustomPyQTGraphWidget
from napari_gapseq2._widget_align_utils import _align_utils

from qtpy.QtWidgets import QFileDialog
import os
from multiprocessing import Pool
import multiprocessing
from functools import partial

if TYPE_CHECKING:
    import napari


class GapSeqWidget(QWidget,
    _undrift_utils, _picasso_detect_utils,
    _import_utils, _events_utils, _export_utils,
    _tranform_utils, _trace_compute_utils, _plot_utils,
    _align_utils, _loc_utils):

    # your QWidget.__init__ can optionally request the napari viewer instance
    # use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer

        from napari_gapseq2.widget_ui import Ui_Frame

        #create UI
        self.setLayout(QVBoxLayout())
        self.form = Ui_Frame()
        self.gapseq_ui = QFrame()
        self.form.setupUi(self.gapseq_ui)
        self.layout().addWidget(self.gapseq_ui)

        #create pyqt graph container
        self.graph_container = self.findChild(QWidget, "graph_container")
        self.graph_container.setLayout(QVBoxLayout())
        self.graph_container.setMinimumWidth(10)
        self.graph_container.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

        self.graph_canvas = CustomPyQTGraphWidget(self)
        self.graph_canvas.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.graph_container.layout().addWidget(self.graph_canvas)

        # register controls
        self.gapseq_import_mode = self.findChild(QComboBox, 'gapseq_import_mode')
        self.gapseq_import_limt = self.findChild(QComboBox, 'gapseq_import_limt')
        self.gapseq_channel_layout = self.findChild(QComboBox, 'gapseq_channel_layout')
        self.gapseq_channel_layout_label = self.findChild(QLabel, 'gapseq_channel_layout_label')
        self.gapseq_alex_first_frame = self.findChild(QComboBox, 'gapseq_alex_first_frame')
        self.gapseq_alex_first_frame_label = self.findChild(QLabel, 'gapseq_alex_first_frame_label')
        self.gapseq_import = self.findChild(QPushButton, 'gapseq_import')
        self.gapseq_import_progressbar = self.findChild(QProgressBar, 'gapseq_import_progressbar')

        self.gapseq_old_dataset_name = self.findChild(QComboBox, 'gapseq_old_dataset_name')
        self.gapseq_new_dataset_name = self.findChild(QLineEdit, 'gapseq_new_dataset_name')
        self.gapseq_update_dataset_name = self.findChild(QPushButton, 'gapseq_update_dataset_name')

        self.gapseq_dataset_selector = self.findChild(QComboBox, 'gapseq_dataset_selector')
        self.gapseq_show_dd = self.findChild(QPushButton, 'gapseq_show_dd')
        self.gapseq_show_da = self.findChild(QPushButton, 'gapseq_show_da')
        self.gapseq_show_aa = self.findChild(QPushButton, 'gapseq_show_aa')
        self.gapseq_show_ad = self.findChild(QPushButton, 'gapseq_show_ad')


        self.import_alex_data = self.findChild(QPushButton, 'import_alex_data')
        self.channel_selector = self.findChild(QComboBox, 'channel_selector')

        self.picasso_dataset = self.findChild(QComboBox, 'picasso_dataset')
        self.picasso_channel = self.findChild(QComboBox, 'picasso_channel')
        self.picasso_min_net_gradient = self.findChild(QLineEdit, 'picasso_min_net_gradient')
        self.picasso_roi_border_width = self.findChild(QLineEdit, 'picasso_roi_border_width')
        self.picasso_box_size = self.findChild(QComboBox, 'picasso_box_size')
        self.picasso_frame_mode = self.findChild(QComboBox, 'picasso_frame_mode')
        self.picasso_detect = self.findChild(QPushButton, 'picasso_detect')
        self.picasso_fit = self.findChild(QPushButton, 'picasso_fit')
        self.picasso_detect_mode = self.findChild(QComboBox, 'picasso_detect_mode')
        self.picasso_window_cropping = self.findChild(QCheckBox, 'picasso_window_cropping')
        self.picasso_progressbar = self.findChild(QProgressBar, 'picasso_progressbar')

        self.picasso_vis_mode = self.findChild(QComboBox, 'picasso_vis_mode')
        self.picasso_vis_size = self.findChild(QComboBox, 'picasso_vis_size')
        self.picasso_vis_opacity = self.findChild(QComboBox, 'picasso_vis_opacity')
        self.picasso_vis_edge_width = self.findChild(QComboBox, 'picasso_vis_edge_width')

        self.picasso_vis_mode.currentIndexChanged.connect(self.draw_fiducials)
        self.picasso_vis_mode.currentIndexChanged.connect(self.draw_bounding_boxes)
        self.picasso_vis_size.currentIndexChanged.connect(self.draw_fiducials)
        self.picasso_vis_size.currentIndexChanged.connect(self.draw_bounding_boxes)
        self.picasso_vis_opacity.currentIndexChanged.connect(self.draw_fiducials)
        self.picasso_vis_opacity.currentIndexChanged.connect(self.draw_bounding_boxes)
        self.picasso_vis_edge_width.currentIndexChanged.connect(self.draw_fiducials)
        self.picasso_vis_edge_width.currentIndexChanged.connect(self.draw_bounding_boxes)

        self.cluster_localisations = self.findChild(QPushButton, 'cluster_localisations')
        self.cluster_mode = self.findChild(QComboBox, 'cluster_mode')
        self.cluster_channel = self.findChild(QComboBox, 'cluster_channel')
        self.cluster_dataset = self.findChild(QComboBox, 'cluster_dataset')

        self.picasso_undrift_mode = self.findChild(QComboBox, 'picasso_undrift_mode')
        self.picasso_undrift_channel = self.findChild(QComboBox, 'picasso_undrift_channel')
        self.detect_undrift = self.findChild(QPushButton, 'detect_undrift')
        self.apply_undrift = self.findChild(QPushButton, 'apply_undrift')
        self.undrift_channel_selector = self.findChild(QComboBox, 'undrift_channel_selector')
        self.undrift_progressbar = self.findChild(QProgressBar, 'undrift_progressbar')

        self.align_reference_dataset = self.findChild(QComboBox, 'align_reference_dataset')
        self.align_reference_channel = self.findChild(QComboBox, 'align_reference_channel')
        self.gapseq_align_datasets = self.findChild(QPushButton, 'gapseq_align_datasets')
        self.align_progressbar = self.findChild(QProgressBar, 'align_progressbar')

        self.gapseq_import_tform = self.findChild(QPushButton, 'gapseq_import_tform')

        self.tform_compute_dataset = self.findChild(QComboBox, 'tform_compute_dataset')
        self.tform_compute_ref_channel = self.findChild(QComboBox, 'tform_compute_ref_channel')
        self.tform_compute_target_channel = self.findChild(QComboBox, 'tform_compute_target_channel')
        self.gapseq_compute_tform = self.findChild(QPushButton, 'gapseq_compute_tform')
        self.tform_apply_target = self.findChild(QComboBox, 'tform_apply_target')
        self.gapseq_apply_tform = self.findChild(QPushButton, 'gapseq_apply_tform')
        self.tform_apply_progressbar = self.findChild(QProgressBar, 'tform_apply_progressbar')
        self.save_tform = self.findChild(QCheckBox, 'save_tform')

        self.gapseq_link_localisations = self.findChild(QPushButton, 'gapseq_link_localisations')

        self.export_dataset = self.findChild(QComboBox, 'export_dataset')
        self.export_channel = self.findChild(QComboBox, 'export_channel')
        self.gapseq_export_data = self.findChild(QPushButton, 'gapseq_export_data')
        self.export_progressbar = self.findChild(QProgressBar, 'export_progressbar')

        self.traces_spot_size = self.findChild(QComboBox, "traces_spot_size")
        self.traces_spot_shape = self.findChild(QComboBox, "traces_spot_shape")
        self.compute_with_picasso = self.findChild(QCheckBox, "compute_with_picasso")
        self.traces_visualise_masks = self.findChild(QPushButton, 'traces_visualise_masks')
        self.compute_traces = self.findChild(QPushButton, 'compute_traces')
        self.compute_traces_progressbar = self.findChild(QProgressBar, 'compute_traces_progressbar')

        self.plot_data = self.findChild(QComboBox, 'plot_data')
        self.plot_channel = self.findChild(QComboBox, 'plot_channel')
        self.plot_metric = self.findChild(QComboBox, 'plot_metric')
        self.plot_background_metric = self.findChild(QComboBox, 'plot_background_metric')
        self.split_plots = self.findChild(QCheckBox, 'split_plots')
        self.normalise_plots = self.findChild(QCheckBox, 'normalise_plots')
        self.plot_compute_progress = self.findChild(QProgressBar, 'plot_compute_progress')
        self.plot_localisation_number = self.findChild(QSlider, 'plot_localisation_number')
        self.plot_localisation_number_label = self.findChild(QLabel, 'plot_localisation_number_label')

        self.gapseq_import.clicked.connect(self.gapseq_import_data)
        self.gapseq_import_mode.currentIndexChanged.connect(self.update_import_options)
        self.gapseq_update_dataset_name.clicked.connect(self.update_dataset_name)

        self.picasso_detect.clicked.connect(self.gapseq_picasso_detect)
        self.picasso_fit.clicked.connect(self.gapseq_picasso_fit)
        self.cluster_localisations.clicked.connect(self.gapseq_cluster_localisations)

        self.gapseq_dataset_selector.currentIndexChanged.connect(self.update_channel_select_buttons)
        self.gapseq_dataset_selector.currentIndexChanged.connect(partial(self.update_active_image,
            dataset = self.gapseq_dataset_selector.currentText()))

        self.detect_undrift.clicked.connect(self.gapseq_picasso_undrift)
        self.apply_undrift.clicked.connect(self.gapseq_undrift_images)

        self.gapseq_align_datasets.clicked.connect(self.align_datasets)
        self.align_reference_dataset.currentIndexChanged.connect(self.update_align_reference_channel)

        self.gapseq_import_tform.clicked.connect(self.import_transform_matrix)
        self.gapseq_compute_tform.clicked.connect(self.compute_transform_matrix)
        self.gapseq_apply_tform.clicked.connect(self.apply_transform_matrix)

        self.picasso_detect_mode.currentIndexChanged.connect(self.update_picasso_options)

        self.gapseq_export_data.clicked.connect(self.export_data)
        self.export_dataset.currentIndexChanged.connect(self.update_export_options)

        self.viewer.dims.events.current_step.connect(self.draw_fiducials)

        self.compute_traces.clicked.connect(self.gapseq_compute_traces)
        self.traces_visualise_masks.clicked.connect(self.visualise_spot_masks)

        self.plot_data.currentIndexChanged.connect(partial(self.update_plot_combos, combo="plot_data"))
        self.plot_channel.currentIndexChanged.connect(partial(self.update_plot_combos, combo="plot_channel"))

        self.plot_data.currentIndexChanged.connect(self.initialize_plot)
        self.plot_channel.currentIndexChanged.connect(self.initialize_plot)
        self.plot_metric.currentIndexChanged.connect(self.initialize_plot)
        self.plot_background_metric.currentIndexChanged.connect(self.initialize_plot)
        self.split_plots.stateChanged.connect(self.initialize_plot)
        self.normalise_plots.stateChanged.connect(self.initialize_plot)

        self.plot_localisation_number.valueChanged.connect(lambda: self.update_slider_label("plot_localisation_number"))
        self.plot_localisation_number.valueChanged.connect(partial(self.plot_traces))

        self.dataset_dict = {}
        self.localisation_dict = {"bounding_boxes": {}, "fiducials": {}}
        self.traces_dict = {}
        self.plot_dict = {}

        self.active_dataset = None
        self.active_channel = None

        self.threadpool = QThreadPool()

        self.transform_matrix = None

        self.update_import_options()

        self.metric_dict = {"spot_mean": "Mean", "spot_sum": "Sum", "spot_max": "Maximum",
                            "spot_std": "std", "snr_mean": "Mean SNR", "snr_std": "std SNR",
                            "snr_max": "Maximum SNR", "snr_sum": "Sum SNR", "spot_photons": "Picasso Photons", }

        self.background_metric_dict = {"bg_mean": "Local Mean", "bg_sum": "Local Sum",
                                       "bg_std": "Local std", "bg_max": "Local Maximum",
                                       "spot_bg": "Picasso Background", }


    def compute_registration_keypoints(self, reference_box_centres, target_box_centres, alignment_distance=20):

        alignment_keypoints = []
        keypoint_distances = []
        target_keypoints = []

        distances = distance.cdist(np.array(reference_box_centres), np.array(target_box_centres))

        for j in range(distances.shape[0]):

            dat = distances[j]

            loc_index = np.nanargmin(dat)
            loc_distance = np.nanmin(dat)
            loc0_index = j

            loc0_centre = reference_box_centres[loc0_index]
            loc_centre = target_box_centres[loc_index]

            x_difference = abs(loc0_centre[0] - loc_centre[0])
            y_difference = abs(loc0_centre[1] - loc_centre[1])

            xy_distance = np.sqrt(x_difference ** 2 + y_difference ** 2)

            if xy_distance < alignment_distance:

                alignment_keypoints.append([loc0_centre[0], loc0_centre[1]])
                target_keypoints.append([loc_centre[0], loc_centre[1]])
                keypoint_distances.append(loc_distance)

        alignment_keypoints = np.array(alignment_keypoints).astype(np.float32)
        target_keypoints = np.array(target_keypoints).astype(np.float32)

        return alignment_keypoints, target_keypoints


    def draw_bounding_boxes(self):

        if hasattr(self, "localisation_dict") and hasattr(self, "active_channel"):

            layer_names = [layer.name for layer in self.viewer.layers]

            if "localisation_centres" in self.localisation_dict["bounding_boxes"].keys():

                localisations = self.localisation_dict["bounding_boxes"]["localisations"]
                localisation_centres = self.get_localisation_centres(localisations, mode="bounding_boxes")

                vis_mode = self.picasso_vis_mode.currentText()
                vis_size = float(self.picasso_vis_size.currentText())
                vis_opacity = float(self.picasso_vis_opacity.currentText())
                vis_edge_width = float(self.picasso_vis_edge_width.currentText())

                if vis_mode.lower() == "square":
                    symbol = "square"
                elif vis_mode.lower() == "disk":
                    symbol = "disc"
                elif vis_mode.lower() == "x":
                    symbol = "cross"


                if "bounding_boxes" not in layer_names:
                    self.bbox_layer = self.viewer.add_points(
                        localisation_centres,
                        edge_color="white",
                        ndim=2,
                        face_color=[0,0,0,0],
                        opacity=vis_opacity,
                        name="bounding_boxes",
                        symbol=symbol,
                        size=vis_size,
                        visible=True,
                        edge_width=vis_edge_width,)

                    self.bbox_layer.mouse_drag_callbacks.append(self._mouse_event)

                else:
                    self.viewer.layers["bounding_boxes"].data = localisation_centres
                    self.viewer.layers["bounding_boxes"].opacity = vis_opacity
                    self.viewer.layers["bounding_boxes"].symbol = symbol
                    self.viewer.layers["bounding_boxes"].size = vis_size
                    self.viewer.layers["bounding_boxes"].edge_width = vis_edge_width
                    self.viewer.layers["bounding_boxes"].edge_color = "white"

            for layer in layer_names:
                self.viewer.layers[layer].refresh()


    def draw_fiducials(self):

        remove_fiducials = True

        if hasattr(self, "localisation_dict") and hasattr(self, "active_channel"):

            layer_names = [layer.name for layer in self.viewer.layers]

            active_frame = self.viewer.dims.current_step[0]

            dataset_name = self.gapseq_dataset_selector.currentText()
            image_channel = self.active_channel

            if image_channel != "" and dataset_name != "":

                if image_channel.lower() in self.localisation_dict["fiducials"][dataset_name].keys():

                    localisation_dict = self.localisation_dict["fiducials"][dataset_name][image_channel.lower()]

                    if "render_locs" in localisation_dict.keys():

                        render_locs = localisation_dict["render_locs"]

                        vis_mode = self.picasso_vis_mode.currentText()
                        vis_size = float(self.picasso_vis_size.currentText())
                        vis_opacity = float(self.picasso_vis_opacity.currentText())
                        vis_edge_width = float(self.picasso_vis_edge_width.currentText())

                        if vis_mode.lower() == "square":
                            symbol = "square"
                        elif vis_mode.lower() == "disk":
                            symbol = "disc"
                        elif vis_mode.lower() == "x":
                            symbol = "cross"

                        if active_frame in render_locs.keys():

                            remove_fiducials = False

                            if "fiducials" not in layer_names:
                                self.fiducial_layer = self.viewer.add_points(
                                    render_locs[active_frame],
                                    ndim=2,
                                    edge_color="red",
                                    face_color=[0,0,0,0],
                                    opacity=vis_opacity,
                                    name="fiducials",
                                    symbol=symbol,
                                    size=vis_size,
                                    edge_width=vis_edge_width, )

                                self.fiducial_layer.mouse_drag_callbacks.append(self._mouse_event)

                            else:
                                self.viewer.layers["fiducials"].data = []

                                self.viewer.layers["fiducials"].data = render_locs[active_frame]
                                self.viewer.layers["fiducials"].opacity = vis_opacity
                                self.viewer.layers["fiducials"].symbol = symbol
                                self.viewer.layers["fiducials"].size = vis_size
                                self.viewer.layers["fiducials"].edge_width = vis_edge_width
                                self.viewer.layers["fiducials"].edge_color = "red"


            if remove_fiducials:
                if "fiducials" in layer_names:
                    self.viewer.layers["fiducials"].data = []

            for layer in layer_names:
                self.viewer.layers[layer].refresh()


    def get_localisation_centres(self, locs, mode = "fiducials"):

        loc_centres = []

        try:

            for loc in locs:
                frame = int(loc.frame)
                if mode == "fiducials":
                    loc_centres.append([frame, loc.y, loc.x])
                else:
                    loc_centres.append([loc.y, loc.x])

        except:
            print(traceback.format_exc())

        return loc_centres

