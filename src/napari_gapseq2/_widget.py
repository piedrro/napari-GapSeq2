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
from qtpy.QtWidgets import (QWidget,QVBoxLayout,QTabWidget,QSizePolicy, QComboBox,QLineEdit, QProgressBar, QLabel, QCheckBox)
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
from napari_gapseq2._widget_import_utils import _import_utils
from napari_gapseq2._widget_events import _events_utils
from napari_gapseq2._widget_export_utils import _export_utils
from napari_gapseq2._widget_transform_utils import _tranform_utils

from qtpy.QtWidgets import QFileDialog
import os
from multiprocessing import Pool
import multiprocessing
from functools import partial

if TYPE_CHECKING:
    import napari



class GapSeqWidget(QWidget, _undrift_utils, _picasso_detect_utils, _import_utils, _events_utils, _export_utils, _tranform_utils):

    # your QWidget.__init__ can optionally request the napari viewer instance
    # use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer

        from napari_gapseq2.widget_ui import Ui_TabWidget

        #create UI
        self.setLayout(QVBoxLayout())
        self.form = Ui_TabWidget()
        self.gapseq_ui = QTabWidget()
        self.form.setupUi(self.gapseq_ui)
        self.layout().addWidget(self.gapseq_ui)



        self.gapseq_import_mode = self.findChild(QComboBox, 'gapseq_import_mode')
        self.gapseq_channel_layout = self.findChild(QComboBox, 'gapseq_channel_layout')
        self.gapseq_channel_layout_label = self.findChild(QLabel, 'gapseq_channel_layout_label')
        self.gapseq_alex_first_frame = self.findChild(QComboBox, 'gapseq_alex_first_frame')
        self.gapseq_alex_first_frame_label = self.findChild(QLabel, 'gapseq_alex_first_frame_label')
        self.gapseq_dataset_name = self.findChild(QLineEdit, 'gapseq_dataset_name')
        self.gapseq_import = self.findChild(QPushButton, 'gapseq_import')
        self.gapseq_import_progressbar = self.findChild(QProgressBar, 'gapseq_import_progressbar')

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
        self.picasso_frame_mode = self.findChild(QComboBox, 'picasso_frame_mode')
        self.picasso_detect = self.findChild(QPushButton, 'picasso_detect')
        self.picasso_fit = self.findChild(QPushButton, 'picasso_fit')
        self.picasso_detect_mode = self.findChild(QComboBox, 'picasso_detect_mode')
        self.picasso_window_cropping = self.findChild(QCheckBox, 'picasso_window_cropping')
        self.picasso_progressbar = self.findChild(QProgressBar, 'picasso_progressbar')

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


        self.gapseq_import.clicked.connect(self.gapseq_import_data)
        self.gapseq_import_mode.currentIndexChanged.connect(self.update_import_options)

        self.picasso_detect.clicked.connect(self.gapseq_picasso_detect)
        self.picasso_fit.clicked.connect(self.gapseq_picasso_fit)
        self.cluster_localisations.clicked.connect(self.gapseq_cluster_localisations)

        self.gapseq_dataset_selector.currentIndexChanged.connect(self.update_channel_select_buttons)
        self.gapseq_dataset_selector.currentIndexChanged.connect(self.update_active_image)

        self.detect_undrift.clicked.connect(self.gapseq_picasso_undrift)
        self.apply_undrift.clicked.connect(self.gapseq_undrift_images)


        self.gapseq_import_tform.clicked.connect(self.import_transform_matrix)
        self.gapseq_compute_tform.clicked.connect(self.compute_transform_matrix)
        self.gapseq_apply_tform.clicked.connect(self.apply_transform_matrix)

        self.picasso_detect_mode.currentIndexChanged.connect(self.update_picasso_options)

        self.gapseq_export_data.clicked.connect(self.export_data)
        self.export_dataset.currentIndexChanged.connect(self.update_export_options)

        self.viewer.dims.events.current_step.connect(self.draw_fiducials)

        self.dataset_dict = {}
        self.localisation_dict = {"bounding_boxes": {}, "fiducials": {}}

        self.active_dataset = None
        self.active_channel = None

        self.threadpool = QThreadPool()

        self.transform_matrix = None

        # transform_matrix_path = r"C:\Users\turnerp\Desktop\PicassoDEV\gapseq_transform_matrix-230719.txt"
        #
        # with open(transform_matrix_path, 'r') as f:
        #     transform_matrix = json.load(f)
        #
        # self.transform_matrix = np.array(transform_matrix)


        self.update_import_options()

    def update_picasso_options(self):

        if self.picasso_detect_mode.currentText() == "Fiducials":
            self.picasso_frame_mode.clear()
            self.picasso_frame_mode.addItems(["Active", "All"])
        else:
            self.picasso_frame_mode.clear()
            self.picasso_frame_mode.addItems(["Active", "All (Linked)"])

    def link_localisations(self):

        try:
            print(f"Linking localisations")

        except:
            print(traceback.format_exc())


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

            if "bounding_boxes" in layer_names:
                visible = self.viewer.layers["bounding_boxes"].visible
            else:
                visible = True

            if visible:

                if "localisation_centres" in self.localisation_dict["bounding_boxes"].keys():

                    localisations = self.localisation_dict["bounding_boxes"]["localisations"]
                    localisation_centres = self.get_localisation_centres(localisations, mode="bounding_boxes")

                    if "bounding_boxes" not in layer_names:
                        self.viewer.add_points(
                            localisation_centres,
                            edge_color="red",
                            ndim=2,
                            face_color=[0,0,0,0],
                            opacity=1.0,
                            name="bounding_boxes",
                            symbol="square",
                            size=2,
                            visible=True)
                    else:
                        self.viewer.layers["bounding_boxes"].data = localisation_centres

            for layer in layer_names:
                self.viewer.layers[layer].refresh()


    def draw_fiducials(self):

        remove_fiducials = True

        if hasattr(self, "localisation_dict") and hasattr(self, "active_channel"):

            layer_names = [layer.name for layer in self.viewer.layers]

            active_frame = self.viewer.dims.current_step[0]

            dataset_name = self.gapseq_dataset_selector.currentText()
            image_channel = self.active_channel

            if image_channel != "":

                if image_channel.lower() in self.localisation_dict["fiducials"][dataset_name].keys():

                    localisation_dict = self.localisation_dict["fiducials"][dataset_name][image_channel.lower()]

                    if "render_locs" in localisation_dict.keys():

                        render_locs = localisation_dict["render_locs"]

                        remove_fiducials = False

                        if active_frame in render_locs.keys():

                            if "fiducials" not in layer_names:
                                self.viewer.add_points(
                                    render_locs[active_frame],
                                    ndim=2,
                                    edge_color="red",
                                    face_color=[0,0,0,0],
                                    opacity=1.0,
                                    name="fiducials",
                                    symbol="disc",
                                    size=5,
                                    edge_width=0.1, )
                            else:
                                self.viewer.layers["fiducials"].data = []
                                self.viewer.layers["fiducials"].data = render_locs[active_frame]

            if remove_fiducials:
                if "fiducials" in layer_names:
                    self.viewer.layers["fiducials"].data = []

            for layer in layer_names:
                self.viewer.layers[layer].refresh()




    def draw_localisations(self):

        if hasattr(self, "image_dict"):

            try:

                layer_names = [layer.name for layer in self.viewer.layers]

                vis_mode = "square"
                vis_size = 10
                vis_opacity = 1.0
                vis_edge_width = 0.1

                if vis_mode.lower() == "square":
                    symbol = "square"
                elif vis_mode.lower() == "disk":
                    symbol = "disc"
                elif vis_mode.lower() == "x":
                    symbol = "cross"

                image_channel = self.channel_selector.currentText()

                channel_dict = self.image_dict[image_channel]

                show_localisaiton = False

                for data_key, data_dict in channel_dict.items():
                    if data_key in ["alignment fiducials","undrift fiducials","bounding boxes"]:

                        if "localisation_centres" in data_dict.keys():

                            localisation_centres = copy.deepcopy(data_dict["localisation_centres"])

                            if len(localisation_centres) > 0:

                                show_localisaiton = True

                                layer_name = data_key

                                if layer_name.lower() == "alignment fiducials":
                                    colour = "blue"
                                elif layer_name.lower() == "undrift fiducials":
                                    colour = "red"
                                else:
                                    colour = "white"

                                if layer_name not in layer_names:

                                    self.viewer.add_points(localisation_centres,
                                        edge_color=colour,
                                        face_color=[0, 0, 0,0],
                                        opacity=vis_opacity,
                                        name=layer_name,
                                        symbol=symbol,
                                        size=vis_size,
                                        edge_width=vis_edge_width, )
                                else:
                                    self.viewer.layers[layer_name].data = []
                                    self.viewer.layers[layer_name].data = localisation_centres
                                    # self.viewer.layers[layer_name].symbol = symbol
                                    # self.viewer.layers[layer_name].size = vis_size
                                    # self.viewer.layers[layer_name].opacity = vis_opacity
                                    # self.viewer.layers[layer_name].edge_width = vis_edge_width
                                    # self.viewer.layers[layer_name].edge_color = colour

                if show_localisaiton == False:
                    for data_key in ["alignment fiducials","undrift fiducials","bounding boxes"]:
                        if data_key in layer_names:
                            self.viewer.layers[data_key.lower()].data = []

                for layer in layer_names:
                    self.viewer.layers[layer].refresh()

            except:
                print(traceback.format_exc())



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


    def apply_transform(self, locs, inverse = False):

        try:

            image_shape = self.image_dict["AA"]["data"].shape[1:]

            tform = self.transform_matrix.copy().astype(np.float32)

            if inverse:
                tform = cv2.invertAffineTransform(tform)

            for loc_index, loc in enumerate(locs):

                loc_centre = np.array([[loc.x, loc.y]], dtype=np.float32)

                transformed_point = cv2.transform(np.array([loc_centre]), tform)

                transformed_loc_centre = transformed_point[0][0]

                transformed_loc = copy.deepcopy(loc)

                transformed_loc.x = transformed_loc_centre[0]
                transformed_loc.y = transformed_loc_centre[1]

                locs[loc_index] = transformed_loc

        except:
            print(traceback.format_exc())
            pass


        return locs


