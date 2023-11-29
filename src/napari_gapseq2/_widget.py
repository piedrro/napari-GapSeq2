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
from qtpy.QtWidgets import (QWidget,QVBoxLayout,QTabWidget,QSizePolicy, QComboBox,QLineEdit, QProgressBar, QLabel)
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

from qtpy.QtWidgets import QFileDialog
import os
from multiprocessing import Pool
import multiprocessing
from functools import partial

if TYPE_CHECKING:
    import napari



class GapSeqWidget(QWidget, _undrift_utils, _picasso_detect_utils, _import_utils, _events_utils):

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

        self.picasso_undrift_mode = self.findChild(QComboBox, 'picasso_undrift_mode')
        self.picasso_undrift_channel = self.findChild(QComboBox, 'picasso_undrift_channel')
        self.detect_undrift = self.findChild(QPushButton, 'detect_undrift')
        self.apply_undrift = self.findChild(QPushButton, 'apply_undrift')

        self.gapseq_compute_tform = self.findChild(QPushButton, 'gapseq_compute_tform')

        self.gapseq_link_localisations = self.findChild(QPushButton, 'gapseq_link_localisations')


        self.gapseq_import.clicked.connect(self.gapseq_import_data)
        self.gapseq_import_mode.currentIndexChanged.connect(self.update_import_options)

        # self.import_alex_data.clicked.connect(self.gapseq_import_alex_data)
        # self.channel_selector.currentIndexChanged.connect(self.update_active_image)

        self.picasso_detect.clicked.connect(self.gapseq_picasso_detect)
        self.picasso_fit.clicked.connect(self.gapseq_picasso_fit)

        self.gapseq_dataset_selector.currentIndexChanged.connect(self.update_channel_select_buttons)
        self.gapseq_dataset_selector.currentIndexChanged.connect(self.update_active_image)

        # self.channel_selector.currentIndexChanged.connect(self.draw_localisations)

        self.detect_undrift.clicked.connect(self.gapseq_picasso_undrift)
        self.apply_undrift.clicked.connect(self.gapseq_undrift_images)

        self.gapseq_compute_tform.clicked.connect(self.compute_transform_matrix)

        # self.gapseq_link_localisations.clicked.connect(self.link_localisations)

        # self.viewer.dims.events.current_step.connect(self.draw_localisations)

        self.dataset_dict = {}
        self.localisation_dict = {"bounding_boxes": {}, "fiducials": {}}

        self.active_dataset = None
        self.active_channel = None

        self.threadpool = QThreadPool()

        self.transform_matrix = None
        self.undrift_channel = None

        # transform_matrix_path = r"C:\Users\turnerp\Desktop\PicassoDEV\gapseq_transform_matrix-230719.txt"
        #
        # with open(transform_matrix_path, 'r') as f:
        #     transform_matrix = json.load(f)
        #
        # self.transform_matrix = np.array(transform_matrix)


        self.update_import_options()



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




    def compute_transform_matrix(self):

        try:
            if self.image_dict != {}:

                reference_points = None
                target_points = None

                for channel_name, channel_data in self.image_dict.items():
                    channel_ex, channel_em = channel_name

                    if "alignment fiducials" in channel_data.keys():

                        localisation_centres = channel_data["alignment fiducials"]["localisation_centres"].copy()

                        if len(localisation_centres) > 0:

                            localisation_centres = [dat[1:] for dat in localisation_centres]

                            if channel_em == "A":
                                reference_points = localisation_centres
                            elif channel_em == "D":
                                target_points = localisation_centres

                if reference_points is not None and target_points is not None:

                    reference_points = [[dat[1], dat[0]] for dat in reference_points]
                    target_points = [[dat[1], dat[0]] for dat in target_points]

                    reference_points, target_points = self.compute_registration_keypoints(reference_points, target_points)

                    reference_points = np.array(reference_points)
                    target_points = np.array(target_points)

                    self.transform_matrix, _ = cv2.estimateAffinePartial2D(reference_points, target_points, method=cv2.RANSAC)

                    print(f"Transform matrix: {self.transform_matrix}")

        except:
            print(traceback.format_exc())
            pass


    def draw_fiducials(self):

        if hasattr(self, "localisation_dict") and hasattr(self, "active_channel"):

            layer_names = [layer.name for layer in self.viewer.layers]

            if "fiducials" in layer_names:
                visible = self.viewer.layers["fiducials"].visible
            else:
                visible = True

            if visible:

                dataset_name = self.gapseq_dataset_selector.currentText()
                image_channel = self.active_channel

                localisation_centres = self.localisation_dict["fiducials"][dataset_name][image_channel.lower()]["localisation_centres"]

                if "fiducials" not in layer_names:
                    self.viewer.add_points(
                        localisation_centres,
                        edge_color="red",
                        face_color=[0,0,0,0],
                        opacity=1.0,
                        name="fiducials",
                        symbol="disc",
                        size=5,
                        edge_width=0.1, )
                else:
                    self.viewer.layers["fiducials"].data = []
                    self.viewer.layers["fiducials"].data = localisation_centres






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



    def get_localisation_centres(self, locs):

        try:
            loc_centres = []
            for loc in locs:
                frame = int(loc.frame)
                # if frame not in loc_centres.keys():
                #     loc_centres[frame] = []
                loc_centres.append([frame, loc.y, loc.x])

        except:
            print(traceback.format_exc())
            loc_centres = []

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


