from PyQt5.QtGui import QCursor
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout, QWidget
import numpy as np
import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import traceback
from functools import partial
import uuid
import copy


class _plot_utils:

    def populate_plot_combos(self):

        print(True)
















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
            self.parent.update_crop_range(xpos)

        elif event.modifiers() & Qt.AltModifier:

            xpos = self.get_event_x_postion(event, mode="click")
            self.parent.update_gamma_ranges(xpos)

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

        # print(f"pos: {pos}, scene_pos: {scene_pos}")

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
