import os.path
import traceback
import numpy as np
from functools import partial
from qtpy.QtWidgets import (QSlider, QLabel)

class _events_utils:

    def update_overlay_text(self):

        try:

            if self.dataset_dict  != {}:

                dataset_name = self.gapseq_dataset_selector.currentText()
                channel_name = self.active_channel

                if dataset_name in self.dataset_dict.keys():
                    if channel_name in self.dataset_dict[dataset_name].keys():

                        channel_dict = self.dataset_dict[dataset_name][channel_name].copy()

                        path = channel_dict["path"]
                        file_name = os.path.basename(path)

                        if channel_name in ["da", "dd", "aa", "ad"]:
                            channel_name = channel_name.upper()
                        else:
                            channel_name = channel_name.capitalize()

                        overlay_string = ""
                        overlay_string += f"File: {file_name}\n"
                        overlay_string += f"Dataset: {dataset_name}\n"
                        overlay_string += f"Channel: {channel_name}\n"

                        if overlay_string != "":
                            self.viewer.text_overlay.visible = True
                            self.viewer.text_overlay.position = "top_left"
                            self.viewer.text_overlay.text = overlay_string.lstrip("\n")
                            self.viewer.text_overlay.color = "red"
                            self.viewer.text_overlay.font_size = 10
                        else:
                            self.viewer.text_overlay.visible = False

        except:
            print(traceback.format_exc())


    def select_channel_da(self, event=None):
        self.update_active_image(channel="da")

    def select_channel_dd(self, event=None):
        self.update_active_image(channel="dd")

    def select_channel_aa(self, event=None):
        self.update_active_image(channel="aa")

    def select_channel_ad(self, event=None):
        self.update_active_image(channel="ad")

    def select_channel_donor(self, event=None):
        self.update_active_image(channel="donor")

    def select_channel_acceptor(self, event=None):
        self.update_active_image(channel="acceptor")

    def update_active_image(self, channel=None, dataset=None, event=None):

        try:

            if dataset == None or dataset not in self.dataset_dict.keys():
                dataset_name = self.gapseq_dataset_selector.currentText()
            else:
                dataset_name = dataset

            if dataset_name in self.dataset_dict.keys():

                channel_names = [channel for channel in self.dataset_dict[dataset_name].keys()]

                if channel not in channel_names:
                    if self.active_channel in channel_names:
                        channel = self.active_channel
                    else:
                        channel = channel_names[0]

                self.active_dataset = dataset_name
                self.active_channel = channel

                if "data" in self.dataset_dict[dataset_name][channel].keys():

                    image = self.dataset_dict[dataset_name][channel]["data"]

                    contrast_range = [np.min(image), np.max(image)]

                    if channel in ["da", "dd", "aa", "ad"]:
                        channel_name = channel.upper()
                    else:
                        channel_name = channel.capitalize()

                    layer_name = f"{dataset_name}: [{channel_name}]"

                    if hasattr(self, "image_layer") == False:

                        self.image_layer = self.viewer.add_image(image,
                            name=layer_name,
                            colormap="gray",
                            blending="additive",
                            visible=True)

                        self.image_layer.mouse_drag_callbacks.append(self._mouse_event)


                    else:
                        self.image_layer.data = image
                        self.image_layer.name = layer_name
                        self.image_layer.refresh()

                    self.viewer.layers[layer_name].contrast_limits = contrast_range

            self.draw_fiducials(update_vis=True)
            self.update_overlay_text()

        except:
            print(traceback.format_exc())
            pass

    def update_channel_selector(self, dataset_selector, channel_selector, event=None, channel_type = "all", efficiency=False, block_signals=False):

        try:

            if hasattr(self, channel_selector) and hasattr(self, dataset_selector):

                channel_selector = getattr(self, channel_selector)
                dataset_selector = getattr(self, dataset_selector)

                dataset_name = dataset_selector.currentText()

                if block_signals == True:
                    channel_selector.blockSignals(True)

                channel_selector_list = []

                if dataset_name in self.dataset_dict.keys():

                    channel_names = [channel.lower() for channel in self.dataset_dict[dataset_name].keys()]

                    if channel_type.lower() == "donor":
                        channel_names = [channel for channel in channel_names if channel in ["dd","ad", "donor"]]
                    elif channel_type.lower() == "acceptor":
                        channel_names = [channel for channel in channel_names if channel in ["da","aa", "acceptor"]]

                    for channel in channel_names:

                        if "efficiency" not in channel.lower():

                            if channel in ["da", "dd", "aa", "ad"]:
                                channel_selector_list.append(channel.upper())
                            elif channel in ["donor", "acceptor"]:
                                channel_selector_list.append(channel.capitalize())

                            if efficiency == True:
                                if set(["donor", "acceptor"]).issubset(set(channel_names)):
                                    channel_selector_list.append("FRET Efficiency")
                                if set(["dd", "da"]).issubset(set(channel_names)):
                                    channel_selector_list.append("ALEX Efficiency")

                elif dataset_name == "All Datasets":

                    channel_names = []

                    for dataset_name in self.dataset_dict.keys():
                        dataset_channels = [channel.lower() for channel in self.dataset_dict[dataset_name].keys()]
                        channel_names.append(dataset_channels)

                    channel_names = list(set.intersection(*map(set, channel_names)))

                    for channel in channel_names:

                        if "efficiency" not in channel.lower():

                            if channel in ["da", "dd", "aa", "ad"]:
                                channel_selector_list.append(channel.upper())
                            elif channel in ["donor", "acceptor"]:
                                channel_selector_list.append(channel.capitalize())

                            if efficiency == True:
                                if set(["donor", "acceptor"]).issubset(set(channel_names)):
                                    channel_selector_list.append("FRET Efficiency")
                                if set(["dd", "da"]).issubset(set(channel_names)):
                                    channel_selector_list.append("ALEX Efficiency")

                if channel_selector_list != []:

                    channel_selector.clear()
                    channel_selector_list = list(set(channel_selector_list))
                    channel_selector.addItems(channel_selector_list)

                channel_selector.blockSignals(False)

        except:
            print(traceback.format_exc())
            pass

    def populate_channel_selectors(self):

        try:

            self.update_channel_selector(dataset_selector="picasso_dataset", channel_selector="picasso_channel")
            self.update_channel_selector(dataset_selector="undrift_dataset_selector", channel_selector="undrift_channel_selector")
            self.update_channel_selector(dataset_selector="cluster_dataset", channel_selector="cluster_channel")
            self.update_channel_selector(dataset_selector="tform_compute_dataset", channel_selector="tform_compute_ref_channel", channel_type="donor")
            self.update_channel_selector(dataset_selector="tform_compute_dataset", channel_selector="tform_compute_target_channel", channel_type="acceptor")
            self.update_channel_selector(dataset_selector="colo_dataset", channel_selector="colo_channel1")
            self.update_channel_selector(dataset_selector="colo_dataset", channel_selector="colo_channel2")

        except:
            print(traceback.format_exc())
            pass



    def update_channel_select_buttons(self):

        try:
            datast_name = self.gapseq_dataset_selector.currentText()

            if datast_name in self.dataset_dict.keys():

                fret_modes = [self.dataset_dict[datast_name][channel]["FRET"] for channel in self.dataset_dict[datast_name].keys()]
                channel_refs = [self.dataset_dict[datast_name][channel]["channel_ref"] for channel in self.dataset_dict[datast_name].keys()]

                channel_refs = list(set(channel_refs))
                fret_mode = list(set(fret_modes))[0]

                self.gapseq_show_dd.clicked.connect(lambda: None)
                self.gapseq_show_da.clicked.connect(lambda: None)
                self.gapseq_show_aa.clicked.connect(lambda: None)
                self.gapseq_show_ad.clicked.connect(lambda: None)

                if fret_mode == True:

                    self.gapseq_show_dd.setVisible(True)
                    self.gapseq_show_da.setVisible(True)
                    self.gapseq_show_aa.setVisible(False)
                    self.gapseq_show_ad.setVisible(False)

                    if "dd" in channel_refs:
                        self.gapseq_show_dd.setEnabled(True)
                        self.gapseq_show_dd.setText("Donor [F1]")
                        self.viewer.bind_key(key="F1", func=self.select_channel_donor, overwrite=True)
                        self.gapseq_show_dd.clicked.connect(partial(self.update_active_image, channel="donor"))
                    else:
                        self.gapseq_show_dd.setEnabled(False)
                        self.gapseq_show_dd.setText("")

                    if "da" in channel_refs:
                        self.gapseq_show_da.setEnabled(True)
                        self.gapseq_show_da.setText("Acceptor [F2]")
                        self.viewer.bind_key(key="F2", func=self.select_channel_acceptor, overwrite=True)
                        self.gapseq_show_da.clicked.connect(partial(self.update_active_image, channel="acceptor"))
                    else:
                        self.gapseq_show_da.setEnabled(False)
                        self.gapseq_show_da.setText("")

                else:

                    self.gapseq_show_dd.setVisible(True)
                    self.gapseq_show_da.setVisible(True)
                    self.gapseq_show_aa.setVisible(True)
                    self.gapseq_show_ad.setVisible(True)

                    if "dd" in channel_refs:
                        self.gapseq_show_dd.setText("DD [F1]")
                        self.gapseq_show_dd.setEnabled(True)
                        self.viewer.bind_key(key="F1", func=self.select_channel_dd, overwrite=True)
                        self.gapseq_show_dd.clicked.connect(partial(self.update_active_image, channel="dd"))

                    else:
                        self.gapseq_show_dd.setText("")
                        self.gapseq_show_dd.setEnabled(False)

                    if "da" in channel_refs:
                        self.gapseq_show_da.setText("DA [F2]")
                        self.gapseq_show_da.setEnabled(True)
                        self.viewer.bind_key(key="F2", func=self.select_channel_da, overwrite=True)
                        self.gapseq_show_da.clicked.connect(partial(self.update_active_image, channel="da"))
                    else:
                        self.gapseq_show_da.setText("")
                        self.gapseq_show_da.setEnabled(False)

                    if "ad" in channel_refs:
                        self.gapseq_show_ad.setText("AD [F3]")
                        self.gapseq_show_ad.setEnabled(True)
                        self.viewer.bind_key(key="F3", func=self.select_channel_ad, overwrite=True)
                        self.gapseq_show_ad.clicked.connect(partial(self.update_active_image, channel="ad"))
                    else:
                        self.gapseq_show_ad.setText("")
                        self.gapseq_show_ad.setEnabled(False)

                    if "aa" in channel_refs:
                        self.gapseq_show_aa.setText("AA [F4]")
                        self.gapseq_show_aa.setEnabled(True)
                        self.viewer.bind_key(key="F4", func=self.select_channel_aa, overwrite=True)
                        self.gapseq_show_aa.clicked.connect(partial(self.update_active_image, channel="aa"))
                    else:
                        self.gapseq_show_aa.setText("")
                        self.gapseq_show_aa.setEnabled(False)

        except:
            print(traceback.format_exc())
            pass

    def update_import_append_options(self):

        try:

            if self.gapseq_append.isChecked():
                self.gapseq_append_dataset.setEnabled(True)
                self.gapseq_append_dataset_label.setEnabled(True)
                self.gapseq_append_dataset.setVisible(True)
                self.gapseq_append_dataset_label.setVisible(True)
            else:
                self.gapseq_append_dataset.setEnabled(False)
                self.gapseq_append_dataset_label.setEnabled(False)
                self.gapseq_append_dataset.setVisible(False)
                self.gapseq_append_dataset_label.setVisible(False)

        except:
            print(traceback.format_exc())
            pass




    def update_import_options(self):

        def update_channel_layout(self, show = True):
            if show:
                self.gapseq_channel_layout.setEnabled(True)
                self.gapseq_channel_layout.clear()
                self.gapseq_channel_layout.addItems(["Donor-Acceptor", "Acceptor-Donor"])
                self.gapseq_channel_layout.setHidden(False)
                self.gapseq_channel_layout_label.setHidden(False)
            else:
                self.gapseq_channel_layout.setEnabled(False)
                self.gapseq_channel_layout.clear()
                self.gapseq_channel_layout.setHidden(True)
                self.gapseq_channel_layout_label.setHidden(True)

        def update_alex_first_frame(self, show = True):
            if show:
                self.gapseq_alex_first_frame.setEnabled(True)
                self.gapseq_alex_first_frame.clear()
                self.gapseq_alex_first_frame.addItems(["Donor", "Acceptor"])
                self.gapseq_alex_first_frame.setHidden(False)
                self.gapseq_alex_first_frame_label.setHidden(False)
            else:
                self.gapseq_alex_first_frame.setEnabled(False)
                self.gapseq_alex_first_frame.clear()
                self.gapseq_alex_first_frame.setHidden(True)
                self.gapseq_alex_first_frame_label.setHidden(True)

        import_mode = self.gapseq_import_mode.currentText()

        if import_mode in ["Donor", "Acceptor"]:
            update_channel_layout(self, show = False)
            update_alex_first_frame(self, show = False)

        elif import_mode == "FRET":
            update_channel_layout(self, show = True)
            update_alex_first_frame(self, show = False)

        elif import_mode == "ALEX":
            update_channel_layout(self, show = True)
            update_alex_first_frame(self, show = True)

        elif import_mode in ["DA", "DD", "AA","AD"]:
            update_channel_layout(self, show = False)
            update_alex_first_frame(self, show = False)


    def gapseq_progress(self, progress, progress_bar):

        progress_bar.setValue(progress)

        if progress == 100:
            progress_bar.setValue(0)
            progress_bar.setHidden(True)
            progress_bar.setEnabled(False)
        else:
            progress_bar.setHidden(False)
            progress_bar.setEnabled(True)

    def _mouse_event(self, viewer, event):

        try:

            event_pos = self.image_layer.world_to_data(event.position)
            image_shape = self.image_layer.data.shape
            modifiers = event.modifiers

            if "Shift" in modifiers or "Control" in modifiers:

                if "Shift" in modifiers:
                    mode = "fiducials"
                elif "Control" in modifiers:
                    mode = "bounding_box"

                [y,x] = [event_pos[-2], event_pos[-1]]

                if (x >= 0) & (x < image_shape[-1]) & (y >= 0) & (y < image_shape[-2]):

                    self.add_manual_localisation(position=[x,y], mode=mode)

        except:
            print(traceback.format_exc())


    def update_dataset_name(self):

        try:

            old_name = self.gapseq_old_dataset_name.currentText()
            new_name = self.gapseq_new_dataset_name.text()

            if old_name != "":

                if new_name == "":
                    raise ValueError("New dataset name cannot be blank")
                elif new_name in self.dataset_dict.keys():
                    raise ValueError("New dataset name must be unique")
                else:
                    dataset_data = self.dataset_dict.pop(old_name)
                    self.dataset_dict[new_name] = dataset_data

                    localisation_data = self.localisation_dict["fiducials"].pop(old_name)
                    self.localisation_dict["fiducials"][new_name] = localisation_data

                    if hasattr(self, "traces_dict"):
                        if old_name in self.traces_dict.keys():
                            print("Updating traces dict")
                            traces_data = self.traces_dict.pop(old_name)
                            self.traces_dict[new_name] = traces_data



                self.populate_dataset_combos()
                self.update_channel_select_buttons()
                self.update_active_image()

                self.populate_plot_combos()
                self.populate_export_combos()
                self.initialize_plot()

        except:
            print(traceback.format_exc())



    def update_slider_label(self, slider_name):

        label_name = slider_name + "_label"

        self.slider = self.findChild(QSlider, slider_name)
        self.label = self.findChild(QLabel, label_name)

        slider_value = self.slider.value()
        self.label.setText(str(slider_value))

    def update_picasso_options(self):

        if self.picasso_detect_mode.currentText() == "Fiducials":
            self.picasso_frame_mode.clear()
            self.picasso_frame_mode.addItems(["Active", "All"])
        else:
            self.picasso_frame_mode.clear()
            self.picasso_frame_mode.addItem("Active")
