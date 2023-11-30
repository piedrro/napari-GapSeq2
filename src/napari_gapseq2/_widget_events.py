import traceback
import numpy as np


class _events_utils:


    def select_channel_da(self):
        self.update_active_image(channel="da")

    def select_channel_dd(self):
        self.update_active_image(channel="dd")

    def select_channel_aa(self):
        self.update_active_image(channel="aa")

    def select_channel_ad(self):
        self.update_active_image(channel="ad")

    def update_active_image(self, channel=None, event=None):

        try:

            dataset_name = self.gapseq_dataset_selector.currentText()

            if dataset_name in self.dataset_dict.keys():

                channel_names = [channel for channel in self.dataset_dict[dataset_name].keys()]

                if channel not in channel_names:

                    channel = channel_names[0]

                if self.active_dataset != dataset_name or self.active_channel != channel:

                    self.active_dataset = dataset_name
                    self.active_channel = channel

                    image_layers = [layer.name for layer in self.viewer.layers if layer.name not in ["bounding_boxes", "fiducials"]]

                    image = self.dataset_dict[dataset_name][channel]["data"]

                    contrast_range = [np.min(image), np.max(image)]

                    if dataset_name not in image_layers:

                        self.viewer.add_image(image,
                            name=dataset_name,
                            colormap="gray",
                            blending="additive",
                            visible=True)

                    else:
                        self.viewer.layers[dataset_name].data = image

                    self.viewer.layers[dataset_name].contrast_limits = contrast_range

            self.draw_fiducials()

        except:
            print(traceback.format_exc())
            pass


    def update_channel_select_buttons(self):

        try:
            datast_name = self.gapseq_dataset_selector.currentText()

            if datast_name in self.dataset_dict.keys():

                fret_modes = [self.dataset_dict[datast_name][channel]["FRET"] for channel in self.dataset_dict[datast_name].keys()]
                channel_refs = [self.dataset_dict[datast_name][channel]["channel_ref"] for channel in self.dataset_dict[datast_name].keys()]

                self.picasso_channel.clear()
                self.undrift_channel_selector.clear()
                self.cluster_channel.clear()

                channel_refs = list(set(channel_refs))
                fret_mode = list(set(fret_modes))[0]

                if fret_mode == True:

                    self.gapseq_show_dd.setVisible(True)
                    self.gapseq_show_da.setVisible(True)
                    self.gapseq_show_aa.setVisible(False)
                    self.gapseq_show_ad.setVisible(False)

                    if "dd" in channel_refs:
                        self.gapseq_show_dd.setEnabled(True)
                        self.gapseq_show_dd.setText("Donor")
                        self.picasso_channel.addItem("Donor")
                        self.undrift_channel_selector.addItem("Donor")
                        self.cluster_channel.addItem("Donor")
                    else:
                        self.gapseq_show_dd.setEnabled(False)
                        self.gapseq_show_dd.setText("")

                    if "da" in channel_refs:
                        self.gapseq_show_da.setEnabled(True)
                        self.gapseq_show_da.setText("Acceptor")
                        self.picasso_channel.addItem("Acceptor")
                        self.undrift_channel_selector.addItem("Acceptor")
                        self.cluster_channel.addItem("Acceptor")
                    else:
                        self.gapseq_show_da.setEnabled(False)
                        self.gapseq_show_da.setText("")

                else:

                    self.gapseq_show_dd.setVisible(True)
                    self.gapseq_show_da.setVisible(True)
                    self.gapseq_show_aa.setVisible(True)
                    self.gapseq_show_ad.setVisible(True)

                    if "dd" in channel_refs:
                        self.gapseq_show_dd.setText("DD")
                        self.gapseq_show_dd.setEnabled(True)
                        self.picasso_channel.addItem("DD")
                        self.undrift_channel_selector.addItem("DD")
                        self.cluster_channel.addItem("DD")
                    else:
                        self.gapseq_show_dd.setText("")
                        self.gapseq_show_dd.setEnabled(False)

                    if "da" in channel_refs:
                        self.gapseq_show_da.setText("DA")
                        self.gapseq_show_da.setEnabled(True)
                        self.picasso_channel.addItem("DA")
                        self.undrift_channel_selector.addItem("DA")
                        self.cluster_channel.addItem("DA")
                    else:
                        self.gapseq_show_da.setText("")
                        self.gapseq_show_da.setEnabled(False)

                    if "aa" in channel_refs:
                        self.gapseq_show_aa.setText("AA")
                        self.gapseq_show_aa.setEnabled(True)
                        self.picasso_channel.addItem("AA")
                        self.undrift_channel_selector.addItem("AA")
                        self.cluster_channel.addItem("AA")
                    else:
                        self.gapseq_show_aa.setText("")
                        self.gapseq_show_aa.setEnabled(False)

                    if "ad" in channel_refs:
                        self.gapseq_show_ad.setText("AD")
                        self.gapseq_show_ad.setEnabled(True)
                        self.picasso_channel.addItem("AD")
                        self.undrift_channel_selector.addItem("AD")
                        self.cluster_channel.addItem("AD")
                    else:
                        self.gapseq_show_ad.setText("")
                        self.gapseq_show_ad.setEnabled(False)

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
