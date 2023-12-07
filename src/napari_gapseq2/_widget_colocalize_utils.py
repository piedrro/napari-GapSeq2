import traceback
import numpy as np
import cv2

class _utils_colocalize:


    def populate_colocalize_combos(self):

        print(True)


    def _colocalize_fiducials(self):

        try:

            dataset = self.colo_dataset.currentText()
            channel1 = self.colo_channel1.currentText()
            channel2 = self.colo_channel2.currentText()
            max_dist = float(self.colo_max_dist.currentText())

            ch1_loc_dict, ch1_n_locs = self.get_loc_dict(dataset, channel1.lower())
            ch2_loc_dict, ch2_n_locs = self.get_loc_dict(dataset, channel2.lower())

            ch1_locs = ch1_loc_dict["localisations"].copy()
            ch2_locs = ch2_loc_dict["localisations"].copy()

            ch1_coords = [[loc.x, loc.y] for loc in ch1_locs]
            ch2_coords = [[loc.x, loc.y] for loc in ch2_locs]

            ch1_coords = np.array(ch1_coords).astype(np.float32)
            ch2_coords = np.array(ch2_coords).astype(np.float32)

            bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

            matches = bf.match(ch1_coords, ch2_coords)
            matches = [m for m in matches if m.distance < max_dist]

            ch1_coords = np.float32([ch1_coords[m.queryIdx] for m in matches]).reshape(-1, 2)
            ch2_coords = np.float32([ch2_coords[m.trainIdx] for m in matches]).reshape(-1, 2)

            filtered_ch1_locs = []
            filter_ch1_render_locs = {}

            for loc in ch1_locs:
                coord = [loc.x, loc.y]
                frame = loc.frame
                if coord in ch1_coords.tolist():
                    filtered_ch1_locs.append(loc)

                    if frame not in filter_ch1_render_locs.keys():
                        filter_ch1_render_locs[frame] = []

                    filter_ch1_render_locs[frame].append([loc.y, loc.x])

            filtered_ch1_locs = np.rec.fromrecords(filtered_ch1_locs, dtype=ch1_locs.dtype)

            filtered_ch2_locs = []
            filter_ch2_render_locs = {}

            for loc in ch2_locs:
                coord = [loc.x, loc.y]
                frame = loc.frame
                if coord in ch2_coords.tolist():
                    filtered_ch2_locs.append(loc)

                    if frame not in filter_ch2_render_locs.keys():
                        filter_ch2_render_locs[frame] = []

                    filter_ch2_render_locs[frame].append([loc.y, loc.x])

            filtered_ch2_locs = np.rec.fromrecords(filtered_ch2_locs, dtype=ch2_locs.dtype)

            filtered_ch1_loc_centers = self.get_localisation_centres(filtered_ch1_locs)
            filtered_ch2_loc_centers = self.get_localisation_centres(filtered_ch2_locs)

            self.localisation_dict["fiducials"][dataset][channel1.lower()]["localisations"] = filtered_ch1_locs
            self.localisation_dict["fiducials"][dataset][channel1.lower()]["localisation_centres"] = filtered_ch1_loc_centers
            self.localisation_dict["fiducials"][dataset][channel1.lower()]["render_locs"] = filter_ch1_render_locs

            self.localisation_dict["fiducials"][dataset][channel2.lower()]["localisations"] = filtered_ch2_locs
            self.localisation_dict["fiducials"][dataset][channel2.lower()]["localisation_centres"] = filtered_ch2_loc_centers
            self.localisation_dict["fiducials"][dataset][channel2.lower()]["render_locs"] = filter_ch2_render_locs

            self.draw_fiducials()

            if self.colo_bboxes.isChecked():

                self.localisation_dict["bounding_boxes"]["localisations"] = filtered_ch1_locs
                self.localisation_dict["bounding_boxes"]["localisation_centres"] = filtered_ch1_loc_centers
                self.draw_bounding_boxes()

        except:
            print(traceback.format_exc())
            pass
