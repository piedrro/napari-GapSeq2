import numpy as np
import cv2
import traceback



class picasso_loc_utils():

    def __init__(self, locs: np.recarray = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.locs = locs

        self.detected_dtype = [('frame', '<i4'),
                               ('x', '<i4'),
                               ('y', '<i4'),
                               ('net_gradient', '<f4')]

        self.fitted_dtype = [('frame', '<u4'),
                           ('x', '<f4'),
                           ('y', '<f4'),
                           ('photons', '<f4'),
                           ('sx', '<f4'),
                           ('sy', '<f4'),
                           ('bg', '<f4'),
                           ('lpx', '<f4'),
                           ('lpy', '<f4'),
                           ('ellipticity', '<f4'),
                           ('net_gradient', '<f4')]

        if self.locs is not None:
            self.get_loc_info()

    def get_loc_info(self):

        self.dtype = self.locs.dtype
        self.columns = self.locs.dtype.names

        if self.locs.dtype == self.fitted_dtype:
            self.loc_type = "fiducial"
        else:
            self.loc_type = "bbox"

    def coerce_new_loc_format(self, new_loc):

        if len(new_loc) != len(self.dtype):
            difference = len(self.dtype) - len(new_loc)
            if difference > 0:
                new_loc = list(new_loc)
                for i in range(difference):
                    new_loc = new_loc + [0]
                new_loc = tuple(new_loc)

        return new_loc

    def remove_duplicate_locs(self, locs = None):

            try:

                if locs is not None:
                    self.locs = locs
                    self.get_loc_info()

                unique_records, indices = np.unique(self.locs, return_index=True)

                self.locs = self.locs[indices]

            except:
                print(traceback.format_exc())
                pass

            return locs


    def add_loc(self, locs = None, new_loc = None):

        try:

            if locs is not None:
                self.locs = locs
                self.get_loc_info()

            if type(new_loc) == list:
                new_loc = tuple(new_loc)
            if type(new_loc) in [np.ndarray, np.recarray]:
                new_loc = tuple(new_loc.tolist())

            new_loc = self.coerce_new_loc_format(new_loc)

            self.locs = np.array(self.locs).tolist()
            self.locs.append(new_loc)
            self.locs = np.rec.fromrecords(self.locs, dtype=self.dtype)

            self.remove_duplicate_locs()

        except:
            print(traceback.format_exc())
            pass

        return self.locs

    def remove_loc(self, locs = None, loc_index = None):

        try:

            if locs is not None:
                self.locs = locs
                self.get_loc_info()

            if loc_index is not None:

                if loc_index < len(self.locs):

                    self.locs = self.locs.view(np.float32).reshape(len(self.locs), -1)
                    self.locs = np.delete(self.locs, loc_index, axis=0)
                    self.locs = self.locs.view(self.dtype)
                    self.locs = np.squeeze(self.locs, axis=1)

                self.remove_duplicate_locs()

        except:
            print(loc_index)
            print(self.locs, self.locs.dtype)
            print(traceback.format_exc())
            pass

        return self.locs

    def create_locs(self, new_loc, fitted = False):

        try:

            if fitted == False:
                dtype = self.detected_dtype
            else:
                dtype = self.fitted_dtype

            if type(new_loc) == list:
                new_loc = [tuple(new_loc)]
            elif type(new_loc) == tuple:
                new_loc = [new_loc]
            elif type(new_loc) in [np.array, np.ndarray]:
                new_loc = [tuple(new_loc.tolist())]

            self.locs = np.rec.fromrecords(new_loc, dtype=dtype)

        except:
            print(traceback.format_exc())
            pass

        return self.locs




class _loc_utils():

    def get_loc_dict(self, dataset_name="", channel_name="", type = "fiducials"):

        loc_dict = {}
        n_localisations = 0

        try:

            if type.lower() == "fiducials":

                if dataset_name not in self.localisation_dict["fiducials"].keys():
                    self.localisation_dict["fiducials"][dataset_name] = {}
                else:
                    if channel_name not in self.localisation_dict["fiducials"][dataset_name].keys():
                        self.localisation_dict["fiducials"][dataset_name][channel_name] = {}
                    else:
                        loc_dict = self.localisation_dict["fiducials"][dataset_name][channel_name].copy()
            else:

                if "bounding_boxes" not in self.localisation_dict.keys():
                    self.localisation_dict["bounding_boxes"] = {}

                loc_dict = self.localisation_dict["bounding_boxes"].copy()

            if "localisations" in loc_dict.keys():
                n_localisations = len(loc_dict["localisations"])

        except:
            print(traceback.format_exc())
            pass

        return loc_dict, n_localisations


    def update_loc_dict(self, dataset_name="", channel_name="", type = "fiducials", loc_dict = {}):

        try:

            if type == "fiducials":
                self.localisation_dict["fiducials"][dataset_name][channel_name] = loc_dict
            else:
                self.localisation_dict["bounding_boxes"] = loc_dict

        except:
            print(traceback.format_exc())
            pass

    def get_bbox_dict(self,dataset_name, channel_name):

        bbox_dict = {}

        if "bounding_boxes" not in self.localisation_dict.keys():
            self.localisation_dict["bounding_boxes"] = {}

        return bbox_dict

    def compute_net_gradient(self, position, box_size = None):

        net_gradient = 0

        try:

            dataset = self.gapseq_dataset_selector.currentText()
            channel = self.active_channel
            frame = self.viewer.dims.current_step[0]

            if box_size is None:
                box_size = self.picasso_box_size.currentText()

            loc_mask, _, loc_bg_mask = self.generate_localisation_mask(
                box_size, spot_shape = "square")

            box_size = len(loc_mask[0])

            x, y = position[0], position[1]

            if box_size % 2 == 0:
                x += 0.5
                y += 0.5
                x, y = round(x), round(y)
                x1 = x - (box_size // 2)
                x2 = x + (box_size // 2)
                y1 = y - (box_size // 2)
                y2 = y + (box_size // 2)
            else:
                # Odd spot width
                x, y = round(x), round(y)
                x1 = x - (box_size // 2)
                x2 = x + (box_size // 2) + 1
                y1 = y - (box_size // 2)
                y2 = y + (box_size // 2) + 1

            loc_spot = self.dataset_dict[dataset][channel]["data"][frame][y1:y2, x1:x2]

            loc_spot_values = np.sum(loc_spot[loc_mask])
            loc_spot_bg_values = np.mean(loc_spot[loc_bg_mask])

            net_gradient = loc_spot_values

        except:
            print(traceback.format_exc())
            pass

        return float(net_gradient)



    def add_manual_localisation(self, position, mode):

        try:

            active_dataset = self.gapseq_dataset_selector.currentText()
            active_channel = self.active_channel
            box_size = int(self.picasso_box_size.currentText())
            frame = self.viewer.dims.current_step[0]
            net_gradient = self.compute_net_gradient(position, box_size=box_size)


            if mode == "fiducials":

                loc_dict, n_locs = self.get_loc_dict(active_dataset, active_channel,
                    type = "fiducials")

                if n_locs > 0:

                    locs = loc_dict["localisations"].copy()
                    render_locs = loc_dict["render_locs"].copy()
                    loc_centers = loc_dict["localisation_centres"].copy()
                    box_size = int(loc_dict["box_size"])
                    dtype = locs.dtype

                    loc_utils = picasso_loc_utils(locs)

                    x,y = position

                    loc_centers = np.array(loc_centers)

                    if loc_centers.shape[-1] !=2:
                        loc_coords = loc_centers[:,1:].copy()
                    else:
                        loc_coords = loc_centers.copy()

                    # Calculate Euclidean distances
                    distances = np.sqrt(np.sum((loc_coords - np.array([y,x])) ** 2, axis=1))

                    # Find the index of the minimum distance
                    min_index = np.argmin(distances)
                    min_distance = distances[min_index]

                    if min_distance < box_size:

                        locs = loc_utils.remove_loc(loc_index=min_index)

                        loc_centers = np.delete(loc_centers, min_index, axis=0)
                        loc_centers = loc_centers.tolist()

                        render_frame_locs = render_locs[frame].copy()
                        render_frame_locs = np.unique(render_frame_locs, axis=0).tolist()
                        distances = np.sqrt(np.sum((np.array(render_frame_locs) - np.array([y,x])) ** 2, axis=1))
                        min_index = np.argmin(distances)
                        render_frame_locs.pop(min_index)
                        render_locs[frame] = render_frame_locs

                        loc_dict["localisations"] = locs
                        loc_dict["localisation_centres"] = loc_centers
                        loc_dict["render_locs"] = render_locs

                        self.update_loc_dict(active_dataset, active_channel, "fiducials", loc_dict)
                        self.draw_fiducials(update_vis=True)

                    else:

                        locs = loc_utils.add_loc(new_loc = [frame, x, y, net_gradient])

                        loc_centers = np.append(loc_centers, np.array([[frame,y,x]], dtype=int), axis=0)
                        loc_centers = loc_centers.tolist()
                        render_locs[frame].append([round(y),round(x)])

                        loc_dict["localisations"] = locs
                        loc_dict["localisation_centres"] = loc_centers
                        loc_dict["render_locs"] = render_locs

                        self.update_loc_dict(active_dataset, active_channel, "fiducials", loc_dict)
                        self.draw_fiducials(update_vis=True)

                else:
                    x, y = position

                    box_size = int(self.picasso_box_size.currentText())

                    new_loc = [frame, position[0], position[1], net_gradient]

                    loc_utils = picasso_loc_utils()
                    locs = loc_utils.create_locs(new_loc=new_loc)

                    loc_centers = [[frame, y, x]]
                    render_locs = {frame: [[y, x]]}

                    loc_dict["localisations"] = locs
                    loc_dict["localisation_centres"] = loc_centers
                    loc_dict["render_locs"] = render_locs
                    loc_dict["fitted"] = False
                    loc_dict["box_size"] = box_size

                    self.update_loc_dict(active_dataset, active_channel, "fiducials", loc_dict)
                    self.draw_fiducials(update_vis=True)

            else:

                loc_dict, n_locs = self.get_loc_dict(active_dataset, active_channel,
                    type = "bounding_box")

                if n_locs > 0:

                    locs = loc_dict["localisations"].copy()
                    loc_centers = loc_dict["localisation_centres"].copy()
                    box_size = int(loc_dict["box_size"])
                    dtype = locs.dtype

                    loc_utils = picasso_loc_utils(locs)

                    x, y = position

                    loc_centers = np.array(loc_centers)

                    if loc_centers.shape[-1] != 2:
                        loc_coords = loc_centers[:, 1:].copy()
                    else:
                        loc_coords = loc_centers.copy()

                    # Calculate Euclidean distances
                    distances = np.sqrt(np.sum((loc_coords - np.array([y, x])) ** 2, axis=1))

                    # Find the index of the minimum distance
                    min_index = np.argmin(distances)
                    min_distance = distances[min_index]

                    if min_distance < box_size:

                        locs = loc_utils.remove_loc(loc_index=min_index)

                        loc_centers = np.delete(loc_centers, min_index, axis=0)
                        loc_centers = loc_centers.tolist()

                        loc_dict["localisations"] = locs
                        loc_dict["localisation_centres"] = loc_centers
                        self.update_loc_dict(active_dataset, active_channel, "bounding_boxes", loc_dict)
                        self.draw_bounding_boxes()

                    else:

                        locs = loc_utils.add_loc(new_loc = [frame, x, y, net_gradient])

                        loc_centers = np.append(loc_centers, np.array([[y,x]], dtype=int), axis=0)
                        loc_centers = loc_centers.tolist()

                        loc_dict["localisations"] = locs
                        loc_dict["localisation_centres"] = loc_centers

                        self.update_loc_dict(active_dataset, active_channel, "bounding_boxes", loc_dict)
                        self.draw_bounding_boxes()

                else:

                    x, y = position

                    box_size = int(self.picasso_box_size.currentText())

                    new_loc = [frame, position[0], position[1], net_gradient]

                    loc_utils = picasso_loc_utils()
                    locs = loc_utils.create_locs(new_loc=new_loc)

                    loc_centers = [[y, x]]

                    loc_dict["localisations"] = locs
                    loc_dict["localisation_centres"] = loc_centers
                    loc_dict["fitted"] = False
                    loc_dict["box_size"] = box_size

                    self.update_loc_dict(active_dataset, active_channel, "bounding_boxes", loc_dict)
                    self.draw_bounding_boxes()

        except:
            print(traceback.format_exc())
            pass


