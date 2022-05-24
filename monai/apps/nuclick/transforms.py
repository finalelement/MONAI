import random
import math
import cv2
import skimage
import numpy as np

from monai.config import KeysCollection
from monai.transforms import (
    MapTransform,
    RandomizableTransform,
    Transform
)

from skimage.morphology import disk, reconstruction, remove_small_holes, remove_small_objects

class FlattenLabeld(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            _, labels, _, _ = cv2.connectedComponentsWithStats(d[key], 4, cv2.CV_32S)
            d[key] = labels.astype(np.uint8)
        return d

class ExtractPatchd(MapTransform):
    def __init__(self, keys: KeysCollection, centroid_key="centroid", patch_size=128):
        super().__init__(keys)
        self.centroid_key = centroid_key
        self.patch_size = patch_size

    def __call__(self, data):
        d = dict(data)

        centroid = d[self.centroid_key]  # create mask based on centroid (select nuclei based on centroid)
        roi_size = (self.patch_size, self.patch_size)

        for key in self.keys:
            img = d[key]
            x_start, x_end, y_start, y_end = self.bbox(self.patch_size, centroid, img.shape[-2:])
            cropped = img[:, x_start:x_end, y_start:y_end]
            d[key] = self.pad_to_shape(cropped, roi_size)
        return d

    @staticmethod
    def bbox(patch_size, centroid, size):
        x, y = centroid
        m, n = size

        x_start = int(max(x - patch_size / 2, 0))
        y_start = int(max(y - patch_size / 2, 0))
        x_end = x_start + patch_size
        y_end = y_start + patch_size
        if x_end > m:
            x_end = m
            x_start = m - patch_size
        if y_end > n:
            y_end = n
            y_start = n - patch_size
        return x_start, x_end, y_start, y_end

    @staticmethod
    def pad_to_shape(img, shape):
        img_shape = img.shape[-2:]
        s_diff = np.array(shape) - np.array(img_shape)
        diff = [(0, 0), (0, s_diff[0]), (0, s_diff[1])]
        return np.pad(
            img,
            diff,
            mode="constant",
            constant_values=0,
        )


class SplitLabeld(Transform):
    def __init__(self, label="label", others="others", mask_value="mask_value", min_area=5):
        self.label = label
        self.others = others
        self.mask_value = mask_value
        self.min_area = min_area

    def __call__(self, data):
        d = dict(data)
        label = d[self.label]
        mask_value = d[self.mask_value]

        mask = np.uint8(label == mask_value)
        others = (1 - mask) * label
        others = self._mask_relabeling(others[0], min_area=self.min_area)[np.newaxis]

        d[self.label] = mask
        d[self.others] = others
        return d

    @staticmethod
    def _mask_relabeling(mask, min_area=5):
        res = np.zeros_like(mask)
        for l in np.unique(mask):
            if l == 0:
                continue

            m = skimage.measure.label(mask == l, connectivity=1)
            for stat in skimage.measure.regionprops(m):
                if stat.area > min_area:
                    res[stat.coords[:, 0], stat.coords[:, 1]] = l
        return res

class FilterImaged(MapTransform):
    def __init__(self, keys: KeysCollection, min_size: int = 500):
        super().__init__(keys)
        self.min_size = min_size

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            d[key] = self.filter(img)
        return d

    def filter(self, rgb):
        mask_not_green = self.filter_green_channel(rgb)
        mask_not_gray = self.filter_grays(rgb)
        mask_gray_green = mask_not_gray & mask_not_green
        mask = (
            self.filter_remove_small_objects(mask_gray_green, min_size=self.min_size) if self.min_size else mask_gray_green
        )

        return rgb * np.dstack([mask, mask, mask])

    def filter_green_channel(self, img_np, green_thresh=200, avoid_overmask=True, overmask_thresh=90, output_type="bool"):
        g = img_np[:, :, 1]
        gr_ch_mask = (g < green_thresh) & (g > 0)
        mask_percentage = self.mask_percent(gr_ch_mask)
        if (mask_percentage >= overmask_thresh) and (green_thresh < 255) and (avoid_overmask is True):
            new_green_thresh = math.ceil((255 - green_thresh) / 2 + green_thresh)
            gr_ch_mask = self.filter_green_channel(img_np, new_green_thresh, avoid_overmask, overmask_thresh, output_type)
        return gr_ch_mask

    def filter_grays(self, rgb, tolerance=15):
        rg_diff = abs(rgb[:, :, 0] - rgb[:, :, 1]) <= tolerance
        rb_diff = abs(rgb[:, :, 0] - rgb[:, :, 2]) <= tolerance
        gb_diff = abs(rgb[:, :, 1] - rgb[:, :, 2]) <= tolerance
        return ~(rg_diff & rb_diff & gb_diff)

    def mask_percent(self, img_np):
        if (len(img_np.shape) == 3) and (img_np.shape[2] == 3):
            np_sum = img_np[:, :, 0] + img_np[:, :, 1] + img_np[:, :, 2]
            mask_percentage = 100 - np.count_nonzero(np_sum) / np_sum.size * 100
        else:
            mask_percentage = 100 - np.count_nonzero(img_np) / img_np.size * 100
        return mask_percentage

    def filter_remove_small_objects(self, img_np, min_size=3000, avoid_overmask=True, overmask_thresh=95):
        rem_sm = remove_small_objects(img_np.astype(bool), min_size=min_size)
        mask_percentage = self.mask_percent(rem_sm)
        if (mask_percentage >= overmask_thresh) and (min_size >= 1) and (avoid_overmask is True):
            new_min_size = round(min_size / 2)
            rem_sm = self.filter_remove_small_objects(img_np, new_min_size, avoid_overmask, overmask_thresh)
        return rem_sm


class AddPointGuidanceSignald(RandomizableTransform):
    def __init__(self, image="image", label="label", others="others", drop_rate=0.5, jitter_range=3):
        super().__init__()

        self.image = image
        self.label = label
        self.others = others
        self.drop_rate = drop_rate
        self.jitter_range = jitter_range

    def __call__(self, data):
        d = dict(data)

        image = d[self.image]
        mask = d[self.label]
        others = d[self.others]

        inc_sig = self.inclusion_map(mask[0])
        exc_sig = self.exclusion_map(others[0], drop_rate=self.drop_rate, jitter_range=self.jitter_range)

        image = np.concatenate((image, inc_sig[np.newaxis, ...], exc_sig[np.newaxis, ...]), axis=0)
        d[self.image] = image
        return d

    @staticmethod
    def inclusion_map(mask):
        point_mask = np.zeros_like(mask)
        indices = np.argwhere(mask > 0)
        if len(indices) > 0:
            idx = np.random.randint(0, len(indices))
            point_mask[indices[idx, 0], indices[idx, 1]] = 1

        return point_mask

    @staticmethod
    def exclusion_map(others, jitter_range=3, drop_rate=0.5):
        point_mask = np.zeros_like(others)
        if drop_rate == 1.0:
            return point_mask

        max_x = point_mask.shape[0] - 1
        max_y = point_mask.shape[1] - 1
        stats = skimage.measure.regionprops(others)
        for stat in stats:
            x, y = stat.centroid
            if np.random.choice([True, False], p=[drop_rate, 1 - drop_rate]):
                continue

            # random jitter
            x = int(math.floor(x)) + random.randint(a=-jitter_range, b=jitter_range)
            y = int(math.floor(y)) + random.randint(a=-jitter_range, b=jitter_range)
            x = min(max(0, x), max_x)
            y = min(max(0, y), max_y)
            point_mask[x, y] = 1

        return point_mask

class AddClickSignalsd(Transform):

    def __init__(self, image, foreground="foreground", bb_size=128):
        self.image = image
        self.foreground = foreground
        self.bb_size = bb_size

    def __call__(self, data):
        d = dict(data)

        location = d.get("location", (0, 0))
        tx, ty = location[0], location[1]
        pos = d.get(self.foreground)
        pos = (np.array(pos) - (tx, ty)).astype(int).tolist() if pos else []

        cx = [xy[0] for xy in pos]
        cy = [xy[1] for xy in pos]

        img = d[self.image].astype(np.uint8)
        img_width = img.shape[-1]
        img_height = img.shape[-2]

        click_map, bounding_boxes = self.get_clickmap_boundingbox(
            cx=cx,
            cy=cy,
            m=img_height,
            n=img_width,
            bb=self.bb_size
        )

        patches, nuc_points, other_points = self.get_patches_and_signals(
            img=img,
            click_map=click_map,
            bounding_boxes=bounding_boxes,
            cx=cx,
            cy=cy,
            m=img_height,
            n=img_width,
            bb=self.bb_size
        )
        patches = patches / 255

        d["bounding_boxes"] = bounding_boxes
        print(bounding_boxes)
        print('###')
        d["img_width"] = img_width
        d["img_height"] = img_height
        d["nuc_points"] = nuc_points
        print(nuc_points)

        d[self.image] = np.concatenate((patches, nuc_points, other_points), axis=1, dtype=np.float32)
        return d


    def get_clickmap_boundingbox(self, cx, cy, m, n, bb=128):
        click_map = np.zeros((m, n), dtype=np.uint8)

        # Removing points out of image dimension (these points may have been clicked unwanted)
        x_del_indices = {i for i in range(len(cx)) if cx[i] >= n or cx[i] < 0}
        y_del_indices = {i for i in range(len(cy)) if cy[i] >= m or cy[i] < 0}
        del_indices = list(x_del_indices.union(y_del_indices))
        cx = np.delete(cx, del_indices)
        cy = np.delete(cy, del_indices)

        click_map[cy, cx] = 1
        bounding_boxes = []
        for i in range(len(cx)):
            x_start = cx[i] - bb // 2
            y_start = cy[i] - bb // 2
            if x_start < 0:
                x_start = 0
            if y_start < 0:
                y_start = 0
            x_end = x_start + bb - 1
            y_end = y_start + bb - 1
            if x_end > n - 1:
                x_end = n - 1
                x_start = x_end - bb + 1
            if y_end > m - 1:
                y_end = m - 1
                y_start = y_end - bb + 1
            bounding_boxes.append([x_start, y_start, x_end, y_end])
        return click_map, bounding_boxes


    def get_patches_and_signals(self, img, click_map, bounding_boxes, cx, cy, m, n, bb=128):
        # total = number of clicks
        total = len(bounding_boxes)
        img = np.array([img])  # img.shape=(1,3,m,n)
        click_map = np.array([click_map])  # clickmap.shape=(1,m,n)
        click_map = click_map[:, np.newaxis, ...]  # clickmap.shape=(1,1,m,n)

        patches = np.ndarray((total, 3, bb, bb), dtype=np.uint8)
        nuc_points = np.ndarray((total, 1, bb, bb), dtype=np.uint8)
        other_points = np.ndarray((total, 1, bb, bb), dtype=np.uint8)

        # Removing points out of image dimension (these points may have been clicked unwanted)
        x_del_indices = {i for i in range(len(cx)) if cx[i] >= n or cx[i] < 0}
        y_del_indices = {i for i in range(len(cy)) if cy[i] >= m or cy[i] < 0}
        del_indices = list(x_del_indices.union(y_del_indices))
        cx = np.delete(cx, del_indices)
        cy = np.delete(cy, del_indices)

        for i in range(len(bounding_boxes)):
            bounding_box = bounding_boxes[i]
            x_start = bounding_box[0]
            y_start = bounding_box[1]
            x_end = bounding_box[2]
            y_end = bounding_box[3]

            patches[i] = img[0, :, y_start : y_end + 1, x_start : x_end + 1]

            this_click_map = np.zeros((1, 1, m, n), dtype=np.uint8)
            this_click_map[0, 0, cy[i], cx[i]] = 1

            others_click_map = np.uint8((click_map - this_click_map) > 0)

            nuc_points[i] = this_click_map[0, :, y_start : y_end + 1, x_start : x_end + 1]
            other_points[i] = others_click_map[0, :, y_start : y_end + 1, x_start : x_end + 1]

        # patches: (total, 3, m, n)
        # nuc_points: (total, 1, m, n)
        # other_points: (total, 1, m, n)
        return patches, nuc_points, other_points


class PostFilterLabeld(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        nuc_points="nuc_points",
        bounding_boxes="bounding_boxes",
        img_height="img_height",
        img_width="img_width",
        thresh=0.33,
        min_size=10,
        min_hole=30,
        do_reconstruction=False,
    ):
        super().__init__(keys)
        self.nuc_points = nuc_points
        self.bounding_boxes = bounding_boxes
        self.img_height = img_height
        self.img_width = img_width

        self.thresh = thresh
        self.min_size = min_size
        self.min_hole = min_hole
        self.do_reconstruction = do_reconstruction

    def __call__(self, data):
        d = dict(data)

        nuc_points = d[self.nuc_points]
        bounding_boxes = d[self.bounding_boxes]
        img_height = d[self.img_height]
        img_width = d[self.img_width]

        for key in self.keys:
            label = d[key].astype(np.uint8)
            masks = self.post_processing(
                label,
                thresh=self.thresh,
                min_size=self.min_size,
                min_hole=self.min_hole,
                do_reconstruction=self.do_reconstruction,
                nuc_points=nuc_points,
            )

            d[key] = self.gen_instance_map(masks, bounding_boxes, img_height, img_width).astype(np.uint8)
        return d

    def post_processing(self, preds, thresh=0.33, min_size=10, min_hole=30, do_reconstruction=False, nuc_points=None):
        masks = preds > thresh
        masks = remove_small_objects(masks, min_size=min_size)
        masks = remove_small_holes(masks, area_threshold=min_hole)
        if do_reconstruction:
            for i in range(len(masks)):
                this_mask = masks[i]
                this_marker = nuc_points[i, 0, :, :] > 0

                try:
                    this_mask = reconstruction(this_marker, this_mask, footprint=disk(1))
                    masks[i] = np.array([this_mask])
                except:
                    print("Nuclei reconstruction error #" + str(i))
        return masks  # masks(no.patches, 128, 128)

    def gen_instance_map(self, masks, bounding_boxes, m, n, flatten=True):
        instance_map = np.zeros((m, n), dtype=np.uint16)
        for i in range(len(masks)):
            this_bb = bounding_boxes[i]
            this_mask_pos = np.argwhere(masks[i] > 0)
            this_mask_pos[:, 0] = this_mask_pos[:, 0] + this_bb[1]
            this_mask_pos[:, 1] = this_mask_pos[:, 1] + this_bb[0]
            instance_map[this_mask_pos[:, 0], this_mask_pos[:, 1]] = 1 if flatten else i + 1
        return instance_map