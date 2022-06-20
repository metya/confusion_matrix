import torch
import numpy as np

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from itertools import product
from matplotlib.collections import QuadMesh
from scipy.spatial.distance import cdist
from matplotlib.figure import Figure
from matplotlib.text import Text
from numpy.typing import NDArray, ArrayLike

Color = tuple[float, float, float, float]

class ConfusionMatrix:
    def __init__(self, thrs_config: dict, class_names: dict, iou_thr=0.5,):
        """
        Class to create and dislpay confusion matrix for MaskRCNN
        or any type of instance segmentation and detection architectures

        Parameters
        ----------
            - iou_thr: IOU threshold
            - thrs_config: dict of thresholds for every class
            - class_names: dict of class names accordingly to class numbers

        Attributes
        ----------
            - box_matrix: confusion matrix that contains results from boxes
            - mask_matrix: confusion matrix that contains results from masks
            - display_labels: labels according to classes + miss
            - figure_: contains last plot of confusion matrix

        Examples
        --------
        >>> from confusion_matrix import MaskRcnnConfusionMatrix

        >>> confusion_matrix = MaskRcnnConfusionMatrix(class_names={0: 'class1', 1: 'class2'},
        ...                                            thrs_config={0: 0.5, 1: 0.5})
        >>> for images, targets in test_dataloader:
        >>>     outputs = model(images)
        >>>     confusion_matrix.update(outputs, targets)

        >>> confusion_matrix.plot(show=True)

        """
        self.iou_thr = iou_thr
        self.num_classes = len(thrs_config)
        self.classes = class_names.values()
        self.box_matrix = np.zeros((self.num_classes + 1, self.num_classes + 1))
        self.mask_matrix = np.zeros((self.num_classes + 1, self.num_classes + 1))
        self.thrs_config = thrs_config
        self.display_labels = list(self.classes) + ["Miss"]

    def update(self, predictions: dict, targets: dict, after_nms=False):
        """
        Update confusion matrix for masks and boxes.
        It is not very performative and effective implementations.
        Needs to be rewritten in vectorize style, cause currently it's loops
        Arguments:
        ---------
            predictions: dict of prediction from mask_rcnn
            targets: dict of targets from dataloader
            after_nms: it is after nms already or it needs to be thresholded here
        Returns:
        -------
            None, updates confusion matrix accordingly
        """
        if isinstance(targets["labels"], torch.Tensor):
            targets = {k: v.to("cpu").numpy() for k, v in targets.items() if type(v) is not str}
        l_classes = targets["labels"]
        l_bboxs = targets["boxes"]
        l_masks = targets["masks"]
        d_confs = predictions["scores"]
        d_bboxs = predictions["boxes"]
        d_masks = predictions["masks"]
        d_classes = predictions["labels"]
        if not after_nms:
            box_thrs = [self.thrs_config[label_id]["box_thr"] for label_id in d_classes]
            mask_thrs = [self.thrs_config[label_id]["mask_thr"] for label_id in d_classes]
            ids = np.where(d_confs > box_thrs)[0]
            d_classes = d_classes[ids]
            d_bboxs = d_bboxs[ids]
            d_masks = d_masks[ids]
        box_labels_detected = np.zeros(len(l_classes))
        mask_labels_detected = np.zeros(len(l_classes))
        box_detections_matched = np.zeros(len(d_classes))
        mask_detections_matched = np.zeros(len(d_classes))
        for l_idx, (l_class, l_bbox, l_mask) in enumerate(zip(l_classes, l_bboxs, l_masks)):
            for d_idx, (d_class, d_bbox, d_mask) in enumerate(zip(d_classes, d_bboxs, d_masks)):
                box_iou = self.box_pairwise_iou(l_bbox, d_bbox)
                mask_iou = self.mask_iou((l_mask, l_class), (d_mask, d_class))
                if box_iou >= self.iou_thr:
                    self.box_matrix[l_class, d_class] += 1
                    box_labels_detected[l_idx] = 1
                    box_detections_matched[d_idx] = 1
                if mask_iou >= self.iou_thr:
                    self.mask_matrix[l_class, d_class] += 1
                    mask_labels_detected[l_idx] = 1
                    mask_detections_matched[d_idx] = 1
        for i in np.where(box_labels_detected == 0)[0]:
            self.box_matrix[l_classes[i], -1] += 1
        for i in np.where(box_detections_matched == 0)[0]:
            self.box_matrix[-1, d_classes[i]] += 1
        for i in np.where(mask_labels_detected == 0)[0]:
            self.mask_matrix[l_classes[i], -1] += 1
        for i in np.where(mask_detections_matched == 0)[0]:
            self.mask_matrix[-1, d_classes[i]] += 1

    def process_batch(self, predictions: dict, targets: dict, after_nms=True):
        """
        Process batch of predictons and targets from model and dataloader
        to update confusion matrix.
        This is supposed to be effective vectorized implementations. Half of that have done, but not masks.
        It means that this implementation only for boxes confusion matrix

        Arguments:
            predictions: dict of prediction from mask_rcnn
            targets: dict of targets from dataloader
            after_nms: it is after nms already or it needs to be thresholded here
        Returns:
            None, updates confusion matrix accordingly
        """
        if isinstance(targets["labels"], torch.Tensor):
            targets = {k: v.to("cpu").numpy() for k, v in targets.items() if type(v) is not str}

        gt_classes = targets["labels"]
        box_thrs = [self.thrs_config[label_id]["box_thr"] for label_id in predictions["labels"]]
        try:
            prediction_indexes = np.where(predictions["scores"] > box_thrs)[0]
            prediction_classes = predictions["labels"][prediction_indexes]
        except IndexError or TypeError as e:
            # detections are empty, end of process
            print("Какая то хуйня произошла!")
            raise e
        if len(prediction_classes) == 0 and len(gt_classes) > 0:
            for gt_class in gt_classes:
                self.box_matrix[self.num_classes, gt_class] += 1
            return
        elif len(prediction_classes) == 0 and len(gt_classes) == 0:
            return

        all_ious = self.box_pairwise_iou(targets["boxes"], predictions["boxes"])
        want_idx = np.where(all_ious > self.iou_thr)

        all_matches = [
            [want_idx[0][i], want_idx[1][i], all_ious[want_idx[0][i], want_idx[1][i]]]
            for i in range(want_idx[0].shape[0])
        ]

        all_matches = np.array(all_matches)
        if all_matches.shape[0] > 0:  # if there is match
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]
            all_matches = all_matches[np.unique(all_matches[:, 1], return_index=True)[1]]
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]
            all_matches = all_matches[np.unique(all_matches[:, 0], return_index=True)[1]]

        for i, gt_class in enumerate(gt_classes):
            if all_matches.shape[0] > 0 and all_matches[all_matches[:, 0] == i].shape[0] == 1:
                detection_class = prediction_classes[int(all_matches[all_matches[:, 0] == i, 1][0])]
                self.box_matrix[detection_class, gt_class] += 1
            else:
                self.box_matrix[self.num_classes, gt_class] += 1

        for i, detection_class in enumerate(prediction_classes):
            if not all_matches.shape[0] or (all_matches.shape[0] and all_matches[all_matches[:, 1] == i].shape[0] == 0):
                detection_class = prediction_classes[i]
                self.box_matrix[detection_class, self.num_classes] += 1

    def box_pairwise_iou(self, boxes1: NDArray[np.float32], boxes2: NDArray[np.float32]) -> NDArray[np.float32]:
        # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            boxes1 (Array[N, 4])
            boxes2 (Array[M, 4])
        Returns:
            iou (Array[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        This implementation is taken from the above link and changed so that it only uses numpy..
        """
        if len(boxes1.shape) < 2:
            boxes1 = boxes1.reshape(1, -1)
        if len(boxes2.shape) < 2:
            boxes2 = boxes2.reshape(1, -1)

        def box_area(box):
            # box = 4xn
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(boxes1.T)
        area2 = box_area(boxes2.T)

        lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

        inter = np.prod(np.clip(rb - lt, a_min=0, a_max=None), 2)  # type: ignore
        return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

    def mask_iou(self, mask_and_label1: tuple[NDArray, NDArray], mask_and_label2: tuple[NDArray, NDArray]):
        """
        Return intersection-over-union (Jaccard index) of masks.
        Masks should be pixel arrays
        Arguments:
            two tuples of mask and label
        """
        mask1, label1 = mask_and_label1
        mask2, label2 = mask_and_label2
        thrs1 = self.thrs_config[label1]["mask_thr"]
        thrs2 = self.thrs_config[label2]["mask_thr"]
        mask1_area = np.count_nonzero(mask1 >= thrs1)
        mask2_area = np.count_nonzero(mask2 >= thrs2)
        intersection = np.count_nonzero(np.logical_and(mask1, mask2))
        iou = intersection / (mask1_area + mask2_area - intersection)
        return iou

    def mask_pairwise_iou(self, masks1: np.ndarray, masks2: np.ndarray, labels1: np.ndarray, labels2: np.ndarray):
        # TODO: implement it finally!
        """Need to have been imnplemented eventually and tested"""

        f1 = np.array(zip(masks1, labels1))
        f2 = np.array(zip(masks2, labels2))
        return cdist(f1, f2, metric=self.mask_iou) # type: ignore

    def return_matrix(self):
        """Returns tuple of box and mask confusion matrix."""
        return self.box_matrix, self.mask_matrix

    def get_matrix_figure(self, type="box", pretty=True):
        """
        Returns figure of confusion matrix of either box or mask type

        Parameters
        ----------
        type: str, either box or mask, default `box`
        pretty: bool, default `True`
            plot pretty, featurize confusion matrix or just regular
        """
        if type == "box":
            if pretty:
                return pp_matrix(
                    self.box_matrix,
                    figsize=(14, 14),
                    rotation=False,
                    display_labels=self.display_labels,
                )
            else:
                return self.plot(figsize=(10, 10), type_matrix="boxes")
        else:
            if pretty:
                return pp_matrix(
                    self.mask_matrix,
                    figsize=(14, 14),
                    rotation=False,
                    display_labels=self.display_labels,
                )
            else:
                return self.plot(figsize=(10, 10), type_matrix="masks")

    def print_matrix(self):
        for i in range(self.num_classes + 1):
            print(" ".join(map(str, self.box_matrix[i])))

    def pretty_plot(
        self,
        type="box",
        figsize=(14, 14),
        rotation=False,
        cmap="viridis",
        ) -> Figure:
        """Plot feature rich confusion matrix.
        """
        if type=="box":
            return pp_matrix(
                        self.box_matrix,
                        figsize=figsize,
                        rotation=rotation,
                        display_labels=self.display_labels,
                        cmap=cmap,
                        show=True,
                    )
        else:
            return pp_matrix(
                        self.mask_matrix,
                        figsize=figsize,
                        rotation=rotation,
                        display_labels=self.display_labels,
                        cmap=cmap,
                        show=True,
                    )

    def plot(
        self,
        include_values=True,
        cmap="viridis",
        xticks_rotation="vertical",
        values_format=None,
        ax=None,
        colorbar=False,
        type_matrix="boxes",
        figsize=(9, 9),
    ) -> Figure:
        """Plot visualization of confusion matrix.

        Parameters
        ----------
        include_values : bool, default=True
            Includes values in confusion matrix.

        cmap : str or matplotlib Colormap, default='viridis'
            Colormap recognized by matplotlib.

        xticks_rotation : {'vertical', 'horizontal'} or float, \
                         default='horizontal'
            Rotation of xtick labels.

        values_format : str, default=None
            Format specification for values in confusion matrix. If `None`,
            the format specification is 'd' or '.2g' whichever is shorter.

        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        colorbar : bool, default=True
            Whether or not to add a colorbar to the plot.

        figsize : tuple, default (9,9)
            Size of figure.

        type_matrix : str, ether box or mask
            Type of matrix that need to plot.

        Returns
        -------
        display : :firuge:`plt.figure`
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        cm = self.box_matrix if type_matrix == "boxes" else self.mask_matrix
        n_classes = cm.shape[0]
        self.im_ = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        self.text_ = None
        cmap_min, cmap_max = self.im_.cmap(0), self.im_.cmap(1.0)

        if include_values:
            self.text_ = np.empty_like(cm, dtype=object)

            # print text with appropriate color depending on background
            thresh = (cm.max() + cm.min()) / 2.0

            for i, j in product(range(n_classes), range(n_classes)):
                color = cmap_max if cm[i, j] < thresh else cmap_min

                if values_format is None:
                    text_cm = format(cm[i, j], ".2g")
                    if cm.dtype.kind != "f":
                        text_d = format(cm[i, j], "d")
                        if len(text_d) < len(text_cm):
                            text_cm = text_d
                else:
                    text_cm = format(cm[i, j], values_format)

                self.text_[i, j] = ax.text(j, i, text_cm, ha="center", va="center", color=color)

        if self.display_labels is None:
            display_labels = np.arange(n_classes)
        else:
            display_labels = self.display_labels
        if colorbar:
            fig.colorbar(self.im_, ax=ax)
        ax.set(
            xticks=np.arange(n_classes),
            yticks=np.arange(n_classes),
            xticklabels=display_labels,
            yticklabels=display_labels,
            ylabel="True label",
            xlabel="Predicted label",
        )

        ax.set_ylim((n_classes - 0.5, -0.5))
        plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)
        plt.tight_layout()
        plt.grid(False)

        self.figure_ = fig
        self.ax_ = ax
        return fig


# Helper function to draw pretty figure
# -------------------------------------


# This is main function to draw pretty confusion matrix

def pp_matrix(
    df_cm: NDArray[np.float64] | pd.DataFrame,
    annot=True,
    cmap="viridis",
    fmt=".2f",
    fz=10,
    lw=1,
    cbar=False,
    figsize=[9, 9],
    show_null_values=False,
    pred_val_axis="x",
    show=False,
    rotation=True,
    display_labels=None,
):
    """
    print conf matrix with default layout (like matlab)
    params:
      df_cm          dataframe (pandas) without totals
      annot          print text in each cell
      cmap           Oranges,Oranges_r,YlGnBu,Blues,RdBu, ... see:
      fz             fontsize
      lw             linewidth
      pred_val_axis  where to show the prediction values (x or y axis)
                      'col' or 'x': show predicted values in columns (x axis) instead lines
                      'lin' or 'y': show predicted values in lines   (y axis)
      show           show the plot or not
      rotation       rotate or not labels on figure
      display_labels None, list of labels that display on figure
    """
    if not isinstance(df_cm, pd.DataFrame):
        df_cm = pd.DataFrame(df_cm, index=display_labels, columns=display_labels)

    if pred_val_axis in ("col", "x"):
        xlbl = "Predicted"
        ylbl = "Actual"
    else:
        xlbl = "Actual"
        ylbl = "Predicted"
        df_cm = df_cm.T

    # create "Total" column
    insert_totals(df_cm)

    # this is for print allways in the same window
    fig, ax1 = get_new_fig("Conf matrix default", figsize)

    ax = sn.heatmap(
        df_cm,
        annot=annot,
        annot_kws={"size": fz},
        linewidths=lw,
        ax=ax1,
        cbar=cbar,
        cmap=cmap,
        linecolor="w",
        fmt=fmt,
    )

    # set ticklabels rotation
    if rotation:
        rotation_x = 45
        rotation_y = 25
    else:
        rotation_x = 0
        rotation_y = 90

    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation_y, fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=rotation_x, fontsize=10)

    # Turn off all the ticks
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # face colors list
    quadmesh = ax.findobj(QuadMesh)[0]
    facecolors = quadmesh.get_facecolors()

    # iter in text elements
    array_df = np.array(df_cm.to_records(index=False).tolist())
    text_add = []
    text_del = []
    posi = -1  # from left to right, bottom to top.
    for t in ax.collections[0].axes.texts:  # ax.texts:
        pos = np.array(t.get_position()) - [0.5, 0.5]
        lin = int(pos[1])
        col = int(pos[0])
        posi += 1

        # set text
        txt_res = configcell_text_and_colors(array_df, lin, col, t, facecolors, posi, fz, fmt, show_null_values)

        text_add.extend(txt_res[0])
        text_del.extend(txt_res[1])

    # remove the old ones
    for item in text_del:
        item.remove()
    # append the new ones
    for item in text_add:
        ax.text(item["x"], item["y"], item["text"], **item["kw"])

    # titles and legends
    ax.set_title("Confusion matrix")
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    # set layout slim
    plt.tight_layout()
    if show:
        plt.show()

    return plt.gcf()


# This is helper functions for pp_print function

def get_new_fig(fn, figsize=[9, 9]):
    """Init graphics"""
    fig1 = plt.figure(fn, figsize)
    ax1 = fig1.gca()  # Get Current Axis
    ax1.cla()  # clear existing plot
    return fig1, ax1


def configcell_text_and_colors(
    array_df: np.ndarray,
    lin: int,
    col: int,
    oText: Text,
    facecolors: list[Color],
    posi: int,
    fz: int,
    fmt: str,
    show_null_values=False,
):
    """
    config cell text and colors
    and return text elements to add and to dell
    """
    text_add = []
    text_del = []
    cell_val = array_df[lin][col]
    tot_all = array_df[-1][-1]
    per = (float(cell_val) / tot_all) * 100
    curr_column = array_df[:, col]
    ccl = len(curr_column)

    # last line  and/or last column
    if (col == (ccl - 1)) or (lin == (ccl - 1)):
        # tots and percents
        if cell_val != 0:
            if (col == ccl - 1) and (lin == ccl - 1):
                tot_rig = 0
                for i in range(array_df.shape[0] - 1):
                    tot_rig += array_df[i][i]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif col == ccl - 1:
                tot_rig = array_df[lin][lin]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif lin == ccl - 1:
                tot_rig = array_df[col][col]
                per_ok = (float(tot_rig) / cell_val) * 100
            per_err = 100 - per_ok  # type: ignore
        else:
            per_ok = per_err = 0

        per_ok_s = ["%.2f%%" % (per_ok), "100%"][per_ok == 100]  # type: ignore

        # text to DEL
        text_del.append(oText)

        # text to ADD
        font_prop = fm.FontProperties(weight="bold", size=fz)
        text_kwargs = dict(
            color="w",
            ha="center",
            va="center",
            gid="sum",
            fontproperties=font_prop,
        )
        lis_txt = ["%d" % (cell_val), per_ok_s, "%.2f%%" % (per_err)]
        lis_kwa = [text_kwargs]
        dic = text_kwargs.copy()
        dic["color"] = "g"
        lis_kwa.append(dic)
        dic = text_kwargs.copy()
        dic["color"] = "r"
        lis_kwa.append(dic)
        lis_pos = [
            (oText._x, oText._y - 0.3),
            (oText._x, oText._y),
            (oText._x, oText._y + 0.3),
        ]
        for i in range(len(lis_txt)):
            new_text = dict(
                x=lis_pos[i][0],
                y=lis_pos[i][1],
                text=lis_txt[i],
                kw=lis_kwa[i],
            )
            text_add.append(new_text)

        # set background color for sum cells (last line and last column)
        carr = (0.27, 0.30, 0.27, 1.0)
        if (col == ccl - 1) and (lin == ccl - 1):
            carr = (0.17, 0.20, 0.17, 1.0)
        facecolors[posi] = carr

    else:
        if per > 0:
            txt = "%s\n%.1f%%" % (cell_val, per)
        else:
            if show_null_values == False:
                txt = ""
            elif show_null_values == True:
                txt = "0"
            else:
                txt = "0\n0.0%"
        oText.set_text(txt)

        # main diagonal
        if col == lin:
            # set color of the textin the diagonal to white
            oText.set_color("w")
            # set background color in the diagonal to blue
            facecolors[posi] = (0.35, 0.8, 0.55, 1.0)
        else:
            oText.set_color("r")

    return text_add, text_del


def insert_totals(df_cm):
    """insert total column and line (the last ones)"""
    sum_col = []
    for c in df_cm.columns:
        sum_col.append(df_cm[c].sum())
    sum_lin = []
    for item_line in df_cm.iterrows():
        sum_lin.append(item_line[1].sum())
    df_cm["sum_lin"] = sum_lin
    sum_col.append(np.sum(sum_lin))
    df_cm.loc["sum_col"] = sum_col






