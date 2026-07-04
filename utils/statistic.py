import numpy as np
import cv2 as cv
import os
import pandas as pd

class Statistic:

    def __init__(self, thresh):
        self.iou_list = []
        self.cle_list = []

        self.thresh = thresh

    def set_rate(self, thresh):
        self.thresh = thresh

    def reset(self):
        self.iou_list = []
        self.cle_list = []

    def get_minMax_iou_frame(slef):
        return np.argmin(slef.iou_list), np.argmax(slef.iou_list)

    def print_statistics(self):
        if not self.iou_list or not self.cle_list:
            print("No statistics to display.")
            return

        def print_stats(name, values):
            print(f"\n{'─' * 40}")
            print(f"{name} Statistics")
            print(f"{'─' * 40}")
            print(f"{'Mean:':<15}{np.mean(values):.2f}")
            print(f"{'Median:':<15}{np.median(values):.2f}")
            print(f"{'Std Dev:':<15}{np.std(values):.2f}")
            print(f"{'Max:':<15}{np.max(values):.2f}  (frame {np.argmax(values)+1})")
            print(f"{'Min:':<15}{np.min(values):.2f}  (frame {np.argmin(values)+1})")
                

        print_stats("IoU", self.iou_list)
        thresh = self.thresh
        rate = np.mean(np.array(self.iou_list) > thresh)
        print(f"{'Success Rate:':<15}{rate:.2f}  (IoU > {thresh})")

        print_stats("CLE", self.cle_list)

    def compute(self, pred_box, gt_box):
        cle = self._compute_cle(pred_box, gt_box)
        iou = self._compute_iou(pred_box, gt_box)

        self.cle_list.append(cle)
        self.iou_list.append(iou)


    def _compute_cle(self, p, g):
        px = p[0] + p[2] / 2
        py = p[1] + p[3] / 2

        gx = g[0] + g[2] / 2
        gy = g[1] + g[3] / 2

        return np.sqrt(((px - gx) ** 2 + (py - gy) ** 2))
    

    def _compute_iou(self, p, g):
        lx = max(p[0], g[0])
        ly = max(p[1], g[1])
        rx = min(p[0] + p[2], g[0] + g[2])
        ry = min(p[1] + p[3], g[1] + g[3])

        intersection = max(0, rx - lx) * max(0, ry - ly)
        union = p[2] * p[3] + g[2] * g[3] - intersection

        return intersection / union if union > 0 else 0