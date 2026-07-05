import numpy as np
import cv2 as cv
import time
import os

from utils.loader import OTB100Loader
from utils.selectors import HSVAppearanceMotionScorer, DebugHSVAppearanceMotionScorer
from utils.statistic import Statistic

class ColorThreshTracker:
    """
    可修改参数：\n
    kernel              核 \n
    \t 默认是全 1 核 \n
    
    kernel_sz           核大小 \n
    \t 默认大小是 (3,3) \n

    thresh              二值化阈值 \n
    \t 默认大小为 10 \n

    template_alpha      模板更新加权系数 \n
    \t 默认大小为 0.05 \n
    """

    def __init__(self, box, selector, first_frame=None, **kwargs):
        self.prev_pred = box

        self.template = self._get_hist(first_frame, box)

        self.box_selector = selector

        """
        可修改参数：
        kernel              核
        kernel_sz           核大小
        thresh              二值化阈值
        template_alpha      模板更新加权系数
        """
        self.thresh = kwargs.get('thresh', 10)
        self.kernel = kwargs.get('kernel', np.ones(
            kwargs.get("kernel_sz", (3, 3)),
            np.uint8
        ))
        self.template_alpha = kwargs.get("template_alpha", 0.05)

    def track(self, frame):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        mask = self._get_mask(hsv)

        pred = self.box_selector.select(
            mask,
            self.prev_pred,
            hsv,
            self.template
        )

        if pred is not None:
            self._update_template(pred, hsv)
            self.prev_pred = pred

        return self.prev_pred, mask

    def _get_hist(self, frame, gt):
        x, y, w, h = gt
        
        roi = frame[y:y+h, x:x+w]
        hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

        hist = cv.calcHist([hsv], [0,1], None, [50,60], [0,180,0,256])
        cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX)

        return hist

    def _get_mask(self, cur):
        proj = cv.calcBackProject([cur], [0,1], self.template, [0,180,0,256], 1)

        _, binary = cv.threshold(proj, self.thresh, 255, cv.THRESH_BINARY)

        opened = cv.morphologyEx(binary, cv.MORPH_OPEN, self.kernel)
        mask   = cv.morphologyEx(opened, cv.MORPH_CLOSE, self.kernel)
        # mask = cv.dilate(opened, self.kernel, iterations=1)


        self._on_step(proj=proj, binary=binary, opened=opened, mask=mask)

        return mask
    
    def _update_template(self, pred, frame):
        if self.template_alpha <= 0:
            return
        
        new_hist = self._get_hist(frame, pred)
        self.template = cv.addWeighted(self.template, 1 - self.template_alpha, new_hist, self.template_alpha, 0)

    def _on_step(self, **kwargs):
        pass


class DebugColorThreshTracker(ColorThreshTracker):
    def __init__(self, box, selector, first_frame=None, idx=0, save_dir="./output", **kwargs):
        super().__init__(box, selector, first_frame, **kwargs)

        self.save_dir = save_dir
        self.idx = idx

    def _on_step(self, **kwargs):
        path = os.path.join(self.save_dir, f"frame{self.idx}")
        os.makedirs(path, exist_ok=True)

        for name, arg in kwargs.items():
            match name:
                case "proj":
                    cv.imwrite(f"{path}/proj.jpg", arg)
                case "binary":
                    cv.imwrite(f"{path}/binary.jpg", arg)
                case "opened":
                    cv.imwrite(f"{path}/opened.jpg", arg)
                case "mask":
                    cv.imwrite(f"{path}/mask.jpg", arg)

        self.idx += 1


DATA_NAME = 'Lemming'
OUTPUT_DIR = f'./output/color_thresh/{DATA_NAME}'
DEBUG = False

parameters = {
    "kernel_sz": (3, 3),
    "thresh": 10,
    "template_alpha": 0
}

st = Statistic(0.5)
loader = OTB100Loader("../OTB100", DATA_NAME, 10)

with loader as l:
    f, b = next(l)
    if DEBUG:
        cs = DebugHSVAppearanceMotionScorer(100, 10000, 0.001, 1, OUTPUT_DIR)
        tracker = DebugColorThreshTracker(b, cs, f, 1, OUTPUT_DIR, **parameters)
    else:
        cs = HSVAppearanceMotionScorer(100, 10000, 0.001)
        tracker = ColorThreshTracker(b, cs, f, **parameters)

    start = time.time()

    for i, (frame, bbox) in enumerate(l, 1):
        pred, mask = tracker.track(frame)
        
        st.compute(pred, bbox)

        img = frame.copy()

        px, py, pw, ph = pred
        bx, by, bw, bh = bbox

        cv.rectangle(img, (bx, by), (bx+bw, by+bh), (0,0,255), 2)
        cv.rectangle(img, (px, py), (px+pw, py+ph), (0,255,0), 2)

        if DEBUG:
            cv.imwrite(f"{OUTPUT_DIR}/frame{i}/result.jpg", img)

        cv.imshow("Frame Difference", img)
        cv.imshow("Mask", mask)

        key = cv.waitKey(5)
        if key == 27:
            break

delta = time.time() - start
print(f"Mean FPS:{len(loader)/delta:.2f}")

cv.destroyAllWindows()

st.print_statistics()