import numpy as np
import cv2 as cv
import time
import os

from utils.loader import OTB100Loader
from utils.selectors import EdgeAppearanceMotionScorer
from utils.statistic import Statistic

class EdgeDetectTracker:
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


        self._init_template(first_frame, box)

    def track(self, frame):
        gray = cv.cvtColor(
            frame,
            cv.COLOR_BGR2GRAY
        )

        mask = self._get_mask(gray)

        pred, score = self.box_selector.select(
            mask,
            self.prev_pred,
            self.template
        )

        if pred is not None:
            self._update_template(pred, mask)
            self.prev_pred = pred

        
        self._on_step(gray=gray)

        return self.prev_pred, mask
    
    def _init_template(self, frame, gt):
        x, y, w, h = gt
        mask = self._get_mask(frame)
        self.template = mask[y:y+h, x:x+w]

    def _get_mask(self, cur):

        edge = cv.Canny(cur, self.thresh, 100)

        mask = cv.morphologyEx(edge, cv.MORPH_CLOSE, self.kernel)

        self._on_step(edge=edge, mask=mask)

        return mask
    
    def _update_template(self, pred, frame):
        if self.template_alpha <= 0:
            return
        
        x, y, w, h = pred
        if w <= 0 or h <= 0:
            return -1
        
        patch = frame[y:y+h, x:x+w]
        th, tw = self.template.shape

        patch = cv.resize(
            patch,
            (tw, th)
        ).astype(np.float32)

        alpha = self.template_alpha

        self.template = (
            (1 - alpha) * self.template.astype(np.float32)
            + alpha * patch
        ).astype(np.uint8)

    def _on_step(self, **kwargs):
        pass


class DebugEdgeDetectTracker(EdgeDetectTracker):
    def __init__(self, box, selector, first_frame=None, save_dir="./output", idx=0, **kwargs):
        self.save_dir = save_dir
        self.idx = idx

        super().__init__(box, selector, first_frame, **kwargs)

    def _on_step(self, **kwargs):
        path = os.path.join(self.save_dir, f"frame{self.idx}")
        os.makedirs(path, exist_ok=True)

        for name, arg in kwargs.items():
            match name:
                case "edge":
                    cv.imwrite(f"{path}/edge.jpg", arg)
                case "mask":
                    cv.imwrite(f"{path}/mask.jpg", arg)
                case "gray":
                    cv.imwrite(f"{path}/gray.jpg", arg)
                    self.idx += 1


DATA_NAME = 'Walking'
OUTPUT_DIR = f'./output/edge/{DATA_NAME}'
DEBUG = False

parameters = {
    "kernel": cv.getStructuringElement(cv.MORPH_CROSS, (3,3)),
    # "kernel_sz": (3, 3),
    "thresh": 50,
    "template_alpha": 0.15
}

st = Statistic(0.5)
cs = EdgeAppearanceMotionScorer(0.3, 5, 2, 10, 0.0001)
loader = OTB100Loader("../OTB100", DATA_NAME, 10)

with loader as l:
    f, b = next(l)
    if DEBUG:
        tracker = DebugEdgeDetectTracker(b, cs, f, OUTPUT_DIR, 1, **parameters)
    else:
        tracker = EdgeDetectTracker(b, cs, f, **parameters)

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