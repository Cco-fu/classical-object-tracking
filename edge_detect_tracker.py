import numpy as np
import cv2 as cv
import time
import os

from utils.loader import OTB100Loader
from utils.selectors import AppearanceMotionScorer
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
        self.prev_frame = cv.cvtColor(
            first_frame,
            cv.COLOR_BGR2GRAY
        )

        self.prev_pred = box

        x, y, w, h = box

        self.template = self.prev_frame[y:y+h, x:x+w]

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
        gray = cv.cvtColor(
            frame,
            cv.COLOR_BGR2GRAY
        )

        mask = self._get_mask(gray)

        contours, _ = cv.findContours(
            mask,
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE
        )

        pred = self.box_selector.select(
            contours,
            self.prev_pred,
            gray,
            self.template
        )

        if pred is not None:
            self._update_template(pred, gray)
            self.prev_pred = pred

        self.prev_frame = gray


        self._on_step(cg=[contours, frame])

        return self.prev_pred, mask

    def _get_mask(self, cur):
        diff = cv.absdiff(cur, self.prev_frame)

        _, binary = cv.threshold(diff, self.thresh, 255, cv.THRESH_BINARY)

        opened = cv.morphologyEx(binary, cv.MORPH_OPEN, self.kernel)
        mask = cv.morphologyEx(opened, cv.MORPH_CLOSE, self.kernel)
        # mask = cv.dilate(opened, self.kernel, iterations=2)


        self._on_step(diff=diff, binary=binary, opened=opened, mask=mask)

        return mask
    
    def _update_template(self, pred, frame):
        x, y, w, h = pred
        if w < 0 or h < 0:
            return -1
        
        patch = frame[y:y+h, x:x+w]

        patch = cv.resize(
            patch,
            (self.template.shape[1], self.template.shape[0])
        ).astype(np.float32)

        alpha = self.template_alpha

        self.template = (
            (1 - alpha) * self.template.astype(np.float32)
            + alpha * patch
        ).astype(np.uint8)

    def _on_step(self, **kwargs):
        """
        钩子函数，用于子类继承实现中间保存，参数有：\n
        diff        相邻帧差分图
        binary      二值掩码图
        opened      开运算结果图
        mask        掩码图
        cg          轮廓信息与帧
        """
        pass


class DebugEdgeDetectTracker(EdgeDetectTracker):
    def __init__(self, box, selector, first_frame=None, save_dir="./output", idx=0, **kwargs):
        super().__init__(box, selector, first_frame, **kwargs)

        self.save_dir = save_dir
        self.idx = idx

    def _on_step(self, **kwargs):
        path = os.path.join(self.save_dir, f"frame{self.idx}")
        os.makedirs(path, exist_ok=True)

        for name, arg in kwargs.items():
            match name:
                case "diff":
                    cv.imwrite(f"{path}/diff.jpg", arg)
                case "binary":
                    cv.imwrite(f"{path}/binary.jpg", arg)
                case "opened":
                    cv.imwrite(f"{path}/opened.jpg", arg)
                case "mask":
                    cv.imwrite(f"{path}/mask.jpg", arg)
                case "cg":
                    self._save_boxes_img(arg, path)

    def _save_boxes_img(self, arg, path):
        contours = arg[0]
        frame = arg[1].copy()
        for c in contours:
            rect = cv.boundingRect(c)
            cv.rectangle(frame, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0, 255, 0), 2)
        cv.imwrite(f"{path}/contours.jpg", frame)

        self.idx += 1


DATA_NAME = 'Walking'
OUTPUT_DIR = f'./output/{DATA_NAME}'
DEBUG = False

parameters = {
    "kernel_sz": (3, 3),
    "thresh": 10,
    "template_alpha": 0.02
}

st = Statistic(0.5)
cs = AppearanceMotionScorer(100, 10000, 0.001)
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