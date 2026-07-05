import cv2 as cv
import numpy as np
import os












class AppearanceMotionScorer:
    def __init__(self, area_below, area_above, dist_alpha):
        self.area_below = area_below
        self.area_above = area_above

        self.dist_alpha = dist_alpha

    def select(self, mask, prev, frame, template):

        self._on_start(frame)
        
        contours, _ = cv.findContours(
            mask,
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE
        )

        best_box = None
        best_score = -1

        for c in contours:
            area = cv.contourArea(c)

            if self.area_below <= area and area <= self.area_above:
                box = cv.boundingRect(c)

                score = appearance_score(frame, box, template) - \
                        center_score(prev, box, self.dist_alpha)

                if score > best_score:
                    best_score = score
                    best_box   = box

                self._on_step(box)

        self._on_finish()
        
        return best_box
    
    def _on_start(self, f):
        pass

    def _on_step(self, box):
        pass

    def _on_finish(self):
        pass


class DebugAppearanceMotionScorer(AppearanceMotionScorer):
    def __init__(self, area_below, area_above, dist_alpha, idx, save_dir="./output"):
        super().__init__(area_below, area_above, dist_alpha)

        self.save_dir = save_dir
        self.idx = idx

    def _on_start(self, f):
        self.f = f.copy()

    def _on_step(self, box):
        x, y, w, h = box
        cv.rectangle(self.f, (x, y), (x+w, y+h), (0, 255, 0), 2)

    def _on_finish(self):
        path = os.path.join(self.save_dir, f"frame{self.idx}")
        os.makedirs(path, exist_ok=True)
        cv.imwrite(f"{path}/contour.jpg", self.f)
        self.idx += 1


def appearance_score(frame, box, template):
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return -1
    
    patch = frame[y:y+h, x:x+w]

    patch = cv.resize(
        patch,
        (template.shape[1], template.shape[0])
    )

    score = cv.matchTemplate(
        patch,
        template,
        cv.TM_CCORR_NORMED
    )

    return score[0, 0]














class HSVAppearanceMotionScorer:
    def __init__(self, area_below, area_above, dist_alpha):
        self.area_below = area_below
        self.area_above = area_above

        self.dist_alpha = dist_alpha

    def select(self, mask, prev, hsv, template):

        self._on_start(hsv)

        contours, _ = cv.findContours(
            mask,
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE
        )
        
        best_box = None
        best_score = -1

        for c in contours:
            area = cv.contourArea(c)

            if self.area_below <= area and area <= self.area_above:
                box = cv.boundingRect(c)

                score = hsv_appearance_score(hsv, box, template) - \
                        center_score(prev, box, self.dist_alpha)

                if score > best_score:
                    best_score = score
                    best_box   = box

                self._on_step(box)

        self._on_finish()

        return best_box

    def _on_start(self, f):
        pass

    def _on_step(self, box):
        pass

    def _on_finish(self):
        pass

class DebugHSVAppearanceMotionScorer(HSVAppearanceMotionScorer):
    def __init__(self, area_below, area_above, dist_alpha, idx, save_dir="./output"):
        super().__init__(area_below, area_above, dist_alpha)

        self.save_dir = save_dir
        self.idx = idx

    def _on_start(self, f):
        self.f =cv.cvtColor(
            cv.cvtColor(f, cv.COLOR_HSV2BGR),
            cv.COLOR_BGR2GRAY
        )

    def _on_step(self, box):
        x, y, w, h = box
        cv.rectangle(self.f, (x, y), (x+w, y+h), (0, 255, 0), 2)

    def _on_finish(self):
        path = os.path.join(self.save_dir, f"frame{self.idx}")
        os.makedirs(path, exist_ok=True)
        cv.imwrite(f"{path}/contour.jpg", self.f)
        self.idx += 1


def hsv_appearance_score(hsv, box, template):
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return -1

    patch = hsv[y:y+h, x:x+w]

    hist = cv.calcHist([patch], [0,1], None, [50,60], [0,180,0,256])
    cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX)

    return cv.compareHist(hist, template, cv.HISTCMP_CORREL)















class EdgeAppearanceMotionScorer:
    def __init__(self, score_thresh, lost_thresh, normal_r, lost_r, dist_alpha):
        self.dist_alpha = dist_alpha
        self.score_thresh = score_thresh
        self.lost_thresh = lost_thresh

        self.NORMAL = normal_r
        self.LOST   = lost_r

        self.lost_cnt = 0
        

    def select(self, mask, prev, template):

        region = self._get_search_region(prev, template, mask)
        if region is None:
            return None, 0.0

        x, y, w, h = region
        search = mask[y:y+h, x:x+w]

        th, tw = template.shape
        res = cv.matchTemplate(
            search,
            template,
            cv.TM_CCOEFF_NORMED
        )

        res_h, res_w = res.shape

        jj, ii = np.meshgrid(np.arange(res_w), np.arange(res_h))
        cx_c = x + jj + tw // 2
        cy_c = y + ii + th // 2

        cx_p = prev[0] + prev[2] // 2
        cy_p = prev[1] + prev[3] // 2

        dist = np.sqrt((cx_c - cx_p) ** 2 + (cy_c - cy_p) ** 2)

        combined = res - self.dist_alpha * dist
        # combined = res

        idx = np.unravel_index(np.argmax(combined), combined.shape)
        best_score = combined[idx]

        px = idx[1] + x
        py = idx[0] + y

        if best_score > self.score_thresh:
            self.lost_cnt = 0
            return [px, py, tw, th], best_score
        
        self.lost_cnt += 1
        return None, 0.0
    

    def _get_search_region(self, prev, template, mask):
        th, tw = template.shape

        if self.lost_cnt > self.lost_thresh:
            r = max(th, tw) * self.LOST
        else:
            r = max(th, tw) * self.NORMAL

        cx = prev[0] + prev[2] // 2
        cy = prev[1] + prev[3] // 2

        lx = max(0, cx - r)
        ly = max(0, cy - r)

        rx = min(mask.shape[1], cx + r)
        ry = min(mask.shape[0], cy + r)

        if rx - lx < tw or ry - ly < th:
            return None

        return (lx, ly, rx - lx, ry - ly)


















def center_score(box1, box2, alpha):
    cx1 = box1[0] + box1[2] // 2
    cy1 = box1[1] + box1[3] // 2

    cx2 = box2[0] + box2[2] // 2
    cy2 = box2[1] + box2[3] // 2

    return alpha * np.sqrt(
        (cx1 - cx2) ** 2 +
        (cy1 - cy2) ** 2
    )
    
