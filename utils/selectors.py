import cv2 as cv
import numpy as np

class AppearanceMotionScorer:
    def __init__(self, area_below, area_above, dist_alpha):
        self.area_below = area_below
        self.area_above = area_above

        self.dist_alpha = dist_alpha

    def select(self, contours, prev, frame, template):
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
        
        return best_box



class HSVAppearanceMotionScorer:
    def __init__(self, area_below, area_above, dist_alpha):
        self.area_below = area_below
        self.area_above = area_above

        self.dist_alpha = dist_alpha

    def select(self, contours, prev, hsv, template):
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
        
        return best_box

def hsv_appearance_score(hsv, box, template):
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return -1

    patch = hsv[y:y+h, x:x+w]

    hist = cv.calcHist([patch], [0,1], None, [50,60], [0,180,0,256])
    cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX)

    return cv.compareHist(hist, template, cv.HISTCMP_CORREL)


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


def center_score(box1, box2, alpha):
    cx1 = box1[0] + box1[2] / 2
    cy1 = box1[1] + box1[3] / 2

    cx2 = box2[0] + box2[2] / 2
    cy2 = box2[1] + box2[3] / 2

    return alpha * np.sqrt(
        (cx1 - cx2) ** 2 +
        (cy1 - cy2) ** 2
    )
    
