#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Template doc"""
import cv2
import numpy as np
from albumentations import ImageOnlyTransform


class ContrastNormalization(ImageOnlyTransform):
    """

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8
    """

    def __init__(self,
                 always_apply=False,
                 p=0.5):
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        return ((img - img.min()) / (img.max() - img.min()) * 255).astype(
            np.uint8)


class SlantCorrection(ImageOnlyTransform):
    """

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8
    """

    def __init__(self,
                 always_apply=False,
                 p=0.5):
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        moments = cv2.moments(img)
        if abs(moments['mu02']) < 1e-2:
            return img.copy()

        skew = moments['mu11'] / moments['mu02']
        h, w = img.shape[:3]

        mtr = np.float32([[1, skew, -0.5 * skew * h],
                          [0, 1, 0]])
        return cv2.warpAffine(img, mtr, (w, h),
                              flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)


class SkewCorrection(ImageOnlyTransform):
    """

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8
    """

    def __init__(self,
                 always_apply=False,
                 p=0.5):
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        gray = cv2.bitwise_not(img)
        thresh = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]

        if angle > 10:
            return img

        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, -angle, 1.0)
        return cv2.warpAffine(img, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
