"""
Part 3: Sliding Window Detection - Object Detector
Integrate image pyramid, sliding window, and NMS for multi-scale object detection.
Implement TODO 12.
"""

from .pyramid import generate_pyramid
from .sliding_window import sliding_window
from .nms import non_maximum_suppression

import numpy as np
from tqdm import tqdm


class ObjectDetector:
    """
    Multi-scale object detector using HOG features and SVM classifier.

    This detector integrates:
    1. Image pyramid for multi-scale detection
    2. Sliding window for exhaustive scanning
    3. HOG feature extraction for each window
    4. SVM classification for detection
    5. Non-maximum suppression for final detections

    Parameters:
        descriptor: HOG feature extractor
        classifier: trained SVM classifier (LinearSVM instance)
        window_size: (width, height) detection window size (default: 64x128)
        scale_factor: pyramid scale factor (default: 1.2)
        step_size: sliding window step size (default: 8)
        score_threshold: minimum detection score (default: 0.0)
        iou_threshold: NMS IoU threshold (default: 0.3)
    """

    def __init__(
        self,
        descriptor,
        classifier,
        window_size=(64, 128),
        scale_factor=1.2,
        step_size=8,
        score_threshold=0.1,
        iou_threshold=0.3
    ):
        self.descriptor = descriptor
        self.classifier = classifier
        
        self.window_size = window_size
        self.scale_factor = scale_factor
        self.step_size = step_size
        
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold


    def detect(self, image, verbose=False):
        """
        Detect objects in one or multiple images using multi-scale sliding window approach.

        Args:
            images: a single (H, W, 3) image
            verbose: whether to print detection progress (default: False)

        Returns:
            final_detections: list of detections (bboxes)
            each detection is a tuple (x, y, width, height, score)
            where (x, y) is top-left corner in original image coordinates

        Hint:
            - Generate image pyramid for each image
            - Apply sliding window detection at each pyramid scale
            - Apply non-maximum suppression
            - Return final detections
        """

        # ========================================
        # TODO 12: Implement Pedestrian Detection pipeline
        # ========================================
        
        detections = []
        win_width, win_height = self.window_size

        # 이미지 피라미드 생성
        pyramid = generate_pyramid(image, self.scale_factor, self.window_size)
        pyramid_iter = tqdm(pyramid, desc="Processing Pyramid", disable=not verbose)
        for (p_img, scale) in pyramid_iter:
            for (x, y, window) in sliding_window(p_img, self.window_size, self.step_size): # 슬라이딩 윈도우
                features = self.descriptor.extract(window) # HOG 특징 추출
                score = self.classifier.predict(features) # 점수 계산

                if score > self.score_threshold: # 분류
                    orig_x = x / scale
                    orig_y = y / scale
                    orig_width = win_width / scale
                    orig_height = win_height / scale
                    
                    detections.append((orig_x, orig_y, orig_width, orig_height, score))

        # NMS 적용
        final_detections = non_maximum_suppression(detections, self.iou_threshold)
        
        # ========================================

        return final_detections


    def set_threshold(self, score_threshold=None, iou_threshold=None):
        """
        Update detection thresholds.

        Args:
            score_threshold: minimum detection score (optional)
            iou_threshold: NMS IoU threshold (optional)
        """
        if score_threshold is not None:
            self.score_threshold = score_threshold

        if iou_threshold is not None:
            self.iou_threshold = iou_threshold
