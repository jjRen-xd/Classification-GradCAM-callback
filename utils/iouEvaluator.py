import numpy as np  
import cv2

_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0

def check_scoremap_validity(scoremap):
    if not isinstance(scoremap, np.ndarray):
        raise TypeError("Scoremap must be a numpy array; it is {}."
                        .format(type(scoremap)))
    if scoremap.dtype != np.float:
        raise TypeError("Scoremap must be of np.float type; it is of {} type."
                        .format(scoremap.dtype))
    if len(scoremap.shape) != 2:
        raise ValueError("Scoremap must be a 2D array; it is {}D."
                         .format(len(scoremap.shape)))
    if np.isnan(scoremap).any():
        raise ValueError("Scoremap must not contain nans.")
    if (scoremap > 1).any() or (scoremap < 0).any():
        raise ValueError("Scoremap must be in range [0, 1]."
                         "scoremap.min()={}, scoremap.max()={}."
                         .format(scoremap.min(), scoremap.max()))


def check_box_convention(boxes, convention):
    """
    Args:
        boxes: numpy.ndarray(dtype=np.int or np.float, shape=(num_boxes, 4))
        convention: string. One of ['x0y0x1y1', 'xywh'].
    Raises:
        RuntimeError if box does not meet the convention.
    """
    if (boxes < 0).any():
        raise RuntimeError("Box coordinates must be non-negative.")

    if len(boxes.shape) == 1:
        boxes = np.expand_dims(boxes, 0)
    elif len(boxes.shape) != 2:
        raise RuntimeError("Box array must have dimension (4) or "
                           "(num_boxes, 4).")

    if boxes.shape[1] != 4:
        raise RuntimeError("Box array must have dimension (4) or "
                           "(num_boxes, 4).")

    if convention == 'x0y0x1y1':
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
    elif convention == 'xywh':
        widths = boxes[:, 2]
        heights = boxes[:, 3]
    else:
        raise ValueError("Unknown convention {}.".format(convention))

    if (widths < 0).any() or (heights < 0).any():
        raise RuntimeError("Boxes do not follow the {} convention."
                           .format(convention))


def compute_bboxes_from_scoremaps(scoremap, scoremap_threshold_list,
                                  multi_contour_eval=False):
    """
    Args:
        scoremap: numpy.ndarray(dtype=np.float32, size=(H, W)) between 0 and 1
        scoremap_threshold_list: iterable
        multi_contour_eval: flag for multi-contour evaluation

    Returns:
        estimated_boxes_at_each_thr: list of estimated boxes (list of np.array)
            at each cam threshold
        number_of_box_list: list of the number of boxes at each cam threshold
    """
    check_scoremap_validity(scoremap)
    height, width = scoremap.shape
    scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)

    def scoremap2bbox(threshold):
        _, thr_gray_heatmap = cv2.threshold(
            src=scoremap_image,
            thresh=int(threshold * np.max(scoremap_image)),
            maxval=255,
            type=cv2.THRESH_BINARY)
        contours = cv2.findContours(
            image=thr_gray_heatmap,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]

        if len(contours) == 0:
            return np.asarray([[0, 0, 0, 0]]), 1

        if not multi_contour_eval:
            contours = [max(contours, key=cv2.contourArea)]

        estimated_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x0, y0, x1, y1 = x, y, x + w, y + h
            x1 = min(x1, width - 1)
            y1 = min(y1, height - 1)
            estimated_boxes.append([x0, y0, x1, y1])

        return np.asarray(estimated_boxes), len(contours)

    estimated_boxes_at_each_thr = []
    number_of_box_list = []
    for threshold in scoremap_threshold_list:
        boxes, number_of_box = scoremap2bbox(threshold)
        estimated_boxes_at_each_thr.append(boxes)
        number_of_box_list.append(number_of_box)

    return estimated_boxes_at_each_thr, number_of_box_list


def calculate_multiple_iou(box_a, box_b):
    """
    Args:
        box_a: numpy.ndarray(dtype=np.int, shape=(num_a, 4))
            x0y0x1y1 convention.
        box_b: numpy.ndarray(dtype=np.int, shape=(num_b, 4))
            x0y0x1y1 convention.
    Returns:
        ious: numpy.ndarray(dtype=np.int, shape(num_a, num_b))
    """
    num_a = box_a.shape[0]
    num_b = box_b.shape[0]

    check_box_convention(box_a, 'x0y0x1y1')
    check_box_convention(box_b, 'x0y0x1y1')

    # num_a x 4 -> num_a x num_b x 4
    box_a = np.tile(box_a, num_b)
    box_a = np.expand_dims(box_a, axis=1).reshape((num_a, num_b, -1))

    # num_b x 4 -> num_b x num_a x 4
    box_b = np.tile(box_b, num_a)
    box_b = np.expand_dims(box_b, axis=1).reshape((num_b, num_a, -1))

    # num_b x num_a x 4 -> num_a x num_b x 4
    box_b = np.transpose(box_b, (1, 0, 2))

    # num_a x num_b
    min_x = np.maximum(box_a[:, :, 0], box_b[:, :, 0])
    min_y = np.maximum(box_a[:, :, 1], box_b[:, :, 1])
    max_x = np.minimum(box_a[:, :, 2], box_b[:, :, 2])
    max_y = np.minimum(box_a[:, :, 3], box_b[:, :, 3])

    # num_a x num_b
    area_intersect = (np.maximum(0, max_x - min_x + 1)
                      * np.maximum(0, max_y - min_y + 1))
    area_a = ((box_a[:, :, 2] - box_a[:, :, 0] + 1) *
              (box_a[:, :, 3] - box_a[:, :, 1] + 1))
    area_b = ((box_b[:, :, 2] - box_b[:, :, 0] + 1) *
              (box_b[:, :, 3] - box_b[:, :, 1] + 1))

    denominator = area_a + area_b - area_intersect
    degenerate_indices = np.where(denominator <= 0)
    denominator[degenerate_indices] = 1

    ious = area_intersect / denominator
    ious[degenerate_indices] = 0
    return ious


class MouseAction():
    def __init__(self):
        self.img = None
        self.point1 = None
        self.point2 = None

        self.min_x = None
        self.min_y = None
        self.width = None
        self.height = None

    def on_mouse(self, event, x, y, flags, param):
        editImg = self.img.copy()
        # 左键点击
        if event == cv2.EVENT_LBUTTONDOWN:        
            self.point1 = (x,y)
            cv2.circle(editImg, self.point1, 10, (0,255,0), 5)
            cv2.imshow('image', editImg)
        # 按住左键拖曳
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):               
            cv2.rectangle(editImg, self.point1, (x,y), (0,255,0), 1)

            blk = np.zeros(self.img.shape, np.float64)  
            cv2.rectangle(blk, self.point1, (x,y), (0,255,0), -1)
            blk = cv2.addWeighted(editImg, 0.9, blk, 0.1, 0)
            cv2.imshow('image', blk)
        # 左键释放
        elif event == cv2.EVENT_LBUTTONUP:         
            self.point2 = (x,y)
            cv2.rectangle(editImg, self.point1, self.point2, (0,0,255), 1) 

            blk = np.zeros(self.img.shape, np.float64)  
            cv2.rectangle(blk, self.point1, (x,y), (0, 0, 255), -1)
            blk = cv2.addWeighted(editImg, 0.9, blk, 0.1, 0)
            cv2.imshow('image', blk)
            # self.min_x = min(self.point1[0], self.point2[0])     
            # self.min_y = min(self.point1[1], self.point2[1])
            # self.width = abs(self.point1[0] - self.point2[0])
            # self.height = abs(self.point1[1] - self.point2[1])



    def label(self, img):
        self.img = img
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.on_mouse)
        cv2.putText(self.img, "<ESC> exit, <Enter> next, <Space> skip", (0, 15), 
                        cv2.FONT_HERSHEY_COMPLEX, 0.3, (100, 200, 200), 1)
        cv2.imshow('image', self.img)
        while 1:
            key = cv2.waitKey()
            if key == 27:   # ESC 退出
                exit()  
            elif key == 13: # Enter 下一张图片
                if self.point1 is None:
                    print('Please select a ROI !!!')
                    continue
                roi_bbox = [self.point1[0], self.point1[1], self.point2[0], self.point2[1]]
                self.point1 = None
                self.point2 = None
                return(np.array(roi_bbox))
            elif key == 32: # Space 跳过标注
                return(None)
            else:
                continue

if __name__ == "__main__":
    img = cv2.imread('acc.jpg')
    mouseAction = MouseAction()
    mouseAction.label(img)