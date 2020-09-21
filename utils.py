import xml.etree.ElementTree as ET
import cv2

gt_boxes = []
NUM_PROPOSALS = 100

def parse_xmlfile(xmlfile):
    root = ET.parse('./Annotations/' + xmlfile + '.xml')
    for bndbox in root.findall('object/bndbox'):
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        gt_boxes.append((xmin, ymin, xmax, ymax))

    return gt_boxes

def draw_groundtruth(image_output):
    for ground_truth in gt_boxes:
        cv2.rectangle(image_output, (ground_truth[0], ground_truth[1]), (ground_truth[2], ground_truth[3]), (0, 0, 255), 1, cv2.LINE_AA)

def intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def compute_recall(test_box):
    for box in gt_boxes:
        if intersection_over_union(test_box, box) > 0.5:
            return True
    return False