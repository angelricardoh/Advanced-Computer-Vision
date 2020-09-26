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

def intersection_over_union(box_a, box_b):
    # determine the (x, y)-coordinates of the intersection rectangle
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    # compute the area of intersection rectangle
    intersection_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(box_a_area + box_b_area - intersection_area)

    return iou

def compare_gt(test_box):
    for box in gt_boxes:
        if intersection_over_union(test_box, box) > 0.5:
            return True
    return False