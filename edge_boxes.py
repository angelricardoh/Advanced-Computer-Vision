import cv2
import numpy as np
import utils

NUM_PROPOSALS = 100

def run_edge_boxes(image_file):
    img = cv2.imread('./JPEGImages/' + image_file + '.jpg')

    edge_detection = cv2.ximgproc.createStructuredEdgeDetection('model.yml.gz')
    rgb_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    edges = edge_detection.detectEdges(np.float32(rgb_im / 255.0))

    orientation = edge_detection.computeOrientation(edges)

    edges = edge_detection.edgesNms(edges, orientation)

    edge_boxes = cv2.ximgproc.createEdgeBoxes()
    boxes, scores = edge_boxes.getBoundingBoxes(edges, orientation)

    # draw ground truth
    utils.draw_groundtruth(img)

    if len(boxes) > 0:
        boxes_scores = zip(boxes, scores)

        for i, b_s in enumerate(boxes_scores):
            if (i < utils.NUM_PROPOSALS):
                box = b_s[0]
                x, y, w, h = box
                if utils.compute_recall((x , y, x + w, y + h)):
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
                    # score = b_s[1][0]
                    # cv2.putText(img, "{:.2f}".format(score), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
                    # print("Box at (x,y)=({:d},{:d}); score={:f}".format(x, y, score))
            else:
                break

    # cv2.imshow("edges", edges)
    cv2.imshow("Edgeboxes output", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
