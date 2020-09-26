import cv2
import numpy as np
import utils

def run_edge_boxes(image_file, num_proposals, draw_only_positives):
    img = cv2.imread('./JPEGImages/' + image_file + '.jpg')

    parameters_combinations = [(0.25, 0.85), (0.85, 0.35), (0.65, 0.75), (0.45, 0.45), (0.85, 0.85)]

    recall_list = []

    for parameter_tuple in parameters_combinations:
        edge_detection = cv2.ximgproc.createStructuredEdgeDetection('model.yml.gz')

        rgb_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        edges = edge_detection.detectEdges(np.float32(rgb_im / 255.0))

        orientation = edge_detection.computeOrientation(edges)

        edges = edge_detection.edgesNms(edges, orientation)
        edge_boxes = cv2.ximgproc.createEdgeBoxes(alpha = parameter_tuple[0], beta = parameter_tuple[1])
        boxes, scores = edge_boxes.getBoundingBoxes(edges, orientation)

        image_output = img.copy()

        true_positives = 0

        if len(boxes) > 0:
            boxes_scores = zip(boxes, scores)

            for i, b_s in enumerate(boxes_scores):
                if (i < num_proposals):
                    box = b_s[0]
                    x, y, w, h = box
                    if draw_only_positives and not utils.compare_gt((x , y, x + w, y + h)):
                        continue
                    true_positives += 1
                    cv2.rectangle(image_output, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
                    score = b_s[1][0]
                    cv2.putText(image_output, "{:.2f}".format(score), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
                    # print("score={:f}".format(score))
                else:
                    break
        
        # draw ground truth
        if draw_only_positives:
            utils.draw_groundtruth(img)

        recall_list.append((true_positives / len(boxes), len(boxes)))
        # cv2.imshow("Edgeboxes output " + str(parameter_tuple), image_output)
        cv2.imwrite("./eb_output_" + str(parameter_tuple) + ".jpg", image_output)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    return recall_list
