import cv2
import utils

def run_selective_search(image_file, num_proposals, draw_only_positives):
    img = cv2.imread('./JPEGImages/' + image_file + '.jpg')

    # create Selective Search Segmentation Object using default parameters
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    ss.setBaseImage(img)

    strategy_color = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyColor()
    strategy_texture = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyTexture()
    strategy_size = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategySize()
    strategy_fill = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyFill()
    strategy_combined = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyMultiple(strategy_color, strategy_texture, strategy_size, strategy_fill)
    
    strategies = [strategy_texture, strategy_color, strategy_size, strategy_fill, strategy_combined]

    recall_list = []

    for strategy in strategies:

        ss.addStrategy(strategy)

        ss.switchToSelectiveSearchFast()
        # ss.switchToSelectiveSearchQuality()

        bboxes = ss.process()

        image_output = img.copy()

        true_positives = 0

        # iterate over all the region proposals
        for i, rect in enumerate(bboxes):
            # draw rectangle for region proposal till numShowRects
            if (i < num_proposals):
                x, y, w, h = rect
                if draw_only_positives and not utils.compare_gt((x , y, x + w, y + h)):
                    continue
                true_positives += 1
                cv2.rectangle(image_output, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
            else:
                break

        if draw_only_positives:
            utils.draw_groundtruth(img)

        recall_list.append((true_positives / len(bboxes), len(boxes)))
        # cv2.imshow("Selective search output " + str(strategy), image_output)
        cv2.imwrite("./ss_output_" + str(strategy) + ".jpg", image_output)
        cv2.waitKey(5)
        ss.clearStrategies()

        # cv2.destroyAllWindows()

    return recall_list
