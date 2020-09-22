import cv2
import utils

def run_selective_search(image_file):
    img = cv2.imread('./JPEGImages/' + image_file + '.jpg')

    # create Selective Search Segmentation Object using default parameters
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    ss.setBaseImage(img)

    strategy_color = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyColor()
    strategy_texture = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyTexture()
    strategy_size = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategySize()
    strategy_fill = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyFill()
    strategy_combined = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyMultiple(strategy_color, strategy_texture, strategy_size, strategy_fill)
    
    strategies = [strategy_color, strategy_texture, strategy_size, strategy_fill, strategy_combined]

    for strategy in strategies:

        ss.clearStrategies()
        ss.addStrategy(strategy)

        ss.switchToSelectiveSearchFast()
        # ss.switchToSelectiveSearchQuality()

        bboxes = ss.process()

        image_output = img.copy()

        # draw ground truth
        # utils.draw_groundtruth(image_output)

        # iterate over all the region proposals
        for i, rect in enumerate(bboxes):
            # draw rectangle for region proposal till numShowRects
            if (i < utils.NUM_PROPOSALS):
                x, y, w, h = rect
                if utils.compute_recall((x , y, x + w, y + h)):
                    cv2.rectangle(image_output, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
            else:
                break

        cv2.imshow("Selective search output " + str(strategy), image_output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
