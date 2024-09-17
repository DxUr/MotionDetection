import cv2
import numpy as np

def get_mask(frame1, frame2, kernel=np.array((9,9), dtype=np.uint8)):
    """ Obtains image mask
        Inputs: 
            frame1 - Grayscale frame at time t
            frame2 - Grayscale frame at time t + 1
            kernel - (NxN) array for Morphological Operations
        Outputs: 
            mask - Thresholded mask for moving pixels
        """

    frame_diff = cv2.subtract(frame2, frame1)

    # blur the frame difference
    frame_diff = cv2.medianBlur(frame_diff, 3)
    
    mask = cv2.adaptiveThreshold(frame_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV, 11, 3)

    mask = cv2.medianBlur(mask, 3)

    # morphological operations
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask

def get_contour_detections(mask, thresh=400):
    """ Obtains initial proposed detections from contours discoverd on the mask. 
        Scores are taken as the bbox area, larger is higher.
        Inputs:
            mask - thresholded image mask
            thresh - threshold for contour size
        Outputs:
            detectons - array of proposed detection bounding boxes and scores [[x1,y1,x2,y2,s]]
        """
    # get mask contours
    contours, _ = cv2.findContours(mask, 
                                   cv2.RETR_EXTERNAL, # cv2.RETR_TREE, 
                                   cv2.CHAIN_APPROX_TC89_L1)
    detections = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        area = w*h
        if area > thresh: 
            detections.append([x,y,x+w,y+h, area])

    return np.array(detections)

def remove_contained_bboxes(boxes):
    """ Removes all smaller boxes that are contained within larger boxes.
        Requires bboxes to be soirted by area (score)
        Inputs:
            boxes - array bounding boxes sorted (descending) by area 
                    [[x1,y1,x2,y2]]
        Outputs:
            keep - indexes of bounding boxes that are not entirely contained 
                   in another box
        """
    check_array = np.array([True, True, False, False])
    keep = list(range(0, len(boxes)))
    for i in keep: # range(0, len(bboxes)):
        for j in range(0, len(boxes)):
            # check if box j is completely contained in box i
            if np.all((np.array(boxes[j]) >= np.array(boxes[i])) == check_array):
                try:
                    keep.remove(j)
                except ValueError:
                    continue
    return keep


def non_max_suppression(boxes, scores, threshold=1e-1):
    """
    Perform non-max suppression on a set of bounding boxes and corresponding scores.
    Inputs:
        boxes: a list of bounding boxes in the format [xmin, ymin, xmax, ymax]
        scores: a list of corresponding scores 
        threshold: the IoU (intersection-over-union) threshold for merging bounding boxes
    Outputs:
        boxes - non-max suppressed boxes
    """
    # Sort the boxes by score in descending order
    boxes = boxes[np.argsort(scores)[::-1]]

    # remove all contained bounding boxes and get ordered index
    order = remove_contained_bboxes(boxes)

    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        for j in order:
            # Calculate the IoU between the two boxes
            intersection = max(0, min(boxes[i][2], boxes[j][2]) - max(boxes[i][0], boxes[j][0])) * \
                           max(0, min(boxes[i][3], boxes[j][3]) - max(boxes[i][1], boxes[j][1]))
            union = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]) + \
                    (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1]) - intersection
            iou = intersection / union

            # Remove boxes with IoU greater than the threshold
            if iou > threshold:
                order.remove(j)
                
    return boxes[keep]


def convert_to_cv8uc1(image):
    # Check if the image is already grayscale and has 2 dimensions
    if len(image.shape) == 2:
        print("Image is already grayscale.")
        gray_image = image
    else:
        # Convert color image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Normalize the image to range [0, 255] if necessary
    if gray_image.dtype != np.uint8:
        gray_image = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return gray_image


def sobel_edge_detection(image):
    # Convert to grayscale if needed
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Sobel edge detection
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=9)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=9)
    edge_strength = cv2.magnitude(grad_x, grad_y)
    
    # Normalize the edge strength to the range [0, 1]
    edge_strength = cv2.normalize(edge_strength, None, 0, 1, cv2.NORM_MINMAX)
    
    return edge_strength

def gaussian_blur(image):
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (19, 19), 0)
    return blurred

def threshold_image(image, thresh):
    # Apply thresholding
    _, thresh_img = cv2.threshold(image, thresh, 1.0, cv2.THRESH_BINARY)
    return thresh_img

def main():
    rtcp_stream_url = "test.mp4"
    cap = cv2.VideoCapture(rtcp_stream_url)
    if not cap.isOpened():
        print("Error: Unable to open RTCP stream")
    else:
        img1 = None
        img2 = None
        latest = None
        counter = 0
        mask = None
        mask_captures = 0
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Unable to read frame from video")
                break
            img2 = img1
            img1 = frame.copy()

            if img2 is None:
                continue

            edge1 = sobel_edge_detection(img1)
            edge2 = sobel_edge_detection(img2)
            
            # Blur to reduce noise
            blurred_edge1 = gaussian_blur(img1)
            blurred_edge2 = gaussian_blur(img2)
            
            # Compute motion by taking the absolute difference
            motion = cv2.absdiff(blurred_edge1, blurred_edge2)
            
            # Threshold to eliminate small changes (noise)
            filtered_motion = threshold_image(motion, 0.001)

            mask = filtered_motion
            
            print(mask.shape)
            detections = get_contour_detections(convert_to_cv8uc1(mask), thresh=40)

            # separate bboxes and scores
            if len(detections) > 0:
                bboxes = detections[:, :4]
                scores = detections[:, 4]
                # Get Non-Max Suppressed Bounding Boxes
                nms_bboxes = non_max_suppression(bboxes, scores, threshold=3)

                for box in nms_bboxes:
                    x1, y1, x2, y2 = box
                    # calc bounding box area
                    area = (x2 - x1) * (y2 - y1)
                    if area > 500:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box


            # Display the frame
            cv2.imshow("frame", frame)
            cv2.imshow("mask", mask * 1.5)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()