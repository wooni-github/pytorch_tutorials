import cv2
import matplotlib
import numpy as np

edges = [
    (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
    (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
    (12, 14), (14, 16), (5, 6)
]

def draw_keypoints(outputs, image, PIL_image = False):

    image = np.array(image, dtype=np.float32)
    if PIL_image:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image / 255.

    # outputs : boxes, labels, scores, keypoints, keypoints_scores

    for i in range(len(outputs[0]['keypoints'])):
        keypoints = outputs[0]['keypoints'][i].cpu().detach().numpy()

        if outputs[0]['scores'][i] > 0.9:
            keypoints = keypoints[:, :].reshape(-1, 3)
            for p in range(keypoints.shape[0]):
                cv2.circle(image, (int(keypoints[p, 0]), int(keypoints[p, 1])), 3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

            boxes = outputs[0]['boxes'][i].cpu().detach().numpy()
            x1, y1, x2, y2 = boxes

            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)

            for ie, e in enumerate(edges):
                rgb = matplotlib.colors.hsv_to_rgb([
                    ie/float(len(edges)), 1.0, 1.0
                ])
                rgb = rgb*255
                cv2.line(image, (int(keypoints[e, 0][0]), int(keypoints[e, 1][0])),
                        (int(keypoints[e, 0][1]), int(keypoints[e, 1][1])),
                        tuple(rgb), 2, lineType=cv2.LINE_AA)
        else:
            continue
    return image