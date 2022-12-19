import torchvision
import cv2
import torch
import argparse
import time
import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default = 'seq1.mp4')
    args = parser.parse_args()

    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

    model = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
    model.to(DEVICE)
    model.eval()

    cap = cv2.VideoCapture(args.input)

    if (not cap.isOpened()):
        print('Video is not available')

    frame_count = 0
    total_fps = 0

    with torch.no_grad():
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                start_time = time.time()
                frame = cv2.resize(frame, (640, 360))
                outputs = utils.get_segment_labels(frame, model, DEVICE)

                end_time = time.time()

                outputs = outputs['out']  # output은 클래스에 속하는 확률이 담긴 'out'과 auxilliary loss가 담긴 'aux'로 구성되어 있음.

                overlay, segmented = utils.visualize(frame, outputs) # 가시화

                fps = 1 / (end_time - start_time)
                total_fps += fps
                frame_count += 1
                wait_time = max(1, int(fps / 4))

                cv2.imshow('Input image', frame)
                cv2.imshow('Overlay image', overlay)
                cv2.imshow('Segmented image', segmented)

                if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()
        avg_fps = total_fps / frame_count
        print(f"Average FPS: {avg_fps:.3f}")
