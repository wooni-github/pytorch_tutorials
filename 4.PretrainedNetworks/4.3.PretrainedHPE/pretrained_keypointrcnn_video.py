import torch
import torchvision
import cv2
import argparse
import utils
import time
from PIL import Image
from torchvision.transforms import transforms as transforms

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default = 'test_video.mp4')
    args = parser.parse_args()

    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True, num_keypoints=17)

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

                pil_image = Image.fromarray(frame)

                outputs = model(transform(pil_image).unsqueeze(0).to(DEVICE))
                end_time = time.time()

                output_image = utils.draw_keypoints(outputs, frame) # 가시화

                fps = 1 / (end_time - start_time)
                total_fps += fps
                frame_count += 1
                wait_time = max(1, int(fps / 4))

                cv2.imshow('Output frame', output_image)

                if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                    break
            else:
                break

    cap.release()
    cv2.destroyAllWindows()
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")