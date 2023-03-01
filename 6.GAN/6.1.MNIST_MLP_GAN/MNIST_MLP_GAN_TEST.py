import torch
import matplotlib.pyplot as plt

from MNIST_MLP_GAN_NETWORK import *

if __name__ == '__main__':

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    G = Generator().to(DEVICE)
    G.eval()

    folder_name = 'result/'

    fig = plt.figure(figsize=(10, 10))
    for i in range(5): # 50 EPOCH마다 저장한 weights를 불러와서
        G.load_state_dict(torch.load(folder_name + 'G_' + str(i * 50) + '.pth'))
        for j in range(5): # 5개의 이미지를 생성
            z = torch.randn(1, latent_size).to(DEVICE) # latent vector로부터
            fake_images = G(z) # G를 통해 이미지 생성

            fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
            subplot = fig.add_subplot(5, 5, j + 1 + i*5, )
            subplot.imshow(denorm(fake_images).cpu().detach().numpy().reshape((28, 28)), cmap=plt.cm.gray_r)
            if j == 2:
                plt.title(str(i*50) + ' Epoch')
            plt.axis('off')
    plt.subplots_adjust(wspace = 1.3, hspace = 1.3)
    plt.show()
