import torch
import matplotlib.pyplot as plt
import random
from MNIST_CGAN_NETWORK import *
import torch.nn.functional as F

if __name__ == '__main__':

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    G = CGAN_Generator().to(DEVICE)
    G.eval()

    folder_name = 'result/'

    fig = plt.figure(figsize=(10, 10))
    for i in range(5): # 50 EPOCH마다 저장한 weights를 불러와서
        G.load_state_dict(torch.load(folder_name + 'G_' + str(i * 50) + '.pth'))
        for j in range(5): # 5개의 이미지를 생성
            z = torch.randn(1, latent_size).to(DEVICE)

            label = torch.tensor(random.randint(0, 9)) # 0~9 사이의 임의의 숫자를 생성
            label_encoded = F.one_hot(label, num_classes=10).to(DEVICE) # one-hot 인코딩
            label_encoded = torch.unsqueeze(label_encoded, dim = 0)
            z_concat = torch.cat((z, label_encoded), 1) # 0~9 사이의 임의의 숫자 + latent vector를 G의 입력으로 사용

            fake_images = G(z_concat) # G를 통해 이미지 생성

            subplot = fig.add_subplot(5, 5, j + 1 + i*5, )
            subplot.imshow(denorm(fake_images).cpu().detach().numpy().reshape((28, 28)), cmap=plt.cm.gray_r)

            if j == 2:
                plt.title(str(i*50) + ' Epoch' + '\n' + str(label.numpy().item()))
            else:
                plt.title('\n' + str(label.numpy().item()))

            plt.axis('off')
    plt.subplots_adjust(wspace = 1.3, hspace = 1.3)
    plt.show()
