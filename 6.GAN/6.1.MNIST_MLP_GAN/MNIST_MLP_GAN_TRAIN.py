import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
import os
import argparse

from MNIST_MLP_GAN_NETWORK import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--img_save_interval', type=int, default=5000)
    args = parser.parse_args()

    folder_name = 'result/'
    if args.img_save_interval > 0:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

    EPOCHS = args.epoch
    BATCH_SIZE = args.batch_size

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])])

    mnist = torchvision.datasets.MNIST(root='../data',
                                       train=True,
                                       transform=transform,
                                       download=True)
    data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True)

    D = Discriminator().to(DEVICE)
    G = Generator().to(DEVICE)

    criterion = nn.BCELoss()  # True vs False만 구분하면 되므로 BCELoss 사용
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

    total_step = len(data_loader)
    iteration = 0
    max_iteration = len(data_loader) * EPOCHS

    for epoch in range(EPOCHS + 1):
        for i, (real_images, _) in enumerate(data_loader):
            real_images = real_images.reshape(BATCH_SIZE, -1).to(DEVICE)
            # CNN이 아닌 FCL만을 사용하였으므로 이미지를 1D Vector로 펴줌 [1 x 784]

            real_labels = torch.ones(BATCH_SIZE, 1).to(DEVICE)
            fake_labels = torch.zeros(BATCH_SIZE, 1).to(DEVICE)
            # Batch 갯수만큼 real label : 1, fake label : 0으로 설정

            # Discriminator 학습
            real_score = D(real_images)
            d_loss_real = criterion(real_score, real_labels)  # real 이미지를 D에 넣었을 때는 real로 판별해야함(1)

            z = torch.randn(BATCH_SIZE, latent_size).to(DEVICE)
            fake_images = G(z)  # latent vector z를 Generator에 넣어 fake 이미지 생성
            fake_score = D(fake_images)
            d_loss_fake = criterion(fake_score, fake_labels)  # 이 때 D(G(z))는 fake 이미지로 판별해야 함 (0)

            d_loss = d_loss_real + d_loss_fake  # D는 E[logD(x)] + E[log(1-D(G(z))]를 모두 학습함
            # 즉, D는 real 이미지를 real로, fake 이미지를 fake로 구분할 수 있도록 학습이 진행됨

            d_optimizer.zero_grad()
            g_optimizer.zero_grad()

            d_loss.backward()
            d_optimizer.step()

            # Generator 학습
            z = torch.randn(BATCH_SIZE, latent_size).to(DEVICE)
            fake_images = G(z)
            fake_score2 = D(fake_images)

            g_loss = criterion(fake_score2, real_labels)
            # D는 E[logD(x)] + E[log(1-D(G(z))]를 모두 학습함
            # G는 G에 관한 term이 없는 좌측을 생략한 E[log(1-D(G(z))]를 학습함
            # 즉, z로부터 G로 생성한 fake 이미지를 real 이미지로 분류하도록 학습

            d_optimizer.zero_grad()
            g_optimizer.zero_grad()

            g_loss.backward()
            g_optimizer.step()
            if iteration % 200 == 0:
                print(('Iteration [{}/{} = {:.3f}%], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(Generator(z)): {:.2f}').format(
                    iteration, max_iteration, iteration / max_iteration * 100.0, d_loss.item(), g_loss.item(), real_score.mean().item(), fake_score.mean().item()))

            if args.img_save_interval > 0 and iteration % args.img_save_interval == 0:
                fig = plt.figure(figsize=(5, 2))

                for it in range(5):
                    subplot = fig.add_subplot(1, 5, it + 1)
                    fake_image = fake_images[it].reshape(28, 28)
                    subplot.imshow(denorm(fake_image).cpu().detach().numpy().reshape((28, 28)), cmap=plt.cm.gray_r)
                    plt.axis('off')
                    if it == 2:
                        plt.title('iteration : ' + str(iteration))
                plt.savefig(folder_name + str(iteration) + '.png')
            iteration += 1

        if epoch % 50 == 0:
            torch.save(G.state_dict(), folder_name + 'G_' + str(epoch) + '.pth')
            # TEST시에는 latent vector로부터 G를 통해 생성된 이미지만을 확인하므로 D는 저장할 필요 없음
