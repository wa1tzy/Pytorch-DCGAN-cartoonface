import torch
from torchvision import transforms
import torchvision
import opt
import torch.utils.data as data
from D_Net import NetD
from G_Net import NetG
import torch.nn as nn
import os
from torchvision.utils import save_image

class Trainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5,],[0.5,0.5,0.5,])
        ])
        self.loss_fn = nn.BCELoss()
    def train(self):
        dataset = torchvision.datasets.ImageFolder(opt.data_path,transform=self.trans)
        dataloader = data.DataLoader(dataset=dataset,batch_size=opt.batch_size,shuffle=True)
        d_net = NetD().to(self.device)
        g_net = NetG().to(self.device)
        if os.path.exists("dcgan_params"):
            # torch.nn.DataParallel(d_net)
            d_net.load_state_dict(torch.load("dcgan_params/d_net.pth"))
        else:
            print("NO d_net Param")

        if os.path.exists("dcgan_params"):
            # torch.nn.DataParallel(g_net)
            g_net.load_state_dict(torch.load("dcgan_params/g_net.pth"))
        else:
            print("NO g_net Param")
        D_optimizer = torch.optim.Adam(d_net.parameters(),lr=opt.lr1,betas=(opt.beta1,0.999))
        G_optimizer = torch.optim.Adam(g_net.parameters(),lr=opt.lr1,betas=(opt.beta1,0.999))
        NUM_EPOHS = opt.epochs
        for epoh in range(NUM_EPOHS):
            for i,(images,_) in enumerate(dataloader):
                N = images.size(0)
                images = images.to(self.device)
                real_labels = torch.ones(N,1,1,1).to(self.device)
                fake_labels = torch.zeros(N,1,1,1).to(self.device)
                real_out = d_net(images)
                d_real_loss = self.loss_fn(real_out,real_labels)

                z = torch.randn(N,100,1,1).to(self.device)
                fake_img = g_net(z)
                fake_out = d_net(fake_img)
                d_fake_loss = self.loss_fn(fake_out,fake_labels)
                d_loss = d_fake_loss+d_real_loss

                D_optimizer.zero_grad()
                d_loss.backward()
                D_optimizer.step()

                z = torch.randn(N,100,1,1).to(self.device)
                fake_img = g_net(z)
                fake_out = d_net(fake_img)
                g_loss = self.loss_fn(fake_out,real_labels)
                G_optimizer.zero_grad()
                g_loss.backward()
                G_optimizer.step()
                if i % 100 == 0:
                    print("Epoch:{}/{},d_loss:{:.3f},g_loss:{:.3f},"
                          "d_real:{:.3f},d_fake:{:.3f}".
                          format(epoh, NUM_EPOHS, d_loss.item(), g_loss.item(),
                                 real_out.data.mean(), fake_out.data.mean()))
                    if not os.path.exists("./dcgan_img"):
                        os.mkdir("./dcgan_img")
                    if not os.path.exists("./dcgan_params"):
                        os.mkdir("./dcgan_params")
                    real_image = images.cpu().data
                    save_image(real_image, "./dcgan_img/epoch{}-iteration{}-real_img.jpg".
                               format(epoh , i), nrow=10, normalize=True, scale_each=True)
                    fake_image = fake_img.cpu().data
                    save_image(fake_image, "./dcgan_img/epoch{}-iteration{}-fake_img.jpg".
                               format(epoh , i), nrow=10, normalize=True, scale_each=True)
                    torch.save(d_net.state_dict(), "dcgan_params/d_net.pth")
                    torch.save(g_net.state_dict(), "dcgan_params/g_net.pth")
if __name__ == '__main__':

    t = Trainer()
    t.train()