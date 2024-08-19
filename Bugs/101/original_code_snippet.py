import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(ResBlock, self).__init__()
        self.conv1x1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        #batch normalization
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x):
        residual = self.conv1x1(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        #adding the skip connection
        out += residual
        out = self.relu(out)

        return out

class ResUnet (nn.Module):

    def __init__(self, in_shape,  num_classes):
        super(ResUnet, self).__init__()
        in_channels, height, width = in_shape
        #
        #self.L1 = IncResBlock(in_channels,64)
        self.e1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2,padding=1),
            ResBlock(64,64))


        self.e2 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.Conv2d(64, 128, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(128),
            ResBlock(128,128))
        #
        self.e2add = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(128))
        #
        ##
        self.e3 = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,),
            nn.Conv2d(128,256, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(256),
            ResBlock(256,256))

        self.e4 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.Conv2d(256,512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            ResBlock(512,512))
        #
        self.e4add = nn.Sequential(
            nn.Conv2d(512,512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(512)) 
        #
        self.e5 = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(512,512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,),
            nn.Conv2d(512,512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            ResBlock(512,512))
        #
        #
        self.e6 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.Conv2d(512,512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512), 
            ResBlock(512,512))
        #
        self.e6add = nn.Sequential(
            nn.Conv2d(512,512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(512)) 
        #
        self.e7 = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(512,512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,),
            nn.Conv2d(512,512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            ResBlock(512,512))
        #
        self.e8 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.Conv2d(512,512, kernel_size=4, stride=2,padding=1))
            #nn.BatchNorm2d(512))

        self.d1 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout(p=0.5),
            ResBlock(512,512))
        #
        self.d2 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout(p=0.5),
            ResBlock(512,512))
        #
        self.d3 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout(p=0.5),
            ResBlock(512,512))
        #
        self.d4 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            ResBlock(512,512))

        #
        self.d5 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(256),
            ResBlock(256,256))
        #
        self.d6 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(128),
            ResBlock(128,128))
        #
        self.d7 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(64),
            ResBlock(64,64))
        #
        self.d8 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2,padding=1))
            #nn.BatchNorm2d(64),
            #nn.ReLU())

        self.out_l = nn.Sequential(
            nn.Conv2d(64,num_classes,kernel_size=1,stride=1))
            #nn.ReLU())

    def forward(self, x):

        #Image Encoder

        #### Encoder #####

        en1 = self.e1(x)

        en2 = self.e2(en1)
        en2add = self.e2add(en2)

        en3 = self.e3(en2add)

        en4 = self.e4(en3)
        en4add = self.e4add(en4)

        en5 = self.e5(en4add)

        en6 = self.e6(en5)
        en6add = self.e6add(en6)

        en7 = self.e7(en6add)

        en8 = self.e8(en7)

        #### Decoder ####
        de1_ = self.d1(en8)
        de1 = torch.cat([en7,de1_],1)

        de2_ = self.d2(de1)
        de2 = torch.cat([en6add,de2_],1)


        de3_ = self.d3(de2)
        de3 = torch.cat([en5,de3_],1)


        de4_ = self.d4(de3)
        de4 = torch.cat([en4add,de4_],1)


        de5_ = self.d5(de4)
        de5 = torch.cat([en3,de5_],1)

        de6_ = self.d6(de5)
        de6 = torch.cat([en2add,de6_],1)

        de7_ = self.d7(de6)
        de7 = torch.cat([en1,de7_],1)
        de8 = self.d8(de7)

        out_l_mask = self.out_l(de8)

        return out_l_mask  
