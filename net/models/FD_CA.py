import torch.nn as nn
import torch
from common import ACmix,h_sigmoid,h_swish,CA

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size//2), bias=bias)


class ECA(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(ECA, self).__init__()

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        pool_h = nn.AdaptiveAvgPool2d((h, 1))
        pool_w = nn.AdaptiveAvgPool2d((1, w))
        x_h = pool_h(x)
        x_w = pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        pool_max_h = nn.AdaptiveMaxPool2d((h, 1))
        pool_max_w = nn.AdaptiveMaxPool2d((1, w))
        x_max_h = pool_max_h(x)
        x_max_w = pool_max_w(x).permute(0, 1, 3, 2)

        y_max_pool = torch.cat([x_max_h,x_max_w], dim=2)
        y_max_pool = self.conv1(y_max_pool)
        y_max_pool = self.bn1(y_max_pool)
        y_max_pool = self.act(y_max_pool)
        
        y = 0.5*y + 0.5* y_max_pool
        #y = 0.8*y + 0.2* y_max_pool

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out
    
class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class Block(nn.Module):
    def __init__(self, conv, dim, kernel_size,):
        super(Block, self).__init__()
        self.conv1=conv(dim, dim, kernel_size, bias=True)
        self.act1=nn.ReLU(inplace=True)
        self.conv2=conv(dim,dim,kernel_size,bias=True)
       # self.ac=ACmix(dim,dim)
        self.calayer=CALayer(dim)
        self.palayer=PALayer(dim)
    def forward(self, x):
        res=self.act1(self.conv1(x))
        res=res+x 
        res=self.conv2(res)
        #res=self.ac(res)
        res=self.calayer(res)
        res=self.palayer(res)
        res += x 
        return res
class RC(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(RC, self).__init__()
        modules = [ Block(conv, dim, kernel_size)  for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)
    def forward(self, x):
        res = self.gp(x)
        res += x
        return res

class FD_CA(nn.Module):
    def __init__(self,gps,blocks,conv=default_conv):
        super(FD_CA, self).__init__()
        self.gps=gps
        self.dim=64
        kernel_size=3
        pre_process = [conv(3, self.dim, kernel_size)]
        assert self.gps==3
        self.g1= RC(conv, self.dim, kernel_size,blocks=blocks)
       # self.ca1=CA(self.dim,self.dim)
        self.g2= RC(conv, self.dim, kernel_size,blocks=blocks)
       # self.ca2= CA(self.dim,self.dim)
        self.g3= RC(conv, self.dim, kernel_size,blocks=blocks)
        self.ca3= CA(self.dim,self.dim)
        self.ca=nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim*self.gps,self.dim//16,1,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim//16, self.dim*self.gps, 1, padding=0, bias=True),
            nn.Sigmoid()
            ])
       # self.palayer=PALayer(self.dim)

        post_precess = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)]

        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_precess)


        self.ECA = ECA(self.dim, self.dim)
       # self.PA = PALayer(64)
        self.calayer=CALayer(64)
        self.palayer = PALayer(64)



    def forward(self, x1):
        x = self.pre(x1)
        res1=self.g1(x)
        #res11=self.ca1(res1)
        res2=self.g2(res1)
        #res22=self.ca2(res2)

        res3=self.g3(res2)

        # 记录差值
        res2_1 = res2 - res1
        res3_2 = res3 - res2
        res3_ = res3 +  res3_2
        res2_ = res2 +  res2_1


        ca1 = self.ECA(res1)
        ca2 = self.ECA(res2_)
        ca3 = self.ECA(res3_)
        sec_c = ca3 + ca2
        fir_c = ca1 + sec_c

        sec1 = res3_ + res2_
        fir1 = res1 + sec1

        #w = self.ca(torch.cat([res1,res2_,res3_],dim=1))
       # x = torch.cat([self.CALyear(fir_c),self.ECA(fir_c)],dim=1)
       # res33=self.ca3(res3)
       # w=self.ca(torch.cat([ca1,ca2,ca3],dim=1))
        w=self.ca(torch.cat([fir_c,sec_c,ca3],dim=1))
        w=w.view(-1,self.gps,self.dim)[:,:,:,None,None]
       # out=w[:,0,::]*res1+w[:,1,::]*res2_+w[:,2,::]*res3_
        out=w[:,0,::]*fir_c+w[:,1,::]*sec_c+w[:,2,::]*ca3
       # out=w[:,0,::]*ca1+w[:,1,::]*ca2+w[:,2,::]*ca3
       # out=self.palayer(out)
       # out=self.calayer(out)
        out=self.ECA(out)
        out=self.ca3(out)  #CA
        x=self.post(out)
        return x + x1
if __name__ == "__main__":
    net=FFA(gps=3,blocks=19)
    print(net)

