import torch
import numpy as np

from model_xss import rid_dense_sub_ca
from torch.autograd import Variable
from load_data import load_sidd

'''------------load model------------'''
model = rid_dense_sub_ca.rid_model(in_channels = 4, channels_basenum = 64)

'''------------hyper param------------'''
Loss = torch.nn.L1Loss()
Loss.cuda()
model.cuda()

'''------------define train------------'''
def train(startepoch,endepoch,lr_):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_, betas=(0.9, 0.999))
    lr_optimizer = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0, last_epoch=-1)

    for ii in range(startepoch,endepoch):
        lr_optimizer.step()
        loss_ = 0
        for idx,(noi,gt) in enumerate(traindataloader):
            noi = Variable(noi).cuda()
            gt = Variable(gt).cuda()

            denoi = model(noi)
            loss = Loss(denoi,gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_ = loss_+loss.item()

        print(loss_)

        # if ii % 100 ==0:
        #     torch.save(model.state_dict(),'./model_param/'+str(ii)+'rid_dense_beta59.pth')
        # torch.save(model.state_dict(), './model_param/' + str(ii) + 'rid_dense_beta59.pth')

rootp = '/home/xss/Project/Dahua/Denoisy/dataset/SIDD_Small_Raw_Only/Data/'
for i in range(10):
    if i == 9:
        patch_size_ = 192
        batch_size_ = 8
    else:
        patch_size_ = 512
        batch_size_ = 1
    # patch_size_ = 32
    # batch_size_ = 1
    dataset = load_sidd.MyTrainData(rootp, patch_size = patch_size_,debug_ = False)
    traindataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_, shuffle=True, num_workers=4, pin_memory = True)
    if i%4 == 0:
        lr_ = 0.0001
    elif i>5:
        lr_ = 0.00001
    train(i*100,(i+1)*100+1,lr_)
    del dataset
    del traindataloader
