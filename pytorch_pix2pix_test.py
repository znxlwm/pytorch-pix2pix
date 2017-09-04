import torch, network, argparse, os
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.autograd import Variable
import util

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='facades',  help='')
parser.add_argument('--test_subfolder', required=False, default='val',  help='')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--input_size', type=int, default=256, help='input size')
parser.add_argument('--save_root', required=False, default='results', help='results save path')
parser.add_argument('--inverse_order', type=bool, default=True, help='0: [input, target], 1 - [target, input]')
opt = parser.parse_args()
print(opt)

# data_loader
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
test_loader = util.data_load('data/' + opt.dataset, opt.test_subfolder, transform, batch_size=1, shuffle=False)

if not os.path.isdir(opt.dataset + '_results/test_results'):
    os.mkdir(opt.dataset + '_results/test_results')

G = network.generator(opt.ngf)
G.cuda()
G.load_state_dict(torch.load(opt.dataset + '_results/' + opt.dataset + '_generator_param.pkl'))

# network
n = 0
print('test start!')
for x_, _ in test_loader:
    if opt.inverse_order:
        y_ = x_[:, :, :, :x_.size()[2]]
        x_ = x_[:, :, :, x_.size()[2]:]
    else:
        y_ = x_[:, :, :, x_.size()[2]:]
        x_ = x_[:, :, :, :x_.size()[2]]

    if x_.size()[2] != opt.input_size:
        x_ = util.imgs_resize(x_, opt.input_size)
        y_ = util.imgs_resize(y_, opt.input_size)

    x_ = Variable(x_.cuda(), volatile=True)
    test_image = G(x_)
    s = test_loader.dataset.imgs[n][0][::-1]
    s_ind = len(s) - s.find('/')
    e_ind = len(s) - s.find('.')
    ind = test_loader.dataset.imgs[n][0][s_ind:e_ind-1]
    path = opt.dataset + '_results/test_results/' + ind + '_input.png'
    plt.imsave(path, (x_[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
    path = opt.dataset + '_results/test_results/' + ind + '_output.png'
    plt.imsave(path, (test_image[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
    path = opt.dataset + '_results/test_results/' + ind + '_target.png'
    plt.imsave(path, (y_[0].numpy().transpose(1, 2, 0) + 1) / 2)

    n += 1

print('%d images generation complete!' % n)
