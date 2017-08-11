import torch, network, argparse, os
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.autograd import Variable
import util

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='facades',  help='')
parser.add_argument('--test_subfolder', required=False, default='val',  help='')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--save_root', required=False, default='results', help='results save path')
opt = parser.parse_args()
print(opt)

# data_loader
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
test_loader = util.data_load('data/' + opt.datset, opt.test_subfolder, transform, 1)

if not os.path.isdir(opt.dataset + '_results/test'):
    os.mkdir(opt.dataset + '_results/test')

G = network.generator(opt.ngf)
G.cuda()
G.load_state_dict(torch.load(opt.dataset + '_results/' + opt.dataset + '_generator_param.pkl'))

# network
n = 0
print('test start!')
for x_, _ in test_loader:
        n += 1
        x_ = x_[:, :, :, x_.size()[2]:]
        x_ = Variable(x_.cuda(), volatile=True)
        test_image = G(x_)
        path = opt.dataset + 'results/test/' + str(n) + '.png'
        plt.imsave(path, (test_image[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

print('%d images generation complete!' % n)