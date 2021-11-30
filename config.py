from lib import *

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

# Anh da qua transform
resize = 224
mean = (0.485, 0.456, 0.406)  # search google: mean std imagenet
std = (0.229, 0.224, 0.225)

num_epochs = 2
batch_size = 4

save_path = ".\weight_fine_tuning.pth"