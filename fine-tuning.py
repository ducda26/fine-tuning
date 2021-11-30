from lib import *
from image_transform import ImageTransform
from dataset import *
from config import *
from utils import make_datapath_list, train_model, params_to_update, load_model

def main():
    train_list = make_datapath_list("train")
    val_list = make_datapath_list("val")

    train_dataset = MyDataset(train_list, transform=ImageTransform(
        resize, mean, std), phase="train")
    val_dataset = MyDataset(val_list, transform=ImageTransform(
        resize, mean, std), phase="val")

    # Buoc 3: Tao Dataloader

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size, shuffle=False)

    # Tạo từ điển để lưu hai thông tin này, sau này sử dụng dễ dàng hơn
    dataloader_dict = {"train": train_dataloader, "val": val_dataloader}


    # Buoc 4: Xay dung mang network
    use_pretrained = True
    net = models.vgg16(pretrained=use_pretrained)
    # chi can phan loai ong va kien ---> 2
    net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

    # Buoc 5: Tao ham loss + optimizer
    # Classify thuong dung ham CrossEntropy

    criterior = nn.CrossEntropyLoss()

    # params_to_update = []

    # update_params_name = ["classifier.6.weight", "classifier.6.bias"]

    # for name, param in net.named_parameters():
    #     if name in update_params_name:
    #         param.requires_grad = True
    #         params_to_update.append(param)
    #     else:
    #         param.requires_grad = False
    params1, params2, params3 = params_to_update(net)
    optimizer = optim.SGD([
        {'params': params1, 'lr': 1e-4}, 
        {'params': params2, 'lr': 5e-4},
        {'params': params3, 'lr': 1e-3}, 
    ], momentum=0.9) #lớp gần input để lr thấp thôi, gần cuối thì để cao

    # Buoc 6 Training
                    
    # training
    train_model(net, dataloader_dict, criterior, optimizer, num_epochs) 

if __name__ == "__main__":
    # training
    main()
    
    #Load model
    # network
    # use_pretrained = True
    # net = models.vgg16(pretrained=use_pretrained)
    # net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

    # load_model(net, save_path)