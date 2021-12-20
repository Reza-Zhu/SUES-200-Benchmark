import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from netvlad import NetVLAD, TripletNet, EmbedNet
from tripleloss import HardTripletLoss


encoder = models.resnet18(pretrained=True)
base_model = nn.Sequential(
    encoder.conv1,
    encoder.bn1,
    encoder.relu,
    encoder.maxpool,
    encoder.layer1,
    encoder.layer2,
    encoder.layer3,
    encoder.layer4,
)

dim = list(base_model.parameters())[-1].shape[0]

netVLAD = NetVLAD(num_clusters=89, dim=dim, alpha=1.0)
model = EmbedNet(base_model, netVLAD).cuda()

criterion = HardTripletLoss(margin=0.1).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
epochs = 50

trans_train_list = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# height = get_yaml_value("height")

save_path = "./save_model_weight"
height = "150"
classes = 89
data_path = "/media/data1/Datasets"

# print(os.path.join(train_data_path, "gallery_satellite"))

train_data_path = data_path+"/Training/{}".format(height)
test_data_path = data_path + "/Testing/{}".format(height)

drone_train_datasets = datasets.ImageFolder(os.path.join(train_data_path, "drone"),
                                            transform=trans_train_list)
satellite_test_datasets = datasets.ImageFolder(os.path.join(test_data_path, "gallery_satellite"),
                                               transform=trans_train_list)

training_data_loader = torch.utils.data.DataLoader(drone_train_datasets,
                                                   batch_size=32,
                                                   shuffle=True,
                                                   num_workers=8)

testing_data_loader = torch.utils.data.DataLoader(satellite_test_datasets,
                                                  batch_size=8,
                                                  shuffle=False,
                                                  num_workers=8)

print("<<<<<<<<<Training Start>>>>>>>>>>>>")
for epoch in range(epochs):
    running_loss = 0
    for batch_idx, (train_img, train_label) in enumerate(training_data_loader):
        out_train = model(train_img.cuda())
        triplet_loss = criterion(out_train, train_label.cuda())
        optimizer.zero_grad()
        triplet_loss.backward()
        optimizer.step()
        running_loss += triplet_loss.item()
    print('epoch : {}, triplet_loss : {}'.format(epoch, running_loss / classes))
    mode_save_name = "model_{:02d}.pt".format(epoch)
    torch.save(model.state_dict(), os.path.join(save_path, mode_save_name))
