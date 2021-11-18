import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from vit_pytorch import ViT, MAE
from torchvision import datasets, transforms
from torch.optim import lr_scheduler

batch_size = 512
data_path = '/media/data1/imagenet-mini/train'
data_transform = [
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(256),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

]
train_datasets = datasets.ImageFolder(data_path,
                                      transform=transforms.Compose(data_transform))
len_train = len(train_datasets)
training_loader = torch.utils.data.DataLoader(train_datasets,
                                              batch_size=batch_size,
                                              # shuffle=True,
                                              num_workers=8,  # 多进程
                                              pin_memory=True)  # 锁页内存

v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048
)

mae = MAE(
    encoder = v,
    masking_ratio = 0.75,   # the paper recommended 75% masked patches
    decoder_dim = 512,      # paper showed good results with just 512
    decoder_depth = 6       # anywhere from 1 to 8
)

# training
if torch.cuda.is_available():
    device = torch.device("cuda:0")
cudnn.benchmark = True
mae = mae.cuda()
num_epochs = 100
lr_base = 0.00015
lr = lr_base * batch_size / 256

optimizer = optim.AdamW(mae.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,20)

for epoch in range(num_epochs):
    running_loss = 0
    count = 0
    for img, label in training_loader:
        count += 1
        img = img.cuda()
        optimizer.zero_grad()

        loss = mae(img)
        loss.backward()
        optimizer.step()

        running_loss = loss.item()
        print("[Epoch: {} | Iter: {}/{}] | Loss: {:.8f}".format(epoch + 1, count, int(len_train/batch_size)+1,
                                                                running_loss / batch_size))
    scheduler.step()
    if (epoch + 1) % 5 == 0:
        torch.save(v.state_dict(), './weights/trained-vit-%d.pt' % epoch)
        torch.save(mae.state_dict(), './weights/trained-mae-%d.pt' % epoch)

    # epoch_loss = running_loss/1000

