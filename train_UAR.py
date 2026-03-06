import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

ABSENCE_THRESHOLD = 0.3

# ================= Dataset ==================
class UARDataset(Dataset):
    def __init__(self, rgb_paths, ir_paths, nir_paths, img_size=640, transform=None,
                 miss_prob=0.3):
        """
        miss_prob: 每个模态缺失的概率
        """
        self.rgb_paths = rgb_paths
        self.ir_paths = ir_paths
        self.nir_paths = nir_paths
        self.img_size = img_size
        self.miss_prob = miss_prob

        self.transform = transform or T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.rgb_paths)

    def random_absence(self):
        """
        随机生成缺失标签 [3]
        至少保留一个模态
        """
        while True:
            abs_flag = torch.bernoulli(
                torch.full((3,), self.miss_prob)
            )
            if abs_flag.sum() < 3:
                return abs_flag

    def __getitem__(self, idx):
        rgb = Image.open(self.rgb_paths[idx]).convert('RGB')
        ir  = Image.open(self.ir_paths[idx]).convert('RGB')
        nir = Image.open(self.nir_paths[idx]).convert('RGB')

        abs_flag = self.random_absence()  # [3]

        # 缺失模态置为全 0
        if abs_flag[0] == 1:
            rgb = Image.new('RGB', rgb.size)
        if abs_flag[1] == 1:
            ir = Image.new('RGB', ir.size)
        if abs_flag[2] == 1:
            nir = Image.new('RGB', nir.size)

        rgb = self.transform(rgb)
        ir  = self.transform(ir)
        nir = self.transform(nir)

        imgs = torch.cat([rgb, ir, nir], dim=0)  # [9, H, W]

        return imgs, abs_flag


def create_uar_dataloader(rgb_paths, ir_paths, nir_paths,
                          batch_size=4, img_size=640,
                          shuffle=True, num_workers=0,
                          miss_prob=0.3):

    dataset = UARDataset(
        rgb_paths, ir_paths, nir_paths,
        img_size=img_size,
        miss_prob=miss_prob
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return dataloader, dataset


# ================= UncertainRouter ==================
class UncertainRouter(nn.Module):
    def __init__(self, input_channels=9):
        super(UncertainRouter, self).__init__()
        # 简单示例，可换成你原来的网络结构
        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(64, 3)

    def forward(self, rgb, ir, nir):
        x = torch.cat([rgb, ir, nir], dim=1)  # [B,9,H,W]
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.fc(x))  # 输出0-1概率
        return x


# ================= Training ==================
def train_one_epoch(router, dataloader, optimizer, device, log_interval=20):
    router.train()
    total_loss = 0.0

    for i, batch in enumerate(tqdm(dataloader, desc='Training UAR')):
        imgs, absences = batch
        imgs = imgs.to(device)
        absences = absences.to(device)

        rgb = imgs[:, 0:3]
        ir  = imgs[:, 3:6]
        nir = imgs[:, 6:9]

        pred = router(rgb, ir, nir)
        loss = F.binary_cross_entropy(pred, absences)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

         # ===== 阈值判断 =====
        pred_bin = (pred >= ABSENCE_THRESHOLD).float()

        # ===== 打印 =====
        if i % log_interval == 0:
            print(f"\n[Batch {i}]")
            print("GT absences      :", absences[0].cpu().numpy())
            print("Pred prob        :", pred[0].detach().cpu().numpy())
            print("Pred bin (>=0.3) :", pred_bin[0].cpu().numpy())
            print("Loss             :", loss.item())
    return total_loss / len(dataloader)


def train(router, dataloader, device, epochs=50, lr=1e-3, save_dir='weights'):
    optimizer = torch.optim.Adam(router.parameters(), lr=lr)
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        loss = train_one_epoch(router, dataloader, optimizer, device, log_interval=20)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
        # 每5个epoch保存一次
        if (epoch+1) % 5 == 0 or (epoch+1) == epochs:
            torch.save(router.state_dict(), os.path.join(save_dir, f'router_epoch{epoch+1}.pth'))
    # 保存最终模型
    torch.save(router.state_dict(), os.path.join(save_dir, 'router_final.pth'))

def load_image_paths(dir_path):
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    return sorted([
        str(p) for p in Path(dir_path).iterdir()
        if p.suffix.lower() in exts
    ])

# ================= Main ==================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-rgb', type=str, required=True)
    parser.add_argument('--train-ir', type=str, required=True)
    parser.add_argument('--train-nir', type=str, required=True)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-dir', type=str, default='weights')
    opt = parser.parse_args()

    rgb_paths = load_image_paths(opt.train_rgb)
    ir_paths  = load_image_paths(opt.train_ir)
    nir_paths = load_image_paths(opt.train_nir)

    assert len(rgb_paths) == len(ir_paths) == len(nir_paths), "RGB / IR / NIR image counts must match"

    dataloader, dataset = create_uar_dataloader(
        rgb_paths,
        ir_paths,
        nir_paths,
        batch_size=opt.batch_size,
        img_size=opt.img_size,
        shuffle=True,
        num_workers=4,
        miss_prob=0.3)


    device = torch.device(opt.device)
    router = UncertainRouter().to(device)
# python train_UAR.py \
#     --train-rgb "/home/bq/data/datasets/DRNT_Our/VIS/test" \
#     --train-ir "/home/bq/data/datasets/DRNT_Our/TIR/test" \
#     --train-nir "/home/bq/data/datasets/DRNT_Our/NIR/test" \
#     --batch-size 4 \
#     --epochs 50
    train(router, dataloader, device, epochs=opt.epochs, lr=opt.lr, save_dir=opt.save_dir)
