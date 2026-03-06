from train_UAR import UncertainRouter
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
class UARTestDataset(Dataset):
    def __init__(self, rgb_paths, ir_paths, nir_paths,
                 img_size=640, transform=None,
                 miss_probs=(0.5, 0.5)):
        """
        miss_probs:
            (p_one_missing, p_two_missing)
        """
        self.rgb_paths = rgb_paths
        self.ir_paths = ir_paths
        self.nir_paths = nir_paths
        self.miss_probs = miss_probs

        self.transform = transform or T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])

        self.combinations_1 = [[0], [1], [2]]
        self.combinations_2 = [[0, 1], [0, 2], [1, 2]]

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb = Image.open(self.rgb_paths[idx]).convert('RGB')
        ir  = Image.open(self.ir_paths[idx]).convert('RGB')
        nir = Image.open(self.nir_paths[idx]).convert('RGB')

        # ===== 决定缺 1 个还是 2 个 =====
        if torch.rand(1).item() < self.miss_probs[0]:
            miss_ids = self.combinations_1[
                torch.randint(0, 3, (1,)).item()
            ]
        else:
            miss_ids = self.combinations_2[
                torch.randint(0, 3, (1,)).item()
            ]

        abs_flag = torch.zeros(3)
        for m in miss_ids:
            abs_flag[m] = 1

        # ===== 置 0 =====
        if 0 in miss_ids:
            rgb = Image.new('RGB', rgb.size)
        if 1 in miss_ids:
            ir = Image.new('RGB', ir.size)
        if 2 in miss_ids:
            nir = Image.new('RGB', nir.size)

        rgb = self.transform(rgb)
        ir  = self.transform(ir)
        nir = self.transform(nir)

        imgs = torch.cat([rgb, ir, nir], dim=0)

        return imgs, abs_flag, abs_flag.clone()


def test_uar(router, dataloader, device, threshold=0.3):
    router.eval()

    total = 0
    correct_all = 0
    correct_per_modal = torch.zeros(3)
    total_per_modal = torch.zeros(3)

    with torch.no_grad():
        for imgs, gt, miss_idx in tqdm(dataloader, desc="Testing UAR"):
            imgs = imgs.to(device)
            gt = gt.to(device)

            rgb = imgs[:, 0:3]
            ir  = imgs[:, 3:6]
            nir = imgs[:, 6:9]

            pred = router(rgb, ir, nir)
            pred_bin = (pred >= threshold).float()

            total += imgs.size(0)

            # ===== 整体样本准确率（3 个都对）=====
            correct_all += (pred_bin == gt).all(dim=1).sum().item()

            # ===== 逐模态准确率 =====
            for m in range(3):
                correct_per_modal[m] += (pred_bin[:, m] == gt[:, m]).sum().item()
                total_per_modal[m] += gt.size(0)

            # ===== 打印错误样本 =====
            for i in range(imgs.size(0)):
                if not torch.equal(pred_bin[i], gt[i]):
                    miss_name = ["RGB", "IR", "NIR"][miss_idx[i]]
                    print("\n[Wrong Prediction]")
                    print("Missing modality:", miss_name)
                    print("GT       :", gt[i].cpu().numpy())
                    print("Pred prob:", pred[i].cpu().numpy())
                    print("Pred bin :", pred_bin[i].cpu().numpy())

    print("\n========== Test Result ==========")
    print(f"Overall accuracy: {correct_all / total:.4f}")
    print(f"RGB accuracy: {correct_per_modal[0] / total_per_modal[0]:.4f}")
    print(f"IR  accuracy: {correct_per_modal[1] / total_per_modal[1]:.4f}")
    print(f"NIR accuracy: {correct_per_modal[2] / total_per_modal[2]:.4f}")

def load_image_paths(dir_path):
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    return sorted([
        str(p) for p in Path(dir_path).iterdir()
        if p.suffix.lower() in exts
    ])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb', type=str, required=True)
    parser.add_argument('--ir', type=str, required=True)
    parser.add_argument('--nir', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    rgb_paths = load_image_paths(args.rgb)
    ir_paths  = load_image_paths(args.ir)
    nir_paths = load_image_paths(args.nir)

    assert len(rgb_paths) == len(ir_paths) == len(nir_paths)

    dataset = UARTestDataset(rgb_paths, ir_paths, nir_paths, img_size=args.img_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device)
    router = UncertainRouter().to(device)
    router.load_state_dict(torch.load(args.weights, map_location=device))

    test_uar(router, dataloader, device, threshold=0.3)
