import os
import json
import csv
import argparse

parser = argparse.ArgumentParser(description="Extract SSIM and PSNR from JSON files and save to CSV.")
parser.add_argument("root_dir", type=str, help="Path to the root directory containing subfolders with JSON files.")
args = parser.parse_args()

root_dir = args.root_dir
output_csv = os.path.join(root_dir, "res_all.csv")

headers = ["Folder", "SSIM-b", "PSNR-b", "SSIM-w", "PSNR-w"]
data = []

for folder_name in [str(i) for i in range(1, 10)] + ["mergeRes"]:
    folder_path = os.path.join(root_dir, folder_name)
    if os.path.isdir(folder_path):
        folder_data = {"Folder": folder_name, "SSIM-b": "NaN", "PSNR-b": "NaN", "SSIM-w": "NaN", "PSNR-w": "NaN"}

        json_black_path = os.path.join(folder_path, 'results-b.json')
        with open(json_black_path, "r") as f:
            json_data = json.load(f)
            metrics = json_data.get("ours_35000", {})

            ssim = metrics.get("SSIM", "NaN")
            psnr = metrics.get("PSNR", "NaN")

            if isinstance(ssim, (int, float)):
                folder_data[f"SSIM-b"] = round(ssim, 2)
            if isinstance(psnr, (int, float)):
                folder_data[f"PSNR-b"] = round(psnr, 2)

        json_white_path = os.path.join(folder_path, 'results-w.json')
        with open(json_white_path, "r") as f:
            json_data = json.load(f)
            metrics = json_data.get("ours_35000", {})

            ssim = metrics.get("SSIM", "NaN")
            psnr = metrics.get("PSNR", "NaN")

            if isinstance(ssim, (int, float)):
                folder_data[f"SSIM-w"] = round(ssim, 2)
            if isinstance(psnr, (int, float)):
                folder_data[f"PSNR-w"] = round(psnr, 2)

        data.append(folder_data)

with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(data)

print(f"Data saved to {output_csv}")
