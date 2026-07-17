"""
result_generator.py

Generate evaluation metrics for the complete validation dataset.
"""

from pathlib import Path
import csv

from src.evaluation.image_evaluator import ImageEvaluator
from src.evaluation.metrics import evaluate_images


class ResultGenerator:

    def __init__(self):

        self.evaluator = ImageEvaluator()

        self.evaluator.load_checkpoint()

        self.output_dir = Path("results")

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def evaluate_dataset(self):

        rows = []

        psnr_total = 0.0
        ssim_total = 0.0
        lpips_total = 0.0

        dataset = self.evaluator.dataset

        for image_index in range(len(dataset)):

            sample = dataset[image_index]

            prediction = self.evaluator.render_prediction(
                image_index=image_index,
            )

            metrics = evaluate_images(
                prediction,
                sample["image"],
            )

            rows.append([
                image_index,
                metrics["psnr"],
                metrics["ssim"],
                metrics["lpips"],
            ])

            psnr_total += metrics["psnr"]
            ssim_total += metrics["ssim"]
            lpips_total += metrics["lpips"]

            print(
                f"Image {image_index:03d} "
                f"PSNR={metrics['psnr']:.2f} "
                f"SSIM={metrics['ssim']:.4f} "
                f"LPIPS={metrics['lpips']:.4f}"
            )

        count = len(dataset)

        rows.append([
            "Average",
            psnr_total / count,
            ssim_total / count,
            lpips_total / count,
        ])

        csv_path = self.output_dir / "evaluation_results.csv"

        with open(
            csv_path,
            "w",
            newline="",
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                "Image",
                "PSNR",
                "SSIM",
                "LPIPS",
            ])

            writer.writerows(rows)

        return csv_path