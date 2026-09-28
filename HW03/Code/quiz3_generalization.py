"""
Quiz 3: data augmentation for both CNNs, kernel visualisation and Grad-CAM.
"""

import os

import numpy as np
import pandas as pd

from .data import CLASS_NAMES
from .evaluation import evaluate, plot_histories, experiment_row, save_fig_path
from .experiments import PLAIN_BASELINE, RESNET_BASELINE, PLAIN_AUG, RESNET_AUG
from .train import train_model
from .xai import plot_kernels, pick_gradcam_samples, plot_gradcam


def run_quiz3(data, ckpt_dir, output_dir):
    y_test = data['y_test'].numpy()

    print("=" * 70)
    print("QUIZ 3 BASIC: DATA AUGMENTATION")
    print("=" * 70)
    rows, figs, models, results = [], {}, {}, {}
    for arch, base_cfg, aug_cfg in [('plain', PLAIN_BASELINE, PLAIN_AUG), ('resnet', RESNET_BASELINE, RESNET_AUG)]:
        hists = []
        for cfg in (base_cfg, aug_cfg):
            model, hist = train_model(cfg, data, ckpt_dir)
            res = evaluate(model, data['x_test'], y_test)
            row = experiment_row(cfg['name'], hist, res)
            row['train_val_acc_gap'] = round(hist['train_acc'][-1] - hist['val_acc'][-1], 4)
            rows.append(row)
            hists.append(hist)
            models[cfg['name']] = model
            results[cfg['name']] = res
        figs[f'{arch}_aug'] = plot_histories(
            hists, ['no augmentation', 'with augmentation'],
            save_fig_path(output_dir, f'quiz3_{arch}_augmentation_curves.png'),
            f"Quiz 3: {'Plain CNN' if arch == 'plain' else 'ResNet-18'} -- Before vs After Data Augmentation")
    aug_tab = pd.DataFrame(rows)
    aug_tab.to_csv(os.path.join(output_dir, 'quiz3_augmentation_results.csv'), index=False)
    print(aug_tab.to_string(index=False))

    print("\n" + "=" * 70)
    print("QUIZ 3 ADVANCED: KERNEL VISUALISATION & GRAD-CAM")
    print("=" * 70)
    plain = models[PLAIN_BASELINE['name']]
    resnet = models[RESNET_BASELINE['name']]

    # Kernel image: first test image that the ResNet classifies correctly with the highest confidence
    res_r = results[RESNET_BASELINE['name']]
    correct = np.where(res_r['pred'] == y_test)[0]
    kernel_img_idx = int(correct[np.argmax(res_r['probs'][correct].max(1))])
    figs['kernels'], kernel_info = plot_kernels(
        resnet, data['x_test'][kernel_img_idx], save_fig_path(output_dir, 'quiz3_kernel_visualization.png'),
        'ResNet-18 conv1 after fine-tuning on CIFAR-10', x_ref=data['x_val'][:1000])
    kernel_info['image_index'] = kernel_img_idx
    kernel_info['image_class'] = CLASS_NAMES[y_test[kernel_img_idx]]
    print(f"Kernel A #{kernel_info['edge_idx']}: {kernel_info['edge_stats']}")
    print(f"Kernel B #{kernel_info['colour_idx']}: {kernel_info['colour_stats']}")

    correct_idx, wrong_idx = pick_gradcam_samples(y_test, results[PLAIN_BASELINE['name']]['pred'], res_r['pred'])
    figs['gradcam'], cam_records = plot_gradcam(
        [plain, resnet], ['Plain CNN', 'ResNet-18'], data['x_test'], y_test, correct_idx + wrong_idx,
        save_fig_path(output_dir, 'quiz3_gradcam.png'))
    cam_tab = pd.DataFrame(cam_records)
    cam_tab.to_csv(os.path.join(output_dir, 'quiz3_gradcam_records.csv'), index=False)
    cam_summary = cam_tab.groupby(['model', 'correct'])['centre_mass'].mean().round(3).reset_index()
    print(cam_tab.to_string(index=False))
    print("\nMean share of Grad-CAM mass in the central 16x16 region (uniform map = 0.25):")
    print(cam_summary.to_string(index=False))

    return {'aug_table': aug_tab, 'kernel_info': kernel_info, 'gradcam_records': cam_tab,
            'gradcam_summary': cam_summary, 'figures': figs}
