"""
Quiz 2: Plain CNN vs classic CNN backbone (ResNet-18) -- training, Top-1/Top-5,
test-set predictions, per-class ROC / Macro-AUC, parameter counts, and
hyperparameter / transfer-learning experiments.
"""

import os

import pandas as pd

from .evaluation import (evaluate, prediction_table, per_class_table, plot_confusion_matrices,
                         plot_sample_predictions, plot_roc_curves, plot_histories, experiment_row,
                         save_fig_path)
from .experiments import PLAIN_BASELINE, RESNET_BASELINE, PLAIN_HPARAM, RESNET_HPARAM
from .train import train_model

PLAIN_TITLE = 'Plain CNN'
RESNET_TITLE = 'ResNet-18 (ImageNet-pretrained, fine-tuned)'


def _run_group(cfgs, data, ckpt_dir):
    rows, hists, models = [], [], {}
    for cfg in cfgs:
        model, hist = train_model(cfg, data, ckpt_dir)
        res = evaluate(model, data['x_test'], data['y_test'])
        rows.append(experiment_row(cfg['name'], hist, res))
        hists.append(hist)
        models[cfg['name']] = model
    return pd.DataFrame(rows), hists, models


def run_quiz2(data, ckpt_dir, output_dir):
    y_test = data['y_test'].numpy()

    print("=" * 70)
    print("QUIZ 2 BASIC: PLAIN CNN vs RESNET-18")
    print("=" * 70)
    plain, plain_hist = train_model(PLAIN_BASELINE, data, ckpt_dir)
    resnet, resnet_hist = train_model(RESNET_BASELINE, data, ckpt_dir)
    plain_res = evaluate(plain, data['x_test'], y_test)
    resnet_res = evaluate(resnet, data['x_test'], y_test)

    plain_pred = prediction_table(plain_res, y_test, os.path.join(output_dir, 'quiz2_plain_cnn_test_predictions.csv'))
    resnet_pred = prediction_table(resnet_res, y_test, os.path.join(output_dir, 'quiz2_resnet18_test_predictions.csv'))
    plain_cls = per_class_table(plain_res, y_test)
    resnet_cls = per_class_table(resnet_res, y_test)

    comparison = pd.DataFrame([
        {'model': PLAIN_TITLE, **{k: v for k, v in experiment_row(PLAIN_BASELINE['name'], plain_hist, plain_res).items()
                                 if k not in ('experiment',)}, 'total_params': plain_hist['total_params']},
        {'model': RESNET_TITLE, **{k: v for k, v in experiment_row(RESNET_BASELINE['name'], resnet_hist, resnet_res).items()
                                  if k not in ('experiment',)}, 'total_params': resnet_hist['total_params']},
    ])
    comparison.to_csv(os.path.join(output_dir, 'quiz2_model_comparison.csv'), index=False)
    print(comparison[['model', 'total_params', 'test_top1', 'test_top5', 'test_macro_auc', 'train_minutes']].to_string(index=False))

    figs = {
        'curves': plot_histories([plain_hist, resnet_hist], [PLAIN_TITLE, 'ResNet-18'],
                                 save_fig_path(output_dir, 'quiz2_baseline_training_curves.png'),
                                 'Quiz 2: Baseline Training Curves (solid = train, dashed = validation)'),
        'confusion': plot_confusion_matrices([plain_res, resnet_res], y_test, [PLAIN_TITLE, RESNET_TITLE],
                                             save_fig_path(output_dir, 'quiz2_confusion_matrices.png')),
        'samples': plot_sample_predictions(data['x_test'], y_test, [plain_res, resnet_res], [PLAIN_TITLE, 'ResNet-18'],
                                           save_fig_path(output_dir, 'quiz2_sample_predictions.png')),
        'roc': plot_roc_curves([plain_res, resnet_res], y_test, [PLAIN_TITLE, RESNET_TITLE],
                               save_fig_path(output_dir, 'quiz2_roc_curves.png')),
    }

    print("\n" + "=" * 70)
    print("QUIZ 2 ADVANCED: HYPERPARAMETER / TRANSFER-LEARNING EXPERIMENTS")
    print("=" * 70)
    plain_tab, plain_hists, _ = _run_group(PLAIN_HPARAM, data, ckpt_dir)
    resnet_tab, resnet_hists, _ = _run_group(RESNET_HPARAM, data, ckpt_dir)
    plain_tab.to_csv(os.path.join(output_dir, 'quiz2_plain_hparam_results.csv'), index=False)
    resnet_tab.to_csv(os.path.join(output_dir, 'quiz2_resnet_hparam_results.csv'), index=False)
    print(plain_tab.to_string(index=False))
    print(resnet_tab.to_string(index=False))
    figs['plain_hparam'] = plot_histories(plain_hists, [c['name'] for c in PLAIN_HPARAM],
                                          save_fig_path(output_dir, 'quiz2_plain_hparam_curves.png'),
                                          'Quiz 2 Advanced: Plain CNN Hyperparameter Experiments')
    figs['resnet_hparam'] = plot_histories(resnet_hists, [c['name'] for c in RESNET_HPARAM],
                                           save_fig_path(output_dir, 'quiz2_resnet_hparam_curves.png'),
                                           'Quiz 2 Advanced: ResNet-18 Transfer-Learning Experiments')

    return {'plain': plain, 'resnet': resnet, 'plain_res': plain_res, 'resnet_res': resnet_res,
            'plain_hist': plain_hist, 'resnet_hist': resnet_hist,
            'plain_pred': plain_pred, 'resnet_pred': resnet_pred,
            'plain_per_class': plain_cls, 'resnet_per_class': resnet_cls,
            'comparison': comparison, 'plain_hparam': plain_tab, 'resnet_hparam': resnet_tab, 'figures': figs}
