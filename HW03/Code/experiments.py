"""
Every experiment in HW03, defined once. Each variant changes exactly one factor
relative to its baseline (one-factor-at-a-time design) so differences can be
attributed to that factor.

Epoch budgets are sized for a Google Colab T4 GPU (the full set of 10
experiments takes roughly 1-1.5 hours there). On the local CPU the same
budgets would take many hours (~60-80 s/epoch Plain CNN, ~5 min/epoch ResNet-18).
"""

PLAIN_EPOCHS = 30
RESNET_EPOCHS = 15

_plain = dict(arch='plain', epochs=PLAIN_EPOCHS, lr=1e-3, batch_size=128, dropout=0.3, augment=False)
_resnet = dict(arch='resnet18', epochs=RESNET_EPOCHS, lr=1e-4, batch_size=128,
               pretrained=True, freeze_backbone=False, input_size=64, augment=False)

PLAIN_BASELINE = dict(_plain, name='plain_baseline')
RESNET_BASELINE = dict(_resnet, name='resnet18_finetune_lr1e-4')

# Quiz 2 Advanced: Plain CNN hyperparameters (learning rate, regularisation)
PLAIN_HPARAM = [
    PLAIN_BASELINE,
    dict(_plain, name='plain_lr1e-2', lr=1e-2),
    dict(_plain, name='plain_lr1e-4', lr=1e-4),
    dict(_plain, name='plain_dropout0', dropout=0.0),
]

# Quiz 2 Advanced: ResNet-18 transfer-learning strategies
RESNET_HPARAM = [
    RESNET_BASELINE,                                                          # pretrained, full fine-tune, lr 1e-4
    dict(_resnet, name='resnet18_finetune_lr1e-3', lr=1e-3),                  # vs baseline: learning rate
    dict(_resnet, name='resnet18_frozen_lr1e-3', lr=1e-3, freeze_backbone=True),  # vs lr1e-3: freeze backbone
    dict(_resnet, name='resnet18_scratch_lr1e-3', lr=1e-3, pretrained=False),     # vs lr1e-3: no pretraining
]

# Quiz 3 Basic: data augmentation (baseline vs baseline + augmentation)
PLAIN_AUG = dict(PLAIN_BASELINE, name='plain_augment', augment=True)
RESNET_AUG = dict(RESNET_BASELINE, name='resnet18_finetune_lr1e-4_augment', augment=True)

ALL_EXPERIMENTS = [PLAIN_BASELINE, RESNET_BASELINE] + PLAIN_HPARAM[1:] + [PLAIN_AUG] + RESNET_HPARAM[1:] + [RESNET_AUG]


def run_all(data, ckpt_dir):
    from .train import train_model
    for cfg in ALL_EXPERIMENTS:
        train_model(cfg, data, ckpt_dir)
