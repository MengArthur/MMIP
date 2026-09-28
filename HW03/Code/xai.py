"""
Quiz 3 Advanced: kernel visualisation and Grad-CAM (Selvaraju et al., 2017).
"""

import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
if 'ipykernel' not in sys.modules:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .data import CLASS_NAMES, normalize


# ----------------------------------------------------------------------------
# Kernel visualisation
# ----------------------------------------------------------------------------
def kernel_stats(w):
    """
    w: (3, k, k) weights of one first-layer kernel.
    colour_ratio: how much the kernel treats R/G/B differently (0 = identical
                  across channels, i.e. colour-blind luminance filter).
    orientation:  dominant spatial gradient direction of the channel-averaged
                  kernel, i.e. which edge direction it responds to.
    """
    w = np.asarray(w, dtype=np.float64)
    lum = w.mean(0)
    colour_ratio = float(np.linalg.norm(w - lum) / (np.linalg.norm(w) + 1e-12))
    gy, gx = np.gradient(lum)
    energy_x, energy_y = float((gx ** 2).sum()), float((gy ** 2).sum())
    if energy_x > 1.5 * energy_y:
        orientation = 'vertical edges (intensity changes left-right)'
    elif energy_y > 1.5 * energy_x:
        orientation = 'horizontal edges (intensity changes top-bottom)'
    else:
        orientation = 'diagonal / mixed-direction structure'
    channel_mean = w.reshape(3, -1).mean(1)
    return {'colour_ratio': colour_ratio, 'orientation': orientation,
            'channel_mean_rgb': channel_mean.round(4).tolist(),
            'spatial_energy': float(np.linalg.norm(lum))}


def select_two_kernels(conv_weight, response_std, gray_max=0.3):
    """
    Picks two contrasting kernels by an explicit rule (no hand-picking):
      - edge kernel:   among near-colour-blind kernels (colour ratio < gray_max), the one
                       whose feature maps vary most across real test images (largest mean
                       spatial std) -> the luminance filter that is most active on natural
                       images. (Ranking by weight energy alone picks very high-frequency
                       kernels that barely respond to 32x32 images.)
      - colour kernel: largest colour ratio -> reacts to colour differences.
    response_std: (n_kernels,) mean spatial std of each kernel's feature map.
    """
    w = conv_weight.detach().cpu().numpy()
    stats = [kernel_stats(k) for k in w]
    colour = np.array([s['colour_ratio'] for s in stats])
    response_std = np.asarray(response_std)
    for s, r in zip(stats, response_std):
        s['mean_response_std'] = round(float(r), 4)
    edge_idx = int(np.argmax(np.where(colour < gray_max, response_std, -np.inf)))
    colour_order = np.argsort(-colour)
    colour_idx = int(colour_order[0] if colour_order[0] != edge_idx else colour_order[1])
    return edge_idx, colour_idx, stats


def _first_conv_input(model, x_uint8):
    x = normalize(x_uint8)
    if hasattr(model, 'input_size') and x.shape[-1] != model.input_size:
        x = F.interpolate(x, size=model.input_size, mode='bilinear', align_corners=False)
    return x


@torch.no_grad()
def kernel_response_std(model, x_uint8, batch_size=250, border=3):
    """Mean (over images) spatial std of every first-layer feature map; the outer
    `border` pixels are excluded so zero-padding edges do not count as responses."""
    conv = model.first_conv()
    device = conv.weight.device
    total = torch.zeros(conv.out_channels)
    for i in range(0, len(x_uint8), batch_size):
        f = conv(_first_conv_input(model, x_uint8[i:i + batch_size]).to(device)).cpu()
        total += f[:, :, border:-border, border:-border].flatten(2).std(2).sum(0)
    return (total / len(x_uint8)).numpy()


def _to_rgb(k):
    k = k.transpose(1, 2, 0)
    return (k - k.min()) / (k.max() - k.min() + 1e-12)


def plot_kernels(model, image_uint8, path, model_title, x_ref):
    """x_ref: uint8 test images used only to rank kernels by how strongly they respond."""
    conv = model.first_conv()
    device = conv.weight.device
    w = conv.weight.detach().cpu()
    edge_idx, colour_idx, stats = select_two_kernels(w, kernel_response_std(model, x_ref))

    with torch.no_grad():
        fmap = conv(_first_conv_input(model, image_uint8.unsqueeze(0)).to(device))[0].cpu()

    fig = plt.figure(figsize=(19, 7.6))
    gs = fig.add_gridspec(2, 9)

    ax_all = fig.add_subplot(gs[:, :3])
    n = w.shape[0]
    cols = 8
    rows = int(np.ceil(n / cols))
    k = w.shape[-1]
    grid = np.ones((rows * (k + 1), cols * (k + 1), 3))
    for i in range(n):
        r, c = divmod(i, cols)
        grid[r * (k + 1):r * (k + 1) + k, c * (k + 1):c * (k + 1) + k] = _to_rgb(w[i].numpy())
    ax_all.imshow(grid, interpolation='nearest')
    ax_all.set_title(f'All {n} first-layer kernels ({k}x{k}x3)\n{model_title}', fontweight='bold')
    ax_all.axis('off')

    ax_img = fig.add_subplot(gs[:, 3])
    ax_img.imshow(image_uint8.permute(1, 2, 0).numpy())
    ax_img.set_title('Input image', fontweight='bold'); ax_img.axis('off')

    for row, (idx, label) in enumerate([(edge_idx, 'Kernel A (edge rule)'), (colour_idx, 'Kernel B (colour rule)')]):
        kw = w[idx].numpy()
        ax = fig.add_subplot(gs[row, 4]); ax.imshow(_to_rgb(kw), interpolation='nearest')
        ax.set_title(f'{label}\n#{idx} as RGB', fontsize=9, fontweight='bold'); ax.axis('off')
        lim = np.abs(kw).max()
        for ch, cname in enumerate(['R', 'G', 'B']):
            ax = fig.add_subplot(gs[row, 5 + ch])
            ax.imshow(kw[ch], cmap='bwr', vmin=-lim, vmax=lim, interpolation='nearest')
            ax.set_title(f'{cname} weights', fontsize=9); ax.axis('off')
        ax = fig.add_subplot(gs[row, 8])
        ax.imshow(fmap[idx].numpy(), cmap='viridis')
        ax.set_title(f'#{idx} feature map', fontsize=9); ax.axis('off')

    fig.suptitle('Quiz 3 Advanced: Two Trained Kernels and Their Feature-Map Responses '
                 '(blue = negative weight, red = positive weight)', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150, bbox_inches='tight')
    return fig, {'edge_idx': edge_idx, 'colour_idx': colour_idx,
                 'edge_stats': stats[edge_idx], 'colour_stats': stats[colour_idx]}


# ----------------------------------------------------------------------------
# Grad-CAM
# ----------------------------------------------------------------------------
def grad_cam(model, image_uint8, layer, target_class=None):
    """
    Grad-CAM: alpha_k = mean over (i, j) of dy_c / dA_k,  CAM = ReLU(sum_k alpha_k * A_k),
    upsampled to 32x32 and normalised to [0, 1].
    """
    acts, grads = {}, {}
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__('a', o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__('g', go[0]))
    model.eval()
    x = normalize(image_uint8.unsqueeze(0)).to(next(model.parameters()).device).requires_grad_(True)
    logits = model(x)
    probs = torch.softmax(logits, 1)[0]
    cls = int(probs.argmax().detach().item()) if target_class is None else target_class
    model.zero_grad()
    logits[0, cls].backward()
    h1.remove(); h2.remove()

    a, g = acts['a'][0].detach(), grads['g'][0].detach()
    weights = g.mean(dim=(1, 2))
    cam = F.relu((weights.view(-1, 1, 1) * a).sum(0))
    cam = F.interpolate(cam[None, None], size=32, mode='bilinear', align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-12)
    return cam.cpu().numpy(), cls, float(probs[cls].detach().item()), tuple(a.shape[1:])


def centre_mass(cam):
    """Fraction of Grad-CAM mass inside the central 16x16 region (25% of the image area)."""
    return float(cam[8:24, 8:24].sum() / (cam.sum() + 1e-12))


def pick_gradcam_samples(y, preds_a, preds_b, n_correct=6, n_wrong=4, seed=0):
    """Images both models classify correctly (one per class, first n_correct classes)
    plus images both models get wrong -- selected by rule, not by hand."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    correct = []
    for c in range(len(CLASS_NAMES)):
        cand = np.where((y == c) & (preds_a == c) & (preds_b == c))[0]
        if len(cand):
            correct.append(int(rng.choice(cand)))
        if len(correct) == n_correct:
            break
    wrong_cand = np.where((preds_a != y) & (preds_b != y))[0]
    wrong = [int(i) for i in rng.choice(wrong_cand, min(n_wrong, len(wrong_cand)), replace=False)]
    return correct, wrong


def plot_gradcam(models, titles, x_test, y, indices, path):
    y = np.asarray(y)
    fig, axes = plt.subplots(1 + len(models), len(indices), figsize=(1.9 * len(indices), 2.3 * (1 + len(models))))
    records = []
    for c, i in enumerate(indices):
        img = x_test[i]
        axes[0, c].imshow(img.permute(1, 2, 0).numpy())
        axes[0, c].set_title(f'true: {CLASS_NAMES[y[i]]}', fontsize=8, fontweight='bold')
        axes[0, c].axis('off')
        for r, (m, t) in enumerate(zip(models, titles), start=1):
            cam, cls, p, fsize = grad_cam(m, img, m.gradcam_layer())
            axes[r, c].imshow(img.permute(1, 2, 0).numpy())
            axes[r, c].imshow(cam, cmap='jet', alpha=0.45)
            ok = cls == y[i]
            axes[r, c].set_title(f'{CLASS_NAMES[cls]} {p:.2f}', fontsize=8, color='green' if ok else 'red')
            axes[r, c].axis('off')
            records.append({'index': int(i), 'model': t, 'true': CLASS_NAMES[y[i]], 'pred': CLASS_NAMES[cls],
                            'confidence': round(p, 3), 'correct': bool(ok),
                            'centre_mass': round(centre_mass(cam), 3), 'feature_map': f'{fsize[0]}x{fsize[1]}'})
    for r, t in enumerate(titles, start=1):
        axes[r, 0].text(-0.15, 0.5, t, transform=axes[r, 0].transAxes, ha='right', va='center',
                        fontsize=9, fontweight='bold', rotation=90)
    fig.suptitle('Quiz 3 Advanced: Grad-CAM (red = regions that most increase the predicted class score)\n'
                 'Title = predicted class & confidence (green = correct, red = wrong)', fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    return fig, records
