import numpy as np
import torch
import pywt
import os
import yaml
from datafilters import analyze_tumor_distribution
from decode import h5_to_imgarray
import dwt_idwt_transforms as dwt_transforms

NUM_SAMPLES = 50  # number of images to average over


def compute_energy(tensor):
    return (tensor ** 2).sum().item()


def analyze_wavelet(wavelet_name, images, image_size):
    """Compute HF energy ratio for a given wavelet across all images."""
    try:
        matrix_Low, matrix_High = dwt_transforms.dwt_matrix(image_size, wavelet_name=wavelet_name)
    except Exception as e:
        return None, str(e)

    LL, LH, HL, HH = dwt_transforms.dwt(images, matrix_Low, matrix_High)

    total_energy = compute_energy(images)
    ll_energy = compute_energy(LL)
    lh_energy = compute_energy(LH)
    hl_energy = compute_energy(HL)
    hh_energy = compute_energy(HH)
    hf_energy = lh_energy + hl_energy + hh_energy

    return {
        'wavelet': wavelet_name,
        'total': total_energy,
        'LL%': ll_energy / total_energy * 100,
        'LH%': lh_energy / total_energy * 100,
        'HL%': hl_energy / total_energy * 100,
        'HH%': hh_energy / total_energy * 100,
        'HF%': hf_energy / total_energy * 100,
        'HF_total': hf_energy,
    }, None


def load_sample_images(config_path='configuration.yml', num_samples=NUM_SAMPLES):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    dataset_config = config['dataset_params']

    sig_df, no_tumor_df = analyze_tumor_distribution(
        csv_path=dataset_config['csv_path'],
        min_slice=dataset_config.get('min_slice', 40),
        max_slice=dataset_config.get('max_slice', 124),
        tumor_threshold=dataset_config.get('tumor_threshold', 100)
    )

    if dataset_config['target_class'] == 'significant_tumor':
        selected_df = sig_df
    else:
        selected_df = no_tumor_df

    num_samples = min(num_samples, len(selected_df))
    indices = np.random.choice(len(selected_df), size=num_samples, replace=False)
    imgs = []

    for idx in indices:
        rel_path = selected_df.iloc[idx]['slice_path']
        filename = os.path.basename(rel_path)
        h5_path = os.path.join(dataset_config.get('data_root', ''), filename)
        img_array = h5_to_imgarray(h5_path, channel_index=dataset_config.get('channel_index', 3), normalize=True)
        if img_array is None:
            continue
        img_tensor = torch.from_numpy(img_array).float().unsqueeze(0) / 255.0
        imgs.append(img_tensor)

    images = torch.stack(imgs)  # (N, 1, 240, 240)
    print(f"\nLoaded {len(imgs)} images for analysis.\n")
    return images, dataset_config['image_size']


def main():
    images, image_size = load_sample_images()

    # all discrete wavelets available in pywt
    wavelet_families = ['haar', 'db2', 'db3', 'db4', 'db5', 'db6', 'db8',
                        'sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym8',
                        'coif1', 'coif2', 'coif3', 'coif4',
                        'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4', 'bior2.6',
                        'bior3.1', 'bior3.3', 'bior3.5',
                        'bior4.4', 'bior5.5', 'bior6.8',
                        'dmey']

    results = []
    print(f"{'Wavelet':<12} {'LL%':>8} {'LH%':>8} {'HL%':>8} {'HH%':>8} {'HF% (total)':>12}")
    print('-' * 60)

    for wname in wavelet_families:
        result, err = analyze_wavelet(wname, images, image_size)
        if err:
            print(f"{wname:<12} FAILED: {err}")
            continue
        results.append(result)
        print(f"{wname:<12} {result['LL%']:>7.2f}% {result['LH%']:>7.2f}% {result['HL%']:>7.2f}% {result['HH%']:>7.2f}% {result['HF%']:>10.2f}%")

    # sort by HF energy
    results.sort(key=lambda x: x['HF%'], reverse=True)
    print(f"\n{'='*60}")
    print("TOP 10 WAVELETS BY HIGH-FREQUENCY ENERGY:")
    print(f"{'='*60}")
    print(f"{'Rank':<6} {'Wavelet':<12} {'HF%':>10} {'LH%':>8} {'HL%':>8} {'HH%':>8}")
    print('-' * 54)
    for i, r in enumerate(results[:10]):
        print(f"{i+1:<6} {r['wavelet']:<12} {r['HF%']:>9.2f}% {r['LH%']:>7.2f}% {r['HL%']:>7.2f}% {r['HH%']:>7.2f}%")


if __name__ == '__main__':
    main()
