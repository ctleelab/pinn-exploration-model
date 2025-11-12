#!/usr/bin/env python3
import argparse
import numpy as np
import mrcfile

from bm4d import bm4d, BM4DProfile, BM4DStages
from skimage.filters import unsharp_mask
from skimage import exposure
from scipy.ndimage import gaussian_filter
from scipy.fft import fftn, ifftn, fftshift, ifftshift

# ----------------------------
# I/O & small utils
# ----------------------------
def load_mrc(path):
    with mrcfile.open(path, permissive=True) as mrc:
        vol = mrc.data.astype(np.float32, copy=True)
        header = mrc.header.copy()
    return vol, header

def save_mrc(path, vol, header_like=None):
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(vol.astype(np.float32, copy=False))
        if header_like is not None:
            try:
                mrc.header.cella.x = header_like.cella.x
                mrc.header.cella.y = header_like.cella.y
                mrc.header.cella.z = header_like.cella.z
                mrc.header.mx = header_like.mx
                mrc.header.my = header_like.my
                mrc.header.mz = header_like.mz
                mrc.header.mapc = header_like.mapc
                mrc.header.mapr = header_like.mapr
                mrc.header.maps = header_like.maps
            except Exception:
                pass
        mrc.update_header_from_data()

def robust_sigma_mad(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad

def minmax_normalize(vol, p_lo=0.5, p_hi=99.5, eps=1e-8):
    vmin = np.percentile(vol, p_lo)
    vmax = np.percentile(vol, p_hi)
    scale = max(vmax - vmin, eps)
    out = (vol - vmin) / scale
    return np.clip(out, 0.0, 1.0), (vmin, scale)

def restore_minmax(vol01, vmin, scale):
    return vol01 * scale + vmin

def match_mean_std(src, ref, eps=1e-8):
    s_mu, s_sd = float(src.mean()), float(src.std() + eps)
    r_mu, r_sd = float(ref.mean()), float(ref.std() + eps)
    return (src - s_mu) * (r_sd / s_sd) + r_mu

# ----------------------------
# Processing blocks
# ----------------------------

def local_background_subtract(vol, sigma_vox=15.0):
    """큰 가우시안으로 저주파만 추출해 빼기 (로컬 배경 평탄화)"""
    low = gaussian_filter(vol, sigma=float(sigma_vox))
    out = vol - low
    # 범위가 바뀌므로 0-1 재스케일 (후단 CLAHE 안정화용)
    out -= out.min()
    denom = max(out.max(), 1e-8)
    out /= denom
    return out

def clahe_2d_slicewise(vol01, clip=0.02, tile=8):
    """Z-슬라이스별 2D CLAHE (3D CLAHE보다 엣지 보존이 안정적)"""
    zdim = vol01.shape[0]
    out = np.empty_like(vol01, dtype=np.float32)
    for z in range(zdim):
        out[z] = exposure.equalize_adapthist(
            vol01[z], clip_limit=float(clip), kernel_size=int(tile)
        ).astype(np.float32)
    return out

def unsharp_3d(vol01, radius=1.0, amount=0.6):
    """3D 언샤프 (소량만, 막/필라멘트 엣지 강화)"""
    return unsharp_mask(
        image=vol01, radius=float(radius), amount=float(amount), preserve_range=True
    ).astype(np.float32)

def smooth_highpass_3d(vol01, hp_low=0.0, hp_high=0.05, roll=0.02):
    """
    부드러운 하이패스 (raised-cosine roll-off).
    hp_high: 하이패스 컷오프(이보다 낮은 주파수 감쇠), Nyquist=1.0 기준
    roll: 완충 대역 폭
    """
    F = fftshift(fftn(vol01))
    nz, ny, nx = vol01.shape
    Z, Y, X = np.ogrid[-nz//2:nz//2, -ny//2:ny//2, -nx//2:nx//2]
    R = np.sqrt((X/(nx/2))**2 + (Y/(ny/2))**2 + (Z/(nz/2))**2)  # 0..1 (Nyquist=1)

    # raised-cosine high-pass
    hp = np.ones_like(R, dtype=np.float32)
    band_start = max(hp_high - roll, 0.0)
    band_end   = min(hp_high + roll, 1.0)

    # below band_start -> 0, above band_end -> 1, in-between -> smooth ramp
    hp[R <= band_start] = 0.0
    ramp = (R - band_start) / max(band_end - band_start, 1e-8)
    mask = (R > band_start) & (R < band_end)
    hp[mask] = 0.5 * (1 - np.cos(np.pi * np.clip(ramp[mask], 0, 1)))

    Ff = F * hp
    out = np.real(ifftn(ifftshift(Ff))).astype(np.float32)
    # 재스케일
    out -= out.min()
    out /= max(out.max(), 1e-8)
    return out

# ----------------------------
# Pipeline
# ----------------------------
def run_pipeline(
    in_path, out_path,
    normalize=True, pclip_lo=0.5, pclip_hi=99.5,
    auto_sigma=True, sigma=None, bm4d_profile="wiener", bm4d_stages="all",
    bg_subtract=True, bg_sigma_vox=15.0,
    do_clahe=True, clahe_clip=0.02, clahe_tile=8,
    do_unsharp=True, unsharp_radius=1.0, unsharp_amount=0.6,
    do_highpass=False, hp_low=0.0, hp_high=0.05, hp_roll=0.02,
    keep_mean_std=True,
):
    raw, hdr = load_mrc(in_path)

    # Step 0) (옵션) 정규화
    if normalize:
        vol01, (vmin, vscale) = minmax_normalize(raw, p_lo=pclip_lo, p_hi=pclip_hi)
        ref_for_stats = raw
    else:
        vol01 = raw.copy()
        vmin, vscale = 0.0, 1.0
        ref_for_stats = raw

    # Step 1) BM4D
    vol01, sigma_used = bm4d_denoise(
        vol01, sigma=sigma, auto_sigma=auto_sigma,
        profile=bm4d_profile, stages=bm4d_stages
    )

    # Step 2) 로컬 배경 보정
    if bg_subtract:
        vol01 = local_background_subtract(vol01, sigma_vox=bg_sigma_vox)

    # Step 3) CLAHE (슬라이스별)
    if do_clahe:
        vol01 = clahe_2d_slicewise(vol01, clip=clahe_clip, tile=clahe_tile)

    # Step 4) 언샤프 (엣지 살짝 강화)
    if do_unsharp:
        vol01 = unsharp_3d(vol01, radius=unsharp_radius, amount=unsharp_amount)

    # Step 5) (옵션) 부드러운 하이패스
    if do_highpass:
        vol01 = smooth_highpass_3d(vol01, hp_low=hp_low, hp_high=hp_high, roll=hp_roll)

    # Step 6) 원 스케일 복원
    out = restore_minmax(vol01, vmin, vscale) if normalize else vol01

    # Step 7) 평균/표준편차 맞춤 (정량/세그 threshold 안정화)
    if keep_mean_std:
        out = match_mean_std(out, ref_for_stats)

    save_mrc(out_path, out, header_like=hdr)

    return {
        "sigma_used": sigma_used,
        "bm4d_profile": bm4d_profile,
        "bm4d_stages": bm4d_stages,
        "normalized": normalize,
        "bg_subtract": bg_subtract,
        "clahe": do_clahe,
        "unsharp": do_unsharp,
        "highpass": do_highpass
    }

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="CryoET Denoise+Contrast pipeline (BM4D→BG-sub→CLAHE→Unsharp→Highpass).")
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)

    p.add_argument("--normalize", action="store_true")
    p.add_argument("--pclip-lo", type=float, default=0.5)
    p.add_argument("--pclip-hi", type=float, default=99.5)

    p.add_argument("--auto-sigma", action="store_true")
    p.add_argument("--sigma", type=float, default=None)
    p.add_argument("--bm4d-profile", type=str, default="wiener",
                help="BM4D profile (np|default|wiener|lc|low|high|hard|basic).")

    p.add_argument("--bm4d-stages", type=str, default="all",
                help="BM4D stages (all|basic|hard|wiener|wi).")


    p.add_argument("--bg-subtract", action="store_true")
    p.add_argument("--bg-sigma-vox", type=float, default=15.0)

    p.add_argument("--clahe", action="store_true")
    p.add_argument("--clahe-clip", type=float, default=0.02)
    p.add_argument("--clahe-tile", type=int, default=8)

    p.add_argument("--unsharp", action="store_true")
    p.add_argument("--unsharp-radius", type=float, default=1.0)
    p.add_argument("--unsharp-amount", type=float, default=0.6)

    p.add_argument("--highpass", action="store_true")
    p.add_argument("--hp-low", type=float, default=0.0)
    p.add_argument("--hp-high", type=float, default=0.05)
    p.add_argument("--hp-roll", type=float, default=0.02)

    p.add_argument("--keep-mean-std", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    info = run_pipeline(
        in_path=args.in_path,
        out_path=args.out_path,
        normalize=args.normalize, pclip_lo=args.pclip_lo, pclip_hi=args.pclip_hi,
        auto_sigma=args.auto_sigma, sigma=args.sigma,
        bm4d_profile=args.bm4d_profile, bm4d_stages=args.bm4d_stages,
        bg_subtract=args.bg_subtract, bg_sigma_vox=args.bg_sigma_vox,
        do_clahe=args.clahe, clahe_clip=args.clahe_clip, clahe_tile=args.clahe_tile,
        do_unsharp=args.unsharp, unsharp_radius=args.unsharp_radius, unsharp_amount=args.unsharp_amount,
        do_highpass=args.highpass, hp_low=args.hp_low, hp_high=args.hp_high, hp_roll=args.hp_roll,
        keep_mean_std=args.keep_mean_std,
    )
    print("Pipeline done:", info)

if __name__ == "__main__":
    main()
