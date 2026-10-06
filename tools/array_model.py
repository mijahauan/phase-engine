#!/usr/bin/env python3
"""
Performance model for a small coherent HF array (docs/MULTI_ANTENNA_PHASING_PLAN.md).

Reproduces the plan's figures for arrival-angle accuracy, beam width and gain,
null depth and cost, and MUSIC resolution. The model is deliberately simple:
plane waves, identical vertical elements, a Gaussian phase error left on each
element after calibration, no mutual coupling, no ground response, no loop.
Read every figure it prints as a ceiling, not a prediction.

Conventions: x east, y north, azimuth clockwise from north, elevation above the
horizon; u points TOWARD the source; element n sees phase +k p_n . u; the array
output is y = w^H x.

Usage:
    python3 tools/array_model.py            # every section
    python3 tools/array_model.py aoa beam   # chosen sections
Sections: aoa, beam, null, pointing, music.
"""

import sys

import numpy as np

C = 299_792_458.0
RNG = np.random.default_rng(7)

# The recommended layout: an equilateral triangle with V1 at the origin,
# V2 along the WWV path from EM38ww (284 deg) and V3 toward 344 deg.
WWV = (284.1, 25.0)    # azimuth, elevation of WWV's one-hop arrival at EM38ww
WWVH = (274.6, 10.0)   # azimuth, elevation of WWVH's low arrival at EM38ww


def triangle(side_m, b1=284.0, b2=344.0):
    pts = [np.zeros(3)]
    for b in (b1, b2):
        r = np.radians(b)
        pts.append(side_m * np.array([np.sin(r), np.cos(r), 0.0]))
    return np.array(pts)


LAYOUTS = {"12 m triangle": triangle(12.0), "20 m triangle": triangle(20.0)}


def unit(az_deg, el_deg):
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)])


def steer(P, f, az, el):
    return np.exp(1j * 2 * np.pi * f / C * (P @ unit(az, el)))


def aoa_rms(P, f, az, el, sigma_deg, n=20_000):
    """Rms azimuth and elevation error of single-wave interferometry.

    Phases are taken as already unwrapped (aided above the ambiguity limit).
    Trials whose horizontal wavenumber exceeds k are clipped to the horizon.
    """
    k = 2 * np.pi * f / C
    base = (P[1:] - P[0])[:, :2]
    true = k * (P[1:] - P[0]) @ unit(az, el)
    eps = np.radians(sigma_deg) * RNG.standard_normal((n, len(P)))
    meas = true[None, :] + (eps[:, 1:] - eps[:, :1])
    uxy = np.linalg.solve(base, (meas / k).T).T
    r = np.hypot(uxy[:, 0], uxy[:, 1])
    invalid = np.mean(r > 1.0)
    az_err = (np.degrees(np.arctan2(uxy[:, 0], uxy[:, 1])) - az + 180) % 360 - 180
    el_err = np.degrees(np.arccos(np.clip(r, 0, 1))) - el
    return np.sqrt(np.mean(az_err**2)), np.sqrt(np.mean(el_err**2)), invalid


def sky_noise_cov(P, f, nel=90, naz=360):
    """Noise covariance for noise spread evenly in azimuth over the upper
    hemisphere, weighted by a short vertical's power pattern (cos^2 el) and
    the solid-angle factor (cos el)."""
    k = 2 * np.pi * f / C
    els = np.radians((np.arange(nel) + 0.5) * 90 / nel)
    azs = np.radians(np.arange(naz) * 360 / naz)
    EL, AZ = np.meshgrid(els, azs, indexing="ij")
    U = np.stack([np.sin(AZ) * np.cos(EL), np.cos(AZ) * np.cos(EL), np.sin(EL)], -1).reshape(-1, 3)
    w = (np.cos(EL) ** 3).reshape(-1)
    A = np.exp(1j * k * U @ P.T)
    return (A.conj().T * w) @ A / w.sum()


def main_lobe_width(P, f, az0, el0, step=0.25):
    """Contiguous half-power width in azimuth around the look direction."""
    a0 = steer(P, f, az0, el0)

    def resp(d):
        return abs(np.vdot(a0, steer(P, f, az0 + d, el0))) ** 2 / len(P) ** 2

    lo = 0.0
    while resp(lo - step) >= 0.5 and lo > -180:
        lo -= step
    hi = 0.0
    while resp(hi + step) >= 0.5 and hi < 180:
        hi += step
    return hi - lo


def lcmv(R, Cm, g):
    Ri = np.linalg.solve(R, Cm)
    return Ri @ np.linalg.solve(Cm.conj().T @ Ri, g)


def section_aoa():
    print("== Arrival angle of one wave: rms az / el error (deg), [share of trials with no valid elevation]")
    for name, P in LAYOUTS.items():
        for f in (2.5e6, 5e6, 10e6, 15e6):
            for el in (10, 25, 45):
                cells = []
                for s in (1, 3, 5):
                    a, e, bad = aoa_rms(P, f, WWV[0], el, s)
                    cells.append(f"{a:5.1f}/{e:5.1f} [{bad:4.0%}]")
                print(f"{name}  {f/1e6:4.1f} MHz  el {el:2d}  err 1/3/5 deg: " + "  ".join(cells))


def section_beam():
    print("== Delay-and-sum toward WWV: main-lobe half-power width and gain against sky noise")
    for name, P in LAYOUTS.items():
        for f in (2.5e6, 5e6, 10e6, 15e6, 20e6):
            Q = sky_noise_cov(P, f)
            a0 = steer(P, f, *WWV)
            g_das = len(P) ** 2 / np.real(a0.conj() @ Q @ a0)
            w_sd = np.linalg.solve(Q, a0)
            g_sd = np.real(a0.conj() @ w_sd)
            wng = abs(np.vdot(w_sd, a0)) ** 2 / np.real(np.vdot(w_sd, w_sd))
            print(f"{name}  {f/1e6:4.1f} MHz: width {main_lobe_width(P, f, *WWV):6.1f} deg; "
                  f"DAS {10*np.log10(g_das):4.1f} dB; superdirective {10*np.log10(g_sd):4.1f} dB "
                  f"at white-noise gain {10*np.log10(wng):5.1f} dB")


def section_null():
    print("== LCMV: unity on WWV, null on WWVH. WWV SNR vs one element (receiver noise / sky noise),")
    print("   and median null depth with 1 and 3 deg of per-element calibration error")
    for name, P in LAYOUTS.items():
        for f in (2.5e6, 5e6, 10e6, 15e6):
            Cm = np.stack([steer(P, f, *WWV), steer(P, f, *WWVH)], 1)
            g = np.array([1, 0])
            snr = []
            for R in (np.eye(len(P)), sky_noise_cov(P, f)):
                w = lcmv(R, Cm, g)
                snr.append(-10 * np.log10(np.real(w.conj() @ R @ w)))
            w = lcmv(np.eye(len(P)), Cm, g)
            depth = []
            for s in (1, 3):
                e = np.exp(1j * np.radians(s) * RNG.standard_normal((4000, len(P))))
                depth.append(20 * np.log10(np.median(np.abs((Cm[:, 1][None, :] * e) @ w.conj()))))
            print(f"{name}  {f/1e6:4.1f} MHz: WWV SNR {snr[0]:6.1f} / {snr[1]:6.1f} dB; "
                  f"null depth {depth[0]:6.1f} / {depth[1]:6.1f} dB")


def section_pointing():
    print("== Null depth when WWVH's true bearing differs from the null's by d (12 m, perfect calibration)")
    P = LAYOUTS["12 m triangle"]
    for f in (5e6, 10e6, 15e6):
        Cm = np.stack([steer(P, f, *WWV), steer(P, f, *WWVH)], 1)
        w = lcmv(np.eye(len(P)), Cm, np.array([1, 0]))
        cells = [f"d={d}: {20*np.log10(abs(np.vdot(w, steer(P, f, WWVH[0] + d, WWVH[1])))):6.1f} dB"
                 for d in (0.5, 1, 2, 3, 5)]
        print(f"  {f/1e6:4.1f} MHz  " + "  ".join(cells))


def music_resolution(P, f, sigma_deg, snr_db=20, K=2000, trials=150, el=25.0, az0=WWV[0]):
    """Smallest azimuth spacing of two equal uncorrelated sources that MUSIC
    resolves in at least 90% of trials. Known elevation, azimuth-only search;
    'resolved' = each of the two highest peaks within half the spacing of its source."""
    for sep in (5, 10, 15, 20, 30, 40, 50, 60, 75, 90, 120):
        ok = 0
        for _ in range(trials):
            azs = (az0 - sep / 2, az0 + sep / 2)
            e = np.exp(1j * np.radians(sigma_deg) * RNG.standard_normal(len(P)))
            A = np.stack([steer(P, f, a, el) * e for a in azs], 1)
            S = (RNG.standard_normal((2, K)) + 1j * RNG.standard_normal((2, K))) / np.sqrt(2)
            N = (RNG.standard_normal((len(P), K)) + 1j * RNG.standard_normal((len(P), K))) / np.sqrt(2)
            X = A @ S + N * 10 ** (-snr_db / 20)
            _, V = np.linalg.eigh(X @ X.conj().T / K)
            En = V[:, : len(P) - 2]
            grid = np.arange(az0 - 90, az0 + 90, 0.5)
            ps = np.array([1 / np.linalg.norm(En.conj().T @ steer(P, f, g, el)) ** 2 for g in grid])
            peaks = [i for i in range(1, len(grid) - 1) if ps[i] > ps[i - 1] and ps[i] >= ps[i + 1]]
            peaks = sorted(peaks, key=lambda i: -ps[i])[:2]
            if len(peaks) == 2:
                est = sorted(grid[peaks])
                ok += all(abs(est[j] - azs[j]) < sep / 2 for j in range(2))
        if ok / trials >= 0.9:
            return sep
    return None


def section_music():
    print("== MUSIC: smallest spacing resolved in >=90% of trials (deg), calibration error 0 / 1 / 3 deg")
    for name, P in LAYOUTS.items():
        for f in (10e6, 15e6):
            print(f"{name}  {f/1e6:4.1f} MHz: {[music_resolution(P, f, s) for s in (0.0, 1.0, 3.0)]}")


SECTIONS = {"aoa": section_aoa, "beam": section_beam, "null": section_null,
            "pointing": section_pointing, "music": section_music}

if __name__ == "__main__":
    for key in (sys.argv[1:] or list(SECTIONS)):
        SECTIONS[key]()
        print()
