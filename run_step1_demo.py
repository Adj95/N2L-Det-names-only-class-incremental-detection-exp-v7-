"""Step 1 runner (Colab / Kaggle / laptop, CPU only, ~10 s).

    !pip -q install numpy scipy pytest
    !python run_step1_demo.py

1. runs the unit tests,
2. prints three small experiments on synthetic features that show the behaviours the
   N2L-Det design depends on (noise removal, no forgetting, threshold vs sample count).

The experiments are synthetic: they validate the MATH of the core, not CLIP/YOLO
quality. Real-feature validation starts in step 2.
"""
import subprocess
import sys

import numpy as np

import n2l_core as nc


def blobs(n_per, classes, means, seed):
    rng = np.random.default_rng(seed)
    X = np.vstack([means[c] + rng.normal(size=(n_per, means.shape[1])) for c in classes])
    y = np.repeat(list(classes), n_per)
    return X, y


def zs_logits(y_local, K, p_noise, rng, systematic=False):
    pred = y_local.copy()
    if systematic:
        pred[y_local == 0] = 1
    else:
        flip = rng.random(len(y_local)) < p_noise
        pred[flip] = (y_local[flip] + rng.integers(1, K, flip.sum())) % K
    L = rng.normal(size=(len(y_local), K))
    L[np.arange(len(y_local)), pred] += 6
    return L, pred


def exp_noise():
    K, d, n_per = 10, 64, 300
    n = K * n_per
    print(f"\n[1] Label refinement vs pseudo-label noise  ({K} classes, {n} samples, d={d}, class sep 0.5)")
    means = np.random.default_rng(1).normal(scale=0.5, size=(K, d))
    X, y = blobs(n_per, range(K), means, 0)
    Xt, yt = blobs(300, range(K), means, 5)
    g = nc.RidgeAccumulator(d, 0.1); g.add_classes(K); g.update(X, y)
    ceil = np.mean((Xt @ g.solve()).argmax(1) == yt)
    rules = {
        "energy50%": nc.N2LConfig(pca_rule="energy", pca_rho=0.5),
        "abs 1.22*sqrt(n)": nc.N2LConfig(pca_rule="abs", pca_theta=float(1.22 * np.sqrt(n))),
    }
    print("    pseudo-label accuracy (vs truth):")
    print("    noise              initial |  refined: " + " | ".join(f"{k:>22s}" for k in rules))
    cases = [(f"random p={p:.1f}", dict(p_noise=p)) for p in (0.1, 0.3, 0.5)]
    cases.append(("systematic 0->1", dict(p_noise=0, systematic=True)))
    for name, kw in cases:
        L, pred = zs_logits(y, K, kw["p_noise"], np.random.default_rng(2), kw.get("systematic", False))
        cells = []
        for cfg in rules.values():
            r = nc.refine_task_labels(X, L, cfg)
            cells.append(f"{np.mean(r.labels == y):.3f} (k={r.k:2d})".rjust(22))
        print(f"    {name:<17s}  {np.mean(pred == y):.3f}  |            " + " | ".join(cells))
    print(f"    ceiling = ridge trained on TRUE labels: test acc {ceil:.3f}")
    print("    -> random noise is largely voted away; the systematic confusion is not.")


def exp_incremental():
    d, per, T = 64, 5, 4
    print(f"\n[2] Incremental learning, {T} tasks x {per} classes, 25% random pseudo-label noise, class sep 0.5")
    means = np.random.default_rng(9).normal(scale=0.5, size=(per * T, d))
    cfg = nc.N2LConfig(pca_rule="energy", pca_rho=0.5)
    L = nc.N2LLearner(d, cfg)
    allX, ally = [], []
    print("    after task | N2L acc, seen classes | N2L acc, task-1 classes | GT-label joint ridge, seen")
    for t in range(T):
        cls = range(t * per, (t + 1) * per)
        X, y = blobs(300, cls, means, 10 + t)
        allX.append(X); ally.append(y)
        Lg, _ = zs_logits(y - t * per, per, 0.25, np.random.default_rng(t))
        L.add_task(X, Lg, per, rng=np.random.default_rng(t))
        seen = range(0, (t + 1) * per)
        Xt, yt = blobs(200, seen, means, 77)
        pred = L.scores(Xt).argmax(1)
        gt = nc.RidgeAccumulator(d, cfg.lam); gt.add_classes((t + 1) * per)
        gt.update(np.vstack(allX), np.concatenate(ally))
        gt_acc = np.mean((Xt @ gt.solve()).argmax(1) == yt)
        print(f"    {t + 1:^10} | {np.mean(pred == yt):^21.3f} | {np.mean(pred[yt < per] == yt[yt < per]):^23.3f} | {gt_acc:^25.3f}")
    print("    -> accuracy on task-1 classes stays at the level the joint ridge gets; the drop with")
    print("       more classes is the harder 20-way problem, not forgetting (exact equivalence to joint")
    print("       training is proved by test_recursive_equals_joint_with_new_class_padding).")


def exp_threshold():
    print("\n[3] How many PCA dimensions k does each rule keep as the number of boxes n grows?")
    print("    (same feature distribution, n repeated; rule thresholds fixed from the n=400 case)")
    rng = np.random.default_rng(6)
    scales = np.array([10, 8, 6, 4, 2, 1, .5, .5, .5, .5, .5, .5])
    X = rng.normal(size=(400, 12)) * scales
    print("        n   abs(sigma>=100)   scaled(ref n=400)   energy(80%)")
    for rep in (1, 4, 16, 64):
        Xr = np.vstack([X] * rep)
        ka = nc.PCABasis.fit(Xr, rule="abs", theta=100).k
        ks = nc.PCABasis.fit(Xr, rule="scaled", theta=100, n_ref=400).k
        ke = nc.PCABasis.fit(Xr, rule="energy", rho=0.8).k
        print(f"    {400 * rep:6d}   {ka:^15d}   {ks:^17d}   {ke:^11d}")
    print("    -> the official absolute threshold drifts with n; 'scaled' and 'energy' do not.")


if __name__ == "__main__":
    print("== running unit tests ==")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "test_n2l_core.py"])
    if r.returncode != 0:
        sys.exit("unit tests FAILED -- do not proceed to step 2")
    exp_noise()
    exp_incremental()
    exp_threshold()
    print("\nSTEP 1 OK")
