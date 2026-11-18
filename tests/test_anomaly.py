import numpy as np

from convae.anomaly import latent_scores, score_metrics


def test_metrics_perfect_and_inverted():
    normal, anom = np.linspace(0, 1, 50), np.linspace(2, 3, 50)
    r = score_metrics(normal, anom)
    assert r["auroc"] == 1.0 and r["fpr_at_95tpr"] == 0.0
    assert score_metrics(anom, normal)["auroc"] == 0.0  # score dans le mauvais sens -> AUROC 0


def test_latent_scores_flag_far_points():
    rng = np.random.default_rng(0)
    Ztr = rng.normal(0, 1, (500, 8))
    Zn, Za = rng.normal(0, 1, (100, 8)), rng.normal(5, 1, (100, 8))
    for sn, sa in latent_scores(Ztr, Zn, Za, k=5).values():
        assert score_metrics(sn, sa)["auroc"] > 0.99
