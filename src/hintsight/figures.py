"""The three figures used in the README and memo."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .experiment import mde

BLUE, GREY, RED, GREEN = "#2b6cb0", "#8a8f98", "#c0392b", "#2f855a"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)


def hint_ladder(h: dict, path):
    rows = [
        ("Naive: hint vs no hint\n(all next problems)", "1_naive_all_pairs", GREY),
        ("Only students who missed\nproblem t", "2_like_for_like_wrong_on_t", GREY),
        ("+ prior ability, skill,\nopportunity", "3_ability_adjusted_skill_fe", BLUE),
        ("Same student\n(student fixed effects)", "4_student_fe", BLUE),
        ("Same student, same problem set,\nsame next problem", "5_student_x_assignment_fe_next_problem_fe", BLUE),
        ("Net of the dip already\nvisible on t-1 (placebo-adjusted)", "7_placebo_adjusted_change_t_minus_1_to_t_plus_1", GREEN),
        ("Placebo: 'effect' of a hint on t\non the answer to t-1", "placebo_hint_on_t_vs_correct_on_t_minus_1", RED),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for i, (label, key, color) in enumerate(rows):
        r = h[key]
        e, (lo, hi) = 100 * r["estimate"], (100 * r["ci95"][0], 100 * r["ci95"][1])
        ax.errorbar(e, i, xerr=[[e - lo], [hi - e]], fmt="o", color=color, capsize=4, ms=7)
        ax.text(max(hi, e) + 0.8, i, f"{e:+.1f} pts", va="center", fontsize=9, color=color)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(-38, 8)
    ax.set_xlabel("Change in chance of getting the next problem right after a hint (points, 95% CI)")
    ax.set_title("Most of the 'hints hurt' signal is selection: struggling students take hints", fontsize=11, loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def calibration(p: dict, path):
    m = p["models"]
    show = [
        ("Item difficulty only", "item_difficulty_only", GREY),
        ("Elo (online Rasch)", "elo_online_rasch", BLUE),
        ("PFA + item + student history", "pfa_item_student_history", GREEN),
        ("Same, item difficulty NOT cross-fitted", "pfa_item_student_history_in_sample_item_encoding", RED),
    ]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})
    a1.plot([0, 1], [0, 1], ls="--", color="black", lw=0.8)
    for label, key, color in show:
        c = np.array(m[key]["calibration_curve"])
        a1.plot(c[:, 0], c[:, 1], "o-", color=color, ms=4,
                label=f"{label} (ECE {m[key]['ece_mean']:.3f})")
    a1.set_xlabel("Predicted chance of a correct first attempt")
    a1.set_ylabel("Observed share correct")
    a1.set_title("Calibration, held-out students (5-fold grouped CV)", fontsize=10, loc="left")
    a1.legend(fontsize=7.5, frameon=False, loc="upper left")
    a1.spines[["top", "right"]].set_visible(False)

    bars = [("Base rate", "base_rate", GREY), ("Item only", "item_difficulty_only", GREY),
            ("PFA", "pfa", BLUE), ("Elo", "elo_online_rasch", BLUE),
            ("PFA+item+hist", "pfa_item_student_history", GREEN),
            ("Leaky\n(peeks ahead)", "LEAKY_pfa_item_student_full_accuracy", RED)]
    vals = [m[k]["auc_mean"] for _, k, _ in bars]
    sds = [m[k]["auc_sd"] for _, k, _ in bars]
    a2.bar(range(len(bars)), vals, yerr=sds, color=[c for *_, c in bars], capsize=3)
    for i, v in enumerate(vals):
        a2.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=8)
    a2.set_xticks(range(len(bars)), [b[0] for b in bars], fontsize=8)
    a2.set_ylim(0.45, 0.82)
    a2.set_ylabel("AUC (mean of 5 folds)")
    a2.set_title("Predicting the next answer", fontsize=10, loc="left")
    a2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def power(e: dict, p_values: dict, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    n = np.arange(200, 6001, 50)
    a1.plot(n, [100 * mde(e["sd_post_accuracy"], k) for k in n], color=GREY, label="Plain difference in means")
    a1.plot(n, [100 * mde(e["sd_post_cuped"], k) for k in n], color=GREEN, label="CUPED (adjust for pre-period accuracy)")
    a1.axhline(2, ls=":", color="black", lw=0.8)
    for key, color in [("n_per_arm_for_2pp_naive", GREY), ("n_per_arm_for_2pp_cuped", GREEN)]:
        a1.axvline(e[key], ls="--", color=color, lw=0.8)
        a1.text(e[key] + 60, 5.2, f"{e[key]:,}", color=color, fontsize=8)
    a1.set_xlabel("Students per arm")
    a1.set_ylabel("Smallest detectable lift (accuracy points)")
    a1.set_ylim(0, 6)
    a1.set_title("Sample needed for a 2-point lift (80% power)", fontsize=10, loc="left")
    a1.legend(fontsize=8, frameon=False)
    a1.spines[["top", "right"]].set_visible(False)

    bins = np.linspace(0, 1, 21)
    a2.hist(p_values["aa_naive"], bins=bins, alpha=0.55, color=GREY, label="Plain")
    a2.hist(p_values["aa_cuped"], bins=bins, alpha=0.55, color=GREEN, label="CUPED")
    a2.axhline(len(p_values["aa_naive"]) / 20, color="black", ls="--", lw=0.8)
    aa = e["aa_test"]
    a2.set_title(f"{aa['n_sims']:,} A/A tests on real students: false positives "
                 f"{100 * aa['rejection_rate_naive']:.1f}% / {100 * aa['rejection_rate_cuped']:.1f}%",
                 fontsize=10, loc="left")
    a2.set_xlabel("p-value (should be flat when nothing changed)")
    a2.legend(fontsize=8, frameon=False)
    a2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
