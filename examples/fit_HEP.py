# -*- coding: utf-8 -*-

# =============================================================================
# File for computing R_pA of J/psi at LO
# Last modified: 20/04/2026
# =============================================================================

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
plots_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'plots'))
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
import lhapdf as lha
from src import Jpsi as jpsi
from src import Collision, Var_choice
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Paramètres physiques
# ─────────────────────────────────────────────────────────────────────────────
# M_JPSI = 3.096   # masse J/ψ [GeV/c²]
M_JPSI = 3
# Pour des sections efficaces intégrées en pT, on prend M_T = M_J/ψ (pT→0)
p_T = 1
M_T = np.sqrt(M_JPSI**2+p_T**2)

proton = "nNNPDF30_nlo_as_0118_noLHCbD_p"
Pb = "nNNPDF30_nlo_as_0118_noLHCbD_A208_Z82"
# proton = "NNPDF40_nlo_as_01180"
# Pb = "nNNPDF30_nlo_as_0118_A208_Z82"

p_set = lha.getPDFSet(proton)
A_set = lha.getPDFSet(Pb)

p_list = p_set.mkPDFs()
A_list = A_set.mkPDFs()

coll_type = 'pp'
# Factor = 3*10**(-8)

#just for the creation of the jpsi object, not used here
Atom = jpsi.Atom
atom = 'Pb'
Z = Atom[atom]['Z']
A = Atom[atom]['A']

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Fonction de fit
# ─────────────────────────────────────────────────────────────────────────────
def fit_func(y, N, n, sqrt_s):
    """
    f(y) = N * (1 - 2*M_T*cosh(y) / sqrt(s)) ** n

    y      : rapidité
    N      : normalisation [même unité que la section efficace, μb]
    n      : exposant (paramètre de forme)
    sqrt_s : énergie dans le centre de masse [GeV]  — fixé, non ajusté
    """
    x = 1.0 - 2.0 * M_T * np.cosh(y) / sqrt_s
    x = np.where(x > 0, x, np.nan)
    return N * x ** n


def make_fit_func(sqrt_s):
    """Renvoie une fonction f(y, N, n) avec sqrt_s fixé (pour curve_fit)."""
    def f(y, N, n):
        return fit_func(y, N, n, sqrt_s)
    return f

def y_to_xf(y,sqrt_s):
    """Renvoie la conversion de y en xf suivant rs"""
    return 2*M_T*np.sinh(y)/sqrt_s

def fit_func_xf(xf, N, n, sqrt_s):
    """
    f(xf) = N * (1 - x') ** n /x'

    xf     : Feynman x 
    N      : normalisation [même unité que la section efficace, μb]
    n      : exposant (paramètre de forme)
    sqrt_s : énergie dans le centre de masse [GeV]  — fixé, non ajusté
    """
    C = 2*M_T/sqrt_s
    x_prime = (xf**2 +xf*np.sqrt(xf**2+C**2)+C**2)/(xf+np.sqrt(xf**2+C**2))
    x = (N*(1-x_prime)**n)/x_prime
    x = np.where(x > 0, x, np.nan)
    return x


def make_xf_fit_func(sqrt_s):
    """Renvoie une fonction f(xf, N, n) avec sqrt_s fixé (pour curve_fit)."""
    def f(xf, N, n):
        return fit_func_xf(xf, N, n, sqrt_s)
    return f

def LO_fit_func(y, N, a,b, sqrt_s):
    Jpsi_LHC = jpsi.SigmaJpsi(p_set,p_list, A_set, A_list,sqrt_s**2, Z, A)
    # dsigma_tot_dvar n'accepte qu'un scalaire → on vectorise manuellement
    y_arr = np.atleast_1d(y)
    result = np.array([
        Jpsi_LHC.dsigma_tot_dvar(yi, p_T, 0,
                                  mu_factor=a, mu_f_factor=b,
                                  is_pp=Collision(coll_type),
                                  Var_y=True)[0]
        for yi in y_arr
    ])
    return N * (result if np.ndim(y) > 0 else result[0])
a = 1
b = a
def make_LO_fit_func(sqrt_s):
    def f(y,N):
        return LO_fit_func(y,N,a,b,sqrt_s)
    return f


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Lecture des fichiers HEPData
# ─────────────────────────────────────────────────────────────────────────────
def read_hepdata_csv(filepath, sqrt_s_target, ncols_required=10):
    """
    Lit un fichier CSV HEPData multi-blocs.

    Sélectionne le bloc correspondant à sqrt_s_target ET ayant au moins
    ncols_required colonnes (pour ne garder que le bloc principal qui
    contient sigma + erreurs, pas les blocs d'erreurs additionnels).

    Retourne un DataFrame avec les colonnes :
        y, y_low, y_high, sigma, err_stat, err_sys_corr, err_sys_uncorr, err_tot
    """
    blocks = []          # liste de (sqrt_s, np.array)
    current_rows = []
    current_sqrts = None
    in_data = False

    with open(os.path.join(data_dir,filepath)) as fh:
        for raw in fh:
            line = raw.strip()

            # Énergie du bloc
            if line.startswith("#: SQRT(S)"):
                parts = line.split(",")
                try:
                    current_sqrts = float(parts[-1])
                except ValueError:
                    pass
                continue

            # Ignorer les autres commentaires
            if line.startswith("#"):
                continue

            # En-tête du tableau
            if "YRAP" in line:
                if current_rows:  # sauvegarder le bloc précédent
                    blocks.append((current_sqrts, np.array(current_rows)))
                    current_rows = []
                in_data = True
                continue

            # Ligne de données
            if in_data and line:
                try:
                    vals = [float(v) for v in line.split(",")]
                    current_rows.append(vals)
                except ValueError:
                    pass
            elif not line:
                # Ligne vide = fin de bloc
                if current_rows:
                    blocks.append((current_sqrts, np.array(current_rows)))
                    current_rows = []
                in_data = False
                current_sqrts = None

    if current_rows:
        blocks.append((current_sqrts, np.array(current_rows)))

    # Sélection : bonne énergie + suffisamment de colonnes
    for sqrts, arr in blocks:
        if sqrts is None:
            continue
        if abs(sqrts - sqrt_s_target) > 1:
            continue
        if arr.shape[1] < ncols_required:
            continue

        df = pd.DataFrame()
        df["y"]      = arr[:, 0]
        df["y_low"]  = arr[:, 1]
        df["y_high"] = arr[:, 2]
        df["sigma"]  = arr[:, 3]

        # Erreurs statistiques (symétrisées)
        df["err_stat"] = (np.abs(arr[:, 4]) + np.abs(arr[:, 5])) / 2.0

        if arr.shape[1] >= 10:
            err_sys1 = (np.abs(arr[:, 6]) + np.abs(arr[:, 7])) / 2.0
            err_sys2 = (np.abs(arr[:, 8]) + np.abs(arr[:, 9])) / 2.0
            df["err_sys_corr"]   = err_sys1
            df["err_sys_uncorr"] = err_sys2
        else:
            df["err_sys_corr"]   = 0.0
            df["err_sys_uncorr"] = 0.0

        # Erreur totale quadratique
        df["err_tot"] = np.sqrt(
            # df["err_stat"]**2 + df["err_sys_corr"]**2 + df["err_sys_uncorr"]**2
            df["err_stat"]**2 + df["err_sys_uncorr"]**2
        )
        return df

    raise RuntimeError(
        f"Aucun bloc valide trouvé pour sqrt(s)={sqrt_s_target} GeV "
        f"avec >= {ncols_required} colonnes dans {filepath}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Chargement des données
# ─────────────────────────────────────────────────────────────────────────────
FILE1 = "HEPData-ins897764-v1-Table_3.csv"
FILE2 = "HEPData-ins1391511-v2-Table_6.csv"

datasets = {}

try:
    datasets["7 TeV\n(ins897764)"] = {
        "df": read_hepdata_csv(FILE1, 7000.0), "sqrt_s": 7000.0
    }
    print("7 TeV chargé  :", len(datasets["7 TeV\n(ins897764)"]["df"]), "points")
except Exception as e:
    print("7 TeV  ERREUR :", e)

try:
    datasets["8 TeV\n(ins1391511)"] = {
        "df": read_hepdata_csv(FILE2, 8000.0), "sqrt_s": 8000.0
    }
    print("8 TeV chargé  :", len(datasets["8 TeV\n(ins1391511)"]["df"]), "points")
except Exception as e:
    print("8 TeV  ERREUR :", e)

try:
    datasets["13 TeV\n(ins1391511)"] = {
        "df": read_hepdata_csv(FILE2, 13000.0), "sqrt_s": 13000.0
    }
    print("13 TeV chargé :", len(datasets["13 TeV\n(ins1391511)"]["df"]), "points")
except Exception as e:
    print("13 TeV ERREUR :", e)

print()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Fits en y 
# ─────────────────────────────────────────────────────────────────────────────
fit_results = {}
LO_fit_results = {}

print("=" * 70)
print(f"{'Dataset':<20}  {'N [μb]':>16}  {'n':>14}  {'χ²/ndf':>8}")
print("=" * 70)

for label, info in datasets.items():
    df     = info["df"]
    sqrt_s = info["sqrt_s"]
    y      = df["y"].values
    sigma  = df["sigma"].values
    err    = df["err_tot"].values

    # Ignorer points sans incertitude (err=0) pour la pondération
    mask = err > 0
    y_m, sig_m, err_m = y[mask], sigma[mask], err[mask]

    f = make_fit_func(sqrt_s)
    LO_f = make_LO_fit_func(sqrt_s)
    try:
        popt, pcov = curve_fit(
            f, y_m, sig_m,
            sigma=err_m, absolute_sigma=True,
            p0=[10.0, 5.0],
            maxfev=20000,
            bounds=([0, 0], [1e6, 200])
        )
        perr     = np.sqrt(np.diag(pcov))
        resid    = (sig_m - f(y_m, *popt)) / err_m
        chi2_ndf = np.sum(resid**2) / max(len(y_m) - 2, 1)

        fit_results[label] = {
            "popt": popt, "perr": perr, "chi2_ndf": chi2_ndf,
            "sqrt_s": sqrt_s, "df": df,
            "y_fit": y_m, "sigma_fit": sig_m, "err_fit": err_m
        }
        lbl = label.replace("\n", " ")
        print(f"{lbl:<20}  N={popt[0]:>10.3f}±{perr[0]:<6.3f}  "
              f"n={popt[1]:>8.3f}±{perr[1]:<6.3f}  χ²/ndf={chi2_ndf:>6.2f}")
    except Exception as e:
        lbl = label.replace("\n", " ")
        print(f"{lbl:<20}  *** Fit échoué : {e}")
    try:
        popt, pcov = curve_fit(
            LO_f, y_m, sig_m,
            sigma=err_m, absolute_sigma=True,
            p0= [1e-6],
            maxfev=20000,
            bounds=([0], [1e-3])
        )
        perr     = np.sqrt(np.diag(pcov))
        resid    = (sig_m - LO_f(y_m, *popt)) / err_m
        chi2_ndf = np.sum(resid**2) / max(len(y_m) - 2, 1)

        LO_fit_results[label] = {
            "popt": popt, "perr": perr, "chi2_ndf": chi2_ndf,
            "sqrt_s": sqrt_s, "df": df,
            "y_fit": y_m, "sigma_fit": sig_m, "err_fit": err_m
        }
        lbl = label.replace("\n", " ")
        print(f"{lbl:<20}  N_LO={popt[0]:10.1e}±{perr[0]:10.1e}  "
            # f"a={popt[1]:>8.3f}±{perr[1]:<6.3f} "
            # f"b={popt[2]:>8.3f}±{perr[2]:<6.3f} "
            f"χ²/ndf={chi2_ndf:>6.2f}")
    except Exception as e:
        lbl = label.replace("\n", " ")
        print(f"{lbl:<20}  *** Fit échoué : {e}")

print("=" * 70)
print()

# ─────────────────────────────────────────────────────────────────────────────
# 5bis.  Fits en xf
# ─────────────────────────────────────────────────────────────────────────────
# fit_xf_results = {}

# print("=" * 70)
# print(f"{'Dataset':<20}  {'N [μb]':>16}  {'n':>14}  {'χ²/ndf':>8}")
# print("=" * 70)

# for label, info in datasets.items():
#     df     = info["df"]
#     sqrt_s = info["sqrt_s"]
#     y      = df["y"].values
#     sigma  = df["sigma"].values
#     err    = df["err_tot"].values

#     # Ignorer points sans incertitude (err=0) pour la pondération
#     mask = err > 0
#     y_m, sig_m, err_m = y[mask], sigma[mask], err[mask]
#     xf_m = y_to_xf(y_m,sqrt_s)
#     sigxf_m = sig_m*sqrt_s/(2*M_T*np.cosh(y_m))
#     errxf_m = err_m*sqrt_s/(2*M_T*np.cosh(y_m))
#     f = make_xf_fit_func(sqrt_s)
#     try:
#         popt, pcov = curve_fit(
#             f, xf_m, sigxf_m,
#             sigma=errxf_m, absolute_sigma=True,
#             p0=[10.0, 5.0],
#             maxfev=20000,
#             bounds=([0, 0], [1e6, 200])
#         )
#         perr     = np.sqrt(np.diag(pcov))
#         resid    = (sigxf_m - f(xf_m, *popt)) / errxf_m
#         chi2_ndf = np.sum(resid**2) / max(len(xf_m) - 2, 1)

#         fit_xf_results[label] = {
#             "popt": popt, "perr": perr, "chi2_ndf": chi2_ndf,
#             "sqrt_s": sqrt_s, "df": df,
#             "xf_fit": xf_m, "sigma_fit": sigxf_m, "err_fit": errxf_m
#         }
#         lbl = label.replace("\n", " ")
#         print(f"{lbl:<20}  N={popt[0]:>10.3f}±{perr[0]:<6.3f}  "
#               f"n={popt[1]:>8.3f}±{perr[1]:<6.3f}  χ²/ndf={chi2_ndf:>6.2f}")
#     except Exception as e:
#         lbl = label.replace("\n", " ")
#         print(f"{lbl:<20}  *** Fit échoué : {e}")

# print("=" * 70)
# print()
# ─────────────────────────────────────────────────────────────────────────────
# 6.  Graphiques
# ─────────────────────────────────────────────────────────────────────────────
color_map = {
    "7 TeV\n(ins897764)":   "C0",
    "8 TeV\n(ins1391511)":  "C1",
    "13 TeV\n(ins1391511)": "C2",
}

n_panels = len(fit_results)
fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5.5), sharey=False)
if n_panels == 1:
    axes = [axes]

for ax, (label, res), (LO_label,LO_res) in zip(axes, fit_results.items(),LO_fit_results.items()):
    color  = color_map.get(label, "C3")
    popt   = res["popt"]
    perr   = res["perr"]
    sqrt_s = res["sqrt_s"]
    df     = res["df"]
    chi2   = res["chi2_ndf"]

    LO_popt   = LO_res["popt"]
    LO_perr   = LO_res["perr"]
    LO_sqrt_s = LO_res["sqrt_s"]
    LO_df     = LO_res["df"]
    LO_chi2   = LO_res["chi2_ndf"]

    # Courbe du fit
    # y_max =  df["y"].max() * 1.15
    y_max = 7.7
    y_plot  = np.linspace(0.0,y_max, 500)
    f       = make_fit_func(sqrt_s)
    sig_fit = f(y_plot, *popt)

    #LO fit
    LO_f = make_LO_fit_func(LO_sqrt_s)
    LO_sig_fit = LO_f(y_plot, *LO_popt)

    # Bande d'incertitude (propagation ±1σ des paramètres)
    sig_lo = f(y_plot, max(popt[0] - perr[0], 0), max(popt[1] - perr[1], 0))
    sig_hi = f(y_plot, popt[0] + perr[0], popt[1] + perr[1])

    LO_sig_lo = LO_f(y_plot, max(LO_popt[0] - LO_perr[0], 0)
                    # ,max(LO_popt[1] - LO_perr[1], 0)
                    # ,max(LO_popt[2]-LO_perr[2],0)
                    )
    LO_sig_hi = LO_f(y_plot, LO_popt[0] + LO_perr[0]
                    # ,LO_popt[1] + LO_perr[1]
                    # ,LO_popt[2]+LO_perr[2]
                    )

    ax.fill_between(y_plot, sig_lo, sig_hi, alpha=0.20, color=color, label="Incertitude fit cosh ±1σ")
    ax.plot(y_plot, sig_fit, "-", color=color, lw=2.0,
            label=r"$N\,\left(1-\frac{2M_T\cosh y}{\sqrt{s}}\right)^{\!n}$")
    ax.fill_between(y_plot, LO_sig_lo, LO_sig_hi, alpha=0.5, color=color, label="Incertitude fit LO ±1σ")
    ax.plot(y_plot, LO_sig_fit, "--", color=color, lw=2.0,
            label=r"$\sigma$ LO")

    # Points de données avec barres d'erreur
    ax.errorbar(df["y"], df["sigma"],
                yerr=[df["err_tot"], df["err_tot"]],
                fmt="o", color=color, label="Données", zorder=5,
                capsize=4, markersize=6, linewidth=1.5)

    ax.set_xlabel("Rapidité $y$", fontsize=13)
    ax.set_ylabel(r"$d\sigma/dy\ [\mu\mathrm{b}]$", fontsize=13)
    sqrt_label = f"{sqrt_s/1000:.0f}"
    ax.set_title(rf"J/$\psi$ en pp à $\sqrt{{s}}={sqrt_label}$ TeV", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)

    txt = (rf"$N  = {popt[0]:.2f} \pm {perr[0]:.2f}\ \mu\mathrm{{b}}$"
           "\n"
           rf"$n  = {popt[1]:.2f} \pm {perr[1]:.2f}$"
           "\n"
           rf"$\chi^2/\mathrm{{ndf}} = {chi2:.2f}$"
           "\n"
           rf"$M_\bot = \sqrt{{M^2_{{J/\psi}}+p_\bot^2}}= {M_T:.3f}\ \mathrm{{GeV}}$"
           "\n"
           rf"$p_\bot = {p_T:.1f}\ \mathrm{{GeV}}$")
    LO_txt = (rf"$N_{{LO}}  = {LO_popt[0]:.2e} \pm {LO_perr[0]:.2e}\ \mu\mathrm{{b}}$"
           "\n"
        #    rf"$\mu  = {LO_popt[1]:.2f} \pm {LO_perr[1]:.2f} * M_{{J/\psi}}$"
        #    "\n"
        #    rf"$\mu_f = {LO_popt[1]:.2f} \pm {LO_perr[1]:.2f} * M_{{J/\psi}}$"
        #    "\n"
           rf"$\mu  = {a:.1f} * M_{{J/\psi}}$"
           "\n"
           rf"$\mu_f = {b:.1f} * M_{{J/\psi}}$"
           "\n"
           rf"$\chi^2/\mathrm{{ndf}} = {LO_chi2:.2f}$"
           "\n"
           rf"$M_\bot = \sqrt{{M^2_{{J/\psi}}+p_\bot^2}}= {M_T:.3f}\ \mathrm{{GeV}}$"
           "\n"
           rf"$p_\bot = {p_T:.1f}\ \mathrm{{GeV}}$")
    ax.text(0.04, 0.05, LO_txt, transform=ax.transAxes, fontsize=9,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.45", fc="white", alpha=0.85, ec=color))

plt.suptitle(
    r"Fit $d\sigma/dy$ J/$\psi$ : $f(y)=N\,(1-2M_T\cosh y/\sqrt{s})^n$",
    fontsize=13, y=1.01
)
plt.tight_layout()
output_fig = "fit_jpsi_rapidity.pdf"
plt.savefig(os.path.join(plots_dir,output_fig), dpi=150, bbox_inches="tight")
print(f"Figure sauvegardée : {output_fig}")

# ─────────────────────────────────────────────────────────────────────────────
# 6bis.  Graphiques en xf
# ─────────────────────────────────────────────────────────────────────────────

# n_panels = len(fit_xf_results)
# fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5.5), sharey=False)
# if n_panels == 1:
#     axes = [axes]

# for ax, (label, res) in zip(axes, fit_xf_results.items()):
#     color  = color_map.get(label, "C3")
#     popt   = res["popt"]
#     perr   = res["perr"]
#     sqrt_s = res["sqrt_s"]
#     df     = res["df"]
#     chi2   = res["chi2_ndf"]

#     # Courbe du fit
#     xf_plot  = np.linspace(0.0, y_to_xf(df["y"],sqrt_s).max() * 1.15, 500)
#     f       = make_xf_fit_func(sqrt_s)
#     sig_fit = f(xf_plot, *popt)

#     # Bande d'incertitude (propagation ±1σ des paramètres)
#     sig_lo = f(xf_plot, max(popt[0] - perr[0], 0), max(popt[1] - perr[1], 0))
#     sig_hi = f(xf_plot, popt[0] + perr[0], popt[1] + perr[1])

#     ax.fill_between(xf_plot, sig_lo, sig_hi, alpha=0.20, color=color, label="Incertitude fit ±1σ")
#     ax.plot(xf_plot, sig_fit, "-", color=color, lw=2.0,
#             label=r"$N\,\frac{\left(1-x'\right)^{\!n}}{x'}$")

#     xf = y_to_xf(df["y"],sqrt_s)
#     sigma = df["sigma"]*sqrt_s/(2*M_T*np.cosh(df["y"]))
#     y_err = [df["err_tot"]*sqrt_s/(2*M_T*np.cosh(df["y"])), df["err_tot"]*sqrt_s/(2*M_T*np.cosh(df["y"]))]
#     # Points de données avec barres d'erreur
#     ax.errorbar(xf, sigma,
#                 yerr=y_err,
#                 fmt="o", color=color, label="Données", zorder=5,
#                 capsize=4, markersize=6, linewidth=1.5)

#     ax.set_xlabel(r"$x_F$", fontsize=13)
#     ax.set_ylabel(r"$d\sigma/dx_F\ [\mu\mathrm{b}]$", fontsize=13)
#     ax.set_yscale('log')
#     sqrt_label = f"{sqrt_s/1000:.0f}"
#     ax.set_title(rf"J/$\psi$ en pp à $\sqrt{{s}}={sqrt_label}$ TeV", fontsize=12)
#     ax.legend(fontsize=9, loc="upper right")
#     ax.set_xlim(left=0.0)
#     # ax.set_ylim(bottom=0.0)

#     txt = (rf"$N  = {popt[0]:.2f} \pm {perr[0]:.2f}\ \mu\mathrm{{b}}$"
#            "\n"
#            rf"$n  = {popt[1]:.2f} \pm {perr[1]:.2f}$"
#            "\n"
#            rf"$\chi^2/\mathrm{{ndf}} = {chi2:.2f}$"
#            "\n"
#            rf"$M_\bot = \sqrt{{M^2_{{J/\psi}}+p_\bot^2}}= {M_T:.3f}\ \mathrm{{GeV}}$"
#            "\n"
#            rf"$p_\bot = {p_T:.1f}\ \mathrm{{GeV}}$")
#     ax.text(0.65, 0.5, txt, transform=ax.transAxes, fontsize=9,
#             va="bottom", ha="left",
#             bbox=dict(boxstyle="round,pad=0.45", fc="white", alpha=0.85, ec=color))

# plt.suptitle(
#     r"Fit $d\sigma/dx_F$ J/$\psi$ : $f(y)=N\,(1-x')^n/x'$",
#     fontsize=13, y=1.01
# )
# plt.tight_layout()
# output_fig = "fit_jpsi_xf.pdf"
# plt.savefig(os.path.join(plots_dir,output_fig), dpi=150, bbox_inches="tight")
# print(f"Figure sauvegardée : {output_fig}")