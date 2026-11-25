#!/usr/bin/env python3
"""
Comparaison de plusieurs sets de PDF (LHAPDF) : x f_i(x, Q^2) en fonction de x.

Le set de référence pour le panneau de ratio est le premier de `sets`.
Les incertitudes sont calculées via PDFSet.uncertainty(), qui gère
automatiquement les sets Hessiens, à réplicas et à paramètres multiples.
"""
import numpy as np
import matplotlib.pyplot as plt
import lhapdf

# --------------------------------------------------------------------------
# SETTINGS
# --------------------------------------------------------------------------
SETTINGS = {
    # label affiché -> nom du set LHAPDF
    "sets": {
        "CT14NLO": "CT14nlo",
        "NNPDF30": "nNNPDF30_nlo_as_0118_p",
        "NNPDF30 noLHCbD": "nNNPDF30_nlo_as_0118_noLHCbD_p"
    },
    "Q2": 10.0,          # échelle Q^2 en GeV^2
    "flavor": "g",       # "g", "u", "d", "s", "c", "b", "ubar", "dbar", ... ou PID entier
    "xmin": 0.4,
    "xmax": 1.0,
    "npts": 200,
    "xf": True,          # True : trace x f(x,Q^2) ; False : f(x,Q^2)
    "cl": 68,            # niveau de confiance des bandes d'incertitude (%)
    "bands": True,       # False : courbe centrale uniquement (plus rapide)
    "ratio": True,       # panneau de ratio au premier set
    "xlog": True,
    "ylog": False,
    "output": None,   # None pour ne pas sauvegarder
}

# Correspondance nom -> PID (convention PDG)
FLAVORS = {
    "g": 21, "gluon": 21,
    "d": 1, "u": 2, "s": 3, "c": 4, "b": 5,
    "dbar": -1, "ubar": -2, "sbar": -3, "cbar": -4, "bbar": -5,
}
FLAVOR_LABELS = {
    21: "g", 1: "d", 2: "u", 3: "s", 4: "c", 5: "b",
    -1: r"\bar{d}", -2: r"\bar{u}", -3: r"\bar{s}", -4: r"\bar{c}", -5: r"\bar{b}",
}


def get_pid(flavor):
    """Convertit un nom de saveur (ou un PID) en PID entier."""
    if isinstance(flavor, (int, np.integer)):
        return int(flavor)
    try:
        return FLAVORS[flavor.lower() if flavor not in FLAVORS else flavor]
    except KeyError:
        raise ValueError(f"Saveur inconnue : {flavor!r}. Choix : {list(FLAVORS)} ou un PID entier.")


def evaluate_set(set_name, pid, xs, q2, xf=True, cl=68, bands=True):
    """
    Retourne (central, err_plus, err_minus) sur la grille xs.
    Les erreurs sont des écarts (positifs) par rapport à la valeur centrale.
    """
    pdfset = lhapdf.getPDFSet(set_name)
    pdfs = pdfset.mkPDFs() if bands else [pdfset.mkPDF(0)]

    # valeurs[membre, ix]
    vals = np.array([[p.xfxQ2(pid, x, q2) for x in xs] for p in pdfs])
    if not xf:
        vals = vals / xs

    if not bands:
        zeros = np.zeros_like(xs)
        return vals[0], zeros, zeros

    central = np.empty_like(xs)
    errp = np.empty_like(xs)
    errm = np.empty_like(xs)
    for i in range(len(xs)):
        unc = pdfset.uncertainty(vals[:, i].tolist(), cl)
        central[i], errp[i], errm[i] = unc.central, unc.errplus, unc.errminus
    return central, errp, errm


def main(cfg=SETTINGS):
    pid = get_pid(cfg["flavor"])
    q2 = cfg["Q2"]
    xs = (np.logspace(np.log10(cfg["xmin"]), np.log10(cfg["xmax"]), cfg["npts"])
          if cfg["xlog"] else np.linspace(cfg["xmin"], cfg["xmax"], cfg["npts"]))

    # Évaluation de tous les sets
    results = {}
    for label, name in cfg["sets"].items():
        print(f"Évaluation de {label} ({name}) ...")
        results[label] = evaluate_set(name, pid, xs, q2, cfg["xf"], cfg["cl"], cfg["bands"])

    # Figure
    if cfg["ratio"]:
        fig, (ax, axr) = plt.subplots(
            2, 1, figsize=(6.5, 7), sharex=True,
            gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.05})
    else:
        fig, ax = plt.subplots(figsize=(6.5, 5))
        axr = None

    fl = FLAVOR_LABELS.get(pid, str(pid))
    ylabel = rf"$x f_{{{fl}}}(x, Q^2)$" if cfg["xf"] else rf"$f_{{{fl}}}(x, Q^2)$"

    ref_label = next(iter(results))
    ref_central = results[ref_label][0]

    for label, (c, ep, em) in results.items():
        line, = ax.plot(xs, c, label=label)
        if cfg["bands"]:
            ax.fill_between(xs, c - em, c + ep, color=line.get_color(), alpha=0.25)
        if axr is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                r = c / ref_central
                rp, rm = (c + ep) / ref_central, (c - em) / ref_central
            axr.plot(xs, r, color=line.get_color())
            if cfg["bands"]:
                axr.fill_between(xs, rm, rp, color=line.get_color(), alpha=0.25)

    ax.set_ylabel(ylabel)
    ax.set_title(rf"$Q^2 = {q2:g}\ \mathrm{{GeV}}^2$")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    if cfg["xlog"]:
        ax.set_xscale("log")
    if cfg["ylog"]:
        ax.set_yscale("log")

    target = axr if axr is not None else ax
    target.set_xlabel(r"$x$")
    if axr is not None:
        axr.axhline(1.0, color="k", lw=0.8)
        axr.set_ylabel(f"ratio / {ref_label}")
        axr.set_ylim(0.6, 1.4)
        axr.grid(alpha=0.3)

    ax.set_xlim(cfg["xmin"], cfg["xmax"])
    fig.tight_layout()
    if cfg["output"]:
        fig.savefig(cfg["output"], bbox_inches="tight")
        print(f"Figure sauvegardée : {cfg['output']}")
    plt.show()


if __name__ == "__main__":
    main()