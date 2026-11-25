import os
import re
import numpy as np
import yaml
import shutil

def read_info_file(path):
    """
    Lit un fichier .info LHAPDF6 et retourne son contenu sous forme de dict
    Python (clés/valeurs, listes converties en list Python).
    """
    with open(path) as f:
        return yaml.safe_load(f)
    
def get_info_value(path, key, default=None):
    """
    Récupère un seul élément du .info par sa clé, ex:
        get_info_value(".../EPPS21..._Pb208.info", "NumMembers")
        get_info_value(..., "ErrorType")
        get_info_value(..., "Flavors")
    Retourne `default` si la clé n'existe pas.
    """
    info = read_info_file(path)
    return info.get(key, default)

def parse_lhapdf_member(path):
    """
    Parse un fichier membre LHAPDF6 existant (.dat) et retourne la liste
    de ses sous-grilles (un set peut avoir plusieurs blocs '---' séparés,
    typiquement un par intervalle de Q entre seuils de masse charm/bottom/top).

    Retour
    ------
    list[dict] : un dict par bloc, avec les clés :
        "xs"      : array 1D des x-knots
        "qs"      : array 1D des Q-knots (en GeV)
        "flavors" : liste des IDs PDG dans l'ordre des colonnes
        "values"  : array (len(xs), len(qs), len(flavors))
    """
    with open(path) as f:
        content = f.read()

    # Le header YAML (PdfType, Format) est suivi du premier '---'
    # puis chaque bloc de données est encadré par '---' ... '---'
    parts = content.split("---")
    data_blocks = [p.strip() for p in parts[1:] if p.strip()]

    blocks = []
    for block in data_blocks:
        lines = block.splitlines()
        xs = np.array([float(v) for v in lines[0].split()])
        qs = np.array([float(v) for v in lines[1].split()])
        flavors = [int(v) for v in lines[2].split()]

        n_flav = len(flavors)
        data_lines = lines[3:]
        assert len(data_lines) == len(xs) * len(qs), (
            f"Nombre de lignes de données ({len(data_lines)}) incohérent "
            f"avec len(xs)*len(qs) = {len(xs) * len(qs)} dans {path}"
        )

        values = np.empty((len(xs), len(qs), n_flav))
        idx = 0
        for ix in range(len(xs)):
            for iq in range(len(qs)):
                row = [float(v) for v in data_lines[idx].split()]
                values[ix, iq, :] = row
                idx += 1

        blocks.append({"xs": xs, "qs": qs, "flavors": flavors, "values": values})

    return blocks


def load_all_members(set_dir, set_name):
    """
    Charge tous les membres d'un set LHAPDF6 existant (ex: EPPS21) depuis
    son dossier, dans l'ordre _0000, _0001, _0002, ...

    Suppose que chaque fichier membre n'a qu'un seul bloc de grille
    (cas d'EPPS21nlo_CT18Anlo). Si un membre a plusieurs blocs, une erreur
    explicite est levée -- il faudrait alors gérer le multi-bloc, ce qui
    n'est pas le cas ici.

    Paramètres
    ----------
    set_dir : str
        Dossier contenant les fichiers <set_name>_NNNN.dat
        (ex: ".../EPPS21nlo_CT18Anlo_Pb208")
    set_name : str
        Préfixe des fichiers membres (ex: "EPPS21nlo_CT18Anlo_Pb208")

    Retour
    ------
    xs, qs, flavors : les grilles communes à tous les membres (issues du
        membre central _0000), à réutiliser telles quelles pour un set dérivé.
    members : list[dict]
        [{"pdftype": ..., "values": ...}, ...] dans l'ordre des fichiers,
        prêt à être passé à create_lhapdf_set.
    """
    # Repère tous les fichiers <set_name>_NNNN.dat, triés par numéro de membre
    pattern = re.compile(rf"^{re.escape(set_name)}_(\d{{4}})\.dat$")
    candidates = []
    for fname in os.listdir(set_dir):
        m = pattern.match(fname)
        if m:
            candidates.append((int(m.group(1)), fname))
    candidates.sort(key=lambda t: t[0])

    if not candidates:
        raise FileNotFoundError(
            f"Aucun fichier '{set_name}_NNNN.dat' trouvé dans {set_dir}"
        )

    members = []
    xs = qs = flavors = None

    for member_index, fname in candidates:
        path = os.path.join(set_dir, fname)
        blocks = parse_lhapdf_member(path)

        if len(blocks) != 1:
            raise ValueError(
                f"{fname} contient {len(blocks)} blocs -- cette fonction "
                f"suppose un seul bloc par membre. Il faut étendre la "
                f"gestion multi-bloc pour ce set."
            )

        block = blocks[0]

        # Vérifie que la grille est bien identique d'un membre à l'autre
        if xs is None:
            xs, qs, flavors = block["xs"], block["qs"], block["flavors"]
        else:
            if not (np.array_equal(xs, block["xs"]) and np.array_equal(qs, block["qs"])):
                raise ValueError(
                    f"{fname} n'a pas la même grille (xs/qs) que le membre "
                    f"central -- vérifiez le fichier."
                )
            if flavors != block["flavors"]:
                raise ValueError(
                    f"{fname} n'a pas le même ordre de flavors que le "
                    f"membre central: {block['flavors']} vs {flavors}"
                )

        pdftype = "central" if member_index == 0 else "error"
        members.append({"pdftype": pdftype, "values": block["values"]})

    print(f"{len(members)} membres chargés depuis {set_dir}")
    return xs, qs, flavors, members


def write_lhagrid_block(f, xs, qs, flavors, values):
    """values: array de forme (len(xs), len(qs), len(flavors))"""
    f.write(" ".join(f"{x:.6e}" for x in xs) + "\n")
    f.write(" ".join(f"{q:.6e}" for q in qs) + "\n")
    f.write(" ".join(str(fl) for fl in flavors) + "\n")
    for ix in range(len(xs)):
        for iq in range(len(qs)):
            row = values[ix, iq, :]
            f.write(" ".join(f"{v:.8e}" for v in row) + "\n")


def write_member(path, pdftype, xs, qs, flavors, values):
    with open(path, "w") as f:
        f.write(f"PdfType: {pdftype}\nFormat: lhagrid1\n---\n")
        write_lhagrid_block(f, xs, qs, flavors, values)
        f.write("---\n")


def write_info_file(path, info_dict):
    """
    Écrit le .info en YAML. info_dict est un dict Python classique ;
    les listes sont rendues au format YAML inline [a, b, c].
    L'ordre des clés est préservé (dict Python 3.7+ = ordonné).
    """
    def format_value(v):
        if isinstance(v, (list, tuple, np.ndarray)):
            items = list(v)
            # format scientifique propre pour les floats, sinon str brut
            formatted = []
            for item in items:
                if isinstance(item, float):
                    formatted.append(f"{item:.6e}")
                else:
                    formatted.append(str(item))
            return "[" + ", ".join(formatted) + "]"
        elif isinstance(v, str):
            return v
        else:
            return str(v)

    with open(path, "w") as f:
        for key, value in info_dict.items():
            f.write(f"{key}: {format_value(value)}\n")


def create_lhapdf_set(
    base_path,
    set_name,
    xs,
    qs,
    flavors,
    members,
    info_dict,
):
    """
    Crée un set LHAPDF6 complet et fonctionnel dans base_path/set_name/.

    Paramètres
    ----------
    base_path : str
        Répertoire parent où créer le dossier du set
        (ce chemin doit ensuite être ajouté via lhapdf.pathsAppend()).
    set_name : str
        Nom du set. Doit être identique au nom du dossier ET au préfixe
        des fichiers .info/.dat -- c'est géré automatiquement ici.
    xs, qs : array-like
        x-knots et Q-knots (Q en GeV, pas Q²), croissants.
    flavors : list[int]
        IDs PDG des flavors, dans l'ordre des colonnes de `values`.
    members : list[dict]
        Un dict par membre, dans l'ordre (membre 0 = central en premier).
        Chaque dict : {"pdftype": "central" ou "error",
                        "values": array (len(xs), len(qs), len(flavors))}
    info_dict : dict
        Contenu du .info (SetDesc, Format, NumMembers, ErrorType, etc.).
        NumMembers doit correspondre à len(members).

    Retour
    ------
    str : chemin du dossier créé
    """
    if info_dict.get("NumMembers") != len(members):
        raise ValueError(
            f"NumMembers ({info_dict.get('NumMembers')}) dans info_dict "
            f"ne correspond pas au nombre de membres fournis ({len(members)})."
        )

    set_dir = os.path.join(base_path, set_name)
    os.makedirs(set_dir, exist_ok=True)

    # 1. Fichier .info
    info_path = os.path.join(set_dir, f"{set_name}.info")
    write_info_file(info_path, info_dict)

    # 2. Fichiers membres, numérotés _0000.dat, _0001.dat, ...
    for i, member in enumerate(members):
        member_path = os.path.join(set_dir, f"{set_name}_{i:04d}.dat")
        write_member(
            member_path,
            pdftype=member["pdftype"],
            xs=xs,
            qs=qs,
            flavors=flavors,
            values=member["values"],
        )

    print(f"Set '{set_name}' créé dans {set_dir} ({len(members)} membres).")
    return set_dir

def reduce_set(replica_list,old_path,new_path,set_name,new_set_name,FCEL = True):
    set_dir = os.path.join(old_path,set_name)
    new_set_dir = os.path.join(new_path,new_set_name)
    counter = 1 
    for i,n in enumerate(replica_list):
        if n == 0 :
            continue
        for j in range(counter,counter+n):
            member_path = os.path.join(set_dir, f"{set_name}_{i+1:04d}.dat") # doit rajouter 1 car c'est un array commençant à 0
            new_member_path = os.path.join(new_set_dir, f"{new_set_name}_{j:04d}.dat")
            shutil.copy2(member_path,new_member_path)
        counter += n
    info_loc = os.path.join(set_dir,set_name+'.info')
    new_info_loc = os.path.join(new_set_dir,new_set_name+'.info')
    info = read_info_file(info_loc)
    if FCEL: 
        desc = "J/psi data (FCEL) + " +set_name
    else : 
        desc = "J/psi data (no FCEL) + " +set_name
    new_info_dict = {
        "SetDesc": desc,
        "SetIndex": 90000,
        "Authors": "Djessy",
        "Reference": "arXiv:xxxx.xxxxx",
        "Format": "lhagrid1",
        "DataVersion": 1,
        "NumMembers": counter , # we take in count the central member, normally computed in the reweighting.py file 
        "Particle": info["Particle"],
        "Flavors": info["Flavors"],
        "OrderQCD": info["OrderQCD"],
        "FlavorScheme": info["FlavorScheme"],
        "NumFlavors": info["NumFlavors"],
        "ErrorType": "replicas",
        "XMin": info["XMin"],
        "XMax": info["XMax"],
        "QMin": info["QMin"],
        "QMax": info["QMax"],
        "MZ": info["MZ"],
        "MUp": info["MUp"],
        "MDown": info["MDown"],
        "MStrange": info["MStrange"],
        "MCharm": info["MCharm"],
        "MBottom": info["MBottom"],
        "MTop": info["MTop"],
        "AlphaS_MZ": info["AlphaS_MZ"],
        "AlphaS_OrderQCD": info["AlphaS_OrderQCD"],
        "AlphaS_Type": info["AlphaS_Type"],
        "AlphaS_Qs": info["AlphaS_Qs"],
        "AlphaS_Vals": info["AlphaS_Vals"]
    }
    write_info_file(new_info_loc,new_info_dict)
    


# if __name__ == "__main__":
#     # ------------------------------------------------------------------
#     # Exemple d'utilisation avec des grilles factices (à remplacer par
#     # vos vraies grilles x/Q et vos valeurs PDF)
#     # ------------------------------------------------------------------
#     xs = np.geomspace(1e-6, 1.0, 20)
#     qs = np.geomspace(1.3, 1e5, 15)
#     flavors = [-5, -4, -3, -2, -1, 21, 1, 2, 3, 4, 5]

#     n_eigen = 2  # exemple réduit (2 eigenvectors -> 1 central + 4 error = 5 membres)
#     rng = np.random.default_rng(0)

#     def fake_values():
#         return rng.uniform(0.01, 1.0, size=(len(xs), len(qs), len(flavors)))

#     members = [{"pdftype": "central", "values": fake_values()}]
#     for k in range(n_eigen):
#         members.append({"pdftype": "error", "values": fake_values()})  # S_k+
#         members.append({"pdftype": "error", "values": fake_values()})  # S_k-

#     info_dict = {
#         "SetDesc": "Set custom de test",
#         "SetIndex": 90000,
#         "Authors": "Djessy",
#         "Reference": "arXiv:xxxx.xxxxx",
#         "Format": "lhagrid1",
#         "DataVersion": 1,
#         "NumMembers": len(members),
#         "Particle": 2212,
#         "Flavors": flavors,
#         "OrderQCD": 1,
#         "FlavorScheme": "variable",
#         "NumFlavors": 5,
#         "ErrorType": "hessian",
#         "ErrorConfLevel": 68.268949,
#         "XMin": float(xs[0]),
#         "XMax": float(xs[-1]),
#         "QMin": float(qs[0]),
#         "QMax": float(qs[-1]),
#         "MZ": 91.1876,
#         "MUp": 0,
#         "MDown": 0,
#         "MStrange": 0,
#         "MCharm": 1.51,
#         "MBottom": 4.92,
#         "MTop": 172.5,
#         "AlphaS_MZ": 0.118,
#         "AlphaS_OrderQCD": 1,
#     }

    # out_path = "/home/claude/lhapdf_test_output"
    # create_lhapdf_set(out_path, "MonSet", xs, qs, flavors, members, info_dict)