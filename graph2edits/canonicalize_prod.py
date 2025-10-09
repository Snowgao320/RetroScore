"""
Canonicalize the product SMILES, and then use substructure matching to infer
the correspondence to the original atom-mapped order. This correspondence is then
used to renumber the reactant atoms.
"""


from rdkit import Chem
import os
import argparse
import pandas as pd


def canonicalize_prod(p):
    import copy
    p = copy.deepcopy(p)
    p = canonicalize(p)
    if p is None:
        return None
    p_mol = Chem.MolFromSmiles(p)
    for atom in p_mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    p = Chem.MolToSmiles(p_mol, kekuleSmiles=True, canonical=True)
    return p


def canonicalize(smiles):
    try:
        tmp = Chem.MolFromSmiles(smiles)
    except:
        print('no mol', flush=True)
        return  None
    if tmp is None:
        return None
    tmp = Chem.RemoveHs(tmp)
    [a.ClearProp('molAtomMapNumber') for a in tmp.GetAtoms()]
    return Chem.MolToSmiles(tmp, kekuleSmiles=True, canonical=True)


def fix_charge(mol):
    # fix simple atomic charge, eg. 'COO-', 'CH3O-', '(S=O)O-', '-NH3+', 'NH4+', 'NH2+', 'S-'
    for atom in mol.GetAtoms():
        explicit_hs = atom.GetNumExplicitHs()
        charge = atom.GetFormalCharge()
        bond_vals = int(sum([b.GetBondTypeAsDouble()
                        for b in atom.GetBonds()]))
        if atom.GetSymbol() == 'O' and bond_vals == 1 and charge == -1 and explicit_hs == 0:
            if atom.GetNeighbors()[0].GetSymbol() != 'N':
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(1)

        if atom.GetSymbol() == 'N' and bond_vals == 1 and charge == 1 and explicit_hs == 3:
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(2)

        if atom.GetSymbol() == 'N' and bond_vals == 0 and charge == 1 and explicit_hs == 4:
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(3)

        if atom.GetSymbol() == 'N' and bond_vals == 2 and charge == 1 and explicit_hs == 2:
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(1)

        if atom.GetSymbol() == 'S' and charge == -1 and explicit_hs == 0 and bond_vals == 1:
            atom.SetNumExplicitHs(1)
            atom.SetFormalCharge(0)
    return mol


def infer_correspondence(p):
    orig_mol = Chem.MolFromSmiles(p)
    orig_mol = Chem.RemoveHs(orig_mol)
    canon_mol = Chem.MolFromSmiles(canonicalize_prod(p))
    matches = list(canon_mol.GetSubstructMatches(orig_mol))
    idx_amap = {atom.GetIdx(): atom.GetAtomMapNum()
                for atom in orig_mol.GetAtoms()}

    correspondence = {}
    if matches:
        for idx, match_idx in enumerate(matches[0]):
            match_anum = canon_mol.GetAtomWithIdx(match_idx).GetAtomMapNum()
            old_anum = idx_amap[idx]
            correspondence[old_anum] = match_anum
    return correspondence


def remap_rxn_smi(rxn_smi):
    r, p = rxn_smi.split(">>")
    canon_p = canonicalize_prod(p)
    if canon_p is None:
        return None, None
    canon_mol = Chem.MolFromSmiles(canon_p)
    correspondence = infer_correspondence(p)
    if len(correspondence) == 0:
        return None, None

    r_split = r.split('.')
    reactant = []
    for r_smi in r_split:
        r_split_mol = Chem.MolFromSmiles(r_smi)
        r_split_mol = Chem.RemoveHs(r_split_mol)
        not_reagent = False
        for a in r_split_mol.GetAtoms():
            anum = a.GetAtomMapNum()
            if anum in correspondence:
                not_reagent = not_reagent or True
        if not_reagent:
            canon_r_smi = Chem.MolToSmiles(r_split_mol, kekuleSmiles=True, canonical=True)
            reactant.append(canon_r_smi)
    r = '.'.join(reactant)

    rmol = Chem.MolFromSmiles(r)
    if rmol is None or rmol.GetNumAtoms() <= 1:
        return None, None

    for atom in rmol.GetAtoms():
        atomnum = atom.GetAtomMapNum()
        if atomnum in correspondence:
            newatomnum = correspondence[atomnum]
            atom.SetAtomMapNum(newatomnum)
        else:
            atom.SetAtomMapNum(0)

    max_amap = max([atom.GetAtomMapNum() for atom in rmol.GetAtoms()])
    for atom in rmol.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            atom.SetAtomMapNum(max_amap + 1)
            max_amap += 1

    # fix simple atomic charge, eg. 'COO-', 'CH3O-', '(S=O)O-', '-NH3+', 'NH4+', 'NH2+', 'S-'
    rmol = fix_charge(rmol)
    canon_mol = fix_charge(canon_mol)
    rmol = Chem.MolFromSmiles(Chem.MolToSmiles(rmol, kekuleSmiles=True, canonical=True))
    pmol = Chem.MolFromSmiles(Chem.MolToSmiles(canon_mol, kekuleSmiles=True, canonical=True))
    rxn_smi_new = Chem.MolToSmiles(rmol) + ">>" + Chem.MolToSmiles(pmol)
    return rxn_smi_new, correspondence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='uspto_full',
                        help='dataset: USPTO_50k or USPTO_full')
    parser.add_argument('--mode', type=str, default='valid',
                        help='Type of dataset being prepared: train or valid or test')
    args = parser.parse_args()

    args.dataset = args.dataset.lower()
    datadir = f'data/{args.dataset}/'
    new_file = f'canonicalized_{args.mode}.csv'
    filename = f'raw_{args.mode}.csv'
    df = pd.read_csv(os.path.join(datadir, filename))
    print(f"Processing file of size: {len(df)}")

    if args.dataset == 'uspto_50k':
        new_dict = {'id': [], 'class': [], 'reactants>reagents>production': []}
    else:
        new_dict = {'id': [], 'reactants>reagents>production': []}
    for idx in range(len(df)):
        element = df.loc[idx]
        if idx % 2000 == 0:
            print('canonicalized:', idx)
        if args.dataset == 'uspto_50k':
            uspto_id, class_id, rxn_smi = element['id'], element['class'], element['reactants>reagents>production']
            rxn_smi_new, _ = remap_rxn_smi(rxn_smi)
            new_dict['id'].append(uspto_id)
            if args.dataset == 'uspto_50k':
                new_dict['class'].append(class_id)
            new_dict['reactants>reagents>production'].append(rxn_smi_new)
        else:
            uspto_id, rxn_smi = element['id'], element['reactants>reagents>production']
            rxn_smi_new, _ = remap_rxn_smi(rxn_smi)
            if rxn_smi_new == None:
                continue
            new_dict['id'].append(uspto_id)
            new_dict['reactants>reagents>production'].append(rxn_smi_new)

    new_df = pd.DataFrame.from_dict(new_dict)
    new_df.to_csv(os.path.join(datadir, new_file), index=False)


if __name__ == "__main__":
    # smi = 'CC(c1cc2ccn(C(OC(C)(C)C)=O)c2cc1)=O'
    # smi = canonicalize_prod(smi)
    # print(smi)
    main()
