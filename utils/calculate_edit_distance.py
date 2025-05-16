from rdkit import Chem
from canonicalize_prod import fix_charge
from rxnmapper import RXNMapper
import numpy as np

def normalize_total_diff(diff_list):
    if len(diff_list) == 0:
        return np.array(diff_list)

    max_diff = max(diff_list)
    min_diff = min(diff_list)
    new_diff_list = []
    for diff in diff_list:
        new_diff = (diff-min_diff)/(max_diff-min_diff+1e-7)
        new_diff_list.append(new_diff)

    return np.array(new_diff_list)

def get_bond_info(mol):
    if mol is None:
        return {}

    bond_info = {}
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()
        bt = int(bond.GetBondType())
        st = int(bond.GetStereo())
        bond_atoms = sorted([a1, a2])
        bond_info[tuple(bond_atoms)] = [bt, st]
    return bond_info


def get_atom_Chiral(mol):
    if mol is None:
        return {}

    atom_Chiral = {}
    for atom in mol.GetAtoms():
        if int(atom.GetChiralTag()) != 0:
            amap_num = atom.GetAtomMapNum()
            atom_Chiral[amap_num] = int(atom.GetChiralTag())
    return atom_Chiral


def map_step_rxn(r_smi, prod_smi):
    rxn_mapper = RXNMapper()
    rxns = [f'{r_smi}>>{prod_smi}']
    results = rxn_mapper.get_attention_guided_atom_maps(rxns)
    mapped_r_smi, mapped_prod_smi = results[0]['mapped_rxn'].split('>>')
    return mapped_r_smi, mapped_prod_smi


def remap_smi_according_to_infer(infer_smi, cur_p_smi, cur_r_smi):
    infer_split = [smi for smi in infer_smi.split('.')]
    cur_p_mol = Chem.MolFromSmiles(cur_p_smi)  # current step p mapped smi mol
    # infer_mol = Chem.MolFromSmiles(infer_smi)  # last step r mapped smi mol
    cur_r_mol = Chem.MolFromSmiles(cur_r_smi)
    max_num = []
    for id, infer in enumerate(infer_split):
        infer_mol = Chem.MolFromSmiles(infer)
        matches = list(infer_mol.GetSubstructMatches(cur_p_mol))
        if len(matches) > 0:
            idx_amap = {atom.GetIdx(): atom.GetAtomMapNum()
            for atom in cur_p_mol.GetAtoms()}  # current mol idx: mapnum

            correspondence = {}
            if matches:
               for idx, match_idx in enumerate(matches[0]):
                    match_anum = infer_mol.GetAtomWithIdx(match_idx).GetAtomMapNum()
                    old_anum = idx_amap[idx]
                    correspondence[old_anum] = match_anum  # cur mapnum: infer mapnum
                    replace_id = id
        else:
            max_amap1 = max([atom.GetAtomMapNum() for atom in infer_mol.GetAtoms()])
            max_num.append(max_amap1)
            continue

    for atom in cur_r_mol.GetAtoms():
        atomnum = atom.GetAtomMapNum()
        if atomnum in correspondence:
            newatomnum = correspondence[atomnum]
            atom.SetAtomMapNum(newatomnum)
        else:
            atom.SetAtomMapNum(0)

    max_amap = max([atom.GetAtomMapNum() for atom in cur_r_mol.GetAtoms()])
    if len(max_num) > 0:
        max_amap = max(max_amap, max(max_num))

    for atom in cur_r_mol.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            atom.SetAtomMapNum(max_amap + 1)
            max_amap += 1

    # fix simple atomic charge, eg. 'COO-', 'CH3O-', '(S=O)O-', '-NH3+', 'NH4+', 'NH2+', 'S-'
    remapped_cur_r_mol = fix_charge(cur_r_mol)
    remapped_cur_p_mol = fix_charge(cur_p_mol)

    remapped_cur_r_smi = Chem.MolToSmiles(remapped_cur_r_mol)

    infer_split[replace_id] = remapped_cur_r_smi

    remapped_cur_r_smi = '.'.join(infer_split)
    remapped_cur_p_smi = Chem.MolToSmiles(remapped_cur_p_mol)

    return remapped_cur_r_smi, remapped_cur_p_smi


def calculate_edits_distance(target_smi, cur_smi):
    target_mol = Chem.MolFromSmiles(target_smi)
    cur_mol = Chem.MolFromSmiles(cur_smi)
    # get bonds info
    target_mol_bonds = get_bond_info(target_mol)
    cur_mol_bonds = get_bond_info(cur_mol)
    # get stereo info
    target_mol_stereo = get_atom_Chiral(target_mol)
    cur_mol_stereo = get_atom_Chiral(cur_mol)

    # calculate bond diff
    bond_diff = 0
    for bond in cur_mol_bonds:
        if bond not in target_mol_bonds:
            bond_diff += cur_mol_bonds[bond][0]
        else:
            if cur_mol_bonds[bond][0] != target_mol_bonds[bond][0]:
                bond_diff += abs((target_mol_bonds[bond][0] - cur_mol_bonds[bond][0]))

    for bond in target_mol_bonds:
        if bond not in cur_mol_bonds:
            bond_diff += target_mol_bonds[bond][0]

    # calculate stereo diff
    stereo_diff = 0
    d_stereo_diff = (target_mol_stereo.keys() | cur_mol_stereo.keys()) - (
                target_mol_stereo.keys() & cur_mol_stereo.keys())
    stereo_diff += len(d_stereo_diff)

    total_diff = abs(bond_diff) + stereo_diff
    return total_diff

