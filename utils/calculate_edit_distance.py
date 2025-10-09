from rdkit import Chem
import numpy as np


def normalize_edit_dist(dist_list):
    if len(dist_list) == 0:
        return np.array(dist_list)

    max_dst = max(dist_list)
    min_dst = min(dist_list)
    new_dist_list = [(d - min_dst) / (max_dst - min_dst + 1e-7) for d in dist_list]

    return np.array(new_dist_list)


def get_bond_info(mol):
    if mol is None:
        return {}

    bond_info = {}
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()
        bt = bond.GetBondTypeAsDouble()
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


def map_step_rxn(r_smi, p_smi, rxn_mapper):
    rxn = [f'{r_smi}>>{p_smi}']
    results = rxn_mapper.get_attention_guided_atom_maps(rxn)
    mapped_r_smi, mapped_prod_smi = results[0]['mapped_rxn'].split('>>')
    return mapped_r_smi, mapped_prod_smi


def remap_smi_according_to_infer(infer_smi, cur_p_smi, cur_r_smi):
    infer_split = [smi for smi in infer_smi.split('.')]
    cur_p_mol = Chem.MolFromSmiles(cur_p_smi)  # current step mapped  p smi
    # infer_mol = Chem.MolFromSmiles(infer_smi)  # last step mapped r smi
    cur_r_mol = Chem.MolFromSmiles(cur_r_smi)

    max_map_num = -1
    correspondence = {}
    replace_id = None
    for idx, infer in enumerate(infer_split):
        infer_mol = Chem.MolFromSmiles(infer)
        matches = list(infer_mol.GetSubstructMatches(cur_p_mol))
        if len(matches) > 0:
            idx_amap = {atom.GetIdx(): atom.GetAtomMapNum()
            for atom in cur_p_mol.GetAtoms()}  # current mol idx: map num

            if matches:
               for i, match_idx in enumerate(matches[0]):
                    match_anum = infer_mol.GetAtomWithIdx(match_idx).GetAtomMapNum()
                    old_anum = idx_amap[i]
                    correspondence[old_anum] = match_anum  # cur mapnum: infer mapnum
                    replace_id = idx

        max_amap = max([atom.GetAtomMapNum() for atom in infer_mol.GetAtoms()])
        max_map_num = max(max_map_num, max_amap)

    assert max_map_num > 0 and replace_id is not None

    for a in cur_r_mol.GetAtoms():
        anum = a.GetAtomMapNum()
        if anum in correspondence:
            new_anum = correspondence[anum]
            a.SetAtomMapNum(new_anum)
        else:
            a.SetAtomMapNum(max_map_num + 1)
            max_map_num += 1

    cur_r_smi = Chem.MolToSmiles(cur_r_mol, isomericSmiles=True)
    infer_split[replace_id] = cur_r_smi
    remapped_r_smi = '.'.join(infer_split)
    return remapped_r_smi


def calculate_edits_distance(target_smi, cur_r_smi):
    target_mol = Chem.MolFromSmiles(target_smi)
    cur_r_mol = Chem.MolFromSmiles(cur_r_smi)
    # get bonds info
    target_mol_bonds = get_bond_info(target_mol)
    cur_r_mol_bonds = get_bond_info(cur_r_mol)
    # get chiral info
    target_mol_chiral = get_atom_Chiral(target_mol)
    cur_r_mol_chiral = get_atom_Chiral(cur_r_mol)

    # calculate bond dist
    bond_dist = 0
    for bond in cur_r_mol_bonds:
        if bond not in target_mol_bonds:
            bond_dist += cur_r_mol_bonds[bond][0]
        else:
            if cur_r_mol_bonds[bond] != target_mol_bonds[bond]:
                bond_dist += abs((target_mol_bonds[bond][0] - cur_r_mol_bonds[bond][0]))

    for bond in target_mol_bonds:
        if bond not in cur_r_mol_bonds:
            bond_dist += target_mol_bonds[bond][0]

    # calculate chiral dist
    atom_dist = 0
    for atom in cur_r_mol_chiral:
        if atom not in target_mol_chiral:
            atom_dist += 1
        else:
            if cur_r_mol_chiral[atom] != target_mol_chiral[atom]:
                atom_dist += 1

    for atom in target_mol_chiral:
        if atom not in cur_r_mol_chiral:
            atom_dist += 1

    total_dist = bond_dist + atom_dist
    return total_dist

