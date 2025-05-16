import pickle
import pandas as pd
import logging
import torch
from models.beam_search import BeamSearch
from retro_star.alg import molstar


class plan_handle:
    def __init__(self, one_step, value_fn, starting_mols, expansion_topk, iterations, max_routes_num):
        self.beam_model = BeamSearch(model=one_step, step_beam_size=expansion_topk,
                            beam_size=expansion_topk, use_rxn_class=False)
        self.value_fn = value_fn
        self.starting_mols = starting_mols
        self.iterations = iterations
        self.max_routes_num = max_routes_num

    def beam_model_run(self, x):
        with torch.no_grad():
            return self.beam_model.run(prod_smi=x, max_steps=9, rxn_class=None)

    def molstar_run(self, x, args):
        m = molstar(
            target_mol=x,
            starting_mols=self.starting_mols,
            expand_fn=self,
            value_fn=self.value_fn,
            iterations=self.iterations,
            max_routes_num = self.max_routes_num,
            args = args
        )
        return m



def prepare_starting_molecules(filename):
    logging.info('Loading starting molecules from %s' % filename)

    if filename[-3:] == 'csv':
        starting_mols = set(list(pd.read_csv(filename)['mol']))
    else:
        assert filename[-3:] == 'pkl'
        with open(filename, 'rb') as f:
            starting_mols = pickle.load(f)

    logging.info('%d starting molecules loaded' % len(starting_mols))
    return starting_mols


# def prepare_molstar_planner(one_step, value_fn, starting_mols, expansion_topk, iterations, max_routes_num):
#
#     beam_model = BeamSearch(model=one_step, step_beam_size=expansion_topk,
#                             beam_size=expansion_topk, use_rxn_class=False)
#     with torch.no_grad():
#         expansion_handle = lambda x: beam_model.run(prod_smi=x, max_steps=9, rxn_class=None)
#
#     assert starting_mols is not None
#     plan_handle = lambda x: molstar(
#         target_mol=x,
#         starting_mols=starting_mols,
#         expand_fn=expansion_handle,
#         value_fn=value_fn,
#         iterations=iterations,
#         max_routes_num = max_routes_num)
#
#     return plan_handle
