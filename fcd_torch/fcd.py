from fcd_torch.utils import SmilesDataset, \
                            calculate_frechet_distance, \
                            todevice, \
                            load_imported_model
import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import warnings


class FCD:
    """
    Computes Frechet ChemNet Distance on PyTorch.
    * You can precalculate mean and sigma for further usage,
      e.g. if you use the statistics from the same dataset
      multiple times.
    * Supports GPU and selection of GPU index
    * Multithread SMILES parsing

    Example 1:
        fcd = FCD(device='cuda:0', n_jobs=8)
        smiles_list = ['CCC', 'CCNC']
        fcd(smiles_list, smiles_list)

    Example 2:
        fcd = FCD(device='cuda:0', n_jobs=8)
        smiles_list = ['CCC', 'CCNC']
        pgen = fcd.precalc(smiles_list)
        fcd(smiles_list, pgen=pgen)
    """
    def __init__(self, device='cpu', n_jobs=1,
                 batch_size=512,
                 model_path=None,
                 canonize=True):
        """
        Loads ChemNet on device
        params:
            device: cpu for CPU, cuda:0 for GPU 0, etc.
            n_jobs: number of workers to parse SMILES
            batch_size: batch size for processing SMILES
            model_path: path to ChemNet_v0.13_pretrained.pt
        """
        if model_path is None:
            model_dir = os.path.split(__file__)[0]
            model_path = os.path.join(model_dir, 'ChemNet_v0.13_pretrained.pt')

        self.device = device
        self.n_jobs = n_jobs if n_jobs != 1 else 0
        self.batch_size = batch_size
        keras_config = torch.load(model_path)
        self.model = load_imported_model(keras_config)
        self.model.eval()
        self.canonize = canonize

    def precalc(self, smiles_list):
        if len(smiles_list) < 2:
            warnings.warn("Can't compute FCD for less than 2 molecules"
                          "({} given)".format(len(smiles_list)))
            return np.nan

        dataloader = DataLoader(
            SmilesDataset(smiles_list, canonize=self.canonize),
            batch_size=self.batch_size,
            num_workers=self.n_jobs
        )
        with todevice(self.model, self.device), torch.no_grad():
            chemnet_activations = []
            for batch in dataloader:
                chemnet_activations.append(
                    self.model(
                        batch.transpose(1, 2).float().to(self.device)
                    ).to('cpu').detach().numpy()
                )
        chemnet_activations = np.row_stack(chemnet_activations)
        mu = chemnet_activations.mean(0)
        sigma = np.cov(chemnet_activations.T)
        return {'mu': mu, 'sigma': sigma}

    def metric(self, pref, pgen):
        return calculate_frechet_distance(
            pref['mu'], pref['sigma'], pgen['mu'], pgen['sigma']
        )

    def __call__(self, ref=None, gen=None, pref=None, pgen=None):
        assert (ref is None) != (pref is None), "specify ref xor pref"
        assert (gen is None) != (pgen is None), "specify gen xor pgen"
        if pref is None:
            pref = self.precalc(ref)
        if pgen is None:
            pgen = self.precalc(gen)
        return self.metric(pref, pgen)
