"""
D-MPNN (directed message-passing neural network, à la Chemprop) molecular graph
encoder — a drop-in alternative to the SELFIES-sequence `SelfiesEncoder`
(property_diffusion/model.py).

Interface contract (matched exactly to SelfiesEncoder):

    memory, pad_mask = encoder(batch)
      memory   : [B, max_atoms, d_model]  per-ATOM contextual embeddings
      pad_mask : [B, max_atoms] bool      True = padding position (ignored by
                                          downstream cross-attention, i.e. passed
                                          as `memory_key_padding_mask`)

So downstream cross-attention treats ATOMS as the "sequence" (one atom == one
token). A molecule with N heavy atoms occupies rows 0..N-1; the rest are padding.

Message passing follows the classic Chemprop D-MPNN:
  h(uv)^0      = ReLU( W_i . [f_atom(u) ; f_bond(uv)] )
  m(uv)^{t+1}  = sum_{w in N(u)\v} h(wu)^t
  h(uv)^{t+1}  = ReLU( h(uv)^0 + W_h . m(uv)^{t+1} )        (depth steps)
  m(v)         = sum_{w in N(v)} h(wv)^T
  h(v)         = ReLU( W_o . [f_atom(v) ; m(v)] )           per-atom readout

Pure PyTorch + RDKit. No torch_geometric / torch_scatter dependency — the
directed message passing is hand-rolled with `torch.index_add_`. (Checked:
neither torch_geometric nor torch_scatter is installed in this venv.)

CPU-testable. Run `python -m pleiome.dmpnn_encoder` for the smoke test.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem

# ---------------------------------------------------------------------------
# Featurization (Chemprop-style atom / bond features)
# ---------------------------------------------------------------------------

# Common organic elements; anything else falls into the "other" bucket.
ATOMIC_NUM_CHOICES = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53]
DEGREE_CHOICES = [0, 1, 2, 3, 4, 5]
FORMAL_CHARGE_CHOICES = [-2, -1, 0, 1, 2]
NUM_H_CHOICES = [0, 1, 2, 3, 4]
HYBRIDIZATION_CHOICES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
BOND_TYPE_CHOICES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BOND_STEREO_CHOICES = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
]


def _one_hot(value, choices: List) -> List[float]:
    """One-hot with a trailing 'other/unknown' bucket -> len(choices)+1 dims."""
    vec = [0.0] * (len(choices) + 1)
    try:
        vec[choices.index(value)] = 1.0
    except ValueError:
        vec[-1] = 1.0
    return vec


def atom_features(atom: Chem.Atom) -> List[float]:
    return (
        _one_hot(atom.GetAtomicNum(), ATOMIC_NUM_CHOICES)
        + _one_hot(atom.GetTotalDegree(), DEGREE_CHOICES)
        + _one_hot(atom.GetFormalCharge(), FORMAL_CHARGE_CHOICES)
        + _one_hot(atom.GetHybridization(), HYBRIDIZATION_CHOICES)
        + _one_hot(atom.GetTotalNumHs(), NUM_H_CHOICES)
        + [1.0 if atom.GetIsAromatic() else 0.0]
        + [1.0 if atom.IsInRing() else 0.0]
        + [atom.GetMass() * 0.01]  # scaled atomic mass
    )


def bond_features(bond: Chem.Bond) -> List[float]:
    return (
        _one_hot(bond.GetBondType(), BOND_TYPE_CHOICES)
        + [1.0 if bond.GetIsConjugated() else 0.0]
        + [1.0 if bond.IsInRing() else 0.0]
        + _one_hot(bond.GetStereo(), BOND_STEREO_CHOICES)
    )


ATOM_FDIM = (
    (len(ATOMIC_NUM_CHOICES) + 1)
    + (len(DEGREE_CHOICES) + 1)
    + (len(FORMAL_CHARGE_CHOICES) + 1)
    + (len(HYBRIDIZATION_CHOICES) + 1)
    + (len(NUM_H_CHOICES) + 1)
    + 3
)
BOND_FDIM = (len(BOND_TYPE_CHOICES) + 1) + 2 + (len(BOND_STEREO_CHOICES) + 1)


class MolGraph:
    """Featurized single molecule.

    f_atoms : [n_atoms, ATOM_FDIM]
    bonds   : list of (src_atom, dst_atom, bond_feat[list]) for each *undirected*
              bond (directed pairs + reverse indices are built in collate_graphs).
    """

    __slots__ = ("f_atoms", "bonds", "n_atoms")

    def __init__(self, f_atoms: torch.Tensor, bonds: list):
        self.f_atoms = f_atoms
        self.bonds = bonds
        self.n_atoms = f_atoms.shape[0]


def smiles_to_graph(smiles: str) -> MolGraph:
    """SMILES -> MolGraph (atom feature matrix, bond features, connectivity).

    Raises ValueError if RDKit cannot parse the SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")

    f_atoms = torch.tensor(
        [atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32
    )
    if f_atoms.numel() == 0:  # e.g. empty molecule -> single dummy row
        f_atoms = torch.zeros((1, ATOM_FDIM), dtype=torch.float32)

    bonds = []
    for b in mol.GetBonds():
        bonds.append((b.GetBeginAtomIdx(), b.GetEndAtomIdx(), bond_features(b)))
    return MolGraph(f_atoms, bonds)


# ---------------------------------------------------------------------------
# Batching: variable-size graphs -> one disconnected big-graph
# ---------------------------------------------------------------------------

def collate_graphs(graphs: List[MolGraph]) -> Dict[str, torch.Tensor]:
    """Batch molecular graphs into a single disconnected big-graph.

    Directed edges: every undirected bond becomes two consecutive directed bonds
    (index 2k and 2k+1) that are each other's reverse -> trivial reverse map.

    Returns a dict of tensors:
      f_atoms      [N_atoms, ATOM_FDIM]
      f_bonds      [N_bonds, ATOM_FDIM + BOND_FDIM]  = [f_atom(src) ; bond_feat]
      b_src        [N_bonds]  source atom of each directed bond
      b_dst        [N_bonds]  destination atom of each directed bond
      b_rev        [N_bonds]  index of the reverse directed bond
      atom_batch   [N_atoms]  molecule index of each atom
      atom_slot    [N_atoms]  position of the atom within its molecule (0..n-1)
      n_mols       int scalar
      max_atoms    int scalar
    where N_atoms / N_bonds are the totals across the whole batch.
    """
    f_atoms_list, f_bonds_list = [], []
    b_src, b_dst, b_rev = [], [], []
    atom_batch, atom_slot = [], []
    n_atoms_offset = 0

    for mol_idx, g in enumerate(graphs):
        n = g.n_atoms
        f_atoms_list.append(g.f_atoms)
        atom_batch.extend([mol_idx] * n)
        atom_slot.extend(range(n))

        _biter = ((int(r[0]), int(r[1]), r[2:]) for r in g.bonds) if isinstance(g.bonds, np.ndarray) else g.bonds
        for (a1, a2, bfeat) in _biter:
            ga1 = a1 + n_atoms_offset
            ga2 = a2 + n_atoms_offset
            bt = torch.as_tensor(bfeat, dtype=torch.float32)
            # directed a1 -> a2
            f_bonds_list.append(torch.cat([g.f_atoms[a1], bt]))
            b_src.append(ga1)
            b_dst.append(ga2)
            # directed a2 -> a1
            f_bonds_list.append(torch.cat([g.f_atoms[a2], bt]))
            b_src.append(ga2)
            b_dst.append(ga1)
            k = len(b_src)  # after appending both, indices are k-2 and k-1
            b_rev.append(k - 1)  # reverse of (a1->a2) is (a2->a1)
            b_rev.append(k - 2)  # reverse of (a2->a1) is (a1->a2)

        n_atoms_offset += n

    f_atoms = torch.cat(f_atoms_list, dim=0)
    if f_bonds_list:
        f_bonds = torch.stack(f_bonds_list, dim=0)
    else:  # batch with no bonds at all (all single-atom mols)
        f_bonds = torch.zeros((0, ATOM_FDIM + BOND_FDIM), dtype=torch.float32)

    n_per_mol = [g.n_atoms for g in graphs]
    return {
        "f_atoms": f_atoms,
        "f_bonds": f_bonds,
        "b_src": torch.tensor(b_src, dtype=torch.long),
        "b_dst": torch.tensor(b_dst, dtype=torch.long),
        "b_rev": torch.tensor(b_rev, dtype=torch.long),
        "atom_batch": torch.tensor(atom_batch, dtype=torch.long),
        "atom_slot": torch.tensor(atom_slot, dtype=torch.long),
        "n_mols": torch.tensor(len(graphs), dtype=torch.long),
        "max_atoms": torch.tensor(max(n_per_mol) if n_per_mol else 0, dtype=torch.long),
    }


def smiles_batch_to_graphs(smiles_list: List[str]) -> Dict[str, torch.Tensor]:
    """Convenience: list of SMILES -> collated batch dict."""
    return collate_graphs([smiles_to_graph(s) for s in smiles_list])


# ---------------------------------------------------------------------------
# D-MPNN encoder
# ---------------------------------------------------------------------------

class DMPNNEncoder(nn.Module):
    """Directed message-passing encoder producing per-atom node embeddings.

    forward(batch) -> (memory [B, max_atoms, d_model], pad_mask [B, max_atoms] bool)
    matching SelfiesEncoder's return contract (True in pad_mask = padding).
    """

    def __init__(self, d_model: int = 512, depth: int = 4, dropout: float = 0.0,
                 atom_fdim: int = ATOM_FDIM, bond_fdim: int = BOND_FDIM):
        super().__init__()
        self.d_model = d_model
        self.depth = depth
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim

        # edge init: [f_atom(src) ; bond_feat] -> hidden
        self.W_i = nn.Linear(atom_fdim + bond_fdim, d_model, bias=False)
        # message update (shared across depth steps)
        self.W_h = nn.Linear(d_model, d_model, bias=False)
        # atom readout: [f_atom(v) ; aggregated edge msg] -> hidden
        self.W_o = nn.Linear(atom_fdim + d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _device(self):
        return self.W_i.weight.device

    def forward(self, batch: Dict[str, torch.Tensor]):
        dev = self._device()
        f_atoms = batch["f_atoms"].to(dev)
        f_bonds = batch["f_bonds"].to(dev)
        b_src = batch["b_src"].to(dev)
        b_dst = batch["b_dst"].to(dev)
        b_rev = batch["b_rev"].to(dev)
        atom_batch = batch["atom_batch"].to(dev)
        atom_slot = batch["atom_slot"].to(dev)
        n_mols = int(batch["n_mols"])
        max_atoms = int(batch["max_atoms"])

        n_atoms = f_atoms.shape[0]
        n_bonds = f_bonds.shape[0]

        # --- directed edge message passing ---
        h0 = F.relu(self.W_i(f_bonds))  # [n_bonds, d_model]
        h = h0
        for _ in range(self.depth):
            if n_bonds > 0:
                # sum of incoming directed-bond states at each atom
                # (a bond w->u is "incoming to u" when its dst == u)
                a_msg = torch.zeros(n_atoms, self.d_model, device=dev, dtype=h.dtype)
                a_msg.index_add_(0, b_dst, h)          # gather at destination atom
                # message into bond (u->v): incoming to u, minus the reverse edge (v->u)
                msg = a_msg[b_src] - h[b_rev]          # [n_bonds, d_model]
                h = F.relu(h0 + self.W_h(msg))
                h = self.dropout(h)
            # if no bonds, h stays empty; atom readout below handles it

        # --- per-atom readout ---
        atom_msg = torch.zeros(n_atoms, self.d_model, device=dev, dtype=h0.dtype)
        if n_bonds > 0:
            atom_msg.index_add_(0, b_dst, h)           # incoming edges of each atom
        atom_hidden = F.relu(self.W_o(torch.cat([f_atoms, atom_msg], dim=1)))
        atom_hidden = self.dropout(atom_hidden)        # [n_atoms, d_model]

        # --- scatter atoms back into padded [B, max_atoms, d_model] ---
        memory = torch.zeros(n_mols, max_atoms, self.d_model,
                             device=dev, dtype=atom_hidden.dtype)
        pad_mask = torch.ones(n_mols, max_atoms, dtype=torch.bool, device=dev)
        memory[atom_batch, atom_slot] = atom_hidden
        pad_mask[atom_batch, atom_slot] = False        # False = real atom
        return memory, pad_mask

    # convenience so callers can pass SMILES directly if desired
    def encode_smiles(self, smiles_list: List[str]):
        return self.forward(smiles_batch_to_graphs(smiles_list))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    smiles = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]
    expected_atoms = [3, 6, 13]  # heavy-atom counts

    print(f"ATOM_FDIM={ATOM_FDIM}  BOND_FDIM={BOND_FDIM}")

    graphs = [smiles_to_graph(s) for s in smiles]
    for s, g, exp in zip(smiles, graphs, expected_atoms):
        print(f"  {s:<28} n_atoms={g.n_atoms:<3} (expected {exp})  n_bonds={len(g.bonds)}")
        assert g.n_atoms == exp, f"atom count mismatch for {s}: {g.n_atoms} != {exp}"

    batch = collate_graphs(graphs)
    B = int(batch["n_mols"])
    max_atoms = int(batch["max_atoms"])
    print(f"\nbatch: n_mols={B}  max_atoms={max_atoms}  "
          f"N_atoms={batch['f_atoms'].shape[0]}  N_bonds={batch['f_bonds'].shape[0]}")

    d_model = 512
    enc = DMPNNEncoder(d_model=d_model, depth=4)
    memory, pad_mask = enc(batch)

    print(f"\nmemory shape   : {tuple(memory.shape)}  (expected ({B}, {max_atoms}, {d_model}))")
    print(f"pad_mask shape : {tuple(pad_mask.shape)}  dtype={pad_mask.dtype}")

    assert memory.shape == (B, max_atoms, d_model), "memory shape wrong"
    assert pad_mask.shape == (B, max_atoms) and pad_mask.dtype == torch.bool, "pad_mask wrong"

    # (b) valid-atom counts per molecule = (~pad_mask).sum(1)
    valid = (~pad_mask).sum(dim=1).tolist()
    print(f"\nvalid atoms per mol (from pad_mask): {valid}  (expected {expected_atoms})")
    assert valid == expected_atoms, f"valid-atom counts wrong: {valid} != {expected_atoms}"

    # padded rows must be exactly zero; real rows generally non-zero
    for i, n in enumerate(expected_atoms):
        assert torch.count_nonzero(memory[i, n:]) == 0, "padding rows should be zero"
        assert torch.count_nonzero(memory[i, :n]) > 0, "real atom rows should be populated"

    # (c) forward + backward, grads finite
    loss = memory.sum()
    loss.backward()
    grads = [p.grad for p in enc.parameters() if p.grad is not None]
    n_grad = len(grads)
    all_finite = all(torch.isfinite(g).all() for g in grads)
    print(f"\nbackward: {n_grad} param tensors got grads; all finite = {all_finite}")
    assert n_grad > 0 and all_finite, "gradient check failed"

    print("\nSMOKE TEST PASSED")
