import torch

from pleiome.dmpnn_encoder import ATOM_FDIM, BOND_FDIM, collate_graphs, smiles_to_graph
from pleiome.unified_model import UnifiedModel


def small_model():
    torch.set_num_threads(1)
    torch.manual_seed(12)
    return UnifiedModel(
        selfies_vocab=12,
        pad_idx=0,
        num_properties=3,
        encoder_type="dmpnn",
        d_model=32,
        nhead=4,
        struct_layers=3,
        slot_layers=1,
        gen_layers=1,
        gen_max_len=16,
    )


def inputs():
    graph = collate_graphs([smiles_to_graph("CCO")])
    pid = torch.tensor([[0, 1, 2]])
    typ = torch.tensor([[0, 2, 0]])
    values = torch.tensor([[1.0, 0.3, 0.0]])
    known = torch.tensor([[True, False, False]])
    slots = torch.ones_like(known)
    return graph, pid, typ, values, known, slots


def test_graph_features_reverse_edges_and_single_atoms():
    graph = collate_graphs([smiles_to_graph("CCO"), smiles_to_graph("[Na+]")])
    assert graph["f_atoms"].shape == (4, ATOM_FDIM)
    assert graph["f_bonds"].shape == (4, ATOM_FDIM + BOND_FDIM)
    rev = graph["b_rev"]
    assert torch.equal(rev[rev], torch.arange(4))
    assert torch.equal(graph["b_src"][rev], graph["b_dst"])
    model = small_model().eval()
    memory, pad = model.encoder(graph)
    assert memory.shape == (2, 3, 32)
    assert pad.tolist() == [[False, False, False], [False, True, True]]


def test_encoder_batching_and_atom_order_preserve_pooling():
    model = small_model().eval()
    with torch.no_grad():
        first, _ = model.encoder(collate_graphs([smiles_to_graph("CCO")]))
        both, _ = model.encoder(collate_graphs([smiles_to_graph("CCO"), smiles_to_graph("CCCCCC")]))
        reverse, _ = model.encoder(collate_graphs([smiles_to_graph("OCC")]))
    torch.testing.assert_close(first[0], both[0, :3])
    torch.testing.assert_close(first.sum(1), reverse.sum(1))


def test_masked_labels_cannot_leak_into_predictions():
    model = small_model().eval()
    args = inputs()
    with torch.no_grad():
        before = model(*args)["binary"]
        changed = args[3].clone()
        changed[~args[4]] = 1000
        after = model(*args[:3], changed, *args[4:])["binary"]
    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_generation_is_causal():
    model = small_model().eval()
    _, pid, typ, values, _, slots = inputs()
    profile = model._profile_slots(pid, typ, values, slots)
    a = torch.tensor([[1, 3, 4, 5]])
    b = torch.tensor([[1, 3, 8, 9]])
    with torch.no_grad():
        before = model.struct_dec(a, profile, ~slots)
        after = model.struct_dec(b, profile, ~slots)
    torch.testing.assert_close(before[:, :2], after[:, :2], rtol=0, atol=1e-6)


def test_joint_loss_has_finite_gradients_and_save_load(tmp_path):
    model = small_model()
    graph, pid, typ, values, known, slots = inputs()
    pred = model(graph, pid, typ, values, known, slots)
    loss, _ = model.compute_loss(pred, pid, typ, values, known, slots)
    loss += model.generation_loss(
        torch.tensor([[1, 3, 4]]), torch.tensor([[3, 4, 2]]), pid, typ, values, slots
    )
    loss.backward()
    for p in [
        model.encoder.W_i.weight,
        model.property_emb.weight,
        model.struct_dec.out.weight,
        model.handlers["numeric"].mask_emb,
    ]:
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        expected = model(graph, pid, typ, values, known, slots)["binary"]
    torch.save(model.state_dict(), tmp_path / "model.pt")
    other = small_model().eval()
    other.load_state_dict(torch.load(tmp_path / "model.pt", weights_only=True), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(
            other(graph, pid, typ, values, known, slots)["binary"], expected, rtol=0, atol=0
        )
