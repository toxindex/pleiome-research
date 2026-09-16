"""
Generic TYPED-PROPERTY model — a strict generalization of CondMultitask
(pleiome/conditional.py) to a REGISTRY of per-type value-encoders (INPUT)
and prediction heads (OUTPUT).

Motivation
----------
CondMultitask handles exactly one property type: per-assay BINARY. The big-run schema
(docs/research/bigrun-preprocessing-plan.md) normalizes every data source into typed
records `(property_id:int32, type:int8, value)` with type in
{binary, categorical, numeric, dose-response, presence/association, transformation,
 text, profile-vector}. To train one model over all of them we need the backbone to be
type-agnostic and the *value semantics* (how an observation enters a slot) and the
*prediction target* (what we read off a slot) to be dispatched by type.

Shared backbone (unchanged from CondMultitask / PropertyDiffusionDenoiser)
--------------------------------------------------------------------------
    structure --> SelfiesEncoder --> memory                 (pretrained, reusable)
    slot[i]   =  property_emb[pid_i] + type_emb[type_i] + value_encoder[type_i](value_i)
                 (masked slots use a per-type learned MASK embedding instead of a value)
    slot_dec  =  TransformerDecoder: self-attn over slots + cross-attn to structure memory
    h[B,P,d]  =  contextual slot representations
    per-type head(h, pid) --> type-specific prediction; loss dispatched by type.

Each slot therefore carries (property_id, type, value); the type selects BOTH the
input value-encoder and the output head. This file is self-contained (plain torch) and
importable. The research tests exercise the D-MPNN path on CPU.

Type registry (this file)
-------------------------
  binary       : value_emb{0,1} in           -> per-property logit out (BCE)           [= CondMultitask]
  categorical  : class-index emb in          -> softmax logits out    (cross-entropy)
  numeric      : Fourier-features + MLP in    -> (mu, log_var) out     (Gaussian NLL, + regression mu)
  association  : learned "present" token in   -> negative-sampled ranking out (PU-learning: observed
                 (presence/assoc)               pairs are positives, sampled unobserved are negatives)
  profile      : MLP over high-D vector in    -> vector-regression out (MSE over the D-vector)
                 (profile-vector modality: cell-painting / L1000 signatures)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- type registry: name <-> stable integer id (matches the typed-record `type` column) -------------
# Defined in pleiome.proptypes so torch-free callers (panel resolution) can read it;
# re-exported here because checkpoints and existing call sites reference this module.
from pleiome.proptypes import TYPE_IDS, TYPE_NAMES  # noqa: E402,F401


# =====================================================================================================
# Per-type handlers. Each owns (a) a value-ENCODER used to turn a known observation into a slot-input
# contribution [N,d], (b) a learned MASK embedding for slots to be predicted, and (c) a prediction HEAD
# read off the decoded slot h plus a per-type loss. All heads run over the full [B,P,...] slot grid; the
# model masks the loss to the slots of that type that are being predicted.
# =====================================================================================================
class _TypeHandler(nn.Module):
    type_name: str

    def __init__(self, d_model, num_properties):
        super().__init__()
        self.d_model = d_model
        self.num_properties = num_properties
        self.type_id = TYPE_IDS[self.type_name]
        self.mask_emb = nn.Parameter(torch.zeros(d_model))   # slot marker for "predict me"

    # value is a float tensor [N]; profile is [N, profile_dim] or None. Returns [N, d_model].
    def encode_value(self, pid, value, profile):
        raise NotImplementedError

    # h [B,P,d], pid [B,P]. Returns a type-specific prediction object (tensor or dict) over all slots.
    def predict(self, h, pid):
        raise NotImplementedError

    # pred = self.predict(...); targets/value [B,P]; mask [B,P] bool = slots to score. Returns scalar loss.
    def loss(self, pred, value, pid, mask, profile=None):
        raise NotImplementedError


class BinaryType(_TypeHandler):
    """Per-property binary logit — identical readout to CondMultitask."""
    type_name = "binary"

    def __init__(self, d_model, num_properties, **_):
        super().__init__(d_model, num_properties)
        self.value_emb = nn.Embedding(2, d_model)              # {0, 1}
        self.head_w = nn.Embedding(num_properties, d_model)    # per-property weight (residual if factors on)
        self.head_b = nn.Embedding(num_properties, 1)          # per-property bias
        nn.init.zeros_(self.head_b.weight)
        self.use_factors = False

    def set_factor_priors(self, gate_init):
        """LLM PRIORS: per-property gate = sign*coupling routing the shared factor potential (a member and
        its counter-screen share ONE |potential| with opposite sign). Initialized from the LLM prior,
        LEARNABLE (data refines / can override). gate_init: FloatTensor [num_properties]."""
        self.factor_gate = nn.Embedding(self.head_w.num_embeddings, 1)
        self.factor_gate.weight.data.copy_(gate_init.reshape(-1, 1))
        self.use_gate = True

    def enable_factors(self, factor_row, n_multi, mode="potential"):
        """FACTOR-graph head: logit = per-property residual(h) + [member] * factor_term. factor_row[pid]
        in [0, n_multi]; multi-member factors are 0..n_multi-1 (shared -> a sparse member inherits its
        well-measured siblings' structure->activity function = parameter-level transfer that works with
        ZERO co-measurement); singletons map to n_multi and are MASKED (pure residual = baseline).
        mode='linear'    : factor term = h . factor_emb + factor_bias   (shared LINEAR readout).
        mode='potential' : factor term = shared NONLINEAR net(h, factor_emb) -> a rich learnable factor
                           potential (a function of structure conditioned on factor identity). This is the
                           principled factor graph's learnable potential. Residual head zero-init so a
                           member starts AS its shared factor function."""
        d = self.head_w.embedding_dim
        self.register_buffer("factor_row", factor_row.long())
        self.n_multi = int(n_multi); self.factor_mode = mode
        self.factor_emb = nn.Embedding(self.n_multi + 1, d)
        nn.init.normal_(self.factor_emb.weight, std=0.02)
        if mode == "linear":
            self.factor_b = nn.Embedding(self.n_multi + 1, 1); nn.init.zeros_(self.factor_b.weight)
        else:
            self.pot_h = nn.Linear(d, d); self.pot_f = nn.Linear(d, d, bias=False)
            self.pot_out = nn.Linear(d, 1)
            nn.init.zeros_(self.pot_out.weight); nn.init.zeros_(self.pot_out.bias)   # potential starts at 0
        nn.init.zeros_(self.head_w.weight)                     # residual starts at 0
        self.use_factors = True

    def encode_value(self, pid, value, profile):
        return self.value_emb(value.long().clamp(0, 1))

    def predict(self, h, pid):
        logit = (h * self.head_w(pid)).sum(-1) + self.head_b(pid).squeeze(-1)   # per-property residual
        if getattr(self, "use_factors", False):
            fr = self.factor_row[pid.clamp(0, self.factor_row.shape[0] - 1)]
            m = (fr < self.n_multi).to(logit.dtype)                    # [B,P] 0 for singletons
            fe = self.factor_emb(fr)                                   # [B,P,d]
            if self.factor_mode == "linear":
                term = (h * fe).sum(-1) + self.factor_b(fr).squeeze(-1)
            else:
                term = self.pot_out(torch.tanh(self.pot_h(h) + self.pot_f(fe))).squeeze(-1)
            if getattr(self, "use_gate", False):                       # LLM prior: sign*coupling routing
                term = term * self.factor_gate(pid).squeeze(-1)
            logit = logit + m * term
        return logit                                                   # [B,P]

    def set_logit_adj(self, la, tau):
        """LOGIT ADJUSTMENT (Menon 2020) vs the base-rate collapse: train on logit + tau*log(pi/(1-pi));
        predict with the RAW logit. Forces the head to learn structure beyond each property's base rate
        (attacks failure-B / macro-AUC~0.5). la = per-property log-prior-odds."""
        self.register_buffer("logit_adj", la, persistent=False); self.la_tau = float(tau)

    def loss(self, pred, value, pid, mask, profile=None):
        ls = float(getattr(self, "label_smooth", 0.0))
        if getattr(self, "la_tau", 0.0):
            pred = pred + self.la_tau * self.logit_adj[pid]        # adjusted logit for the LOSS only
        tgt = value.float()
        if ls:
            tgt = tgt * (1 - ls) + 0.5 * ls                        # symmetric label smoothing
        pw = float(getattr(self, "pos_weight", 1.0)); gamma = float(getattr(self, "focal_gamma", 0.0))
        if pw != 1.0:
            el = F.binary_cross_entropy_with_logits(pred, tgt, reduction="none",
                                                    pos_weight=torch.tensor(pw, device=pred.device))
        else:
            el = F.binary_cross_entropy_with_logits(pred, tgt, reduction="none")
        if gamma > 0:                                             # focal: down-weight easy examples
            p = torch.sigmoid(pred); ptt = torch.where(value > 0.5, p, 1 - p)
            el = el * (1 - ptt).clamp(min=1e-6) ** gamma
        w = mask.float()
        if getattr(self, "prop_weight", None) is not None:
            w = w * self.prop_weight[pid]                          # gentle per-property tail upweighting (Option-1)
        return (el * w).sum() / w.sum().clamp_min(1)


class CategoricalType(_TypeHandler):
    """K-way softmax. Class index enters via an embedding; head projects to K global classes.
    (Per-property class subsets are disambiguated by the property_emb already in the slot.)"""
    type_name = "categorical"

    def __init__(self, d_model, num_properties, num_categories=16, **_):
        super().__init__(d_model, num_properties)
        self.num_categories = num_categories
        self.value_emb = nn.Embedding(num_categories, d_model)
        self.head = nn.Linear(d_model, num_categories)

    def encode_value(self, pid, value, profile):
        return self.value_emb(value.long().clamp(0, self.num_categories - 1))

    def predict(self, h, pid):
        return self.head(h)                                    # [B,P,K]

    def loss(self, pred, value, pid, mask, profile=None):
        if mask.sum() == 0:
            return pred.sum() * 0.0
        logits = pred[mask]                                    # [M,K]
        tgt = value[mask].long().clamp(0, self.num_categories - 1)
        return F.cross_entropy(logits, tgt)


class FourierFeatures(nn.Module):
    """Random-Fourier positional encoding of a scalar: v -> [sin(2π f_k v), cos(2π f_k v)]_k."""
    def __init__(self, n_freq=32, sigma=4.0):
        super().__init__()
        self.register_buffer("freqs", torch.randn(n_freq) * sigma)

    def forward(self, v):                                      # v [N]
        x = 2 * math.pi * v.unsqueeze(-1) * self.freqs         # [N, n_freq]
        return torch.cat([x.sin(), x.cos()], dim=-1)           # [N, 2*n_freq]


class NumericType(_TypeHandler):
    """Continuous value. INPUT: Fourier features -> MLP. OUTPUT: distributional (mu, log_var) head
    trained by Gaussian NLL; `mu` is the point (regression) prediction."""
    type_name = "numeric"

    def __init__(self, d_model, num_properties, n_freq=32, fourier_sigma=4.0, **_):
        super().__init__(d_model, num_properties)
        self.fourier = FourierFeatures(n_freq, fourier_sigma)
        self.value_mlp = nn.Sequential(
            nn.Linear(2 * n_freq, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.head = nn.Linear(d_model, 2)                      # -> (mu, log_var)

    def encode_value(self, pid, value, profile):
        return self.value_mlp(self.fourier(value.float()))

    def predict(self, h, pid):
        mu, log_var = self.head(h).unbind(-1)                  # each [B,P]
        return {"mu": mu, "log_var": log_var.clamp(-2.0, 6.0)}

    def loss(self, pred, value, pid, mask, profile=None):
        mu, log_var = pred["mu"], pred["log_var"]
        y = value.float().clamp(-10, 10)
        nll = (0.5 * (log_var + (y - mu) ** 2 / log_var.exp())).clamp(max=30.0)   # bounded Gaussian NLL
        return (nll * mask).sum() / mask.sum().clamp_min(1)


class AssociationType(_TypeHandler):
    """Presence / association (compound<->target|disease|drug edges). Only POSITIVES are observed, so
    this is PU-learning: we learn a ranking where an observed (compound, property) pair scores above
    sampled unobserved properties (assumed negative). INPUT: a single learned 'present' token. OUTPUT:
    a negative-sampled softmax ranking loss over {true property} ∪ {K sampled negatives}."""
    type_name = "association"

    def __init__(self, d_model, num_properties, n_neg=8, **_):
        super().__init__(d_model, num_properties)
        self.n_neg = n_neg
        self.present_emb = nn.Parameter(torch.zeros(d_model))  # "this association is observed/present"
        self.assoc_emb = nn.Embedding(num_properties, d_model) # scored against the decoded slot

    def encode_value(self, pid, value, profile):
        # observed association slots are positives ("present"); value is ignored / always 1.
        return self.present_emb.expand(pid.shape[0], -1)

    def predict(self, h, pid):
        return h                                               # ranking is computed in loss (needs negatives)

    def loss(self, pred, value, pid, mask, profile=None):
        h = pred                                               # [B,P,d]
        if mask.sum() == 0:
            return h.sum() * 0.0
        hp = h[mask]                                           # [M,d] positive slots
        pos_pid = pid[mask]                                    # [M]
        neg_pid = torch.randint(0, self.num_properties, (hp.shape[0], self.n_neg), device=h.device)
        pos = (hp * self.assoc_emb(pos_pid)).sum(-1, keepdim=True)          # [M,1]
        neg = torch.einsum("md,mkd->mk", hp, self.assoc_emb(neg_pid))       # [M,K]
        logits = torch.cat([pos, neg], dim=1)                              # [M,1+K], index 0 = positive
        tgt = torch.zeros(hp.shape[0], dtype=torch.long, device=h.device)
        return F.cross_entropy(logits, tgt)                                # softmax ranking (PU-style)


class ProfileType(_TypeHandler):
    """High-D profile vector modality (cell-painting features / L1000 signature). INPUT: MLP over the
    D-vector -> slot. OUTPUT: vector-regression (reconstruct the D-vector) trained by MSE."""
    type_name = "profile"

    def __init__(self, d_model, num_properties, profile_dim=978, **_):
        super().__init__(d_model, num_properties)
        self.profile_dim = profile_dim
        self.value_mlp = nn.Sequential(
            nn.Linear(profile_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.head = nn.Linear(d_model, profile_dim)

    def encode_value(self, pid, value, profile):
        return self.value_mlp(profile)

    def predict(self, h, pid):
        return self.head(h)                                    # [B,P,profile_dim]

    def loss(self, pred, value, pid, mask, profile=None):
        if mask.sum() == 0:
            return pred.sum() * 0.0
        return F.mse_loss(pred[mask], profile[mask])


_HANDLERS = {
    "binary": BinaryType,
    "categorical": CategoricalType,
    "numeric": NumericType,
    "association": AssociationType,
    "profile": ProfileType,
}


# =====================================================================================================
# The model.
# =====================================================================================================
class TypedPropertyModel(nn.Module):
    """Generic typed-property predictor.

    A batch is a set of slots per compound. Each slot carries:
        property_id [B,P]  int   global property registry id
        type_id     [B,P]  int   TYPE_IDS[...]
        value       [B,P]  float scalar value (binary 0/1, categorical class idx, numeric value,
                                 association ignored); profile slots read from `profile` instead
        value_mask  [B,P]  bool  True = observed (conditioning) slot, False = masked (predict) slot
        slot_mask   [B,P]  bool  True = real slot, False = padding
        profile     [B,P,profile_dim] float | None  target/observed profile vectors (profile slots only)

    forward -> dict {type_name: prediction} over the whole [B,P,...] grid.
    compute_loss -> (total_loss, {type_name: loss}) reducing each head over its predict-slots.
    """
    def __init__(self, selfies_vocab, pad_idx, num_properties, d_model=512, nhead=8,
                 struct_layers=6, slot_layers=6, type_names=None, type_cfg=None, encoder_type="selfies"):
        super().__init__()
        type_names = type_names or TYPE_NAMES
        type_cfg = type_cfg or {}
        self.num_properties = num_properties
        self.d_model = d_model
        self.nhead = nhead
        self.encoder_type = encoder_type

        if encoder_type == "dmpnn":
            # BET A: GRAPH molecular encoder (D-MPNN) built from ORIGINAL SMILES. Returns per-ATOM
            # (memory, pad_mask) — same contract as SelfiesEncoder, so the slot decoder's cross-attention
            # is unchanged. forward(structure) receives a graph-batch dict instead of SELFIES tokens.
            from pleiome.dmpnn_encoder import DMPNNEncoder
            self.encoder = DMPNNEncoder(d_model=d_model, depth=max(3, struct_layers), dropout=0.0)
        else:
            raise ValueError("This research package supports encoder_type='dmpnn' only")
        self.property_emb = nn.Embedding(num_properties, d_model)
        self.type_emb = nn.Embedding(len(TYPE_NAMES), d_model)
        layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward=4 * d_model,
            batch_first=True, activation="gelu", norm_first=True)
        self.slot_dec = nn.TransformerDecoder(layer, slot_layers)

        self.handlers = nn.ModuleDict(
            {name: _HANDLERS[name](d_model, num_properties, **type_cfg) for name in type_names})
        # Kendall multi-task uncertainty weighting: a learned log-variance s_t = log σ²_t per type.
        # The per-type losses live on wildly different scales (BCE ~0.7, 16-way CE ~2.7, Gaussian NLL
        # can be negative/large, high-D profile MSE potentially huge, ranking ~2); an unweighted sum
        # lets numeric/profile dominate and destabilize joint training. Weighted term = ½·e^{-s_t}·L_t
        # + ½·s_t auto-balances them and is self-tuning (hard/noisy types get down-weighted).
        self.log_var = nn.Parameter(torch.zeros(len(TYPE_NAMES)))

    @classmethod
    def from_tokenizer(cls, tokenizer, num_properties, extra_vocab=0, **kw):
        """Build against the shared SelfiesPropertyValTokenizer (extra_vocab=1 for a [MASK] row,
        matching CondMultitask's pretrained-encoder loading path)."""
        return cls(selfies_vocab=tokenizer.selfies_offset + extra_vocab,
                   pad_idx=tokenizer.PAD_IDX, num_properties=num_properties, **kw)

    def _build_slots(self, property_id, type_id, value, value_mask, slot_mask, profile):
        """property_emb + type_emb + (per-type value encoding | per-type MASK emb), dispatched by type."""
        slots = self.property_emb(property_id) + self.type_emb(type_id)
        contrib = torch.zeros_like(slots)
        for name, h in self.handlers.items():
            tsel = (type_id == h.type_id) & slot_mask
            known = tsel & value_mask
            masked = tsel & ~value_mask
            if known.any():
                prof_known = profile[known] if profile is not None else None
                contrib[known] = h.encode_value(property_id[known], value[known], prof_known)
            if masked.any():
                contrib[masked] = h.mask_emb.to(contrib.dtype)
        return slots + contrib

    def enable_factor_gcn(self, factor_row, n_multi):
        """Turn the slot decoder into a FACTOR-GRAPH GCN: slots exchange messages only along factor edges."""
        self.register_buffer("_gcn_factor_row", factor_row.long())
        self._gcn_n_multi = int(n_multi); self.use_factor_gcn = True

    def _factor_gcn_mask(self, property_id, slot_mask):
        """FACTOR-GRAPH GCN: additive slot-attention mask so a slot only exchanges messages with its
        factor-neighbors (share the same multi-member factor) + itself. Attention = message passing;
        this mask = the factor-graph adjacency. Non-members attend to self only (still cross-attend to
        structure). -> [B*nhead, P, P]."""
        fr = self._gcn_factor_row[property_id.clamp(0, self._gcn_factor_row.shape[0] - 1)]  # [B,P] factor/slot
        member = fr < self._gcn_n_multi
        same = (fr.unsqueeze(2) == fr.unsqueeze(1)) & member.unsqueeze(2) & member.unsqueeze(1)  # [B,P,P]
        eye = torch.eye(fr.shape[1], dtype=torch.bool, device=fr.device).unsqueeze(0)
        allow = same | eye                                            # factor-mates + self
        allow = allow | (~slot_mask).unsqueeze(2)                     # padded query rows attend freely (ignored)
        mask = torch.zeros(allow.shape, dtype=torch.float32, device=fr.device).masked_fill(~allow, float("-inf"))
        return mask.repeat_interleave(self.nhead, dim=0)              # [B*nhead, P, P]

    def forward(self, selfies, property_id, type_id, value, value_mask, slot_mask, profile=None):
        memory, mem_pad = self.encoder(selfies)
        property_id = property_id.clamp(0, self.num_properties - 1)
        slots = self._build_slots(property_id, type_id, value, value_mask, slot_mask, profile)
        tgt_mask = self._factor_gcn_mask(property_id, slot_mask) if getattr(self, "use_factor_gcn", False) else None
        h = self.slot_dec(tgt=slots, memory=memory, tgt_mask=tgt_mask,
                          tgt_key_padding_mask=~slot_mask, memory_key_padding_mask=mem_pad)
        return {name: handler.predict(h, property_id) for name, handler in self.handlers.items()}

    def compute_loss(self, preds, property_id, type_id, value, value_mask, slot_mask, profile=None,
                     uncertainty_weighting=True):
        """Predict-slots = real & masked. Each head is scored only on its own-type predict-slots.

        `parts` holds the RAW (unweighted) per-type losses for interpretable logging; `total` is the
        Kendall-uncertainty-weighted sum used for the backward pass. Types absent from the batch add
        neither their loss nor a log-variance penalty (so σ_t isn't pushed around by empty batches)."""
        property_id = property_id.clamp(0, self.num_properties - 1)
        predict = slot_mask & ~value_mask
        parts, total = {}, None
        for name, handler in self.handlers.items():
            mask = predict & (type_id == handler.type_id)
            l = handler.loss(preds[name], value, property_id, mask, profile=profile)
            parts[name] = l
            if uncertainty_weighting and bool(mask.any()):
                s = self.log_var[handler.type_id].clamp(-2.0, 4.0)   # bound Kendall weight: exp(-s) in [0.018,7.39], stops numeric runaway
                term = 0.5 * torch.exp(-s) * l + 0.5 * s          # ½·e^{-s}·L + ½·s (Kendall)
            else:
                term = l                                          # empty batches contribute 0·grad only
            total = term if total is None else total + term
        return total, parts
