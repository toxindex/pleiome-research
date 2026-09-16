"""
B-FULL — ONE model that PREDICTS (D-MPNN quality) and GENERATES (inverse design), unified by a SHARED
property-slot representation.

  PREDICT   : SMILES -> D-MPNN -> atom memory -> property slots (masked vals) cross-attend -> per-type heads
              (the prediction path is TypedPropertyModel)
  GENERATE  : property slots (KNOWN target vals, structure-free) -> StructureDecoder -> SELFIES tokens
              (autoregressive; cross-attends ONLY to the property slots, so it needs no input molecule)
  SHARED    : property_emb + type_emb + per-type value_emb  (the slot construction is identical in both
              directions -> the generator is conditioned on the same property semantics the predictor learns)
  CO-TRAIN  : L = L_predict  +  lambda * L_generate

The generator conditions on the compound's FULL known-property profile to reconstruct its structure; at
inference, feed a TARGET profile -> generate a molecule that should exhibit it.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from pleiome.typed_model import TYPE_IDS, TypedPropertyModel

SOS, EOS, PADT = 1, 2, 0                                          # SELFIES tokenizer special-token ids


class StructureDecoder(nn.Module):
    """Autoregressive SELFIES decoder cross-attending to property slots (the generation conditioning)."""
    def __init__(self, vocab, d_model=512, nhead=8, layers=6, max_len=128, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model, padding_idx=PADT)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerDecoderLayer(d_model, nhead, 4 * d_model, dropout=dropout,
                                           batch_first=True, activation="gelu", norm_first=True)
        self.dec = nn.TransformerDecoder(layer, layers)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab)
        self.max_len = max_len

    def forward(self, tok_in, slots, slot_pad):
        """tok_in [B,L] -> logits [B,L,vocab]; slots [B,P,d] memory, slot_pad [B,P] True=pad."""
        B, L = tok_in.shape
        pos = torch.arange(L, device=tok_in.device).clamp(max=self.max_len - 1)
        x = self.tok_emb(tok_in) + self.pos_emb(pos)[None]
        causal = torch.triu(torch.ones((L, L), dtype=torch.bool, device=tok_in.device), 1)
        h = self.dec(tgt=x, memory=slots, tgt_mask=causal,
                     tgt_key_padding_mask=(tok_in == PADT), memory_key_padding_mask=slot_pad)
        return self.out(self.norm(h))


class UnifiedModel(TypedPropertyModel):
    def __init__(self, selfies_vocab, *a, gen_layers=6, gen_max_len=128, **kw):
        super().__init__(selfies_vocab, *a, **kw)
        self.encoder_vocab = selfies_vocab
        self.struct_dec = StructureDecoder(vocab=selfies_vocab, d_model=self.d_model,
                                           nhead=self.nhead, layers=gen_layers, max_len=gen_max_len)

    def _profile_slots(self, property_id, type_id, value, slot_mask, profile=None):
        """Structure-free property slots with ALL known values (the generation conditioning)."""
        pid = property_id.clamp(0, self.num_properties - 1)
        return self._build_slots(pid, type_id, value, slot_mask, slot_mask, profile)   # value_mask = slot_mask

    def generation_loss(self, tok_in, tok_tgt, property_id, type_id, value, slot_mask, profile=None):
        slots = self._profile_slots(property_id, type_id, value, slot_mask, profile)
        logits = self.struct_dec(tok_in, slots, ~slot_mask)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tok_tgt.reshape(-1), ignore_index=PADT)

    @torch.no_grad()
    def generate(self, property_id, type_id, value, slot_mask, profile=None, max_len=128, greedy=True):
        slots = self._profile_slots(property_id, type_id, value, slot_mask, profile)
        B = property_id.shape[0]; dev = property_id.device
        seq = torch.full((B, 1), SOS, dtype=torch.long, device=dev)
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        for _ in range(max_len):
            logits = self.struct_dec(seq, slots, ~slot_mask)[:, -1]     # [B,vocab]
            nxt = logits.argmax(-1) if greedy else torch.multinomial(logits.softmax(-1), 1).squeeze(-1)
            nxt = torch.where(done, torch.full_like(nxt, PADT), nxt)
            seq = torch.cat([seq, nxt[:, None]], 1)
            done = done | (nxt == EOS)
            if done.all():
                break
        return seq[:, 1:]                                               # drop SOS
