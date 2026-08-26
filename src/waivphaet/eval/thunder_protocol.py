"""THUNDER's per-backbone pooling protocol -- the ONE definition, dependency-free.

This lived inside :mod:`waivphaet.eval.thunder_model`, which cannot be imported outside
the THUNDER venv (it does ``from thunder.models import PretrainedModel`` at module
scope). The reporting scripts run in the main venv, so they grew their own copy of the
rule -- and the copy was WRONG: ``scoreboard._thunder_pooling`` returned ``clsmean`` for
every backbone that was not phikon-v2, which is right for midnight and Virchow2 by luck
and wrong for anything else.

So the tables live here, in a module with no third-party imports at all, and both the
runner and the reporters read them. A protocol constant with two definitions is a
protocol constant with none.
"""

from __future__ import annotations

#: THUNDER pooling is **per backbone**, and it is not our choice -- it is Waiv's.
#: arXiv:2607.22861 3, line 106: CLS+mean-pool concatenation was used for ALL models in
#: PathoROB, but in THUNDER only for Virchow2, AquaViT, H0-mini and **Midnight-12k**.
#: So phikon-v2 must be scored CLS-only here (which is also THUNDER's own published
#: phikon2 protocol, ``pretrained_models.py:303``) while midnight must be clsmean. Get
#: this backwards and the base-vs-fine-tuned rank sums are not comparable to their table.
#: ``paige-ai/Virchow2`` is in that same clsmean list, named explicitly on line 106, so it
#: is a transcription and not an inference. Its clsmean width is 2560 (2 x 1280), not
#: Midnight's 3072 -- but the *reason* the segmentation branch has to fall back to cls is
#: not the number 3072, it is that clsmean advertises ``emb_dim = 2 * hidden`` while
#: ``get_segmentation_embeddings`` returns raw hidden-d patch tokens. That inequality holds
#: for every backbone with clsmean pooling, Virchow2 included (2560 != 1280), so
#: ``resolve_pooling`` in ``thunder_model`` applies to it unchanged.
THUNDER_CLSMEAN_BACKBONES = frozenset({"kaiko-ai/midnight", "paige-ai/Virchow2"})

#: The other half of the same published table: backbones Waiv scored CLS-only.
#: Both sets are transcriptions of a paper, so membership cannot be inferred for a
#: backbone that is not in the paper -- which is why an unlisted backbone is an error
#: below rather than a default. Silently taking "cls" would produce a number that looks
#: like a THUNDER result and is not comparable to their table.
#:
#: ``bioptimus/H-optimus-0`` and ``MahmoodLab/UNI2-h`` are both in Table 2 and NEITHER is
#: in the line-106 clsmean list, so both are cls. The trap is ``H0-mini``, which IS in
#: that list: it is a *distillation of* H-Optimus-0 and a separate row of Table 2
#: (see ``docs/waiv_published.json`` -- "H0-mini" and "H-Optimus-0" are distinct models
#: with distinct numbers). Reading H0-mini's protocol onto H-Optimus-0 would silently
#: double its THUNDER feature width and put it on a different protocol from the paper.
THUNDER_CLS_BACKBONES = frozenset({
    "owkin/phikon-v2",
    "bioptimus/H-optimus-0",
    "MahmoodLab/UNI2-h",
})

_overlap = THUNDER_CLS_BACKBONES & THUNDER_CLSMEAN_BACKBONES
if _overlap:
    raise RuntimeError(
        f"backbones {sorted(_overlap)} are in BOTH THUNDER pooling sets; the protocol "
        "for a backbone is a single published fact and cannot be both."
    )
del _overlap


def default_pooling(backbone: str | None) -> str:
    """The published THUNDER pooling protocol for ``backbone``. Raises if unlisted."""
    if backbone is None:
        # Deliberately lazy: this module is imported by reporting scripts that have no
        # torch, and ``encoder`` pulls in torch/transformers at import time.
        from waivphaet.models.encoder import DEFAULT_BACKBONE

        backbone = DEFAULT_BACKBONE
    if backbone in THUNDER_CLSMEAN_BACKBONES:
        return "clsmean"
    if backbone in THUNDER_CLS_BACKBONES:
        return "cls"
    raise RuntimeError(
        f"no published THUNDER pooling protocol for backbone {backbone!r}. "
        "Set WAIV_POOLING=cls|mean|clsmean explicitly for this run, and if the choice is "
        "a protocol decision (i.e. it comes from arXiv:2607.22861 3), record it by adding "
        "the backbone to THUNDER_CLSMEAN_BACKBONES or THUNDER_CLS_BACKBONES in "
        "src/waivphaet/eval/thunder_protocol.py."
    )
