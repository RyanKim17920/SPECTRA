"""waivphaet -- registered-pair contrastive fine-tuning of pathology foundation models.

See PLAN.md for the full experimental design. In short: PLISM (7 scanners x 13 stains x
16,278 Elastix-aligned tiles) is our *training* set, PathoROB is the primary (untouched)
robustness metric, and PLISM retrieval is a training diagnostic only.
"""

__version__ = "0.1.0"
