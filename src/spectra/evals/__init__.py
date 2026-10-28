"""spectra.evals: model specs, the frozen EvalModel (member: base / sweep-LoRA loader), protocol tables and the result writer
shared by the HEST / THUNDER / CPTAC / PathoROB harnesses (scripts/evals/).

Importing the package is cheap (stdlib only); torch/timm load with ``spectra.evals.model``.
See REQUIREMENTS.txt for the dependency set.
"""

__all__ = ["keys", "spec", "member", "model", "protocol", "results"]
