# Cleanup audit tools

These scripts implement the read-only repository audits used by the phased
cleanup process. Their existing paths are stable entry points during Phase 3.

## Generators

| Command | Outputs |
| --- | --- |
| `python tools/phase0_inventory.py` | `docs/phase0-inventory.json` and `docs/phase1-runtime-audit.md` |
| `python tools/phase1_config_contracts.py` | `docs/phase1-config-contracts.json` |
| `python tools/phase1_external_audit.py` | `docs/phase1-external-audit.json` |
| `python tools/phase2_model_contracts.py` | `docs/phase2-model-contracts.json` |

Run generators from the repository root. Fetch and prune all remote refs before
regenerating reports whose contents include remote-branch evidence:

```bash
git fetch --all --tags --prune
python tools/phase0_inventory.py
python tools/phase1_config_contracts.py
python tools/phase1_external_audit.py
python tools/phase2_model_contracts.py
```

## Tests

Run the complete audit and compatibility suite with:

```bash
python -m unittest discover -s tools -p 'test_*.py' -v
```

The Phase 2 runtime tests skip explicitly when PyTorch is unavailable; the CIIP
checks additionally require TorchGeo. A Phase 2 validation run is successful
only when all seven runtime tests pass without skips in the declared ML
environment.

## Safety

The generators inspect repository state and write only their declared files
under `docs/`. Their results are review evidence, not an unused-code detector.
Do not move or delete a path solely because it lacks a static reference.
