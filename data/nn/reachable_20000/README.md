# Reachable data: 20,000 training poses per robot

Each robot has 26,000 accepted reachable poses: 20,000 training, 5,000 validation and 1,000 test. `train_large.csv` and `validation_large.csv` retain their existing names for trainer compatibility. Small/medium subsets are also derived by the master generator.

The original seed-42 raw audits were copied into `data/raw/reachable_20000/{robot}_master_audit.csv` and resumed to 26,000 accepted rows each in headless Webots. All newly accepted poses use the existing settling, clearance and FK-versus-measured-TCP error checks. Both successful and rejected attempts are retained. The first 16,000 accepted samples and existing test/small/medium split memberships are preserved. The original files remain separate.

To reproduce in a new destination, first copy each original `data/raw/{robot}_master_audit.csv` to a new raw directory. Then run these commands, replacing the destination names if they already exist:

```bash
python scripts/run_dataset_generation.py ned 26000 42 data/raw/reachable_20000/ned_master_audit.csv --large 25000 --splits-root data/nn/reachable_20000
python scripts/run_dataset_generation.py scara 26000 42 data/raw/reachable_20000/scara_master_audit.csv --large 25000 --splits-root data/nn/reachable_20000
```

Add `--webots PATH` if Webots is not on PATH. Repeating these commands resumes the copied audit and stops once 26,000 accepted rows exist. The independent `--splits-root` prevents rewriting original splits.

The dataset-only fallback generator uses this training bank to label certified unreachable requests. It writes `data/nn/directional_fallback_20000`, with 20,000 reachable + 6,667 unreachable training rows per robot (approximately 3:1), 5,000 + 1,667 validation rows and 1,000 + 1,000 test rows. It reuses previous unreachable test coordinates but chooses labels from this larger training bank. No test or validation pose supplies a training fallback label. The request audit links each label to this reachable training bank and its recorded Webots pose.
