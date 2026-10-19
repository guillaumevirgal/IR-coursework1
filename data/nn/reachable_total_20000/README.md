# Reachable subset for 20,000 total training examples

Per robot: 15,000 reachable training rows, 3,750 reachable validation rows, and the original 1,000 reachable test rows. These are prefixes of the corresponding `data/nn/reachable_20000` splits, whose poses were accepted and audited in Webots. Fallback labels use only this 15,000-pose training bank. The raw audit stays at `data/raw/reachable_20000`. Adding 5,000 unreachable training rows gives exactly 20,000 training rows per robot, at a 3:1 ratio.
