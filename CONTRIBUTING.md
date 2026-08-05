# Contributing

Contributions should stay within the AoH head-classification implementation. Please keep model-specific changes isolated, avoid committing model weights or generated outputs, and include a minimal reproducible test for behavior changes.

Before submitting a change, run:

```bash
python -m compileall -q aoh examples visualization
python -m aoh --help
```
