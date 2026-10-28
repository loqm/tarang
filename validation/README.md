# Validation scripts

Scripts that check TARANG's results with other tools. They are not part of the installed
package and are not needed to use it.

| Folder | What | Needs |
|---|---|---|
| `pylle_soliton/` | Soliton formation for a TARANG design: hands `<design>/lle/` to pyLLE, sweeps the pump detuning, saves the soliton steps, the comb spectrum and the intracavity pulse (with their numbers) in `<design>/validation/pylle/` | Julia >= 1.8, `pip install pyLLE` |

Run `python validation/pylle_soliton/run_pylle.py <design> --dry-run` to see the inputs it
would use without running anything.
