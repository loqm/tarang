# Validation scripts

Scripts that check TARANG's results with other tools. They are not part of the installed
package and are not needed to use it.

| Folder | What | Needs |
|---|---|---|
| `pylle_soliton/` | Soliton formation for a TARANG design: hands `<design>/lle/` to pyLLE, prints the normalised pump f² and the soliton limit ζ_max = π²f²/8, sweeps the pump detuning (the soliton steps, the comb spectrum and the intracavity pulse), holds a single soliton seeded at a ζ inside the soliton range and checks it stays one steady pulse, and compares the D_int pyLLE used with TARANG's. Figures (with their numbers) and `pylle_summary.json` go to `<design>/validation/pylle/` | Julia >= 1.8, `pip install pyLLE` |
| `mesh_study/` | Why D of a thin core moves between meshes: the same cross-section at 50-90 steps/λ with the paper's dispersion band (0.2 µm) and with the band scaled to the wavelength; prints D against the mesh for each | nothing beyond TARANG |
| `literature/` | TARANG's D2 against seven published Si3N4 microresonators (measured and simulated, 780-1596 nm, air and SiO2 cladding, Damascene, a 23 µm ring); see its README | nothing beyond TARANG |

Run `python validation/pylle_soliton/run_pylle.py <design> --dry-run` to see the inputs it
would use without running anything. `--no-sweep` runs only the held single soliton (about
10 s of solver time); `--hold-zeta` sets its ζ.
