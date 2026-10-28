# TARANG web page

`index.html` is the TARANG page for the LOQM website, in the style of the lab's software pages.
It is one static HTML file with its pictures in `assets/`: no script, no build step, nothing
loaded from elsewhere except the two Google fonts (system fonts are used without them).

**To put it on the site:** copy this folder (`index.html` and `assets/`) to the web server as
it is, e.g. as `loqm.tech/tarang/`. It goes live with the public release of the repository.

**Before it goes live:**
1. Replace the *Paper · in preparation* button (search `btn off`) with a link to the paper.
2. Check that the GitHub and documentation links open (the repository must be public).

**The pictures** come from the free local steps of a 1550 nm demonstration design
(materials, cross-section, ring) and from the GUI running on it; no measured data and no
cloud results are in them. To make them again after a change of the GUI or the figures,
run those steps on a design like `examples/design_sin1550.toml` and replace the files in
`assets/` with the same names:

| file | from |
|---|---|
| `gui_guided.png`, `gui_cross_section.png` | screenshots of `python -m tarang.gui` (1500 px wide window) |
| `fig_design_map.png` | `figures/2_dispersion_map/design_map.png` |
| `fig_profile.png` | `figures/3_dispersion_profile/profile_H700_W1000.png` |
| `fig_gamma.png` | `figures/3_dispersion_profile/gamma_H700_W1000.png` |

Keep each picture under about 400 kB (a width of 1200 to 1600 px is enough).
