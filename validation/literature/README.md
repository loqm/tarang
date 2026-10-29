# TARANG against published Si3N4 microresonators

`compare.py` solves the cross-section of each resonator in `papers.csv` with TARANG's mode
solver and predicts the second-order dispersion D2 of its resonances from the paper's FSR:

    ω_μ = ω0 + D1 μ + D2 μ²/2,   D1 = 2π FSR,   D2 = −(c / n_g) D1² β2

β2 and n_g come from TARANG (the bent waveguide when the paper's radius is given). Only the
paper's geometry, cladding, mode, pump wavelength and FSR go in; nothing is fitted.

```
python validation/literature/compare.py              # all rows, about an hour, free
python validation/literature/compare.py --only liu2021_TE
```

Results go to `validation/literature/results/` (not committed).

## Kinds of rows

| kind | test | passes when |
|---|---|---|
| `measured`, `simulated` | D2/2π against the paper's | same sign, and within 20 % or inside TARANG's range for the film ±10 nm |
| `bracket` | a Damascene waveguide (SiO2 beside the core, air on top), which one top cladding cannot draw | the paper's value lies between TARANG's all-air and all-SiO2 values |
| `window` | the paper's statement that D stays anomalous over a range of widths | TARANG's D > 0 at five widths across it |
| `sign` | dispersion so close to zero that only its sign is robust | same sign |

## Results (v0.1.1, grid snapped to the core; mesh 50 steps per wavelength, the two closest rows also at 70)

| row | paper | TARANG | |
|---|---|---|---|
| Liu 2018 A, TE00, 1.58×0.75 µm, Damascene, 88.6 GHz | 1.10 MHz (meas.) | 0.73 (all SiO2) – 2.27 (all air) | inside |
| Liu 2018 A, TM00 | 0.967 MHz (meas.) | 0.16 – 2.11 | inside |
| Liu 2018 B, TE00, 1.58×0.81 µm, SiO2, 98.9 GHz | 1.23 MHz (meas.) | 1.288 MHz (70 steps: 1.282; film ±10 nm: 1.16–1.34) | +4.7 % (+4.2 %) |
| Liu 2021, TE00, 2.2×0.95 µm, SiO2, 40.6 GHz | 0.224 MHz (meas.) | 0.227 MHz (70 steps: 0.226; film ±10 nm: 0.218–0.226) | +1.1 % (+0.9 %) |
| Karpov 2018, TE00, 1.45×0.74 µm, R ≈ 23 µm, 1 THz, 1060 nm | −32 MHz (their sim.) | −37.9 MHz | −18.5 % |
| Karpov 2018, TM00 | +11 MHz (their sim.) | +10.1 MHz | −8.4 % |
| Moille 2024, TM00, H 0.43 µm, air, 780 nm | anomalous for W 0.575–0.775 µm | D = 152–169 ps/(nm km) | all anomalous |

Seven of seven agree. The finer mesh moves the two closest values by less than 0.5 %; at 50
steps the ±10 nm film values are noisy (both below the nominal one for Liu 2018 B), at 70 they
change smoothly with the film, so 70 is the default. Two lessons:

- **The bend matters in small rings.** At R ≈ 23 µm the straight waveguide gives the wrong
  sign for TE00 (+10.8 MHz against −32 MHz) and five times the TM00 value; with the bend both
  agree within 20 %. The ring stage computes D2 from the straight waveguide; for rings of
  about 100 µm and more that is fine (the 1550 nm design, R = 107.5 µm: the bend changes D by
  0.7 ps/(nm km) of 362, figure `bend_dispersion`), but a ring of a few tens of µm needs the
  bent mode's D.
- **Damascene waveguides lie between the two claddings** TARANG can draw. They agree as a
  bracket only, which is wide for TM00.

Known differences: the papers' films (index, thickness), sidewall angles (80–82° in the EPFL
devices; TARANG draws vertical sidewalls) and, for Karpov 2018, an assumed SiO2 cladding.
