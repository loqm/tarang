# Material data: sources and coverage

Every material TARANG can fit comes from a CSV in this folder, registered under a
key in [`../media.json`](../media.json). Columns are `wl` (wavelength in µm, strictly
increasing), `n` and, where measured, `k`. `get_medium(key, wl_range=...)` fits a
pole-residue model to the rows inside `wl_range`.

**Check the range before you simulate.** Outside the tabulated range the fitted
index is an extrapolation. TARANG warns when `wl_range` extends beyond the data and
refuses a range with no data at all.

## Summary

| Key | File | Material | Range (µm) | n at 1.55 µm | Columns |
|---|---|---|---|---|---|
| `si3n4` | `Luke_Si3N4.csv` | Stoichiometric Si₃N₄ film | 0.310 – 5.504 | 1.9963 | wl, n |
| `si3n4_hong` | `Hong_Si3N4_LPCVD.csv` | LPCVD SiN | 0.423 – 1.633 | 1.9897 | wl, n |
| `sio2` | `Malitson_fusedSilica.csv` | Fused silica (bulk) | 0.400 – 3.000 | 1.4440 | wl, n |
| `sio2_nyakuchena` | `Nyakuchena_fusedSilica.csv` | Silica glass microspheres | 1.100 – 1.650 | 1.4204 | wl, n |
| `si` | `Wang_Si_25C.csv` | Crystalline Si (thin SOI film), 25 °C | 0.400 – 1.305 | not covered | wl, n, k |
| `si_franta` | `_Franta_Si_300K.csv` | Float-zone crystalline Si, 300 K | 0.031 – 309.96 | 3.4795 | wl, n, k |
| `air` | none | Vacuum / air | any | 1 | handled in code |

n at 1.55 µm is interpolated from the tabulated data, not from a fit.

### Which one to use at 1550 nm

* **Silicon nitride:** `si3n4` (Luke) covers the whole telecom and mid-IR range.
  `si3n4_hong` ends at 1.633 µm, so it cannot describe a comb spanning well beyond
  the C/L bands.
* **Oxide:** `sio2` (Malitson, bulk fused silica, n = 1.444) is the standard choice
  for thermal or deposited oxide claddings and the buried oxide. `sio2_nyakuchena`
  is about 0.024 lower at 1550 nm. It was measured on glass microspheres, not on
  bulk fused silica or a thermal oxide film. Before v0.1.0 it was what `sio2`
  pointed to.
* **Silicon** (e.g. the handle wafer under a grating coupler): use **`si_franta`**.
  `si` (Wang) ends at 1.305 µm, so `get_medium("si", wl_range=(1.45, 1.65))` raises
  an error, and a wider range that only overlaps the data extrapolates. `si_franta`
  fits to within 1e-6 in index over 1.45–1.65 µm.

## References

Each dataset was taken from [refractiveindex.info](https://refractiveindex.info). The
details below are as listed there for each dataset.

**`si3n4`: Luke_Si3N4.csv**
K. Luke, Y. Okawachi, M. R. E. Lamont, A. L. Gaeta, M. Lipson, "Broadband
mid-infrared frequency comb generation in a Si₃N₄ microresonator,"
*Opt. Lett.* **40**, 4823–4826 (2015).
[doi:10.1364/OL.40.004823](https://doi.org/10.1364/OL.40.004823).
Sample: 340 nm stoichiometric Si₃N₄ on 3.1 µm thermal SiO₂ on Si. 101 points.

**`si3n4_hong`: Hong_Si3N4_LPCVD.csv**
Y. Hong, Y. Hong, J. Hong, G.-W. Lu, "Dispersion optimization of silicon nitride
waveguides for efficient four-wave mixing," *Photonics* **8**, 161 (2021).
[doi:10.3390/photonics8050161](https://doi.org/10.3390/photonics8050161).
LPCVD SiN. 35 points about 0.035 µm apart at non-round wavelengths, consistent with
values read from the paper's dispersion figure.

**`sio2`: Malitson_fusedSilica.csv**
I. H. Malitson, "Interspecimen comparison of the refractive index of fused silica,"
*J. Opt. Soc. Am.* **55**, 1205–1208 (1965).
[doi:10.1364/JOSA.55.001205](https://doi.org/10.1364/JOSA.55.001205).
Bulk fused silica, 20 °C. The file tabulates Malitson's Sellmeier formula from 0.4 to
3.0 µm in 0.01 µm steps (261 points; agrees with the formula to 1e-10).

**`sio2_nyakuchena`: Nyakuchena_fusedSilica.csv**
M. Nyakuchena, C. Juntunen, P. Shea, Y. Sung, "Refractive index dispersion
measurement in the short-wave infrared range using synthetic phase microscopy,"
*Phys. Chem. Chem. Phys.* **25**, 23141–23149 (2023).
[doi:10.1039/D3CP03158F](https://doi.org/10.1039/D3CP03158F).
Silica glass microspheres, 23 °C. 101 points.

**`si`: Wang_Si_25C.csv**
X. Wang, C. Shi, Q. Zheng, D. Thompson, "Resonant enhancement of the
thermoreflectance response of silicon nanodisks," *Appl. Phys. Lett.* **127**,
061103 (2025). [doi:10.1063/5.0276240](https://doi.org/10.1063/5.0276240).
Crystalline Si film (silicon-on-insulator, about 224 nm) on SiO₂, ellipsometry, 25 °C.
216 points with n and k.

**`si_franta`: _Franta_Si_300K.csv**
D. Franta, A. Dubroka, C. Wang, A. Giglia, J. Vohánka, P. Franta, I. Ohlídal,
"Temperature-dependent dispersion model of float zone crystalline silicon,"
*Appl. Surf. Sci.* **421**, 405–419 (2017).
[doi:10.1016/j.apsusc.2017.02.021](https://doi.org/10.1016/j.apsusc.2017.02.021).
Float-zone c-Si at 300 K, combined ellipsometry and spectrophotometry. 4001 points
with n and k.

## Adding a material

1. Put the CSV in this folder with a header line `wl,n` or `wl,n,k` (wavelength in
   µm, strictly increasing). A stacked refractiveindex.info export (an `n` block
   followed by a `wl,k` block) also works; it is merged in memory and the file is
   never modified.
2. Add a lower-case key to `tarang/media/media.json`, with the path relative to `tarang/media/`
   (`"data/<file>.csv"`).
3. Add a row to the summary table and a reference here.
4. Run `python -m pytest tests/test_packaging.py tests/test_docs.py`. These tests
   fail if a registered file is missing, is not shipped with the package, or is not
   listed in this file.

Changing an existing file's bytes (even its line endings) invalidates the fitted
material cache for that key, and with it the cache key of every simulation that uses
it. Two files (`Luke_Si3N4.csv`, `Nyakuchena_fusedSilica.csv`) are stored with
Windows line endings for that reason; `.gitattributes` leaves CSVs untouched.
