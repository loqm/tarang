"""Test waveguides for propagation loss (run.goal = "loss_test"): bend loss against radius,
the spiral radius, and cutback lengths long enough to measure the loss.

From first principles
---------------------
Cutback: structures of lengths L_i, each with the same two couplers, transmit
T_i(dB) = -alpha L_i - C_i, where alpha is the propagation loss (dB/cm) and C_i the
coupling loss of structure i. C_i scatters from structure to structure by sigma (fibre
placement, cleanliness: typically 0.2-0.5 dB). A straight-line fit of T against L gives
alpha with the uncertainty of a least-squares slope,

    sigma_alpha = sigma / sqrt( sum_i (L_i - mean L)^2 ),

so the lengths must spread far enough that sigma_alpha is a small fraction of alpha.
Scaling every length by k divides sigma_alpha by k; the stage recommends the smallest k
(lengths rounded up to 0.5 cm) that reaches waveguides.target_precision.

Spirals keep long waveguides compact. Their bends must not add measurable loss: the bend
loss per cm of a bent waveguide (local bent-mode solve: the imaginary part of the bent
mode's effective index, k_eff, gives 4 pi k_eff / lambda per unit length) must stay below a
tenth of the expected propagation loss at the smallest radius used.

A ring resonator measures low loss far more precisely (alpha = 2 pi n_g / (Q_i lambda),
read from the linewidth of one under-coupled resonance); the comb_ring goal designs one.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

DB = 10 / math.log(10)


def slope_sigma(lengths_cm, sigma_db: float) -> float:
    """Uncertainty (dB/cm) of the loss fitted to structures of these lengths."""
    L = np.asarray(lengths_cm, float)
    s = float(np.sum((L - L.mean()) ** 2))
    return math.inf if s <= 0 else sigma_db / math.sqrt(s)


def recommend_lengths(lengths_cm, sigma_db: float, alpha_db_cm: float,
                      precision: float) -> tuple[list, float, float]:
    """(recommended lengths, their sigma_alpha, scale factor): the given pattern scaled up
    (never down) so sigma_alpha <= precision * alpha, in steps of 0.5 cm."""
    L = [float(x) for x in lengths_cm]
    now = slope_sigma(L, sigma_db)
    need = precision * alpha_db_cm
    k = 1.0 if now <= need else now / need
    out = [math.ceil(x * k * 2 - 1e-9) / 2 for x in L]
    return out, slope_sigma(out, sigma_db), k


FLOOR_DB_CM = 1e-3        # below this the mode solver cannot resolve the loss


def bend_loss_db_cm(ctx, H: float, W: float, R: float) -> Optional[float]:
    """Loss (dB/cm) of the pump mode (TE00 or TM00) bent at radius R: local mode solve in the bend's
    conformal frame with absorbing layers, the domain reaching past the radiation caustic
    x_c = R (n_eff / n_BOX - 1) so the radiated light is absorbed, and the bent mode
    picked by its overlap with the straight TE00 mode (in a tight bend the highest-index
    solutions are substrate whispering-gallery modes, not the waveguide mode)."""
    import tidy3d as td
    from tidy3d.plugins.mode import ModeSolver
    from tarang.waveguide.dispersion_mapping import core as dm
    cfg = ctx.cfg
    wl = cfg.run.wavelength_um
    xs = (ctx.state["stages"].get("cross_section") or {}).get("results") or {}
    n_eff = float(xs.get("n_eff") or ctx.n(cfg.materials.core) * 0.85)
    n_box = ctx.n(cfg.materials.box)
    margin = min(R * max(n_eff / n_box - 1, 0.0) + 3.0, 16.0)
    sim = dm.build_sim(H, W, wl0=wl, steps=cfg.cross_section.map_steps, box=cfg.stack.box_height_um,
                       core=cfg.materials.core, substrate=cfg.materials.box,
                       cladding=cfg.materials.cladding, cladding_height=cfg.stack.cladding_height_um,
                       band=ctx.band, margin_y=margin)
    plane = td.Box(center=(0, sim.center[1], sim.center[2]), size=(0, sim.size[1], sim.size[2]))

    def solve(**kw):
        spec = dict(num_pml=(15, 15), **kw)
        return ModeSolver(simulation=sim, plane=plane, freqs=[td.C_0 / wl],
                          mode_spec=td.ModeSpec(**spec)).solve()
    from tarang.design.stages import pump_mode
    te_pump = pump_mode(cfg) == "TE00"
    nm = 2 if te_pump else 6                     # TM00 lies below one or more TE modes
    straight = solve(num_modes=nm, target_neff=ctx.n(cfg.materials.core))
    bent = solve(num_modes=12, target_neff=n_eff + 0.01, bend_radius=float(R), bend_axis=1)

    def field(d, i, c):
        return np.squeeze(getattr(d, c).isel(mode_index=i, f=0).values)
    def is_pump_pol(i):
        ey = float(np.sum(np.abs(field(straight, i, "Ey")) ** 2))
        ez = float(np.sum(np.abs(field(straight, i, "Ez")) ** 2))
        return (ey > ez) == te_pump
    te = max(range(nm), key=lambda i: float(np.real(straight.n_eff.isel(mode_index=i, f=0)))
             if is_pump_pol(i) else -1)              # the pump mode (TE00 or TM00)
    s2 = sum(float(np.sum(np.abs(field(straight, te, c)) ** 2)) for c in ("Ey", "Ez"))

    def overlap(i):
        num = abs(sum(np.sum(np.conj(field(straight, te, c)) * field(bent, i, c)) for c in ("Ey", "Ez")))
        b2 = sum(float(np.sum(np.abs(field(bent, i, c)) ** 2)) for c in ("Ey", "Ez"))
        return float(num / math.sqrt(s2 * b2 + 1e-300))
    ovs = [overlap(i) for i in range(int(bent.n_eff.sizes["mode_index"]))]
    i = int(np.argmax(ovs))
    if ovs[i] < 0.5:
        return None                               # no bound mode left: radiates away
    k = float(np.real(bent.k_eff.isel(mode_index=i, f=0)))
    # power decays as exp(-4 pi k z / lambda); in dB per cm: x 10 / ln 10, lambda in cm
    return max(4 * math.pi * abs(k) / (wl * 1e-4) * DB, 0.0)


def default_radii(wl: float) -> list:
    base = [5, 7.5, 10, 15, 20, 30, 50, 100]
    return [round(r * wl / 1.55, 1) for r in base]


def choose_radius(rows: list, alpha_db_cm: float) -> Optional[float]:
    """Smallest radius whose bend loss is at most a tenth of the propagation loss (and every
    larger radius too, so a solver glitch at one radius does not decide)."""
    ok = None
    for r in sorted(rows, key=lambda r: -r["radius_um"]):
        if r["bend_loss_db_cm"] is None or r["bend_loss_db_cm"] > 0.1 * alpha_db_cm:
            break
        ok = r["radius_um"]
    return ok


def stage_waveguides(ctx) -> dict:
    from tarang.design import assumptions, visuals
    from tarang.design.stages import _losses, measured_loss
    from tarang.design.ui import parse_list
    cfg, ui = ctx.cfg, ctx.ui
    xs = ctx.results("cross_section")
    H, W, wl = xs["height_um"], xs["width_um"], cfg.run.wavelength_um
    ui.form(cfg, "Test waveguides (propagation loss)", [
        dict(section="ring", key="loss_preset",
             check=lambda v: "" if str(v).lower() in assumptions.LOSS_PRESETS else
             "choose one of " + ", ".join(assumptions.LOSS_PRESETS)),
        dict(section="ring", key="scattering_loss_db_cm",
             default=lambda c: assumptions.preset(c.ring.loss_preset)["scattering_db_cm"]),
        dict(section="ring", key="absorption_loss_db_cm",
             default=lambda c: assumptions.preset(c.ring.loss_preset)["absorption_db_cm"]),
        dict(section="waveguides", key="lengths_cm", cast=parse_list),
        dict(section="waveguides", key="coupler_scatter_db"),
        dict(section="waveguides", key="target_precision"),
        dict(section="waveguides", key="bend_radii_um", default=lambda c: default_radii(wl),
             cast=parse_list),
        dict(section="waveguides", key="bend_radius_um"),
        dict(section="waveguides", key="spiral_gap_um")])
    scat, absl = _losses(cfg)
    alpha = scat + absl
    wg = cfg.waveguides
    radii = [float(r) for r in (wg.bend_radii_um or default_radii(wl))]
    measured = measured_loss(cfg, xs.get("n_group", 2.0), wl)
    if measured:
        alpha = measured["straight_db_cm"]
        ui.say(f"Expected propagation loss {alpha:.3g} dB/cm ({measured['what']}, "
               f"{measured['source']}).")
    else:
        ui.say(f"Expected propagation loss {alpha:g} dB/cm (scattering {scat:g} + absorption "
               f"{absl:g}; {assumptions.loss_source(cfg, 'scattering')}).")
    ui.say(f"Bend loss of the {H * 1e3:.0f} × {W * 1e3:.0f} nm waveguide at {len(radii)} radii "
           "(local bent-mode solve, free):")
    rows = []
    for R in radii:
        try:
            b = bend_loss_db_cm(ctx, H, W, R)
        except Exception as err:                                  # noqa: BLE001
            ui.say(f"  (R = {R:g} µm: {type(err).__name__}: {err})")
            b = None
        rows.append(dict(radius_um=R, bend_loss_db_cm=b,
                         loss_per_90deg_db=None if b is None else b * math.pi * R / 2 * 1e-4))
    def shown(r):                       # below the solver's resolution: printed as an upper bound
        b = r["bend_loss_db_cm"]
        if b is None or b >= FLOOR_DB_CM:
            return r
        return dict(r, bend_loss_db_cm=f"< {FLOOR_DB_CM:g}",
                    loss_per_90deg_db=f"< {FLOOR_DB_CM * math.pi * r['radius_um'] / 2 * 1e-4:.1e}")
    ui.table([shown(r) for r in rows], formats={"bend_loss_db_cm": ".3g", "loss_per_90deg_db": ".2e"})
    if any(r["bend_loss_db_cm"] is not None and r["bend_loss_db_cm"] < FLOOR_DB_CM for r in rows):
        ui.say(f"  Below {FLOOR_DB_CM:g} dB/cm the mode solver cannot resolve the bend loss (the "
               "table's csv keeps the computed numbers).")
    if any(r["bend_loss_db_cm"] is None for r in rows):
        ui.say("  Empty: no bound mode at that radius (the light radiates out of the bend).")
    from tarang.design.checks import _write_csv
    _write_csv(ctx.path("tables", "bend_loss.csv"), rows)
    R_rec = choose_radius(rows, alpha)
    R_use = float(wg.bend_radius_um) if wg.bend_radius_um else (R_rec or max(radii))
    if R_rec is None:
        ui.say(f"  No radius in the list keeps the bend loss below {0.1 * alpha:.3g} dB/cm "
               f"(a tenth of {alpha:g}); using the largest, {R_use:g} µm. Add larger radii.")
    else:
        ui.say(f"  Recommended smallest bend radius: {R_rec:g} µm (bend loss ≤ {0.1 * alpha:.3g} "
               "dB/cm, a tenth of the propagation loss)" + ("" if R_use == R_rec else
                                                            f"; using your {R_use:g} µm") + ".")
    lengths = [float(x) for x in wg.lengths_cm]
    from tarang.design.stages import coupling
    edge = coupling(cfg) == "edge"
    chip_cm = float(cfg.edge.chip_length_um) * 1e-4
    if edge:
        # facet to facet: the shortest structure is the straight across the chip
        ui.say(f"Edge coupling: every structure runs facet to facet; 0 means the straight across the "
               f"{chip_cm * 1e4:g} µm chip ({chip_cm:g} cm), longer lengths are total lengths with a "
               "spiral in the middle.")
        short = [x for x in lengths if 0 < x <= chip_cm]
        if short:
            ui.say(f"  Lengths {', '.join(f'{x:g}' for x in short)} cm are not longer than the chip: "
                   "the layout cannot draw them (use longer lengths or a shorter chip).")
    sig_now = slope_sigma([chip_cm if (edge and x == 0) else x for x in lengths], wg.coupler_scatter_db)
    rec, sig_rec, k = recommend_lengths(lengths, wg.coupler_scatter_db, alpha, wg.target_precision)
    ui.say(f"Cutback with {', '.join(f'{x:g}' for x in lengths)} cm and ±{wg.coupler_scatter_db:g} dB "
           f"scatter per structure: loss known to ±{sig_now:.3g} dB/cm "
           f"({100 * sig_now / alpha:.0f} % of {alpha:g}).")
    if k > 1:
        ui.say(f"  Recommended for ±{100 * wg.target_precision:.0f} %: {', '.join(f'{x:g}' for x in rec)} "
               f"cm (±{sig_rec:.3g} dB/cm). Set waveguides.lengths_cm to use them.")
    else:
        ui.say(f"  This reaches the target precision (±{100 * wg.target_precision:.0f} %).")
    ui.say("  For losses below about 1 dB/cm an under-coupled ring measures them more precisely "
           "(Q_i from one linewidth): design one with run.goal = \"comb_ring\".")
    pitch = W + float(wg.spiral_gap_um)
    spirals = [dict(length_cm=x, **spiral_size((x - chip_cm) * 1e4 if edge else x * 1e4, 2 * R_use, pitch))
               for x in lengths if x > 0 and (not edge or x > chip_cm)]
    if spirals:
        ui.table(spirals, ["length_cm", "turns", "inner_radius_um", "outer_radius_um"],
                 {"inner_radius_um": ".1f", "outer_radius_um": ".1f"})
    fig = visuals.safe(ctx, "bend loss", draw_bend_loss, ctx, rows, alpha, R_rec, R_use)
    recommend = dict(
        choices=[dict(option="bend radius", value=R_rec, unit="um"),
                 dict(option="cutback lengths", value=rec, unit="cm")],
        recommended=dict(bend_radius_um=R_rec, lengths_cm=rec),
        why=(f"bend loss ≤ {0.1 * alpha:.3g} dB/cm; fitted loss to ±{100 * wg.target_precision:.0f} % "
             f"with ±{wg.coupler_scatter_db:g} dB scatter per structure"))
    return dict(alpha_db_cm=alpha, bend=rows, bend_radius_recommended_um=R_rec,
                bend_radius_um=R_use, lengths_cm=lengths, sigma_alpha_db_cm=sig_now,
                lengths_recommended_cm=rec, sigma_alpha_recommended_db_cm=sig_rec,
                spiral_pitch_um=pitch, spirals=spirals, n_group=xs.get("n_group"),
                figure=str(fig) if fig else None, recommend=recommend)


def spiral_size(length_um: float, r0: float, pitch: float) -> dict:
    """Turns and radii of the double spiral (two interleaved Archimedean arms, centre
    S-bend of radius r0 / 2) with this total length (see layout.spiral_path)."""
    from tarang.design.layout import spiral_path
    pts, info = spiral_path(length_um, r0, pitch)
    return dict(turns=info["turns"], inner_radius_um=info["r0"], outer_radius_um=info["r_out"])


def draw_bend_loss(ctx, rows, alpha, R_rec, R_use):
    """Figure 3w: bend loss against radius, with the limit (a tenth of the propagation
    loss) and the radius used."""
    from tarang.design import visuals
    plt = visuals._plt()
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    pts = [(r["radius_um"], max(r["bend_loss_db_cm"], FLOOR_DB_CM)) for r in rows
           if r["bend_loss_db_cm"] is not None]
    if pts:
        x, y = zip(*pts)
        ax.semilogy(x, y, "o-", color=visuals.PALETTE[0], label="bend loss (TE00)")
    ax.axhspan(1e-9, FLOOR_DB_CM, color="0.85", label="below the solver's resolution")
    ax.set_ylim(FLOOR_DB_CM / 3, None)
    ax.axhline(alpha, color="0.4", ls="--", label=f"expected propagation loss {alpha:g} dB/cm")
    ax.axhline(0.1 * alpha, color=visuals.PALETTE[1], ls=":", label="limit: a tenth of it")
    if R_rec:
        ax.axvline(R_rec, color=visuals.PALETTE[2], ls="-.", label=f"recommended {R_rec:g} µm")
    if R_use and R_use != R_rec:
        ax.axvline(R_use, color="k", ls=":", label=f"used {R_use:g} µm")
    ax.set(xlabel="Bend radius (µm)", ylabel="Loss (dB/cm)",
           title="Bend loss of the test waveguide (local bent-mode solve)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    return visuals._save(fig, ctx.path("figures", "3w_bend_loss.png"))
