# Equation-to-code mapping (PyLBM-Isca v1.0.0)

This document maps every tendency term in the governing equations of the model
description (manuscript Sect. 2, Eqs. 2–10) to its implementation, and records
the explicit/implicit split of the semi-implicit solver. It is the authoritative,
machine-readable companion to Appendix A of the manuscript.

Prognostic state: `x' = (zeta', D', T', q', pi')` with
`pi' = ln(ps_bar + ps') - ln(ps_bar)` (exact; the quadratic term is retained).

Code: explicit tendencies in `pylbm_isca/dynamics.py`; the semi-implicit
gravity-wave solve and the linear dissipation `D` in `pylbm_isca/integrator.py`
and `pylbm_isca/vertical.py`.

## Resolved review point: geopotential is counted once

The divergence tendency (Eq. 7) is

    dD'/dt = div(A') - laplacian(E' + Phi_v') - K_D D'

with `E' = u'·u_bar + v'·v_bar + 0.5(u'^2 + v'^2)` the **kinetic-energy term
only** (geopotential is *not* part of `E'`), and `Phi_v'` the virtual-temperature
hydrostatic geopotential. Geopotential therefore enters the divergence tendency
exactly once, through `Phi_v'`.

In the code the Laplacian argument is split:

- explicit (`dynamics.py`): `ke = hydrostatic(Tv' - T') + u'·u_bar + v'·v_bar
  [+ 0.5(u'^2+v'^2)]`, then `wt_div -= laplacian(grid_to_wave(ke))`.
  Here `hydrostatic(Tv'-T') = Phi(Tv') - Phi(T')` is only the **virtual
  correction**.
- implicit (`integrator.py`/`vertical.py`): the **dry** geopotential `Phi(T')`
  and the reference pressure-gradient / energy-conversion couplings are advanced
  by the per-wavenumber gravity-wave matrices.

Explicit + implicit = `laplacian(E' + Phi_v')`, as written. An earlier manuscript
draft defined `E' = Phi_v' + ...`, which double-counted geopotential on paper;
that was a notation error in the write-up, not in the code.

## Simmons–Burridge omega/sigma-dot factor X (Isca `four_in_one`)

    X_k   = alpha_k * D_k + wP_k * (V_k · grad ln ps) + wS_k * S_{k+1}
    S_k   = sum_{l >= k} dsig_l * (D_l + V_l · grad ln ps)      # mass-flux column integral
    sdot_k = sigm_k * S_0 - S_k                                 # zero at both boundaries

`alpha_k, wP_k, wS_k` are the pure-sigma Simmons–Burridge weights
(`vgrid.alpha`, `vgrid.wP`, `vgrid.w3`). Basic-state `X_bar` (`bst.x5`) and
perturbation `X'` (`x5p`) use the corresponding fields; the product
`T_v' X_bar + T_v_bar X'` is the exact linearization.

Code: `dynamics.py` `_column_sum`, `_sigma_dot`, `x5p = alpha*div + wP*padv +
w3*ssum[...,1:]`; basic state in `BasicStateTerms` (`ssum`, `sdot`, `x5`).

## Centered vertical advection V_sigma (Isca SECOND_CENTERED, advective form)

    V_sigma[phi]_k = -(1/dsig_k) * [ F_{k+1} - F_k - phi_k*(sdot_{k+1} - sdot_k) ]
    F_k            = sdot_k * 0.5*(phi_k + phi_{k-1}),   sdot = 0 at top & surface

Bilinear in `(phi, sdot)`, so the linearization is exact:

    V_sigma[phi'] = V_sigma[phi'; sdot_bar] + V_sigma[phi_bar; sdot']

It vanishes identically for isothermal / solid-body basic states
(`d(phi_bar)/dsig = 0` and `sdot_bar = 0`). Code: `dynamics.py` `_vert_advect`.
(The shared `_sigma_dot` returns sdot in the Watanabe–Kimoto sign convention,
opposite to Isca's half-level mass flux; the overall `+` sign in `_vert_advect`
restores the physical `-sdot d(phi)/dsig` — see `tests/test_vert_advection_sign.py`.)

## Term-by-term map

| Term | Eq. | Code (`pylbm_isca`) | Explicit / implicit |
|---|---|---|---|
| Absolute-vorticity flux `A'` | 5,6 | `dynamics.py`: `fu,fv` via `absvor`, `bs.vor_g` | explicit |
| Pressure gradient `-R*Tv*grad(ln ps)` | 5,6 | `dynamics.py`: `rcpg*(tv*bs.pix + tvdev*pix)` | explicit (pert + `Tv_bar-Tref` dev); reference part implicit |
| Vertical advection `V_sigma[u'],[v']` | 5,6 | `dynamics.py`: `_vert_advect(up/vp,...)` | explicit |
| Vorticity `curl(A') - K_zeta*zeta'` | 5 | `uv_to_vordiv(fu,fv)`; damping in `integrator.py` | explicit + damping |
| Divergence `div(A') - lap(E'+Phi_v') - K_D*D'` | 7 | `uv_to_vordiv`; `wt_div -= laplacian(ke)` | explicit (KE + virtual geopot.); dry geopot. + gravity wave implicit |
| Kinetic energy `E'` | 7 | `ke = ... + u'*bs.u + v'*bs.v [+0.5*(u'^2+v'^2)]` | explicit (quadratic term optional via `strictly_linear`) |
| Geopotential `Phi_v'` | 7 | `phiv = hydrostatic(tv-tp)` (explicit) + `Phi(T')` (implicit) | split |
| Adiabatic `-kappa*(Tv'*X_bar + Tv_bar*X')` | 8 | `dynamics.py`: `adiab`; `-kappa*Tref` part via `div @ tmtx.T` | explicit + reference implicit |
| Temperature advection (horiz + vert) | 8 | `hadv`, `vadv`, `neg_divergence(xu,xv)` | explicit |
| Humidity advection | 9 | `yu,yv,g`, `_vert_advect(qp/bs.q,...)` | explicit |
| Continuity `dpi'/dt` | 10 | `gt_pi = gpid + sum(div*dsig)` | explicit; divergence part implicit |
| Omega diagnostic `omega'` | — | `dynamics.py`: `omega = -sig*(x5p*ps_bar + x5_bar*ps')` | diagnostic |
| Dissipation `D` (hyperdiff, Rayleigh, Newtonian, sponge, vert. diff.) | 3,4 | `integrator.py` | implicit/explicit per operator |
| Semi-implicit gravity-wave solve | 4 | `integrator.py` `prepare/solve`; `vertical.py` `tmtx`,`gg`,`whmtx` | implicit |

## Nonlinearities retained

1. `0.5*(u'^2 + v'^2)` in `E'` — the single quadratic term Isca keeps; dropped by
   `Dynamics(strictly_linear=True)`.
2. The exact log-pressure `pi'` (and `ps' = ps_bar*(exp(pi')-1)`) — an exact
   change of variable, not a nonlinear tendency; the *evolution* of `pi'` is
   linear.

Everything else in `L(x_bar) x'` is strictly basic-state × perturbation.
