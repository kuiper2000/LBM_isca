# Copyright (C) 2026  PyLBM-Isca authors.  GPLv3 (or later).
"""Human-editable run configuration files -- the PyLBM-Isca *namelist*.

The Watanabe-Kimoto LBM drives every run from a ``SYSIN`` file: a stack of
Fortran ``&nmXXX ... &end`` groups read on stdin that fully specify the
integration.  PyLBM-Isca keeps that tradition -- one editable text file that
determines every parameter of a run -- but in a modern, dependency-light
format: **TOML** (stdlib :mod:`tomllib`) or **YAML** (if :mod:`PyYAML` is
installed), chosen by file extension.

The tables map one-to-one onto the old namelist groups::

    [time]    -> &nmtime / &nmdelt      (ndays, delt, inistp)
    [hdif]    -> &nmhdif                (order, tefold, tunit)
    [damp]    -> &nmdamp                (ddragv/d/t, tunit, sponge, ray0)
    [vdif]    -> &nmvdif                (vdifv/d/t/q)
    [integration]                       (robert, strictly_linear, nmem)
    [input]   -> &nminit                (basic_state, kmo, navg, zonal_mean)
    [forcing] -> &nmfrc / &nmsfrc       (mjo, steady, case, spinup, scale, ...)
    [output]  -> &nmhist / &nmhisd      (out, netcdf, interval_days, ...)
    [backend]                           (device)

A loaded file yields a :class:`RunSpec` with everything :mod:`run_lbm` needs.
Any value a user does *not* set keeps the same default the CLI uses, so a
namelist and the equivalent command line produce identical runs.
"""

import os

# ---------------------------------------------------------------------------
# Schema: section -> {toml_key: (target, dest)}
#   target "arg"    -> goes to run_lbm's argparse namespace (dest = argparse dest)
#   target "config" -> goes straight onto Config (dest = Config field name)
# Booleans whose argparse dest is store_true-negated are handled explicitly
# below (sponge, instantaneous) so the file reads naturally.
# ---------------------------------------------------------------------------
_SCHEMA = {
    "run": {
        "label": ("meta", "label"),
    },
    "time": {
        "ndays": ("arg", "days"),
        "delt": ("arg", "delt"),
        "inistp": ("arg", "inistp"),
    },
    "hdif": {
        "order": ("config", "hdif_order"),
        "tefold": ("config", "hdif_tefold"),
        "tunit": ("config", "hdif_tunit"),
    },
    "damp": {
        "ddragv": ("config", "ddragv"),
        "ddragd": ("config", "ddragd"),
        "ddragt": ("config", "ddragt"),
        "ddragq": ("config", "ddragq"),
        "tunit": ("config", "damp_tunit"),
        "ray0": ("config", "ray0"),
        # "sponge" handled specially (-> arg no_sponge)
    },
    "vdif": {
        "vdifv": ("config", "vdifv"),
        "vdifd": ("config", "vdifd"),
        "vdift": ("config", "vdift"),
        "vdifq": ("config", "vdifq"),
    },
    "integration": {
        "robert": ("arg", "robert"),
        "strictly_linear": ("arg", "strictly_linear"),
        "nmem": ("config", "nmem"),
    },
    "input": {
        "basic_state": ("arg", "basic_state"),
        "kmo": ("arg", "kmo"),
        "navg": ("arg", "navg"),
        "zonal_mean": ("arg", "zonal_mean_bs"),
    },
    "forcing": {
        "mjo": ("arg", "mjo_forcing"),
        "steady": ("arg", "steady_forcing"),
        "case": ("arg", "case"),
        "spinup": ("arg", "spinup"),
        "scale": ("arg", "forcing_scale"),
        "south_first": ("arg", "yflip"),
    },
    "output": {
        "out": ("arg", "out"),
        "netcdf": ("arg", "netcdf"),
        "interval_days": ("arg", "out_interval"),
        # "instantaneous" handled specially (-> arg instantaneous)
    },
    "backend": {
        "device": ("arg", "device"),
    },
}


class RunSpec:
    """Everything a namelist file specifies about a run.

    Attributes
    ----------
    arg_defaults : dict
        Keyed by :mod:`run_lbm` argparse *dest* names; fed to
        ``parser.set_defaults`` so the command line still overrides the file.
    config_extra : dict
        Config-only knobs the CLI does not expose (``hdif_*``, ``ddrag*``,
        ``vdif*``, ``ray0``, ``nmem``); merged into :class:`Config`.
    label : str or None
        The optional ``[run] label``.
    """

    def __init__(self, arg_defaults, config_extra, label=None):
        self.arg_defaults = arg_defaults
        self.config_extra = config_extra
        self.label = label

    def build_config(self, **overrides):
        """Construct a :class:`~pylbm_isca.model.Config` from this spec.

        ``overrides`` (typically the resolved argparse values) win over the
        file, mirroring "command line beats namelist".
        """
        from .model import Config

        a = dict(self.arg_defaults)
        a.update({k: v for k, v in overrides.items() if v is not None})
        kw = dict(self.config_extra)
        if "days" in a:
            kw["ndays"] = a["days"]
        for src in ("delt", "inistp", "robert"):
            if src in a:
                kw[src] = a[src]
        if "strictly_linear" in a:
            kw["strictly_linear"] = a["strictly_linear"]
        if "out_interval" in a:
            kw["out_interval_days"] = a["out_interval"]
        if "no_sponge" in a:
            kw["sponge"] = not a["no_sponge"]
        if "instantaneous" in a:
            kw["time_average"] = not a["instantaneous"]
        if "device" in a:
            kw["device"] = a["device"]
        return Config(**kw)


def _read_mapping(path):
    """Parse a TOML or YAML file into a nested dict (by extension)."""
    ext = os.path.splitext(path)[1].lower()
    with open(path, "rb") as fh:
        raw = fh.read()
    if ext in (".toml", ".nml", ".ini", ""):
        import tomllib
        return tomllib.loads(raw.decode("utf-8"))
    if ext in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                f"{path}: YAML config needs PyYAML (pip install pyyaml)") from exc
        return yaml.safe_load(raw.decode("utf-8")) or {}
    raise SystemExit(f"{path}: unknown config extension {ext!r} "
                     "(use .toml or .yaml)")


def load_namelist(path):
    """Load a TOML/YAML namelist file into a :class:`RunSpec`.

    Unknown sections or keys raise, so typos fail loudly rather than being
    silently ignored -- the same contract the Fortran namelist reader gave.
    """
    data = _read_mapping(path)
    arg_defaults, config_extra = {}, {}
    label = None

    for section, body in data.items():
        if section not in _SCHEMA:
            raise SystemExit(f"{path}: unknown section [{section}]")
        if not isinstance(body, dict):
            raise SystemExit(f"{path}: [{section}] must be a table")
        spec = _SCHEMA[section]

        for key, val in body.items():
            # specially-handled negated booleans
            if section == "damp" and key == "sponge":
                arg_defaults["no_sponge"] = not bool(val)
                continue
            if section == "output" and key == "instantaneous":
                arg_defaults["instantaneous"] = bool(val)
                continue
            if key not in spec:
                raise SystemExit(
                    f"{path}: unknown key '{key}' in [{section}]")
            target, dest = spec[key]
            if target == "meta":
                label = val
            elif target == "arg":
                arg_defaults[dest] = val
            else:
                config_extra[dest] = val

    return RunSpec(arg_defaults, config_extra, label=label)
