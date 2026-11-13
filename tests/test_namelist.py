# Copyright (C) 2026  PyLBM-Isca authors.  GPLv3 (or later).
"""The TOML/YAML namelist loader -- the PyLBM-Isca equivalent of SYSIN.

Checks that (1) the tables map onto the right Config fields, (2) unknown
sections/keys fail loudly, (3) the shipped ``examples/mjo_phase1.toml``
reproduces the CLI defaults of the realistic experiment, and (4) run_lbm's
argparse still overrides the file.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pylbm_isca import Config, load_namelist          # noqa: E402
from pylbm_isca.model import DDRAG_DEFAULT             # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(ROOT, "examples", "mjo_phase1.toml")


def _write(tmp_path, text, name="run.toml"):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_config_fields_mapped(tmp_path):
    cfg = Config.from_namelist(_write(tmp_path, """
        [time]
        ndays = 10
        delt = 600.0
        inistp = 3
        [hdif]
        order = 2
        tefold = 4.0
        tunit = "DAY"
        [damp]
        ray0 = 15.0
        sponge = false
        [vdif]
        vdift = 2500.0
        [integration]
        robert = 0.02
        strictly_linear = true
        nmem = 4
        [output]
        interval_days = 2.0
        instantaneous = true
    """))
    assert cfg.ndays == 10
    assert cfg.delt == 600.0
    assert cfg.inistp == 3
    assert cfg.hdif_order == 2
    assert cfg.hdif_tefold == 4.0
    assert cfg.hdif_tunit == "DAY"
    assert cfg.ray0 == 15.0
    assert cfg.sponge is False
    assert cfg.vdift == 2500.0
    assert cfg.robert == 0.02
    assert cfg.strictly_linear is True
    assert cfg.nmem == 4
    assert cfg.out_interval_days == 2.0
    assert cfg.time_average is False        # instantaneous=true -> no averaging


def test_defaults_when_absent(tmp_path):
    """An empty-ish file leaves every Config default untouched."""
    cfg = Config.from_namelist(_write(tmp_path, "[time]\nndays = 5\n"))
    default = Config()
    assert cfg.ndays == 5
    assert cfg.delt == default.delt
    assert cfg.robert == default.robert
    assert cfg.sponge == default.sponge
    assert cfg.ddragv == default.ddragv


def test_unknown_section_and_key_raise(tmp_path):
    with pytest.raises(SystemExit):
        load_namelist(_write(tmp_path, "[nonsense]\nx = 1\n"))
    with pytest.raises(SystemExit):
        load_namelist(_write(tmp_path, "[time]\nbogus = 1\n"))


def test_sample_file_matches_cli_defaults():
    """examples/mjo_phase1.toml reproduces run_lbm.py's realistic defaults."""
    spec = load_namelist(SAMPLE)
    a = spec.arg_defaults
    assert a["basic_state"] == "data/bs.1979.bin"
    assert a["mjo_forcing"] == "data/frc.phase1.bin"
    assert a["days"] == 35
    assert a["spinup"] == 435
    assert a["out"] == "out/isca.1979.phase1"
    assert a["no_sponge"] is False          # sponge = true
    assert a["instantaneous"] is False

    cfg = spec.build_config()
    assert cfg.delt == 1200.0
    assert cfg.robert == 0.04
    assert cfg.ddragv == DDRAG_DEFAULT       # the linearneutral.t42l20 profile
    assert cfg.vdifv == 1.0e3


def test_run_lbm_cli_overrides_namelist():
    """--days on the command line beats [time] ndays in the file."""
    import run_lbm
    parser = run_lbm.build_parser()
    spec = load_namelist(SAMPLE)
    parser.set_defaults(**spec.arg_defaults)
    args = parser.parse_args(["--days", "7", "--device", "cpu"])
    assert args.days == 7                     # overridden
    assert args.basic_state == "data/bs.1979.bin"   # from the file
    assert args.spinup == 435                 # from the file
