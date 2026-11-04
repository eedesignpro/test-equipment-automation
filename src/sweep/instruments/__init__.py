"""Ready-made sources and measurements for every instrument with a driver here.

Each one takes a driver handle (for example from a user's ``bench.py``) and
uses only that driver's public methods. Equipment not listed here needs no
change to this package: pass a plain function, see :mod:`sweep.interfaces`.
"""

from sweep.instruments.keysight_33120a import Keysight33120ASource
from sweep.instruments.keysight_33500 import Keysight33500Source
from sweep.instruments.rohde_schwarz_rtb2000 import (
    RohdeSchwarzRtb2000AfgSource,
    RohdeSchwarzRtb2000Receiver,
)
from sweep.instruments.scalar import (
    ScalarMeasurement,
    keysight_b2900_current,
    keysight_b2900_voltage,
    keysight_e364xa_current,
    keysight_e364xa_power,
    keysight_e364xa_voltage,
    owon_odp_current,
    owon_odp_power,
    owon_odp_voltage,
    rigol_dl3000_current,
    rigol_dl3000_power,
    rigol_dl3000_voltage,
    siglent_sdm_ac_current,
    siglent_sdm_ac_voltage,
    siglent_sdm_frequency,
)
from sweep.instruments.siglent_sdg3000x import SiglentSdg3000xSource
from sweep.instruments.tektronix_afg31000 import TektronixAfg31000Source
from sweep.instruments.tektronix_mso import TektronixMsoAfgSource, TektronixMsoReceiver

__all__ = [
    "Keysight33120ASource",
    "Keysight33500Source",
    "RohdeSchwarzRtb2000AfgSource",
    "RohdeSchwarzRtb2000Receiver",
    "ScalarMeasurement",
    "SiglentSdg3000xSource",
    "TektronixAfg31000Source",
    "TektronixMsoAfgSource",
    "TektronixMsoReceiver",
    "keysight_b2900_current",
    "keysight_b2900_voltage",
    "keysight_e364xa_current",
    "keysight_e364xa_power",
    "keysight_e364xa_voltage",
    "owon_odp_current",
    "owon_odp_power",
    "owon_odp_voltage",
    "rigol_dl3000_current",
    "rigol_dl3000_power",
    "rigol_dl3000_voltage",
    "siglent_sdm_ac_current",
    "siglent_sdm_ac_voltage",
    "siglent_sdm_frequency",
]
