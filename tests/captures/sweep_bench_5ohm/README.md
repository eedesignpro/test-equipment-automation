# 5 ohm resistor captures, MSO58, 2026-11-03

Three points of the bench run `2026-11-03_11-33-38_5ohm_impedance` on the
development bench: 33522A CH1 into a 5 ohm resistor at 0.5 Vpp (load setting 5 ohm), V
across the resistor on CH8 (differential probe, reported as 1X), current on
CH3 (TCP202 current probe), trigger on CH5 (33522A Sync). Samples are stored as
float32. `z_re`, `z_im` hold the impedance that run computed at that point.
The current probe faces the opposite way to the voltage, so the phase is near
180 degrees.
