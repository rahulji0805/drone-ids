"""
command_injection_attack.py  (TC-02)

Simulates an unauthorized/abnormal flight-control command being injected
into the MAVLink command stream — e.g. an attacker on the same network/RF
link sends a COMMAND_LONG (e.g. MAV_CMD_DO_SET_MODE, MAV_CMD_NAV_LAND,
MAV_CMD_COMPONENT_ARM_DISARM) that did NOT originate from the legitimate
ground control station.

Detection signal: our detector will check
  (a) sender system/component ID not matching the known-authorized GCS ID
  (b) command arriving with no corresponding operator context (e.g. a LAND
      command while mid-mission with no prior mode-change pattern)
  (c) command rate/sequence anomalies (a real GCS doesn't spam commands)

This module injects synthetic COMMAND_LONG messages into the stream at
the flight-physics-record level (as sidecar events keyed by timestamp),
which the MAVLink encoder later turns into real COMMAND_LONG messages
from a spoofed system ID.
"""

import numpy as np
from base_attack import BaseAttack

ATTACKER_SYSTEM_ID = 99  # anything != the legitimate GCS system id (255)
LEGIT_GCS_SYSTEM_ID = 255

INJECTED_COMMANDS = [
    "MAV_CMD_NAV_LAND",
    "MAV_CMD_DO_SET_MODE",
    "MAV_CMD_COMPONENT_ARM_DISARM",
    "MAV_CMD_DO_REPOSITION",
]


class CommandInjectionAttack(BaseAttack):
    name = "command_injection"

    def __init__(self, start_t, duration, rng, injection_rate_hz=2.0):
        super().__init__(start_t, duration, rng)
        self.injection_rate_hz = injection_rate_hz

    def apply(self, records):
        """
        Returns (records, injected_events).
        records: labeled the same as input, but with label='attack' during
                 the injection window (the anomaly is the EXTRA commands,
                 not a change to physics, so trajectory fields are untouched).
        injected_events: list of dicts describing each fake command, with
                 timestamps, to be turned into COMMAND_LONG MAVLink messages.
        """
        out = []
        injected_events = []
        next_inject_t = self.start_t

        for rec in records:
            r = dict(rec)
            t = r["t"]

            if self.is_active(t):
                r["label"] = "attack"
                r["attack_type"] = self.name

                if t >= next_inject_t:
                    cmd = self.rng.choice(INJECTED_COMMANDS)
                    injected_events.append({
                        "t": t,
                        "command": cmd,
                        "source_system": ATTACKER_SYSTEM_ID,
                        "source_component": 1,
                    })
                    next_inject_t += 1.0 / self.injection_rate_hz

            out.append(r)
        return out, injected_events


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
    from flight_physics import NormalFlightSimulator

    sim = NormalFlightSimulator()
    records = sim.generate(20)
    rng = np.random.default_rng(2)
    atk = CommandInjectionAttack(start_t=5.0, duration=4.0, rng=rng, injection_rate_hz=2.0)
    out, events = atk.apply(records)

    print(f"Attack-labeled records: {sum(1 for r in out if r['label']=='attack')} / {len(out)}")
    print(f"Injected fake commands: {len(events)}")
    for e in events:
        print(f"  t={e['t']:.2f}s  cmd={e['command']}  from sysid={e['source_system']} (illegit)")
