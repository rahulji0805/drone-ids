"""
base_attack.py

Common interface for all 6 attack injectors. Each attack takes the
ground-truth flight records (before MAVLink encoding) and returns a
MODIFIED copy where the relevant field(s) are tampered with, plus
correct ground-truth labels for evaluation.

Design choice: we inject at the flight-physics-record level (pre-MAVLink),
which is more realistic for GPS/telemetry/command attacks (they corrupt
what the message CONTAINS), and we inject at the message-stream level for
protocol-level attacks like DoS and MAVLink-anomaly (they corrupt HOW/WHEN
messages arrive). See dos_attack.py and mavlink_anomaly_attack.py.
"""

from abc import ABC, abstractmethod
import copy


class BaseAttack(ABC):
    name = "base"

    def __init__(self, start_t, duration, rng):
        """
        start_t : seconds into the flight when the attack begins
        duration: seconds the attack lasts
        rng     : numpy random Generator, for reproducibility
        """
        self.start_t = start_t
        self.duration = duration
        self.rng = rng

    def is_active(self, t):
        return self.start_t <= t < self.start_t + self.duration

    @abstractmethod
    def apply(self, records):
        """Return a new list of records with the attack applied + labeled."""
        ...

    def _label(self, rec, t):
        rec = copy.deepcopy(rec)
        if self.is_active(t):
            rec["label"] = "attack"
            rec["attack_type"] = self.name
        return rec
