"""
mavlink_encoder.py

Takes ground-truth flight-physics records and encodes them as REAL MAVLink v2
messages (HEARTBEAT, GPS_RAW_INT, GLOBAL_POSITION_INT, ATTITUDE, SYS_STATUS),
same message types a real PX4/ArduPilot autopilot emits.

This gives us genuine protocol-level fidelity: our detection engine reads
actual MAVLink message objects, not a made-up schema. Attack injection later
will corrupt/spoof/flood these same message streams.
"""

import time
from pymavlink import mavutil
from pymavlink.dialects.v20 import common as mavlink2


SYSTEM_ID = 1
COMPONENT_ID = mavlink2.MAV_COMP_ID_AUTOPILOT1


def make_mav():
    """Create a standalone mavlink message-encoder (no real connection)."""
    mav = mavlink2.MAVLink(None, srcSystem=SYSTEM_ID, srcComponent=COMPONENT_ID)
    mav.robust_parsing = True
    return mav


def record_to_messages(mav, rec, boot_time_us):
    """
    Convert one flight-physics record into the set of MAVLink messages
    an autopilot would emit for that instant. Returns list of (msg_type, packed_bytes, decoded_fields).

    Honors attack-injected override fields when present:
      - spoofed_lat/spoofed_lon   -> used for GPS_RAW_INT (raw receiver),
                                      GLOBAL_POSITION_INT keeps TRUE lat/lon
                                      (fused estimate resists pure-GPS spoofing)
      - telemetry_alt/vz/roll     -> used for GLOBAL_POSITION_INT/ATTITUDE
                                      (fused/reported telemetry is what's
                                      manipulated; GPS_RAW_INT keeps truth)
    """
    t_us = int(rec["t"] * 1e6) + boot_time_us
    msgs = []

    gps_lat = rec.get("spoofed_lat", rec["lat"])
    gps_lon = rec.get("spoofed_lon", rec["lon"])
    tel_alt = rec.get("telemetry_alt", rec["alt"])
    tel_vz = rec.get("telemetry_vz", rec["vz"])
    tel_roll = rec.get("telemetry_roll", rec["roll"])

    # HEARTBEAT — sent regularly, its absence/delay is our DoS signal
    hb = mav.heartbeat_encode(
        type=mavlink2.MAV_TYPE_QUADROTOR,
        autopilot=mavlink2.MAV_AUTOPILOT_PX4,
        base_mode=mavlink2.MAV_MODE_FLAG_SAFETY_ARMED | mavlink2.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        custom_mode=0,
        system_status=mavlink2.MAV_STATE_ACTIVE,
    )
    msgs.append(("HEARTBEAT", hb))

    # GLOBAL_POSITION_INT — fused position/telemetry (what telemetry-manipulation corrupts)
    gpi = mav.global_position_int_encode(
        time_boot_ms=t_us // 1000,
        lat=int(rec["lat"] * 1e7),
        lon=int(rec["lon"] * 1e7),
        alt=int(tel_alt * 1000),
        relative_alt=int(tel_alt * 1000),
        vx=int(rec["vx"] * 100),
        vy=int(rec["vy"] * 100),
        vz=int(tel_vz * 100),
        hdg=int(rec["heading"] * 100),
    )
    msgs.append(("GLOBAL_POSITION_INT", gpi))

    # GPS_RAW_INT — raw GPS receiver output (what GPS spoofing corrupts)
    gps = mav.gps_raw_int_encode(
        time_usec=t_us,
        fix_type=3,  # 3D fix
        lat=int(gps_lat * 1e7),
        lon=int(gps_lon * 1e7),
        alt=int(rec["alt"] * 1000),
        eph=100, epv=100,
        vel=int(((rec["vx"]**2 + rec["vy"]**2) ** 0.5) * 100),
        cog=int(rec["heading"] * 100),
        satellites_visible=12,
    )
    msgs.append(("GPS_RAW_INT", gps))

    # ATTITUDE — IMU-derived orientation, independent of GPS
    att = mav.attitude_encode(
        time_boot_ms=t_us // 1000,
        roll=tel_roll, pitch=rec["pitch"], yaw=rec["yaw"],
        rollspeed=0.0, pitchspeed=0.0, yawspeed=0.0,
    )
    msgs.append(("ATTITUDE", att))

    # SYS_STATUS — battery/load, useful for computational-overhead framing later
    sys_status = mav.sys_status_encode(
        onboard_control_sensors_present=0, onboard_control_sensors_enabled=0,
        onboard_control_sensors_health=0,
        load=300, voltage_battery=12400, current_battery=1500,
        battery_remaining=80, drop_rate_comm=0, errors_comm=0,
        errors_count1=0, errors_count2=0, errors_count3=0, errors_count4=0,
    )
    msgs.append(("SYS_STATUS", sys_status))

    return t_us, msgs


def encode_stream(records):
    """
    Generator: yields (t_us, msg_type, MAVLink_message_object) for a full
    list of flight-physics records — this is our 'wire stream'.
    """
    mav = make_mav()
    boot_time_us = int(time.time() * 1e6)
    for rec in records:
        t_us, msgs = record_to_messages(mav, rec, boot_time_us)
        for msg_type, msg in msgs:
            yield t_us, msg_type, msg, rec


if __name__ == "__main__":
    from flight_physics import NormalFlightSimulator

    sim = NormalFlightSimulator()
    records = sim.generate(5)
    count = 0
    for t_us, msg_type, msg, rec in encode_stream(records):
        count += 1
        if count <= 6:
            print(msg_type, "->", msg.to_dict())
    print(f"\nTotal MAVLink messages generated: {count}")
