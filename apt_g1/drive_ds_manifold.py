"""ds_manifold: Phase 1 official-loop collection driver (DS_GAIT_MANIFOLD_PLAN §3).

Same pty driver pattern as drive_ds_smoke.py. Collects the 4-family gait
manifold grid from the official deploy loop (deploy C++ + run_sim_loop WBC
+ MuJoCo), plus inter-family transition segments:

  SLOW(1)   "1"           speeds {0.2,0.4,0.6,0.8} via '0' presses x dirs x 2 reps
  HAPPY(23) 'n' -> "4"    (styled set) 8 direction keyings x 2 reps
  RUN(3)    "3"           {default,2.0,2.5,3.0} via '0' presses x dirs x 2 reps
  JUMP(17)  "4"           (default set) {fwd, adjL, adjR} x 2 reps
  TRANS     A 10s -> 'r' 2s -> +{0,2,4}s -> B 10s, 12 ordered pairs x 3 phases

CRITICAL set-state machine: default set "1"-"6" = SLOW/WALK/RUN/FORWARD_JUMP/
STEALTH/INJURED; styled set (toggled with 'n', first = LEDGE) position 4 =
HAPPY. Pressing "4" in the wrong set collects the wrong gait -- enter_mode()
tracks and switches sets explicitly.

Every segment self-reports by grepping the tail of deploy.log for the last
"Replanning with mode: ..., target_vel: ..., movement: [...], facing: [...]"
line (plan §1 gotcha 6: never trust keypresses, verify mode/speed/direction).
Keyboard semantics (09-04 probe-calibrated): 'a'/'d' rotate movement+facing
by ~+-5.73 deg PER PRESS (accumulating -- holding spins!), 'q'/'e' rotate
facing only by +-30 deg, 'w' sustains momentum along the CURRENT movement
direction (never resets it), 's' is backward momentum, 'r' zeroes movement,
9/0 speed -/+ (SLOW base 0.2 / RUN base 1.5, +0.1 per press). Diagonal
directions = press 'a'/'d' N times (45 deg ~= 8 presses), then hold 'w'.

Run on lab-ts:  python3 /tmp/drive_ds_manifold.py [--smoke]
(after /tmp/setup_ds_manifold.sh; recordings land in /tmp/ds_manifold/)
"""
import argparse
import json
import os
import pty
import select
import subprocess
import sys
import time

REC = "/tmp/ds_manifold"
LOG = REC + "/deploy.log"

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true", help="reduced schedule for key/mode verification")
cli = ap.parse_args()

master, slave = pty.openpty()
proc = subprocess.Popen(
    ["bash", REC + "/run_deploy.sh"],
    stdin=slave,
    stdout=slave,
    stderr=slave,
    close_fds=True,
)
os.close(slave)
log = open(LOG, "wb", buffering=0)
events = []
start = time.time()


def drain(timeout):
    end = time.time() + timeout
    while time.time() < end:
        r, _, _ = select.select([master], [], [], 0.1)
        if not r:
            continue
        try:
            data = os.read(master, 65536)
        except OSError:
            return
        if not data:
            return
        log.write(data)
        if not cli.smoke:
            return  # keep non-smoke stdout clean
        sys.stdout.buffer.write(data)
        sys.stdout.flush()


def send(s):
    os.write(master, s.encode())
    drain(0.25)


def phase(name, detail=""):
    events.append([round(time.time() - start, 2), name, detail])
    print("T=%.2f %s %s" % (time.time() - start, name, detail), flush=True)


def hold(keys, seconds, interval=0.05):
    """Repeatedly send key(s) to hold momentum (decay without input)."""
    if isinstance(keys, str):
        keys = [keys]
    end = time.time() + seconds
    while time.time() < end and proc.poll() is None:
        for k in keys:
            os.write(master, k.encode())
        time.sleep(interval)
        drain(0.01)


def falls():
    try:
        with open(REC + "/sim.log") as f:
            return f.read().count("Robot has fallen")
    except Exception:
        return 0


def last_replanning():
    """Tail deploy.log for the most recent planner-command line."""
    try:
        with open(LOG, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 40000))
            tail = f.read().decode(errors="replace")
        lines = [l for l in tail.splitlines() if "Replanning with mode" in l]
        return lines[-1].strip() if lines else "(none)"
    except Exception as e:
        return "(err %s)" % e


current_set = 0  # motion-set index: 0 standing / 1 squat / 2 boxing / 3 styled
# (localmotion_kplanner.hpp get_motion_set; 'n' next set, 'p' previous;
#  styled set position 4 = HAPPY_DANCE_WALK -- smoke-verified 09-04)
SET_KEYS = {"SLOW": (0, "1"), "RUN": (0, "3"), "JUMP": (0, "4"),
            "HAPPY": (3, "4")}


def enter_mode(family):
    """Step the motion-set cycle explicitly (up='n', down='p') to the family's
    set, then select the family's mode key."""
    global current_set
    want_set, key = SET_KEYS[family]
    while current_set != want_set:
        step = 1 if want_set > current_set else -1
        send("n" if step > 0 else "p")
        current_set += step
        time.sleep(1.0)
        drain(0.3)
        phase("set_switch", "set%d" % current_set)
    send(key)
    time.sleep(1.0)
    drain(0.3)


prev_falls = [0]

# direction presets (09-04 probe): (presses, press_key, sustain_key);
# 45 deg ~= 8 presses of 'a'(left)/'d'(right); 'w' sustains current direction
DIR_PRESETS = {
    "fwd":  (0, "a", "w"),
    "back": (0, "a", "s"),
    "L45":  (8, "a", "w"),
    "L90":  (16, "a", "w"),
    "L135": (24, "a", "w"),
    "R45":  (8, "d", "w"),
    "R90":  (16, "d", "w"),
    "R135": (24, "d", "w"),
}


def segment(name, family, dir_preset, hold_secs, speed_presses=0, speed_label=""):
    """Enter mode, set speed, rotate direction (point presses), then hold the
    sustain key. Caller must emit the seg_start phase event first."""
    global prev_falls
    enter_mode(family)
    for _ in range(speed_presses):
        send("0")  # +0.1 m/s per press (deploy clamps per mode)
        time.sleep(0.3)
    drain(0.3)
    presses, press_key, sustain = dir_preset
    for _ in range(presses):
        send(press_key)
        time.sleep(0.08)
    drain(0.3)
    hold(sustain, hold_secs)
    f = falls()
    delta = f - prev_falls[-1]
    prev_falls.append(f)
    line = last_replanning()
    print("RESULT seg=%s falls=%d(+%d) replan: %s" % (name, f, delta, line), flush=True)
    events.append([round(time.time() - start, 2), "seg_end", name])
    return delta, line


def cool(name="idle_10s", secs=10):
    phase(name)
    send("r")
    time.sleep(secs)
    drain(0.3)


# ---- startup: wait for deploy + planner ready (same budget as ds_smoke) ----
deadline = start + 120
while time.time() < deadline and proc.poll() is None:
    drain(0.5)
print("startup done, alive:", proc.poll() is None, "falls:", falls(), flush=True)

phase("start_control")
send("]")
time.sleep(12)
drain(0.5)
phase("planner_mode")
send("\r")
time.sleep(6)
drain(0.5)

manifest = []


def run_segment(name, family, dirs, secs, presses=0, speed="default"):
    phase("seg_start", name)
    t0 = events[-1][0]
    d, line = segment(name, family, dirs, secs, presses, speed)
    manifest.append({"seg": name, "family": family, "dirs": dirs, "secs": secs,
                     "speed_presses": presses, "target_speed": speed,
                     "t": t0, "replan": line})
    cool()


# ---- smoke: reduced key/mode verification (plan §1 gotcha 6) ----
if cli.smoke:
    run_segment("smoke_slow_fwd_p2", "SLOW", "w", 15, presses=2, speed="0.4?")
    run_segment("smoke_run_fwd_p5", "RUN", "w", 15, presses=5, speed="2.0?")
    run_segment("smoke_happy_adjL", "HAPPY", ["a"], 15)
    run_segment("smoke_jump_fwd", "JUMP", "w", 15)
    phase("smoke_stop")
    send("o")
    time.sleep(3)
    drain(0.5)
    with open(REC + "/events.json", "w") as f:
        json.dump(events, f, indent=1)
    with open(REC + "/manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print("SMOKE_DONE alive:", proc.poll() is None, "falls:", falls(), flush=True)
    if proc.poll() is None:
        proc.terminate()
    drain(2)
    sys.exit(0)

# ---- full grid (plan §3) ----
# smoke-verified (09-04): SLOW base 0.2, RUN base 1.5, '0' = +0.1/press
SLOW_PRESSES = {"0.2": 0, "0.4": 2, "0.6": 4, "0.8": 6}
RUN_PRESSES = {"1.5": 0, "2.0": 5, "2.5": 10, "3.0": 15}

# SLOW(1): 4 speeds x 4 directions x 2 reps (dirs: fwd/back/adjL/adjR)
slow_dirs = {"fwd": "w", "back": "s", "adjL": "a", "adjR": "d"}
for speed, presses in SLOW_PRESSES.items():
    for dname, key in slow_dirs.items():
        for rep in (1, 2):
            run_segment("slow_%s_%s_r%d" % (dname, speed, rep), "SLOW", key, 60,
                        presses=presses, speed=speed)

# HAPPY(23): 8 direction keyings x 2 reps (bins verified from recordings at build)
happy_dirs = {"fwd": "w", "back": "s", "adjL": "a", "adjR": "d",
              "arcL": ["w", "q"], "arcR": ["w", "e"], "backarcL": ["s", "q"],
              "backarcR": ["s", "e"]}
for dname, keys in happy_dirs.items():
    for rep in (1, 2):
        run_segment("happy_%s_r%d" % (dname, rep), "HAPPY", keys, 60)

# RUN(3): fwd at 4 speeds + 4 dirs at {1.5, 2.5} -- all x 2 reps
run_dirs = {"back": "s", "adjL": "a", "adjR": "d", "arcL": ["w", "q"]}
for speed, presses in RUN_PRESSES.items():
    for rep in (1, 2):
        run_segment("run_fwd_%s_r%d" % (speed, rep), "RUN", "w", 60,
                    presses=presses, speed=speed)
for speed, presses in (("1.5", 0), ("2.5", 10)):
    for dname, keys in run_dirs.items():
        for rep in (1, 2):
            run_segment("run_%s_%s_r%d" % (dname, speed, rep), "RUN", keys, 60,
                        presses=presses, speed=speed)

# JUMP(17): fwd/adjL/adjR x 2 reps
for dname, key in {"fwd": "w", "adjL": "a", "adjR": "d"}.items():
    for rep in (1, 2):
        run_segment("jump_%s_r%d" % (dname, rep), "JUMP", key, 60)

# ---- transitions: 12 ordered pairs x 3 onset phases {0,2,4}s ----
fams = ["SLOW", "HAPPY", "RUN", "JUMP"]
for a in fams:
    for b in fams:
        if a == b:
            continue
        for ph in (0, 2, 4):
            name = "trans_%s2%s_p%d" % (a.lower(), b.lower(), ph)
            phase("seg_start", name)
            t0 = events[-1][0]
            enter_mode(a)
            hold("w", 10)
            send("r")
            time.sleep(2)          # idle 2s
            time.sleep(ph)         # onset phase after idle
            enter_mode(b)
            hold("w", 10)
            f = falls()
            delta = f - prev_falls[-1]
            prev_falls.append(f)
            print("RESULT seg=%s falls=%d(+%d) replan: %s" % (name, f, delta,
                                                              last_replanning()), flush=True)
            manifest.append({"seg": name, "family": "TRANS", "pair": (a, b),
                             "phase": ph, "t": t0,
                             "replan": last_replanning()})
            events.append([round(time.time() - start, 2), "seg_end", name])
            cool()

phase("stop")
send("o")
time.sleep(3)
drain(0.5)
with open(REC + "/events.json", "w") as f:
    json.dump(events, f, indent=1)
with open(REC + "/manifest.json", "w") as f:
    json.dump(manifest, f, indent=1)
print("alive_after_stop:", proc.poll() is None, "total_falls:", falls(), flush=True)
if proc.poll() is None:
    proc.terminate()
drain(2)
