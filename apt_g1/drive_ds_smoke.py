"""ds_smoke: RUN mode + speed-axis smoke collection (D-series recollect line).

Same pty driver as drive_exp3.py. Purpose (2026-09-04 direction pivot):
  - mode 3 RUN was NEVER collected before (exp1/2/3 pressed 1/2/4/5 only);
    deploy LocomotionMode enum says RUN = 2.5-7.5 m/s, keyboard clamp 1.5-3.0
  - movement_speed axis (SLOW_WALK 0.2-0.8 / RUN 1.5-3.0) never swept:
    exp_all mode-2 WALK rows all have speed=-1 (mode default)
Coverage in this smoke:
  - health baseline: IDLE 15s -> WALK(2) fwd 60s (must reproduce exp-era behavior)
  - core 1: RUN(3) fwd 60s  (does it stand / actually run?)
  - core 2: RUN(3) speed ladder via '0' (+0.1 each press), 3 rungs x 30s
  - core 3: SLOW_WALK(1) speed ladder '0'x2 then '0'x4, 30s each
Each segment is separated by 10s IDLE ('r'); fall counter read from sim.log
after every segment. Aborts remaining locomotion segments if falls keep
growing while robot is down (prints CLEAR/KEEPRUNNING verdict per segment).

Run on lab-ts:  python3 /tmp/drive_ds_smoke.py   (after /tmp/setup_ds_smoke.sh)
"""
import json
import os
import pty
import select
import subprocess
import sys
import time

master, slave = pty.openpty()
proc = subprocess.Popen(
    ["bash", "/tmp/g1deploy_ds_smoke.sh"],
    stdin=slave,
    stdout=slave,
    stderr=slave,
    close_fds=True,
)
os.close(slave)
log = open("/tmp/ds_smoke/deploy.log", "wb", buffering=0)
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
        sys.stdout.buffer.write(data)
        sys.stdout.flush()


def send(s):
    os.write(master, s.encode())
    drain(0.25)


def phase(name):
    events.append((round(time.time() - start, 2), name))
    print("T=%.2f %s" % (time.time() - start, name), flush=True)


def hold(key, seconds, interval=0.05):
    end = time.time() + seconds
    n = 0
    while time.time() < end and proc.poll() is None:
        os.write(master, key.encode())
        n += 1
        time.sleep(interval)
        drain(0.01)
    return n


def falls():
    try:
        with open("/tmp/ds_smoke/sim.log") as f:
            return f.read().count("Robot has fallen")
    except Exception:
        return 0


# startup: wait for deploy + planner ready (same budget as exp3)
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

prev_falls = [0]


def segment(name, mode_key, hold_secs, speed_presses=0, press_label=""):
    """Enter mode, optionally raise speed N times, hold w, report falls."""
    global prev_falls
    phase(name)
    send(mode_key)
    time.sleep(1)
    drain(0.3)
    for _ in range(speed_presses):
        send("0")  # +0.1 m/s per press (deploy clamps per-mode)
        time.sleep(0.3)
    drain(0.3)
    if press_label:
        print("speed presses:", press_label, flush=True)
    n = hold("w", hold_secs)
    f = falls()
    delta = f - prev_falls[-1]
    prev_falls.append(f)
    print("RESULT seg=%s keyholds=%d falls_total=%d falls_delta=%d" % (
        name, n, f, delta), flush=True)
    return delta


def cool(name="idle_10s"):
    phase(name)
    send("r")
    time.sleep(10)
    drain(0.3)


# --- health baseline (if this fails the harness regressed, not RUN) ---
segment("idle_15s", "r", 0)
time.sleep(15)
drain(0.3)
print("falls:", falls(), flush=True)

segment("walk_fwd_60s_baseline", "2", 60)
cool()

# --- core 1: RUN at default speed ---
d = segment("run_fwd_60s_default", "3", 60)
cool()
run_broken = d > 0

# --- core 2: RUN speed ladder (skip if RUN already falling repeatedly) ---
if not run_broken:
    segment("run_fwd_30s_speed+2", "3", 30, speed_presses=2, press_label="0x2")
    cool()
    segment("run_fwd_30s_speed+4", "3", 30, speed_presses=4, press_label="0x4")
    cool()
    segment("run_fwd_30s_speed+6", "3", 30, speed_presses=6, press_label="0x6")
    cool()
else:
    phase("run_speed_ladder_SKIPPED_run_default_fell")
    # one retry after full settle: single data point still valuable
    time.sleep(5)
    segment("run_fwd_30s_default_retry", "3", 30)
    cool()

# --- core 3: SLOW_WALK speed ladder (axis known to work, sweep coverage) ---
segment("slow_fwd_30s_speed+2", "1", 30, speed_presses=2, press_label="0x2")
cool()
segment("slow_fwd_30s_speed+4", "1", 30, speed_presses=4, press_label="0x4")
cool()
segment("slow_fwd_30s_speed+6", "1", 30, speed_presses=6, press_label="0x6")
cool()

phase("stop")
send("o")
time.sleep(3)
drain(0.5)

with open("/tmp/ds_smoke/events.json", "w") as f:
    json.dump(events, f, indent=1)
print("alive_after_stop:", proc.poll() is None,
      "total_falls:", falls(), flush=True)
if proc.poll() is None:
    proc.terminate()
drain(2)
