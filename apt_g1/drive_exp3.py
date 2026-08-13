"""exp3: scripted official closed-loop collection for missing walk directions.

Same pty driver as drive_exp2.py. Coverage targets:
  - WALK mode (2) heading sweeps (q/e) to populate bins 1/2/3/5/6/7
  - WALK strafes (,. keys)
  - SLOW_WALK speed variants (9/0) for continuous-speed coverage
"""
import os
import pty
import select
import subprocess
import sys
import time
import json

master, slave = pty.openpty()
proc = subprocess.Popen(
    ["bash", "/tmp/g1deploy_exp3.sh"],
    stdin=slave,
    stdout=slave,
    stderr=slave,
    close_fds=True,
)
os.close(slave)
log = open("/tmp/exp3/deploy.log", "wb", buffering=0)
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


def hold(key, seconds, interval=0.05, extra=None, extra_interval=0.0):
    end = time.time() + seconds
    n = ne = 0
    while time.time() < end and proc.poll() is None:
        os.write(master, key.encode())
        n += 1
        if extra is not None:
            os.write(master, extra.encode())
            ne += 1
        time.sleep(interval)
        if extra_interval > 0:
            time.sleep(extra_interval)
        drain(0.01)
    return n, ne


def fall_count():
    try:
        with open("/tmp/exp3/sim.log") as f:
            return f.read().count("Robot has fallen")
    except Exception:
        return 0


deadline = start + 120
while time.time() < deadline and proc.poll() is None:
    drain(0.5)
print("startup done, alive:", proc.poll() is None, "falls:", fall_count(), flush=True)

phase("start_control")
send("]")
time.sleep(12)
drain(0.5)

phase("planner_mode")
send("\r")
time.sleep(6)
drain(0.5)

phase("idle_12s")
time.sleep(12)
drain(0.3)
print("falls:", fall_count(), flush=True)

phase("walk_fwd_20s")
send("2")
time.sleep(1)
drain(0.3)
hold("w", 20)
print("falls:", fall_count(), flush=True)

phase("idle_8s")
send("r")
time.sleep(8)
drain(0.3)

phase("walk_heading_right_sweep_40s")
send("2")
time.sleep(1)
drain(0.3)
end = time.time() + 40
while time.time() < end and proc.poll() is None:
    os.write(master, b"w")
    time.sleep(0.05)
    if int(time.time() * 4) % 3 == 0:
        os.write(master, b"q")
    time.sleep(0.2)
    drain(0.01)
print("falls:", fall_count(), flush=True)

phase("idle_8s")
send("r")
time.sleep(8)
drain(0.3)

phase("walk_heading_left_sweep_40s")
send("2")
time.sleep(1)
drain(0.3)
end = time.time() + 40
while time.time() < end and proc.poll() is None:
    os.write(master, b"w")
    time.sleep(0.05)
    if int(time.time() * 4) % 3 == 0:
        os.write(master, b"e")
    time.sleep(0.2)
    drain(0.01)
print("falls:", fall_count(), flush=True)

phase("idle_8s")
send("r")
time.sleep(8)
drain(0.3)

phase("walk_strafe_right_20s")
send("2")
time.sleep(1)
drain(0.3)
hold(".", 20)
print("falls:", fall_count(), flush=True)

phase("idle_8s")
send("r")
time.sleep(8)
drain(0.3)

phase("walk_strafe_left_20s")
send("2")
time.sleep(1)
drain(0.3)
hold(",", 20)
print("falls:", fall_count(), flush=True)

phase("idle_8s")
send("r")
time.sleep(8)
drain(0.3)

phase("slow_speed_plus_20s")
send("1")
time.sleep(1)
drain(0.3)
for _ in range(4):
    send("0")
hold("w", 20)
print("falls:", fall_count(), flush=True)

phase("idle_8s")
send("r")
time.sleep(8)
drain(0.3)

phase("slow_speed_minus_20s")
send("1")
time.sleep(1)
drain(0.3)
for _ in range(6):
    send("9")
hold("w", 20)
print("falls:", fall_count(), flush=True)

phase("idle_12s")
send("r")
time.sleep(12)
drain(0.3)

phase("stop")
send("o")
time.sleep(3)
drain(0.5)

with open("/tmp/exp3/events.json", "w") as f:
    json.dump(events, f, indent=1)
print("alive_after_stop:", proc.poll() is None, "total falls:", fall_count(), flush=True)
if proc.poll() is None:
    proc.terminate()
drain(2)
