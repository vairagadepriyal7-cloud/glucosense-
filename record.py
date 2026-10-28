import serial, csv, time, os, sys

PORT = "COM9"          # your ESP32 port
BAUD = 115200


def open_port():
    """Open the port WITHOUT resetting / freezing the ESP32."""
    ser = serial.Serial()
    ser.port = PORT
    ser.baudrate = BAUD
    ser.timeout = 1
    ser.dtr = False    # don't pull the ESP32 into reset
    ser.rts = False    # don't put it into upload (BOOT) mode
    ser.open()
    return ser


def record(seconds=60, label="rec"):
    """Read the sensor for some seconds and save ms,red,ir to data/<label>_<time>.csv"""
    os.makedirs("data", exist_ok=True)
    ser = open_port()
    time.sleep(2)                 # let the connection settle
    ser.reset_input_buffer()      # throw away half-finished lines

    rows = []
    other_lines = 0
    start = time.time()
    last_shown = None

    while time.time() - start < seconds:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) != 3:
            other_lines += 1
            if other_lines <= 3:
                print(f"  (ESP32 says: {line})")
            continue
        try:
            ms, red, ir = (int(p) for p in parts)
        except ValueError:
            other_lines += 1
            continue
        rows.append((ms, red, ir))

        left = seconds - int(time.time() - start)
        if left % 10 == 0 and left != last_shown:
            print(f"  {left} s left ... {len(rows)} readings so far")
            last_shown = left

    ser.close()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = f"data/{label}_{stamp}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ms", "red", "ir"])
        writer.writerows(rows)

    print(f"Saved {len(rows)} readings to {path}")

    if rows:
        duration = (rows[-1][0] - rows[0][0]) / 1000
        rate = len(rows) / duration if duration > 0 else 0
        irs = [r[2] for r in rows]
        maxed = sum(1 for v in irs if v >= 262143)
        print(f"  Sample rate : {rate:.0f} readings per second")
        print(f"  IR range    : {min(irs)} to {max(irs)}")
        if maxed:
            print(f"  WARNING: {maxed} IR readings were maxed out (262143)")
    else:
        print("  No data received. Press EN on the ESP32 and try again.")

    return path


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "rec"
    print("Place your finger on the sensor and keep still for 60 seconds...")
    record(60, label)