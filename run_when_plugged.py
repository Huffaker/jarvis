"""
Run the web app only when the laptop is plugged into power.
Exits on startup if on battery, and stops the app when power is unplugged.

While running, this script (and app.py when run directly) asks Windows to
keep the system awake so the app stays available. You do not need to change
global sleep settings. When you stop the app, normal sleep/lid behavior returns.

To also keep running with the lid closed when plugged in, set one Windows
power option: "When plugged in, closing the lid" -> "Do nothing".
"""
import sys
import subprocess
import time
import os

# Default: check power every 30 seconds
CHECK_INTERVAL = int(os.environ.get("OLLAMA_AGENT_POWER_CHECK_SEC", "30"))


def request_stay_awake():
    """
    Ask Windows to avoid sleeping while this process is running.
    Has no effect on other OSes. When the process exits, the request is cleared.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    except Exception:
        pass


def is_plugged_in_windows():
    """True if AC power is connected (Windows)."""
    try:
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        class SYSTEM_POWER_STATUS(ctypes.Structure):
            _fields_ = [
                ("ACLineStatus", wintypes.BYTE),
                ("BatteryFlag", wintypes.BYTE),
                ("BatteryLifePercent", wintypes.BYTE),
                ("Reserved1", wintypes.BYTE),
                ("BatteryLifeTime", wintypes.DWORD),
                ("BatteryFullLifeTime", wintypes.DWORD),
            ]
        status = SYSTEM_POWER_STATUS()
        if kernel32.GetSystemPowerStatus(ctypes.byref(status)) == 0:
            return True  # assume plugged in if we can't read
        # ACLineStatus: 0 = offline (battery), 1 = online (AC)
        return status.ACLineStatus == 1
    except Exception:
        return True  # if we can't detect, allow running


def is_plugged_in():
    if sys.platform == "win32":
        return is_plugged_in_windows()
    # Linux: often in /sys/class/power_supply/AC/online or BAT0/status
    try:
        for name in ("AC", "ACAD", "AC0"):
            p = os.path.join("/sys/class/power_supply", name, "online")
            if os.path.isfile(p):
                with open(p) as f:
                    return f.read().strip() == "1"
    except Exception:
        pass
    return True  # allow running if we can't detect


def main():
    if not is_plugged_in():
        print("Not plugged in. Exiting. (Run when on AC power to start the server.)")
        sys.exit(1)

    request_stay_awake()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_py = os.path.join(script_dir, "app.py")
    if not os.path.isfile(app_py):
        print("app.py not found next to run_when_plugged.py")
        sys.exit(1)

    proc = subprocess.Popen(
        [sys.executable, app_py],
        cwd=script_dir,
        stdin=subprocess.DEVNULL,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    try:
        while True:
            time.sleep(CHECK_INTERVAL)
            if not is_plugged_in():
                print("\nPower unplugged. Stopping server.")
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                sys.exit(0)
            if proc.poll() is not None:
                sys.exit(proc.returncode or 0)
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        sys.exit(0)


if __name__ == "__main__":
    main()
