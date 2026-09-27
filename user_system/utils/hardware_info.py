import os, platform, subprocess, json, re, shutil

def run(cmd, shell=False):
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=shell, text=True)
    return out.strip()

def to_gib(bytes_val):
     return f"{bytes_val / (1024**3):.1f} GB"
 
 
def cpu_info():
    system = platform.system()
    try:
        if system == "Darwin":
            phys = int(run(["sysctl", "-n", "hw.physicalcpu"]) or 0)
            logi = int(run(["sysctl", "-n", "hw.logicalcpu"]) or 0)
        elif system == "Linux":
            if shutil.which("lscpu"):
                out = run(["lscpu"])
                cores_per_socket = re.search(r"Core\(s\) per socket:\s*(\d+)", out)
                sockets = re.search(r"Socket\(s\):\s*(\d+)", out)
                logical = re.search(r"CPU\(s\):\s*(\d+)", out)
                phys = int(cores_per_socket.group(1)) * int(sockets.group(1)) if cores_per_socket and sockets else 0
                logi = int(logical.group(1)) if logical else os.cpu_count() or 0
            else:
                phys = 0
                logi = os.cpu_count() or 0
        elif system == "Windows":
            ps = (
                "(Get-CimInstance Win32_Processor | "
                "Measure-Object -Property NumberOfCores -Sum).Sum"
            )
            phys = int(run(["powershell", "-NoProfile", "-Command", ps]) or 0)
            ps = (
                "(Get-CimInstance Win32_Processor | "
                "Measure-Object -Property NumberOfLogicalProcessors -Sum).Sum"
            )
            logi = int(run(["powershell", "-NoProfile", "-Command", ps]) or 0)
        else:
            phys = 0
            logi = os.cpu_count() or 0
    except Exception:
        phys = 0
        logi = os.cpu_count() or 0

    if phys and logi:
        return f"CPU: {phys} cores ({logi} threads)"
    elif logi:
        return f"CPU: {logi} threads (physical cores unknown)"
    else:
        return "CPU: Unknown"

def ram_info():
    system = platform.system()
    try:
        if system == "Darwin":
            bytes_total = int(run(["sysctl", "-n", "hw.memsize"]) or 0)
        elif system == "Linux":
            with open("/proc/meminfo", "r") as f:
                m = re.search(r"MemTotal:\s+(\d+)\s+kB", f.read())
            bytes_total = int(m.group(1)) * 1024 if m else 0
        elif system == "Windows":
            ps = "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"
            bytes_total = int(run(["powershell", "-NoProfile", "-Command", ps]) or 0)
        else:
            bytes_total = 0
    except Exception:
        bytes_total = 0

    return f"RAM: {to_gib(bytes_total)}"

def parse_nvidia_smi():
    if not shutil.which("nvidia-smi"):
        return []
    out = run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    if not out:
        return []
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            name, mb = parts[0], parts[1]
            try:
                vram_bytes = int(mb) * 1024 * 1024
            except Exception:
                vram_bytes = 0
            gpus.append((name, vram_bytes))
    return gpus

def gpu_info():
    system = platform.system()

    # Try NVIDIA first (all OSes)
    nvidia = parse_nvidia_smi()
    if nvidia:
        entries = [f"GPU: {name} (VRAM: {to_gib(v)})" for name, v in nvidia]
        return entries

    if system == "Darwin":
        # system_profiler JSON for clean parsing
        out = run(["system_profiler", "SPDisplaysDataType", "-json"])
        try:
            data = json.loads(out)
            displays = data.get("SPDisplaysDataType", [])
            entries = []
            for d in displays:
                name = d.get("_name") or d.get("sppci_model") or "Unknown GPU"
                vram_text = d.get("spdisplays_vram") or d.get("spdisplays_vram_shared") or ""
                if vram_text:
                    entries.append(f"GPU: {name} (VRAM: {vram_text})")
                else:
                    entries.append(f"GPU: {name} (VRAM:0)")
            if entries:
                return entries
        except Exception:
            pass
        return ["GPU: Not detected"]

    if system == "Windows":
        ps = (
            "Get-CimInstance Win32_VideoController | "
            "Where-Object { $_.Name -ne 'Microsoft Basic Display Adapter' } | "
            "Select-Object Name,AdapterRAM | "
            "ForEach-Object { "
            "$name=$_.Name; $ram=[int64]($_.AdapterRAM); "
            "if($ram -gt 0){ \"$name|$ram\" } else { \"$name|0\" } }"
        )
        out = run(["powershell", "-NoProfile", "-Command", ps])
        entries = []
        for line in out.splitlines():
            if "|" in line:
                name, ram = line.split("|", 1)
                try:
                    vram = int(ram)
                except Exception:
                    vram = 0
                if vram > 0:
                    entries.append(f"GPU: {name} (VRAM: {to_gib(vram)})")
                else:
                    entries.append(f"GPU: {name} (VRAM:0)")
        return entries or ["GPU: Not detected"]

    if system == "Linux":
        # Try glxinfo -B for a concise device line
        if shutil.which("glxinfo"):
            out = run("glxinfo -B | grep -E 'Device|Video memory|Memory' || true", shell=True)
            dev = None
            vram = None
            for line in out.splitlines():
                if "Device:" in line:
                    dev = line.split(":", 1)[1].strip()
                if "Video memory:" in line or "Memory:" in line:
                    m = re.search(r"(\d+)\s*MB", line)
                    if m: vram = int(m.group(1)) * 1024 * 1024
            if dev:
                return [f"GPU: {dev} (VRAM: {to_gib(vram) if vram else 'Unknown/Shared'})"]
        # Fallback: list PCI graphics devices
        if shutil.which("lspci"):
            out = run("lspci | grep -E 'VGA|3D|Display' || true", shell=True)
            entries = [f"GPU: {line.split(':', 2)[-1].strip()} (VRAM:0)" for line in out.splitlines() if line]
            return entries or ["GPU: Not detected"]
        return ["GPU: Not detected"]

    return ["GPU: Not detected"]

if __name__ == "__main__":
    print(f"=======System Information=======")
    print(f"CPU : {cpu_info()}")
    print(f"GPU : {gpu_info()[0]}")
    print(f"Ram : {ram_info()}")
    print(f"=================================")
