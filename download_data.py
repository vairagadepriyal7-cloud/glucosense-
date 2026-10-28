import os, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE_URL = "https://physionet.org/files/big-ideas-glycemic-wearable/1.1.3"

def download(job):
    url, dest = job
    tmp = dest + ".part"
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(256 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(tmp, dest)            # file appears only when complete
            print("Finished", dest)
            return
        except Exception as e:
            print(f"Retry {attempt} for {dest} ({e})")
            time.sleep(5)
    print("FAILED", dest, "- run the script again later")

jobs = []
for i in range(1, 17):
    pid = f"{i:03d}"
    os.makedirs(f"physionet/{pid}", exist_ok=True)
    for name in [f"Dexcom_{pid}.csv", f"IBI_{pid}.csv", f"Food_Log_{pid}.csv"]:
        dest = f"physionet/{pid}/{name}"
        if not os.path.exists(dest):
            jobs.append((f"{BASE_URL}/{pid}/{name}", dest))

print(f"{len(jobs)} files to download, 4 at a time...")
with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(download, jobs))
print("Done. If any file says FAILED, run this script again.")