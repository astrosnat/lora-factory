"""Release the Colab runtime when nobody is looking: /content/keepalive untouched for 45 minutes and no job running."""
import os, subprocess, time, urllib.request
PAT = "[w]s_|[p]ip install|[f]etch_state|[j]udge_openrouter|[s]core_answers|[p]ush_"
while True:
    time.sleep(60)
    idle = time.time() - os.path.getmtime("/content/keepalive")
    busy = subprocess.run(["pgrep", "-f", PAT], capture_output=True).returncode == 0
    if idle > 45 * 60 and not busy:
        print(time.strftime("%H:%M:%S", time.gmtime()), "no keepalive for", int(idle), "s and nothing running: unassigning", flush=True)
        urllib.request.urlopen(urllib.request.Request("http://" + os.environ["TBE_RUNTIME_ADDR"] + "/unassign", method="POST", data=b""), timeout=60)
        break
