import time
import subprocess
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

WATCH_DIR = Path("data/manual")

class Handler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith((".txt", ".md")):
            print(f"New file detected: {event.src_path}")
            subprocess.run(["python3", "ingest_manual_local.py"])

if __name__ == "__main__":
    print("Watching for new documents in:", WATCH_DIR)

    event_handler = Handler()
    observer = Observer()
    observer.schedule(event_handler, str(WATCH_DIR), recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
