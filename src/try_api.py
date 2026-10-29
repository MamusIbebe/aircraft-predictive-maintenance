"""Step 12: act like another system and send one engine's flights to the API."""
import json
import urllib.request

from data import load

engine = load("data/raw/test_FD001.txt").query("unit == 34")
sensors = [f"s{i}" for i in (2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21)]
body = {
    "engine_id": "ENG-34",
    "flights": [{"cycle": int(r.cycle), "sensors": {s: float(getattr(r, s)) for s in sensors}}
                for r in engine.itertuples()],
}
request = urllib.request.Request("http://127.0.0.1:8000/predict", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
print(f"Sending {len(body['flights'])} flights for ENG-34 ...")
print(json.loads(urllib.request.urlopen(request).read()))
