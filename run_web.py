#!/usr/bin/env python3
"""Development launcher — runs VidTighten in the browser at http://127.0.0.1:9877"""

import os
import sys

_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if os.path.isdir(_src):
    sys.path.insert(0, _src)

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from preprod.web import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 9877))
    print(f"VidTighten: http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
