#!/usr/bin/bash
rsync -avrz \
      --exclude '.venv' \
      --exclude 'data' \
      --exclude 'logs' \
      --exclude 'runs' \
      --exclude '__pycache__' \
      --include '*.py' \
      -e 'ssh -p 58022' \
      * w0126@atbeetz21.informatik.tu-muenchen.de:opt_lat
