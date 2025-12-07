#!/bin/bash
# entrypoint.sh
echo "machine urs.earthdata.nasa.gov" > ~/.netrc
echo "login $EARTHDATA_USER" >> ~/.netrc
echo "password $EARTHDATA_PASS" >> ~/.netrc
chmod 600 ~/.netrc
exec gunicorn --bind :$PORT --workers 2 --threads 8 --timeout 0 run:app