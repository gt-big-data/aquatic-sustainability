# entrypoint.sh
#!/bin/bash
echo "machine urs.earthdata.nasa.gov" > ~/.netrc
echo "login $EARTHDATA_USERNAME" >> ~/.netrc
echo "password $EARTHDATA_PASSWORD" >> ~/.netrc
chmod 600 ~/.netrc
exec gunicorn --bind :$PORT --workers 2 --threads 8 --timeout 0 run:app