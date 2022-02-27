docker build -t heartfail-ml .
docker run --name heartfailure -p 8888:8888 -i heartfail-ml
docker exec –it heartfailure /bin/bash

Other:
docker container rm skillzcard_api



jupyter notebook --allow-root
