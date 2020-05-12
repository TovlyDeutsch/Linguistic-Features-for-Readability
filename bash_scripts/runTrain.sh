CONTAINER_NAME="dimages/pipinstall"
docker run --runtime=nvidia --name=mlrun3 --rm -dit -v /home/tovlydeutsch/seniorThesis:/seniorThesis -w /seniorThesis $CONTAINER_NAME \
python -W ignore runner.py -v configs/WeeBit/cnnLimit600.yaml
# rm -f output.log
# docker logs -f mlrun3 > output.log
# sleep 10s
# docker stop $(docker ps -q)
docker wait mlrun3
# git add .
# git commit -m "ran experiment"
# git push
# poweroff