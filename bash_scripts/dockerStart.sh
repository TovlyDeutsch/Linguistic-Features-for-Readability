docker run --runtime=nvidia -it -v /home/tovlydeutsch/seniorThesis:/seniorThesis nvcr.io/nvidia/tensorflow:19.09-py3

docker run --runtime=nvidia -it -v /home/tovlydeutsch/seniorThesis:/seniorThesis -w /seniorThesis dimages/pipinstall
sudo -- sh -c '(nohup python -W ignore runner.py -v configs/WeeBit/cnn.yaml > console3.out; sleep 1m; poweroff) &'

export CLASSPATH="C:/Users/we890/seniorThesis/stanford-parser-full-2018-10-17/stanford-parser.jar;C:/Users/we890/seniorThesis/stanford-parser-full-2018-10-17/stanford-parser-3.9.2-models.jar"