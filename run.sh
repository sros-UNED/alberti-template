#!/bin/bash
# TERM=xterm-256color
# curl -o run.sh -q https://raw.githubusercontent.com/linhd-postdata/alberti-template/master/run.sh
# chmod +x run.sh


say() {
    echo "$@" | sed \
        -e "s/\(\(@\(red\|green\|yellow\|blue\|magenta\|cyan\|white\|reset\|b\|i\|u\)\)\+\)[[]\{2\}\(.*\)[]]\{2\}/\1\4@reset/g" \
        -e "s/@red/$(tput setaf 1)/g" \
        -e "s/@green/$(tput setaf 2)/g" \
        -e "s/@yellow/$(tput setaf 3)/g" \
        -e "s/@blue/$(tput setaf 4)/g" \
        -e "s/@magenta/$(tput setaf 5)/g" \
        -e "s/@cyan/$(tput setaf 6)/g" \
        -e "s/@white/$(tput setaf 7)/g" \
        -e "s/@reset/$(tput sgr0)/g" \
        -e "s/@b/$(tput bold)/g" \
        -e "s/@i/$(tput sitm)/g" \
        -e "s/@u/$(tput sgr 0 1)/g"
}

if [ -z "${NODEPS}" ]; then
    say @b"Installing dependencies" @reset
    sudo apt-get -y update
    sudo apt-get install -y jq byobu git
    sudo apt-get install -y nfs-common
    pip install -qU pip
    pip install -r https://raw.githubusercontent.com/linhd-postdata/alberti-template/master/requirements.txt
    wandb login
fi

if [ -n "${TAG}" ]; then
    sudo mkdir -p /shared
    sudo mount ${NFS-10.139.154.226:/shared} /shared
    sudo chmod go+rw /shared
    df -h --type=nfs
    mkdir -p "/shared/$TAG"
    mkdir -p "/shared/$TAG/runs"
    mkdir -p "/shared/$TAG/data"
    mkdir -p "/shared/$TAG/models"
    ln -s "/shared/$TAG/runs" runs
    ln -s "/shared/$TAG/data" data
    ln -s "/shared/$TAG/models" models
else
    mkdir -p runs
    mkdir -p data
    mkdir -p models
fi

curl -o shutdown.sh -q https://raw.githubusercontent.com/linhd-postdata/alberti-template/master/shutdown.sh
chmod +x shutdown.sh

if [ -n "${SCRIPT}" ]; then
    case "${SCRIPT}" in
    stanzas)
        say @b"Downloading stanzas-evaluation scripts" @reset
        curl -o clean-checkpoints.sh -q https://raw.githubusercontent.com/linhd-postdata/alberti-template/master/clean-checkpoints.sh
        chmod +x clean-checkpoints.sh
        curl -o stanzas-evaluation.py -q https://raw.githubusercontent.com/linhd-postdata/alberti-template/master/stanzas-evaluation.py
        chmod +x stanzas-evaluation.py
        byobu new-session -d -s "alberti" "watch -n 1 nvidia-smi"
        byobu new-window -t "alberti" "TAG=${TAG} MODELNAME=\"${ST_MODELNAME}\" OVERWRITE=${ST_OVERWRITE} python -W ignore stanzas-evaluation.py 2>&1 | tee -a \"runs/$(date +\"%Y-%m-%dT%H%M%S\").log\""
        sleep 10
        byobu new-window -t "alberti" "tail -f runs/*.log"
        byobu new-window -t "alberti" "tail -f models/*.log"
        byobu new-window -t "alberti" "tensorboard dev upload --logdir ./runs"
        if [ -z "${NOAUTOKILL}" ]; then
            byobu new-window -t "alberti" "./shutdown.sh $(cat pid)"
        fi
        say @green "--------------------------------" @reset
        say @green "| Run: byobu attach -t alberti |" @reset
        say @green "--------------------------------" @reset
        ;;
    *)
        echo "No SCRIPT specified."
        exit 1
        ;;
    esac
fi
