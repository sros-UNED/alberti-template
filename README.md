# alberti-template
alBERTi template code

Start a Google Cloud Deep Learning machine with PyTorch 1.4 and Debian 9 and run the next command (the hostname of the machine will be used to tag the experiment):

```bash
TAG=$(hostname) bash <(curl -s "https://raw.githubusercontent.com/linhd-postdata/alberti-template/master/run.sh")
```

Parameters (as envornment variables):

- `TAG`. How to tag the experiments. e.g., `alberti-roberta-base-es`. If should default to the hostname.
- `NFS`. Network filesystem to mount and save all the experiment runs and data to. If not given, local filesystem will be used, so be careful with the volumes termination policy.
- `NODEPS`. When set, no dependencies will be installed. This is useful for debugging.
- `SCRIPT`. The script to execute. It must be defined in `run.sh`. Parameters to the script must be passed in with a prefix. For example, if the `SCRIPT` is `stanzas`, all paremeters passed in must be prefixed with `ST_` by convention. An example [`stanzas-evaluation.py`](./stanzas-evaluation.py) is added to fill in the blanks.
