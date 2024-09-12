"""
Goal
---
1. Read test results from log.txt files
2. Compute mean and std across different folders (seeds)

Usage
---
Assume the output files are saved under output/my_experiment,
which contains results of different seeds, e.g.,

my_experiment/
    seed1/
        log.txt
    seed2/
        log.txt
    seed3/
        log.txt

Run the following command from the root directory:

$ python tools/parse_test_res.py output/my_experiment

Add --ci95 to the argument if you wanna get 95% confidence
interval instead of standard deviation:

$ python tools/parse_test_res.py output/my_experiment --ci95

If my_experiment/ has the following structure,

my_experiment/
    exp-1/
        seed1/
            log.txt
            ...
        seed2/
            log.txt
            ...
        seed3/
            log.txt
            ...
    exp-2/
        ...
    exp-3/
        ...

Run

$ python tools/parse_test_res.py output/my_experiment --multi-exp
"""
import re
import numpy as np
import os.path as osp
import argparse
from collections import OrderedDict, defaultdict

from dassl.utils import check_isfile, listdir_nohidden


def compute_ci95(res):
    return 1.96 * np.std(res) / np.sqrt(len(res))


def parse_function(*metrics, directory="", args=None, end_signal=None):
    #print(f"Parsing files in {directory}")
    subdirs = listdir_nohidden(directory, sort=True)

    outputs = []
    
    for subdir in subdirs:
        
        fpath = osp.join(directory, subdir, "log.txt")
        assert check_isfile(fpath)
        good_to_go = True
        output = OrderedDict()

        with open(fpath, "r") as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()

                if line == end_signal:
                    good_to_go = True

                for metric in metrics:
                    match = metric["regex"].search(line)
                    if match and good_to_go:
                        if "file" not in output:
                            output["file"] = fpath
                        num = float(match.group(1))
                        name = metric["name"]
                        output[name] = num

        if output:
            outputs.append(output)

    assert len(outputs) > 0, f"Nothing found in {directory}"

    metrics_results = defaultdict(list)

    for output in outputs:
        msg = ""
        for key, value in output.items():
            if isinstance(value, float):
                msg += f"{key}: {value:.2f}%. "
            else:
                msg += f"{key}: {value}. "
            if key != "file":
                metrics_results[key].append(value)

    output_results = OrderedDict()

    #print("===")
    #print(f"Summary of directory: {directory}")
    dir_sets = directory.split('/')
    #print(dir_sets)
    ci95=True
    for key, values in metrics_results.items():
        avg = np.mean(values)
        std = compute_ci95(values) if ci95 else np.std(values)
        #print(f"* {dir_sets[-1]} {key}: {avg:.2f}% +- {std:.2f}%")
        output_results[key] = avg

    return avg


def main(args):
    keyword = "accuracy"
    metric = {
        "name": keyword,
        "regex": re.compile(fr"\* {keyword}: ([\.\deE+-]+)%"),
    }

    datasets=['eurosat', 'dtd', 'fgvc_aircraft', 'food101', 'oxford_flowers', 'oxford_pets', 'stanford_cars', 'ucf101', 'caltech101','sun397','imagenet']

    train_avg_=0.0
    test_avg_=0.0
    
    for i in range(11):
        if i<9:
            config_file = 'vit_b32_ep50_ctxv1'
        else:
            config_file = 'vit_b128_ep10_ctxv1'
        directory='{}/base2new/train_base/{}/shots_16_4.0/TCP/'.format(args.directory,datasets[i])
        train_avg = parse_function(metric,directory=directory)
        directory='{}/base2new/test_new/{}/shots_16_4.0/TCP/'.format(args.directory,datasets[i])
        test_avg = parse_function(metric,directory=directory)
        H = 2*train_avg*test_avg/(train_avg+test_avg)
        print(f'{datasets[i]}:Train:{train_avg:.2f},Test:{test_avg:.2f},Final:{H:.2f}')
        train_avg_ +=train_avg
        test_avg_ +=test_avg
    train_avg_ /=11.0
    test_avg_ /=11.0
    H_ = 2*train_avg_*test_avg_/(train_avg_+test_avg_)
    print(f'Final Performance:Base:{train_avg_:.2f},New:{test_avg_:.2f},H:{H_:.2f}')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str, help="path to directory")
    args = parser.parse_args()

    main(args)
