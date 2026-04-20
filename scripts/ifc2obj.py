# batch convert ifc files to obj files. requires IfcConvert

import argparse
import os
import subprocess
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('output_dir')
    #parser.add_argument('ifcconvert_dir')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    files = os.listdir(args.input_dir)
    metadata_path = args.output_dir + "occluded.json"

    for f in files:
        print(f)
        out = f.split(".")[0] + ".obj"

        if len(f.split(".")) > 1:
            if f.split(".")[1] =='ifc':
                cmds = ("./IfcConvert", os.path.join(args.input_dir, f), os.path.join(args.output_dir, out))
                popen = subprocess.Popen(cmds, stdout=subprocess.PIPE)
                popen.wait()
                output = popen.stdout.read()
                print (output)

                # if args.sample_points:
