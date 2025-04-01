#!/usr/bin/env python3

from __future__ import print_function
import glob, sys, os, optparse, shutil, subprocess
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,currentdir) 

"""
Simple little script meant to normalize the outputs as 
things are converted over to new test system
Meant to be deleted after conversion
"""

def mvfiles(directory):
  for ofile in glob.glob(directory+"/*.out"):
    rootdir=os.path.dirname(ofile)
    base=os.path.splitext(os.path.basename(ofile))[0]
    if not "_" in base:
      newname=os.path.join(rootdir,base+"_1.out")
      subprocess.call("git mv "+ofile+" "+newname,shell=True)

  return

def main():
    parser = optparse.OptionParser(usage="%prog [options]")
    parser.add_option('-d', '--directory', dest='directory',
                      help='Directory containing results of PETSc test system',
                      default='output')
    options, args = parser.parse_args()

    # Process arguments
    if len(args) > 0:
      parser.print_usage()
      return

    directory=options.directory
    if not os.path.isdir(directory):
      print(directory+' is not a directory')
      return

    mvfiles(directory)


if __name__ == "__main__":
        main()
