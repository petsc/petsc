#!/usr/bin/env python
from __future__ import print_function
import glob
import sys
import os
import optparse
import re
import inspect

thisfile = os.path.abspath(inspect.getfile(inspect.currentframe()))
pdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(thisfile)))))
sys.path.insert(0, os.path.join(pdir, 'config'))

import testparse


"""
  Simple walker for the new test harness
  Currently prints out the list of datafiles
"""


def get_datafiles(args):
    """ Go through args and pull out datafiles"""
    mylist = []
    for subarg in args.split():
        if 'DATAFILESPATH' in subarg:
            dfile = re.sub('\${DATAFILESPATH}/', '', subarg)
            dfile = re.sub('\$DATAFILESPATH/', '', dfile)
            mylist.append(dfile)
    return mylist


def walktree(top, action, datafilespath=None):
    """
    Walk a directory tree, starting from 'top'
    """
    verbose = False
    d = {}
    alldatafiles = []
    for root, dirs, files in os.walk(top, topdown=False):
        if "examples" not in root: continue
        if root == 'output': continue
        if '.dSYM' in root: continue
        if verbose: print(root)

        for exfile in files:
            # Ignore emacs files
            if exfile.startswith("#") or exfile.startswith(".#"): continue
            ext=os.path.splitext(exfile)[1]
            if ext[1:] not in ['c','cxx','cpp','cu','F90','F']: continue

            # Convenience
            fullex = os.path.join(root, exfile)
            if verbose: print('   --> '+fullex)
            d[root] = testparse.parseTestFile(fullex, 0)
            if exfile in d[root]:
                for test in d[root][exfile]:
                    if 'args' in d[root][exfile][test]:
                        args = d[root][exfile][test]['args']
                        alldatafiles += get_datafiles(args)
                        if 'subtests' in d[root][exfile][test]:
                            for s in d[root][exfile][test]['subtests']:
                                if 'args' in d[root][exfile][test][s]:
                                    args = d[root][exfile][test][s]['args']
                                    alldatafiles += get_datafiles(args)

    # Make unique and sort
    alldatafiles = list(set(alldatafiles))
    alldatafiles.sort()

    if datafilespath:
        action = 'gen_dl_path'
    if action == 'print_datafiles':
        print('\n'.join(alldatafiles))
    if action == 'gen_dl_path':
        ftproot='http://ftp.mcs.anl.gov/pub/petsc/Datafiles/'
        for dfile in alldatafiles:
           fulldfile=os.path.join(datafilespath,dfile)
           if not os.path.exists(fulldfile):
              dl_dir=os.path.dirname(fulldfile)
              if not os.path.isdir(dl_dir):
                  try:
                     os.mkdir(dl_dir)
                  except:
                     os.mkdir(os.path.dirname(os.path.dirname(dl_dir)))
                     os.mkdir(os.path.dirname(dl_dir))
                     os.mkdir(dl_dir)
              dl_dfile=ftproot+dfile
              print('cd '+dl_dir+' && wget '+dl_dfile+'\n')

    return


def main():
    parser = optparse.OptionParser(usage="%prog [options] startdir")
    parser.add_option('-s', '--startdir', dest='startdir',
                      help='Where to start the recursion',
                      default='')
    parser.add_option('-d', '--datafilespath', dest='datafilespath',
                      help='Location of datafilespath for action gen_dl_script',
                      default=None)
    parser.add_option('-a', '--action', dest='action',
                      help='action to take from traversing examples: print_datafiles, gen_dl_script',
                      default=None)
    parser.add_option('-p', '--petsc_dir', dest='petsc_dir',
                      help='Location of petsc_dir',
                      default='')
    options, args = parser.parse_args()

    # Process arguments

    if options.petsc_dir:
        petsc_dir = options.petsc_dir
    else:
        petsc_dir = pdir

    startdir = os.path.join(petsc_dir, 'src')
    if len(args) > 1:
        parser.print_usage()
        return
    elif len(args) == 1:
        startdir = args[0]
    else:
        if not options.startdir == '':
            startdir = options.startdir

    # Do actual work

    action = 'print_datafiles' if not options.action else options.action
    walktree(startdir, action, datafilespath=options.datafilespath)


if __name__ == "__main__":
        main()
