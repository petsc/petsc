#!/usr/bin/env python
import glob
import sys
import os
import optparse
import re
import inspect

thisfile = os.path.abspath(inspect.getfile(inspect.currentframe()))
pdir = os.path.dirname(os.path.dirname(os.path.dirname(thisfile)))
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


def walktree(top, action):
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

    alldatafiles = list(set(alldatafiles))
    alldatafiles.sort()

    if action == 'print_datafiles':
        print('\n'.join(alldatafiles))

    return


def main():
    parser = optparse.OptionParser(usage="%prog [options] startdir")
    parser.add_option('-s', '--startdir', dest='startdir',
                      help='Where to start the recursion',
                      default='')
    parser.add_option('-a', '--action', dest='action',
                      help='action to take from traversing examples',
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
    walktree(startdir, action)


if __name__ == "__main__":
        main()
