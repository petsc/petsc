#!/usr/bin/env python3
"""
# Created: Wed Feb 23 17:32:36 2022 (-0600)
# @author: jacobfaibussowitsch
"""
import sys
if sys.version_info < (3,5):
  raise RuntimeError('requires python 3.5')
import os
import re
import subprocess
import pathlib
import collections
import itertools

class Replace:
  __slots__ = 'verbose','special'

  def __init__(self,verbose,special=None):
    """
    verbose: (bool)                verbosity level
    special: (set-like of strings) list of functions/symbols that will remain untouched
    """
    self.verbose = bool(verbose)
    if special is None:
      special = {
        'PetscOptionsBegin','PetscObjectOptionsBegin','PetscOptionsEnd',
        'MatPreallocateInitialize','MatPreallocateFinalize',
        'PetscDrawCollectiveBegin','PetscDrawCollectiveEnd'
      }
    self.special = special
    return

  def __call__(self,match):
    """
    match: a match object from re.match containing 2 or 3 groups
    """
    if any(map(match.group(0).__contains__,self.special)):
      if self.verbose:
        print('SKIPPED',match.group(0))
      return match.group(0)
    ierr,chkerr = match.group(1),match.group(2)
    chkerr_suff = chkerr.replace('CHKERR','')
    replace     = 'PetscCall'
    if chkerr_suff == 'Q':
      pass
    elif chkerr_suff == 'V':
      replace += 'Void'
    elif chkerr_suff in {'ABORT','CONTINUE'}:
      replace += chkerr_suff.title()
      if chkerr_suff == 'ABORT':
        comm = match.group(3).split(',')[0]
        ierr = ','.join((comm,ierr))
    elif chkerr_suff == 'XX':
      replace += 'Throw'
    else:
      replace += chkerr_suff
    return '{}({})'.format(replace,ierr)

class Processor:
  __slots__ = (
    'chkerr_re','pinit_re','pfinal_re','retierr_re','cleanup_re','edecl_re','euses_re',
    'addcount','delcount','verbose','dry_run','del_empty_last_line','replace_chkerrs'
  )

  def __init__(self,verbose,dry_run,del_empty_last_line):
    """
    verbose:             (int)  verbosity level
    dry_run:             (bool) is this a dry-run
    del_empty_last_line: (bool) should we try and delete empty last (double) lines in the file
    """
    self.chkerr_re  = re.compile(r'(?:\w+ +)?\w*(?:err|stat|ccer)\w* *= *(.*?);(CHKERR.*)\((.*?)\)')
    self.pinit_re   = re.compile(r'(?:\w+ +)?ierr *= *(PetscInitialize.*);\s*if\s+\(ierr\)\s*return\s+ierr.*')
    self.pfinal_re  = re.compile(r'(?:\w+ +)?ierr *= *(PetscFinalize.*);.*')
    self.retierr_re = re.compile(r'(?:\w+ +)?(return)\s+ierr;.*')
    self.cleanup_re = re.compile(r'{\s*(PetscCall[^;]*;)\s*}')
    self.edecl_re   = re.compile(r'\s*PetscErrorCode\s+ierr;.*')
    self.euses_re   = re.compile(r'.*ierr\s*=\s*.*')
    self.addcount   = 0
    self.delcount   = 0
    self.verbose    = verbose
    self.dry_run    = dry_run

    self.del_empty_last_line = del_empty_last_line
    self.replace_chkerrs     = Replace(verbose > 2)
    return

  def __call__(self,path):
    new_lines,changes    = [],[]
    last                 = collections.deque(('',''),maxlen=2)
    error_code_decls     = []
    error_code_uses      = []
    petsc_finalize_found = False
    delete_set           = set()
    is_fortran_binding   = any(p.startswith('ftn-') for p in path.parts)

    for lineno,line in enumerate(path.read_text().splitlines()):
      if line.lstrip().startswith('PetscFunctionBegin') and last[0] == '' and last[1] == '{':
        # found
        # {
        #   <should delete this empty line>
        #   PetscFunctionBegin;
        delete_set.add(lineno-1)
        changes.append((lineno,last[0],None))
      # check for trivial unused variable
      if self.euses_re.match(line):
        error_code_uses.append((line,lineno))
      if self.edecl_re.match(line):
        error_code_decls.append((line,lineno))
      # check for PetscInitialize() to wrap
      repl = self.pinit_re.sub(r'PetscCall(\1);',line)
      if repl == line:
        if is_fortran_binding:
          petsc_finalize_found = False
        else:
          repl = self.pfinal_re.sub(r'PetscCall(\1);',line)
        if repl == line:
          repl = self.chkerr_re.sub(self.replace_chkerrs,line)
          if petsc_finalize_found and repl == line:
            repl = self.retierr_re.sub(r'\1 0;',line)
          petsc_finalize_found = False
        else:
          petsc_finalize_found = True
      if repl != line:
        repl = self.cleanup_re.sub(r'\1',repl)
        self.add()
        self.delete()
        changes.append((lineno,line,repl))
      new_lines.append(repl)
      last.appendleft(line.strip())

    self.delete_unused_error_code_decls(error_code_decls,error_code_uses,new_lines,delete_set,changes)

    if len(new_lines) and new_lines[-1] == '':
      if self.del_empty_last_line:
        self.delete()
        changes.append((len(new_lines),new_lines[-1],None))
      else:
        new_lines[-1] = '\n'

    self.delete(len(delete_set))
    if delete_set:
      new_lines = [l for i,l in enumerate(new_lines) if i not in delete_set]

    return new_lines,changes,delete_set


  def delete_unused_error_code_decls(self,error_code_decls,error_code_uses,new_lines,delete_set,changes):
    def pairwise(iterable,default=None):
      "s -> (s0,s1,..s(n-1)), (s1,s2,.., sn), (s2, s3,..,s(n+1)), ..."
      n      = 2
      iters  = iter(iterable)
      result = tuple(itertools.islice(iters,n))
      if len(result) == n:
        yield result
      for elem in iters:
        result = result[1:]+(elem,)
        yield result
      if default is not None:
        yield result[-1],default


    if not len(error_code_decls):
      return # nothing to do

    # see if we can find consecutive PetscErrorCode ierr; without uses, if so, delete
    # them
    default_entry = (None,len(new_lines))
    for (cur_line,cur_lineno),(_,next_lineno) in pairwise(error_code_decls,default=default_entry):
      line_range = range(cur_lineno,next_lineno)
      if not any(ln in line_range for _,ln in error_code_uses):
        # the ierr is unused
        assert new_lines[cur_lineno] == cur_line # don't want to delete the wrong line
        delete_set.add(cur_lineno)
        if self.dry_run:
          change = (cur_lineno,cur_line,None)
          added  = False
          for i,(cln,_,_) in enumerate(changes):
            if cln > cur_lineno:
              changes.insert(i,change)
              added = True
              break
          if not added:
            changes.append(change)
    return

  def add(self,n=1):
    self.addcount += n
    return

  def delete(self,n=1):
    self.delcount += n
    return

  def summary(self):
    if self.verbose:
      print(self.delcount,'deletions and',self.addcount,'insertions')
    return


def path_resolve_strict(path):
  path = pathlib.Path(path)
  return path.resolve() if sys.version_info < (3,6) else path.resolve(strict=True)

def subprocess_run(*args,**kwargs):
  if sys.version_info < (3,7):
    kwargs.setdefault('stdout',subprocess.PIPE)
    kwargs.setdefault('stderr',subprocess.PIPE)
  else:
    kwargs.setdefault('capture_output',True)
  return subprocess.run(args,**kwargs)

def get_paths_list(start_path,search_tool,force):
  if start_path.is_dir():
    if search_tool == 'rg':
      extra_flags = ['-T','fortran','--no-stats','-j','5']
    else: # grep
      extra_flags = ['-E','-r']

    if force:
      import glob
      file_list = glob.iglob(str(start_path/'**'),recursive=True)
    else:
      ret = subprocess_run(search_tool,*extra_flags,'-l','CHKERR',str(start_path))
      try:
        ret.check_returncode()
      except subprocess.CalledProcessError as cpe:
        print('command:',ret.args)
        print('stdout:\n',ret.stdout.decode())
        print('stderr:\n',ret.stderr.decode())
        raise RuntimeError from cpe
      else:
        file_list = ret.stdout.decode().splitlines()

    filter_file = lambda x: x.endswith(('.c','.cpp','.cxx','.h','.hpp','.C','.H','.inl','.c++','.cu'))
    found_list  = [x.resolve() for x in map(pathlib.Path,filter(filter_file,map(str,file_list)))]
    found_list  = [f for f in found_list if not f.is_dir()]
  else:
    found_list = [start_path]

  assert 'chkerrconvert.py' not in found_list
  return found_list

def main(petsc_dir,search_tool,start_path,dry_run,verbose,force,del_empty_last_line):
  if petsc_dir == '${PETSC_DIR}':
    try:
      petsc_dir = os.environ['PETSC_DIR']
    except KeyError as ke:
      mess = 'Must either define PETSC_DIR as environment variable or pass it via flags'
      raise RuntimeError(mess) from ke
  petsc_dir = path_resolve_strict(petsc_dir)

  if start_path == '${PETSC_DIR}/src':
    start_path = petsc_dir/'src'
  start_path = path_resolve_strict(start_path)

  found_list = get_paths_list(start_path,search_tool,force)
  processor  = Processor(verbose,dry_run,del_empty_last_line)

  for path in found_list:
    # check if this is a fortran binding
    if any(p.startswith('ftn-auto') for p in path.parts):
      if verbose > 2:
        print('skipping',str(path),'because it is an auto-generated fortran binding')
      continue # skip auto-generated files

    if path.stem.endswith('feopencl'):
      # the feopencl has some exceptions
      continue

    new_lines,changes,delete_set = processor(path)

    if dry_run:
      if len(changes):
        print(str(path.relative_to(petsc_dir))+':')
        for lineno,line,repl in changes:
          lineno += 1
          print(f'{lineno}: - {line}')
          if repl is not None:
            print(f'{lineno}: + {repl}')
    elif processor.delcount or processor.addcount:
      output = '\n'.join(new_lines)
      if not output.endswith('\n'):
        output += '\n'
      path.write_text(output)

  processor.summary()
  return


if __name__ == '__main__':
  import argparse
  import signal

  signal.signal(signal.SIGPIPE,signal.SIG_DFL) # allow the output of this script to be piped
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--petsc-dir',default='${PETSC_DIR}',help='petsc directory (containing src, include, config, etc.)')
  parser.add_argument('--search-tool',default='rg',metavar='<executable>',choices=['rg','grep'],help='search tool to use to find files containing matches (rg or grep)')
  parser.add_argument('path',nargs='?',default='${PETSC_DIR}/src',metavar='<path>',help='path to directory base or file')
  parser.add_argument('-n','--dry-run',action='store_true',help='print what the result would be')
  parser.add_argument('-v','--verbose',action='count',default=0,help='verbose')
  parser.add_argument('-f','--force',action='store_true',help='don\'t narrow search using SEARCH TOOL, just replace everything under PATH')
  parser.add_argument('--delete-empty-last-line',action='store_true',help='remove empty lines at the end of the file')

  if len(sys.argv) == 1:
    parser.print_help()
  else:
    args = parser.parse_args()
    main(args.petsc_dir,args.search_tool,args.path,args.dry_run,args.verbose,args.force,args.delete_empty_last_line)
