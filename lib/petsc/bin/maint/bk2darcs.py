#!/usr/bin/env python3
#
# Usage bk2darcs local-bk-repo local-darcs-repo
#
#  options:
#    -i : create/initialize a new darcs repo
#  example:
#    bk2darcs.py /sandbox/petsc/petsc-dev-bk /sandbox/petsc/petsc-dev-darcs
#
# local-bk-repo is a valid bk repository
# local-darcs-repo is a new location

from __future__ import print_function
import sys
import os

def main():
  createdarcsrepo=0
  if '-i' in sys.argv:
    sys.argv.remove('-i')
    createdarcsrepo=1

  arg_len = len(sys.argv)
  if arg_len != 3:
    print('Error Insufficient arguments.')
    print('Usage:', sys.argv[0], '[-i] local-bk-repo local-darcs-repo')
    print('Example:')
    print('  bk2darcs.py /sandbox/petsc/petsc-dev-bk /sandbox/petsc/petsc-dev-darcs')
    sys.exit()
  bk_repo = sys.argv[1]
  darcs_repo= sys.argv[2]

  # get absolute paths - this way - os.chdir() works
  bk_repo = os.path.realpath(bk_repo)
  darcs_repo = os.path.realpath(darcs_repo)

  # verify if bkdir exists
  if not os.path.exists(bk_repo):
    print('Error! specified path does not exist: ' + bk_repo)
    sys.exit()

  # if createdarcsrepo - then create & initialize the repo [if dir exists]
  # otherwise - make sure the dir exists [if not error]
  if os.path.exists(darcs_repo):
    if createdarcsrepo:
      print('Warning! ignoring option -i as specified darcsrepo exists: ' + darcs_repo)
  else:
    if createdarcsrepo:
      print('Creating darcsrepo: ' + darcs_repo)
      os.mkdir(darcs_repo)
      os.chdir(darcs_repo)
      os.system("darcs initialize")
    else:
      print('Error! specified path does not exist: ' + darcs_repo)
      print('If you need to create a new darcs-repo, use option: -i')
      sys.exit()

  # verify the specified dirs are valid repositories
  os.chdir(bk_repo)
  if os.system("bk changes -r+ > /dev/null 2>&1"):
    print('Error! specified path is not a bk repository: ' + bk_repo)
    sys.exit()

  os.chdir(darcs_repo)
  if os.system("darcs changes --last=1 > /dev/null 2>&1"):
    print('Error! specified path is not a darcs repository: ' + darcs_repo)
    sys.exit()

  # now get the latest bk cset number
  os.chdir(bk_repo)
  fd=os.popen('bk changes -k -r+')
  bk_cset_max=fd.read().strip()
  fd.close()

  # similarly get the latest darcs cset number
  os.chdir(darcs_repo)
  fd=os.popen('darcs changes --last=1')
  buf=fd.read()
  fd.close()
  if buf == '':
    bk_cset_min = '1.0'
  else:
    bk_cset_min = buf.splitlines()[2].strip()

  if bk_cset_min == bk_cset_max:
    print('No new changesets Quitting! Last commit:', bk_cset_min)
    sys.exit()

  log_file=os.path.join(bk_repo,'darcs_log.tmp')
  if os.path.isfile(log_file):
    os.remove(log_file)

  # find the bk-changesets that need to be exported to darcs
  # using -end:KEY avoids duplicate listing of TAGS [causes too much grief]
  os.chdir(bk_repo)
  fd=os.popen('bk changes -end:KEY: -f -r'+'"'+bk_cset_min+'".."'+bk_cset_max+'"')
  buf=fd.read()
  fd.close()
  revs=buf.splitlines()

  if revs: lastrev = revs[-1]
  #now process each revision [ignore the first]
  for rev in revs:
    # rev  - basic string
    # revq - rev with quotes  [usable with -r]
    # revn - rev number [ 1.234.4 etc..]
    # revi - just the 2nd number in rev
    os.chdir(bk_repo)
    revq='"'+rev+'"'
    # get the rev-number
    fd=os.popen('bk changes -and:I: -r'+revq)
    revn = fd.read().splitlines()[0]
    fd.close()
    # Don't know how to handle branch changesets
    if len(revn.split('.')) > 2:
      print('Ignoring changeset  : '+revn)
      continue

    print('Processing changeset: '+revn)
    # get revi
    revi = int(revn.split('.')[1])
    # get username
    fd=os.popen('bk changes -and:USER:@:HOST: -r'+revq)
    auth_email=fd.read().splitlines()[0].strip()
    fd.close()
    auth_email=auth_email.replace('.(none)','')

    #get comment string
    fd=os.popen('bk changes -r'+revq+' | grep -v ^ChangeSet@')
    buf=fd.read()
    fd.close()
    msg = 'bk-changeset-'+revn + '\n' + rev + '\n'+ buf.strip() + '\n'
    fd=open(log_file,'w')
    fd.write(msg)
    fd.close()

    os.chdir(darcs_repo)
    # verify darcs again
    if os.system("darcs changes --last=1 > /dev/null 2>&1"):
      print('Error! specified path is not a darcs repository: ' + darcs_repo)
      sys.exit()

    # Now remove the old files - and export the new modified files
    os.system('ls -a | grep -v _darcs | xargs rm -rf >/dev/null 2>&1')
    os.system('bk export -r'+revq+' ' + bk_repo + ' ' + darcs_repo)
    os.system('darcs record --test -l -a -A ' + auth_email + ' --delete-logfile --logfile='+log_file + '> /dev/null 2>&1')

    # optimize/checkpoint every 250 patches
    if revi%250 == 0  and revi != 0 and rev != lastrev:
      print('checkpointing/optimizing changeset-'+ revn)
      os.system('darcs tag -A snapshot@petsc snapshot-'+revn +'> /dev/null 2>&1')
      os.system('darcs optimize --checkpoint -t snapshot-'+revn +'> /dev/null 2>&1')
  return 0

# The classes in this file can also
# be used in other python-programs by using 'import'
if __name__ ==  '__main__':
  main()


