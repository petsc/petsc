#!/usr/bin/env python
#
# Usage bk2darcs bk-repo darcs-repo
#
#  options:
#    -i : create/initialize a new darcs repo
#
# bk-repo is a valid bk repository
# darcs-repo is a new location

import sys
import os

def main():
  createdarcsrepo=0
  if '-i' in sys.argv:
    sys.argv.remove('-i')
    createdarcsrepo=1

  arg_len = len(sys.argv)
  if arg_len != 3:
    print 'Error Insufficient arguments.'
    print 'Usage:', sys.argv[0], '[-i] bk-repo darcs-repo'
    sys.exit()
  bk_repo = sys.argv[1]
  darcs_repo= sys.argv[2]

  # get absolute paths - this way - os.chdir() works
  bk_repo = os.path.abspath(bk_repo)
  darcs_repo = os.path.abspath(darcs_repo)

  # verify if bkdir exists
  if not os.path.exists(bk_repo):
    print 'Error! specified path does not exist: ' + bk_repo
    sys.exit()

  # if createdarcsrepo - then create & initialize the repo [if dir exists]
  # otherwise - make sure the dir exists [if not error]
  if os.path.exists(darcs_repo):
    if createdarcsrepo:
      print 'Warning! ignoring option -i as specified darcsrepo exists: ' + darcs_repo
  else:
    if createdarcsrepo:
      print 'Creating darcsrepo: ' + darcs_repo
      os.mkdir(darcs_repo)
      os.chdir(darcs_repo)
      os.system("darcs initialize")
    else:
      print 'Error! specified path does not exist: ' + darcs_repo
      print 'If you need to create a new darcs-repo, use option: -i'
      sys.exit()

  # verify the specified dirs are valid repositories
  os.chdir(bk_repo)
  if os.system("bk changes -r+ > /dev/null 2>&1"):
    print 'Error! specified path is not a bk repository: ' + bk_repo
    sys.exit()
    
  os.chdir(darcs_repo)
  if os.system("darcs changes --last=1 > /dev/null 2>&1"):
    print 'Error! specified path is not a darcs repository: ' + darcs_repo
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
    bk_cset_min = buf.splitlines()[1].strip()[2:]

  if bk_cset_min == bk_cset_max:
    print 'No new changesets Quitting! Last commit:', bk_cset_min
    sys.exit()

  log_file=os.path.join(bk_repo,'darcs_log.tmp')
  if os.path.isfile(log_file):
    os.remove(log_file)

  # find the bk-changesets that need to be exported to darcs
  os.chdir(bk_repo)
  fd=os.popen('bk changes -k -f -r'+'"'+bk_cset_min+'".."'+bk_cset_max+'"')
  buf=fd.read()
  fd.close()
  revs=buf.splitlines()


  #now process each revision [ignore the first]
  for rev in revs:
    # rev  - basic string
    # revq - rev with quotes  [useable with -r]
    # revn - rev number [ 1.234.4 etc..]
    # revi - just the 2nd number in rev
    os.chdir(bk_repo)
    revq='"'+rev+'"'
    # get the rev-number
    fd=os.popen('bk changes -and:I: -r'+revq)
    revn = fd.read()
    fd.close()
    print 'processing changeset: '+revn
    # get revi
    fd=os.popen('bk changes -and:L: -r'+revq)
    revi = int(fd.read().strip())
    fd.close()
    # get username
    fd=os.popen('bk changes -and:USER:@:HOST: -r'+revq)
    auth_email=fd.read().strip()
    fd.close()
    auth_email=auth_email.replace('.(none)','')
    
    #get comment string
    fd=os.popen('bk changes -r'+revq+' | grep -v ^ChangeSet@')
    buf=fd.read()
    fd.close()
    msg = rev+ '\n' + buf.strip() + '\n'
    fd=open(log_file,'w')
    fd.write(msg)
    fd.close()

    os.chdir(darcs_repo)
    # verify darcs again
    if os.system("darcs changes --last=1 > /dev/null 2>&1"):
      print 'Error! specified path is not a darcs repository: ' + darcs_repo
      sys.exit()

    # Now remove the old files - and export the new modified files
    os.system('ls -a | grep -v _darcs | xargs rm -rf >/dev/null 2>&1')
    os.system('bk export -r'+revq+' ' + bk_repo + ' ' + darcs_repo)
    os.system('darcs record --test -l -a -A ' + auth_email + ' --delete-logfile --logfile='+log_file)
    # optimize/checkpoint every 250 patches
    if revi%250 == 0 :
      os.system('darcs tag -A snapshot@petsc snapshot-'+revn)
      os.system('darcs optimize --checkpoint -t snapshot-'+revn)
  return 0

# The classes in this file can also
# be used in other python-programs by using 'import'
if __name__ ==  '__main__':
  main()

        
