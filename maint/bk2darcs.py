#!/usr/bin/env python
#
# Usage bk2darcs bk-repo darcs-repo
#
# bk-repo is a valid bk repository
# darcs-repo is a new location

import sys
import os

def main():
  arg_len = len(sys.argv)
  if arg_len != 3:
    print 'Error Insufficient arguments.'
    print 'Usage:', sys.argv[0], 'bk-repo darcs-repo'
    sys.exit()
  bk_repo = sys.argv[1]
  darcs_repo= sys.argv[2]

  # verify the dirs exist 
  if not os.path.exists(bk_repo):
    print 'Error! specified path does not exist: ' + bk_repo
    sys.exit()
  if not os.path.exists(darcs_repo):
    print 'Error! specified path does not exist: ' + darcs_repo
    sys.exit()

  # get absolute paths - this way - os.chdir() works
  bk_repo = os.path.abspath(bk_repo)
  darcs_repo = os.path.abspath(darcs_repo)
  
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
  fd=os.popen('bk changes -r+')
  buf=fd.read()
  fd.close()
  bk_cset_max = ((buf.split(',')[0]).split('.')[1]).strip()
  if not bk_cset_max.isdigit():
    print 'Error! bk_cset_max not a digit!',bk_cset_max
    sys.exit()

  # similarly get the latest darcs cset number
  os.chdir(darcs_repo)
  fd=os.popen('darcs changes --last=1')
  buf=fd.read()
  fd.close()
  if buf == '':
    bk_cset_min = '0'
  else:
    bk_cset_min = buf.split('bk-import-')[1].splitlines()[0].split('.')[1].strip()

  if not bk_cset_min.isdigit():
    print 'Error! bk_cset_min not a digit!',bk_cset_min
    sys.exit()

  if eval(bk_cset_min) > eval(bk_cset_max):
    print 'Error! min > max - repositories are not related?'
    sys.exit()
    
  print 'Processing changesets from: ' + bk_cset_min + ' to: ' + bk_cset_max

  log_file=os.path.join(bk_repo,'darcs_log.tmp')
  if os.path.isfile(log_file):
    os.remove(log_file)
  for i in range(eval(bk_cset_min)+1,eval(bk_cset_max)+1):
    rev = '1.'+str(i)
    print 'processing changeset-'+rev
    os.chdir(bk_repo)
    # get username
    fd=os.popen('bk changes -k -r'+rev)
    buf=fd.read()
    fd.close()
    auth_email=buf.split('|')[0]
    # fix things like knepley@khan.(none)
    auth_email=auth_email.replace('.(none)','')
    
    #get comment string
    fd=os.popen('bk changes -r'+rev+' | grep -v ^ChangeSet@')
    buf=fd.read()
    fd.close()
    msg = 'bk-import-1.'+str(i) + '\n' + buf.strip() + '\n'
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
    os.system('bk export -r1.'+str(i) +' ' + bk_repo + ' ' + darcs_repo)
    os.system('darcs record --test -l -a -A ' + auth_email + ' --delete-logfile --logfile='+log_file)
    # optimize/checkpoint every 250 patches
    if  i%250 == 0 :
      os.system('darcs tag -A snapshot@petsc snapshot-'+rev)
      os.system('darcs optimize --checkpoint -t snapshot-'+rev)
  return 0

# The classes in this file can also
# be used in other python-programs by using 'import'
if __name__ ==  '__main__':
  main()

        
