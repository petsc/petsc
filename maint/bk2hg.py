#!/usr/bin/env python
#
# Usage bk2hg local-bk-repo local-hg-repo
#
#  options:
#    -i : create/initialize a new hg repo
#  example:
#    bk2hg.py /sandbox/petsc/petsc-dev-bk /sandbox/petsc/petsc-dev-hg
#
# local-bk-repo is a valid bk repository
# local-hg-repo is a new location

import sys
import os
import time

def main():
  createhgrepo=0
  if '-i' in sys.argv:
    sys.argv.remove('-i')
    createhgrepo=1

  arg_len = len(sys.argv)
  if arg_len != 3:
    print 'Error Insufficient arguments.'
    print 'Usage:', sys.argv[0], '[-i] local-bk-repo local-hg-repo'
    print 'Example:'
    print '  bk2hg.py /sandbox/petsc/petsc-dev-bk /sandbox/petsc/petsc-dev-hg'
    sys.exit()
  bk_repo = sys.argv[1]
  hg_repo= sys.argv[2]

  # get absolute paths - this way - os.chdir() works
  bk_repo = os.path.realpath(bk_repo)
  hg_repo = os.path.realpath(hg_repo)

  # verify if bkdir exists
  if not os.path.exists(bk_repo):
    print 'Error! specified path does not exist: ' + bk_repo
    sys.exit()

  # if createhgrepo - then create & initialize the repo [if dir exists]
  # otherwise - make sure the dir exists [if not error]
  if os.path.exists(hg_repo):
    if createhgrepo:
      print 'Warning! ignoring option -i as specified hgrepo exists: ' + hg_repo
  else:
    if createhgrepo:
      print 'Creating hgrepo: ' + hg_repo
      os.mkdir(hg_repo)
      os.chdir(hg_repo)
      os.system("hg init")
    else:
      print 'Error! specified path does not exist: ' + hg_repo
      print 'If you need to create a new hgrepo, use option: -i'
      sys.exit()

  # verify the specified dirs are valid repositories
  os.chdir(bk_repo)
  if os.system("bk changes -r+ > /dev/null 2>&1"):
    print 'Error! specified path is not a bk repository: ' + bk_repo
    sys.exit()
    
  os.chdir(hg_repo)
  if os.system("hg tip > /dev/null 2>&1"):
    print 'Error! specified path is not a hg repository: ' + hg_repo
    sys.exit()

  # now get the latest bk cset number
  os.chdir(bk_repo)
  fd=os.popen('bk changes -k -r+')
  bk_cset_max=fd.read().strip()
  fd.close()

  # similarly get the latest hg cset number
  os.chdir(hg_repo)
  fd=os.popen('hg tip -v')
  buf=fd.read()
  fd.close()
  if buf == '':
    print 'ErrorNo new changesets Quitting! Last commit:', bk_cset_min
    sys.exit()
  elif buf.splitlines()[0].find(' -1:') >=0:
    # a new repository has -1 changeset number.
    bk_cset_min = '1.0'
  else:
    bk_cset_min = buf.splitlines()[7].strip()

  if bk_cset_min == bk_cset_max:
    print 'No new changesets Quitting! Last commit:', bk_cset_min
    sys.exit()

  log_file=os.path.join(bk_repo,'hg_log.tmp')
  if os.path.isfile(log_file):
    os.remove(log_file)

  # find the bk-changesets that need to be exported to hg
  # using -end:KEY avoids duplicate listing of TAGS [causes too much grief]
  os.chdir(bk_repo)
  fd=os.popen('bk changes -end:KEY: -f -r'+'"'+bk_cset_min+'".."'+bk_cset_max+'"')
  buf=fd.read()
  fd.close()
  revs=buf.splitlines()

  if revs == []:
    print 'No new revisions found in bk. Perhaps some internal error!'
    print 'bk_cset_min:',bk_cset_min
    print 'bk_cset_max:',bk_cset_max
    sys.exit()
            
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
    revn = fd.read().splitlines()[0]
    fd.close()
    # Don't know how to handle branch changesets
    if len(revn.split('.')) > 2:
      print 'Ignoring changeset  : '+revn
      continue
    
    print 'Processing changeset: '+revn
    # get revi
    revi = int(revn.split('.')[1])
    # get username
    fd=os.popen('bk changes -and:USER:@:HOST: -r'+revq)
    auth_email=fd.read().splitlines()[0].strip()
    fd.close()
    auth_email=auth_email.replace('.(none)','')

    fd=os.popen('bk changes -and:TIME_T: -r'+revq)
    gtime = fd.read().splitlines()[0].strip()
    timestr = '"' + str(gtime) + ' ' +str(time.timezone) + '"'
    
    #get comment string
    fd=os.popen('bk changes -r'+revq+' | grep -v ^ChangeSet@')
    buf=fd.read()
    fd.close()
    msg = 'bk-changeset-'+revn + '\n' + rev + '\n'+ buf.strip() + '\n'
    fd=open(log_file,'w')
    fd.write(msg)
    fd.close()

    os.chdir(hg_repo)
    # verify hg again
    if os.system("hg tip > /dev/null 2>&1"):
      print 'Error! specified path is not a hg repository: ' + hg_repo
      sys.exit()

    # Now remove the old files - and export the new modified files
    os.system('ls -a | grep -v .hg | xargs rm -rf >/dev/null 2>&1')
    os.system('bk export -r'+revq+' ' + bk_repo + ' ' + hg_repo)
    # somehow add in --date as well
    if os.system('hg commit --addremove --user ' + auth_email + ' --date ' + timestr + ' --logfile '+log_file + ' --exclude '+log_file):
      print 'Exiting due to the previous error!'
      sys.exit()
    os.unlink(log_file)
    
  return 0

# The classes in this file can also
# be used in other python-programs by using 'import'
if __name__ ==  '__main__':
  main()

        
