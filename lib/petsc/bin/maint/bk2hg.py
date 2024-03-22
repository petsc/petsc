#!/usr/bin/env python3
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

from __future__ import print_function
import sys
import os
import time
import shutil
import tempfile

# Get the key for the tip revision - so the bk <-> hg key mappings can be stored
def hg_get_tip_key(hg_repo):
  cur_path=os.path.abspath(os.path.curdir)
  os.chdir(hg_repo)
  fd=os.popen('hg tip -v')
  buf = fd.read()
  fd.close()
  os.chdir(cur_path)
  return buf.splitlines()[0].split(':')[2]

def main():
  createhgrepo=0
  if '-i' in sys.argv:
    sys.argv.remove('-i')
    createhgrepo=1

  arg_len = len(sys.argv)
  if arg_len != 3:
    print('Error Insufficient arguments.')
    print('Usage:', sys.argv[0], '[-i] local-bk-repo local-hg-repo')
    print('Example:')
    print('  bk2hg.py /sandbox/petsc/petsc-dev-bk /sandbox/petsc/petsc-dev-hg')
    sys.exit()
  bk_repo = sys.argv[1]
  hg_repo= sys.argv[2]

  # get absolute paths - this way - os.chdir() works
  bk_repo = os.path.realpath(bk_repo)
  hg_repo = os.path.realpath(hg_repo)
  hg_repo_child = hg_repo+'-child'
  hg_repo_child_merge = hg_repo+'-child-merge'
  hg_rev = {}
  os.environ['HGMERGE'] = '/bin/true'
  # create a file with 'd' for delete - required for hg merge
  fd,tmp_file = tempfile.mkstemp()
  err=os.write(fd,'d\nd\nd\nd\nd\nd\nd\nd\nd\nd\nd\nd\nd\nd\nd\nd\nd\nd\nd\nd\nd\nd\nd\nd\n')
  os.close(fd)

  # verify if bkdir exists
  if not os.path.exists(bk_repo):
    print('Error! specified path does not exist: ' + bk_repo)
    sys.exit()

  # if createhgrepo - then create & initialize the repo [if dir exists]
  # otherwise - make sure the dir exists [if not error]
  if os.path.exists(hg_repo):
    if createhgrepo:
      print('Warning! ignoring option -i as specified hgrepo exists: ' + hg_repo)
      sys.stdout.flush()
  else:
    if createhgrepo:
      print('Creating hgrepo: ' + hg_repo)
      sys.stdout.flush()
      os.mkdir(hg_repo)
      os.chdir(hg_repo)
      if os.system("hg init"):
        print('Error during hg init!')
        sys.exit()
      hg_rev['1.0'] = hg_get_tip_key(hg_repo)
    else:
      print('Error! specified path does not exist: ' + hg_repo)
      print('If you need to create a new hgrepo, use option: -i')
      sys.exit()

  # verify the specified dirs are valid repositories
  os.chdir(bk_repo)
  if os.system("bk changes -r+ > /dev/null 2>&1"):
    print('Error! specified path is not a bk repository: ' + bk_repo)
    sys.exit()

  os.chdir(hg_repo)
  if os.system("hg tip > /dev/null 2>&1"):
    print('Error! specified path is not a hg repository: ' + hg_repo)
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
    print('ErrorNo new changesets Quitting! Last commit:', bk_cset_min)
    sys.exit()
  elif buf.splitlines()[0].find(' -1:') >=0:
    # a new repository has -1 changeset number.
    bk_cset_min = '1.0'
  else:
    bk_cset_min =''
    for line in buf.splitlines():
      if line.find('|ChangeSet|') >=0:
        bk_cset_min = line.strip()
        break
    if bk_cset_min == '':
      print('Error! bk changeset tag not found at the tip of hg repo!')
      sys.exit()

  if bk_cset_min == bk_cset_max:
    print('No new changesets Quitting! Last commit:', bk_cset_min)
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
    print('No new revisions found in bk. Perhaps some internal error!')
    print('bk_cset_min:',bk_cset_min)
    print('bk_cset_max:',bk_cset_max)
    sys.exit()

  #now process each revision [ignore the first]
  for rev in revs:
    # rev  - basic string
    # revq - rev with quotes  [usable with -r]
    # revn - rev number [ 1.234.4 etc..]
    os.chdir(bk_repo)
    revq='"'+rev+'"'
    # get the rev-number
    fd=os.popen('bk changes -and:I: -r'+revq)
    revn = fd.read().splitlines()[0]
    fd.close()

    print('Processing changeset: '+revn)
    sys.stdout.flush()
    # get username
    fd=os.popen('bk changes -and:USER:@:HOST: -r'+revq)
    auth_email=fd.read().splitlines()[0].strip()
    fd.close()
    auth_email=auth_email.replace('.(none)','')

    fd=os.popen('bk changes -and:TIME_T: -r'+revq)
    gtime = fd.read().splitlines()[0].strip()
    timestr = '"' + str(gtime) + ' ' +str(time.timezone) + '"'

    #get comment string
    fd=os.popen('bk changes -v -r'+revq)
    buf=fd.read()
    fd.close()
    msg = 'bk-changeset-'+revn + '\n' + rev + '\n'+ buf.strip() + '\n'
    fd=open(log_file,'w')
    fd.write(msg)
    fd.close()

    # prev parent revision
    # mrev merge revision
    # crev current revision
    if revn == '1.0':
      mrev = None
      prev = None
      crev = '1.0'
    else:
      fd=os.popen('bk rset -r'+revq)
      buf =fd.read()
      fd.close()
      tmpstr = buf.splitlines()[0].strip().split('|')[1]
      if len(tmpstr.split('+')) == 2:
        mrev,pstr= tmpstr.split('+')
      else:
        mrev = None
        pstr = tmpstr
      prev,crev = pstr.split('..')
      print(mrev,prev,crev,revn)
      sys.stdout.flush()
      # crev should be same as revn
      if crev != revn:
        print('Error! crev and revn do not match!', crev, revn)
        sys.exit()

    # Now start the converion process
    if prev:
      hg_prev_val = hg_rev[prev]
    else:
      hg_prev_val = '000000000000'
    if mrev:
      hg_mrev_val = hg_rev[mrev]
    else:
      hg_mrev_val = '000000000000'

    # merge if necessary
    if mrev != None:
      if os.system('hg clone -r'+str(hg_prev_val) +' '+ hg_repo +' '+ hg_repo_child + '> /dev/null 2>&1'):
        print('Error during hg clone!')
        sys.exit()

      if os.system('hg clone -r'+str(hg_mrev_val) +' '+ hg_repo +' '+ hg_repo_child_merge + '> /dev/null 2>&1'):
        print('Error during hg clone!')
        sys.exit()

      # now merge these 2 branches
      os.chdir(hg_repo_child)
      if os.system('hg pull -f '+hg_repo_child_merge):
        print('Error during hg pull!')
        sys.exit()

      # look for the tip - and tag the non-tip version [among prev and mrev]
      tip_key = hg_get_tip_key(hg_repo_child)
      if tip_key == hg_rev[prev]:
        tagrev = mrev
      elif tip_key == hg_rev[mrev]:
        tagrev = prev
      else:
        print('Error tip does not match either prev or mrev!')
        sys.exit()

      if os.system('hg tag -l -r'+str(hg_rev[tagrev]) +' '+ str(tagrev)):
        print('Error during hg tag prev')
        sys.exit()

      # now attempt to merge
      os.system('ls -a | grep -v .hg | xargs rm -rf >/dev/null 2>&1')
      if os.system('hg co -C -f'):
        print('Error during checkout')
        sys.exit()
      if os.system('hg merge -f -b'+str(tagrev) + '< '+ tmp_file):
        print('******* Error during merge! Ignoring*********')
        sys.stdout.flush()
      # remove the temp clone
      shutil.rmtree(hg_repo_child_merge)
    else:
      # clone the prev version[parent to the current change]
      if os.system('hg clone -r'+str(hg_prev_val) +' '+ hg_repo +' '+ hg_repo_child + '> /dev/null 2>&1'):
        print('Error during hg clone!')
        sys.exit()

    # Now remove the old files - and export the new modified files
    os.chdir(hg_repo_child)
    os.system('ls -a | grep -v .hg | xargs rm -rf >/dev/null 2>&1')
    if os.system('bk export -r'+revq+' ' + bk_repo + ' ' + hg_repo_child):
      print('Error during bk export!')
      sys.exit()
    # somehow add in --date as well
    if os.system('hg commit --addremove --user ' + auth_email + ' --date ' + timestr + ' --logfile '+log_file + ' --exclude '+log_file):
      print('Exiting due to the previous error!')
      sys.exit()
    os.unlink(log_file)
    # now extract the changeset id - and store
    hg_rev[revn] = hg_get_tip_key(hg_repo_child)
    if os.system('hg push -f ' +hg_repo):
      print('********** Push returned error code! Ignoring! *************')
      sys.stdout.flush()
    # remove the temp clone
    shutil.rmtree(hg_repo_child)
  os.unlink(tmp_file)
  return 0

# The classes in this file can also
# be used in other python-programs by using 'import'
if __name__ ==  '__main__':
  main()
  print('******** Done Conversion ***************')
