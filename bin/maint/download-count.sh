#!/bin/bash

#copy ftp and http logs and uncompress
cd /sandbox/balay/tmp
rm -f /sandbox/balay/tmp/*
cp /mcs/logs/ftp/ftp.mcs.anl.gov/xferlog.2011* .
cp /mcs/logs/http/ftp.mcs.anl.gov/access_log.2011* .
gunzip -f *.gz

echo "total ftp downloads"
cat xferlog.2011* | egrep \(petsc-dev\|\(petsc-\|petsc-lite-\)\[1-3]\) | grep 'tar.gz ' | grep c\$ | wc -l
cat xferlog.2011* | egrep \(petsc-dev\|\(petsc-\|petsc-lite-\)\[1-3]\) | grep 'tar.gz ' | grep c\$ | tr -s ' '| cut -d ' ' -f 7 > /sandbox/balay/tmp/machines-ftp
echo "unique ftp downloads"
cat /sandbox/balay/tmp/machines-ftp| sort | uniq| wc -l

echo "total http downloads"
cat access_log.2011* | egrep \(petsc-dev\|\(petsc-\|petsc-lite-\)\[1-3]\) | grep 'tar.gz ' | grep ' 200 ' | wc -l
cat access_log.2011* | egrep \(petsc-dev\|\(petsc-\|petsc-lite-\)\[1-3]\) | grep 'tar.gz ' | grep ' 200 ' | cut -d ' ' -f 1 > /sandbox/balay/tmp/machines-http
echo "unique http downloads"
cat /sandbox/balay/tmp/machines-http | sort | uniq | wc -l

echo "unique ftp/http downloads"
cat /sandbox/balay/tmp/machines-ftp /sandbox/balay/tmp/machines-http | sort | uniq | wc -l

# mailing list user count
# <download/save the following URLs as text [with admin authentication]>
# https://lists.mcs.anl.gov/mailman/roster/petsc-dev
# https://lists.mcs.anl.gov/mailman/roster/petsc-announce
# https://lists.mcs.anl.gov/mailman/roster/petsc-users
# cat petsc-announce.html petsc-users.html petsc-dev.html | grep ' \* ' |sort |  uniq |wc -l

