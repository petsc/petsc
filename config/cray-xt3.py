#!/usr/bin/env python

# After running configure - remove the following flags from petscconf.h
#
# PETSC_HAVE_SYS_PROCFS_H
# PETSC_HAVE_DLFCN_H
# PETSC_HAVE_SYS_SOCKET_H
# PETSC_HAVE_SYS_UTSNAME_H
# PETSC_HAVE_PWD_H
# PETSC_HAVE_GETWD
# PETSC_HAVE_UNAME
# PETSC_HAVE_GETHOSTNAME
# PETSC_HAVE_GETDOMAINNAME
# PETSC_HAVE_NETINET_IN_H
# PETSC_HAVE_NETDB_H
#
# And add the following
#
# PETSC_HAVE_LSEEK
# PETSC_HAVE_GETCWD
#

configure_options = [
  '--with-batch=1',
  '--with-memcmp-ok',
  '--sizeof_char=1',
  '--sizeof_void_p=8',
  '--sizeof_short=2',
  '--sizeof_int=4',
  '--sizeof_long=8',
  '--sizeof_long_long=8',
  '--sizeof_float=4',
  '--sizeof_double=8',
  '--bits_per_byte=9',
  '--sizeof_MPI_Comm=4',
  '--sizeof_MPI_Fint=4',
  
  '--with-debugging=0',
  'CFLAGS=-fastsse -O3 -Munroll=c:4 -tp k8-64',
  'FFLAGS=-fastsse -O3 -Munroll=c:4 -tp k8-64',
  '--with-x=0',
  '--with-mpi-dir=/opt/xt-mpt/1.3.15/mpich2-64/P2'
  ]
if __name__ == '__main__':
  import os
  import sys
  sys.path.insert(0, os.path.abspath(os.path.join('config')))
  import configure
  configure.petsc_configure(configure_options)

    
