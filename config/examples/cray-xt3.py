#!/usr/bin/env python

# This script will generate configuration files for a Cray XT3/4 system 
# running either Catamount or Compute Node Linux (CNL).  With either OS, 
# some manual modifications must be made to petscconf.h.
#
# If running CNL, you must add PETSC_HAVE_F90_2PTR_ARG to petscconf.h 
# after running configure.
#
# If running Catamount:
# After running configure, remove the following flags from petscconf.h
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
###### On Cray XT4 the following additional flags need removal ########
#
# PETSC_USE_SOCKET_VIEWER
# PETSC_HAVE_GETPWUID
#
# And add the following
#
# PETSC_HAVE_LSEEK
# PETSC_HAVE_GETCWD
# PETSC_HAVE_F90_2PTR_ARG

# Configure script for building PETSc on the Cray XT3/4 ("Red Storm").
configure_options = [
  '--with-batch=1',
  '--with-mpi-shared=0',
  '--with-memcmp-ok',
  '--sizeof_char=1',
  '--sizeof_void_p=8',
  '--sizeof_short=2',
  '--sizeof_int=4',
  '--sizeof_long=8',
  '--sizeof_size_t=8',
  '--sizeof_long_long=8',
  '--sizeof_float=4',
  '--sizeof_double=8',
  '--bits_per_byte=8',
  '--sizeof_MPI_Comm=4',
  '--sizeof_MPI_Fint=4',
  '--have-mpi-long-double=1',
  '--sizeof_size_t=8',  
  '--with-debugging=0',
  'COPTFLAGS=-fastsse -O3 -Munroll=c:4 -tp k8-64',
  'FOPTFLAGS=-fastsse -O3 -Munroll=c:4 -tp k8-64',
  '--with-x=0',
  '--with-mpi-dir=/opt/xt-mpt/default/mpich2-64/P2'
  ]

if __name__ == '__main__':
  import os
  import sys
  sys.path.insert(0, os.path.abspath(os.path.join('config')))
  import configure
  configure.petsc_configure(configure_options)
