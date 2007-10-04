#ifndef __ZOPE_H__
#define __ZOPE_H__

#define PETSC_DLL

#include "petsc.h"
#include "petscerror.h"
#include <stdio.h>
#include <sys/types.h>
#ifdef PETSC_HAVE_SYS_SOCKET_H
  #include <sys/socket.h>
#endif
#include <netinet/in.h>
#include <netdb.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/errno.h>

PetscErrorCode PETSC_DLLEXPORT PetscOpenSocket(char * hostname, int portnum, int *clientfd);
PetscErrorCode PETSC_DLLEXPORT PetscFdRecv(int fd, void *buf, size_t len, int flags, unsigned int *size);
PetscErrorCode PETSC_DLLEXPORT PetscFdWrite(int fd, void *buf, size_t len, unsigned int *size);
PetscErrorCode PETSC_DLLEXPORT PetscSocketListen(char * hostname, int portnum, int *listenfd);

#endif
