#ifndef __ZOPE_H__
#define __ZOPE_H__

#define PETSC_DLL

#include "petsc.h"
#include "petscerror.h"
#include <stdio.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/errno.h>

extern int PETSC_SOCKFD;
extern int PETSC_LISTENFD;
extern int PETSC_LISTEN_CHECK;

PetscErrorCode PETSC_DLLEXPORT PetscOpenSocket(char * hostname, int portnum, int *clientfd);
PetscErrorCode PETSC_DLLEXPORT Recv(int fd, void *buf, size_t len, int flags, unsigned int *size);
PetscErrorCode PETSC_DLLEXPORT Write(int fd, void *buf, size_t len, unsigned int *size);
PetscErrorCode PETSC_DLLEXPORT PetscSocketListen(char * hostname, int portnum, int *listenfd);

#endif
