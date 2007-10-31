#ifndef __ZOPE_H__
#define __ZOPE_H__

#define PETSC_DLL

#include "petsc.h"
#include "petscerror.h"
#include <stdio.h>
#ifdef PETSC_HAVE_STDLIB_H
  #include <stdlib.h>
#endif
#ifdef PETSC_HAVE_SYS_TYPES_H
  #include <sys/types.h>
#endif
#ifdef PETSC_HAVE_SYS_SOCKET_H
  #include <sys/socket.h>
#endif
#ifdef PETSC_HAVE_NETINET_IN_H
  #include <netinet/in.h>
#endif
#ifdef PETSC_HAVE_NETDB_H
  #include <netdb.h>
#endif
#ifdef PETSC_HAVE_UNISTD_H
  #include <unistd.h>
#endif
#ifdef PETSC_HAVE_FCNTL_H
  #include <fcntl.h>
#endif
#include <errno.h>
#if !defined(PETSC_HAVE_SYS_SOCKET_H) /* handle windows/cygwin */
#ifdef PETSC_HAVE_WINSOCK2_H
  #include <Winsock2.h>
#endif
#ifdef PETSC_HAVE_WS2TCPIP_H
  #include <Ws2tcpip.h>
#endif
#endif /* PETSC_HAVE_SYS_SOCKET_H */

PetscErrorCode PETSC_DLLEXPORT PetscOpenSocket(char * hostname, int portnum, int *clientfd);
PetscErrorCode PETSC_DLLEXPORT PetscFdRecv(int fd, void *buf, size_t len, int flags, unsigned int *size);
PetscErrorCode PETSC_DLLEXPORT PetscFdWrite(int fd, void *buf, size_t len, unsigned int *size);
PetscErrorCode PETSC_DLLEXPORT PetscSocketListen(char * hostname, int portnum, int *listenfd);

#endif
