#ifndef __ZOPE_H__
#define __ZOPE_H__

#include <stdio.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/errno.h>
#include "petsc.h"

int PetscOpenSocket(char * hostname, int portnum);
ssize_t Recv(int fd, void *buf, size_t len, int flags);
ssize_t Write(int fd, void *buf, size_t len);
int PetscSocketListen(char * hostname, int portnum);


#endif
