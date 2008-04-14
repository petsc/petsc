#include "zope.h"

/*
 * Opens a socket to the PETSc remote server
 * returns a file descriptor for the socket
 *
 * Note: PETSC_HAVE_SYS_SOCKET_H is not defined on Windows
 * and SO_REUSEADDR is not defined on BGL. In either case,
 * this functionality gets disabled.  
 */  

extern FILE *petsc_history;

PetscErrorCode PETSC_DLLEXPORT PetscOpenSocket(char * hostname, int portnum, int * clientfd){
#if defined(PETSC_HAVE_SYS_SOCKET_H) && defined(SO_REUSEADDR)
    struct sockaddr_in sin;
    typedef struct sockaddr SA;
    struct hostent *host;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    /* retrieve information of host */
    host = gethostbyname(hostname);
    if(!host){
        SETERRQ(PETSC_ERR_ARG_CORRUPT, "unknown host");}
    sin.sin_family = AF_INET;
    ierr = PetscMemcpy((char *)&sin.sin_addr,host->h_addr, host->h_length); CHKERRQ(ierr);
    sin.sin_port = htons(portnum);
    /* active open */
    if((*clientfd = socket(AF_INET, SOCK_STREAM, 0)) < 0 ){
        SETERRQ(PETSC_ERR_ARG_CORRUPT,"could not create new socket for client");}
    if(connect(*clientfd, (SA*)&sin, sizeof(sin)) < 0){
        SETERRQ(PETSC_ERR_ARG_CORRUPT,"could not create new connection for client");
        close(*clientfd);}
    PetscFunctionReturn(0);
#else
  SETERRQ(PETSC_ERR_SUP,"Sockets not supported");
#endif
}

/*
 * Recieve function with error handeling
 *
 */
PetscErrorCode PETSC_DLLEXPORT PetscFdRecv(int fd, void *buf, size_t len, int flags, unsigned int *size){
#if defined(PETSC_HAVE_SYS_SOCKET_H) && defined(SO_REUSEADDR)
  ssize_t recvLen;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_SYS_SOCKET_H) && defined(SO_REUSEADDR)
  recvLen = recv(fd, buf, len, flags);
  if(recvLen < 0) {SETERRQ(PETSC_ERR_ARG_CORRUPT,"Could not complete recv");}
  *size = (unsigned int) recvLen;
#else
  SETERRQ(PETSC_ERR_SUP,"Sockets not supported");
#endif
  PetscFunctionReturn(0);
}   

/*
 * Write function with error handeling
 *
 */
PetscErrorCode PETSC_DLLEXPORT PetscFdWrite(int fd, void *buf, size_t len, unsigned int *size){
#if defined(PETSC_HAVE_SYS_SOCKET_H) && defined(SO_REUSEADDR)
  ssize_t sendLen;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_SYS_SOCKET_H) && defined(SO_REUSEADDR)
  sendLen = write(fd, buf, len);
  if(sendLen < 0) {SETERRQ(PETSC_ERR_ARG_CORRUPT, "Could not complete write: ");}
  *size = (unsigned int) sendLen;
#else
  SETERRQ(PETSC_ERR_SUP,"Sockets not supported");
#endif
  PetscFunctionReturn(0);
}  

/*
 * Opens a listening socket to allow for the PETSc remote to set options
 *
 */
PetscErrorCode PETSC_DLLEXPORT PetscSocketListen(char * hostname, int portnum, int *listenfd){
    const int MAX_BUF = 256;
    int MAX_PENDING = 1;
    int optval = 1;
    char iname[256];
    char value[256];
    unsigned int len = 0;
    unsigned int len2 = 0;
    int newfd,flags;
    PetscErrorCode ierr;
#if defined(PETSC_HAVE_SYS_SOCKET_H) && defined(SO_REUSEADDR)
    struct sockaddr_in sin;
    typedef struct sockaddr SA;
    socklen_t sin_size, sout_size;
    int PETSC_LISTEN_CHECK = 0;
#endif
    
    PetscFunctionBegin;
#if defined(PETSC_HAVE_SYS_SOCKET_H) && defined(SO_REUSEADDR)
    sin.sin_family = AF_INET;
    sin.sin_addr.s_addr = INADDR_ANY;
    sin.sin_port = htons(portnum);
    /* passive open */
    if((*listenfd = socket(PF_INET, SOCK_STREAM, 0)) < 0){
        SETERRQ(PETSC_ERR_ARG_CORRUPT, "could not make a new socket for server");}

    /* Allow for non-blocking on the socket */
    if(!(flags = fcntl(*listenfd, F_GETFL, NULL)))
      SETERRQ(PETSC_ERR_ARG_CORRUPT,"flags error");
    flags = 0 | O_NONBLOCK;
    if(fcntl(*listenfd, F_SETFL, 0 | O_NONBLOCK)) 
      SETERRQ(PETSC_ERR_ARG_CORRUPT,"flags error");
    /* so it is possible to reuse port numbers */
    if(setsockopt(*listenfd, SOL_SOCKET, SO_REUSEADDR,
                (const void *)&optval, sizeof(int)) < 0)
        SETERRQ(PETSC_ERR_ARG_CORRUPT,"Could not open a new socket");
    if((bind(*listenfd, (SA*)&sin, sizeof(sin))) < 0)
      SETERRQ(PETSC_ERR_ARG_CORRUPT,"Could not complete bind");
    listen(*listenfd, MAX_PENDING);
    /* Non-blocking busy loop waiting for a connnection */
    if(fork() == 0){
      /* PETSC_LISTEN_CHECK is set in PetscInitilize and PetscFinalize to
         tell the forked process when the program has finished */
      while(PETSC_LISTEN_CHECK){
	sin_size = sizeof(struct sockaddr_in);
	/* non-blocking listen */
        if((newfd = accept(*listenfd, (SA*)&sin, &sout_size)) < 0){
	  if(errno == EAGAIN) break;
	  SETERRQ(PETSC_ERR_ARG_CORRUPT,"Could not complete accept");}
        /* If a connection is found, fork off process to handle the connection */
        if(fork() == 0){
          close(*listenfd);
          PetscFdRecv(newfd, iname, MAX_BUF, 0, &len);
	  iname[len] = '\0';
	  printf("len = %d iname = %s\n",len, iname);
	  PetscFdWrite(newfd, iname, MAX_BUF, &len2);
          PetscFdRecv(newfd, value, MAX_BUF, 0, &len);
	  value[len] = '\0';
	  printf("len = %d value = %s\n", len, value);
	  ierr = PetscOptionsSetValue(iname, value); CHKERRQ(ierr);
	  close(newfd);
	  exit(0);}
	close(newfd);
	if(!PETSC_LISTEN_CHECK) exit(0);}
      exit(0);}
#else
  SETERRQ(PETSC_ERR_SUP,"Sockets not supported");
#endif
    PetscFunctionReturn(0);}
