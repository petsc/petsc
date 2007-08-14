#include "zope.h"

/*
 * Opens a socket to the PETSc remote server
 * returns a file descriptor for the socket
 *
 */  

extern FILE *petsc_history;

PetscErrorCode PETSC_DLLEXPORT PetscOpenSocket(char * hostname, int portnum, int * clientfd){
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
}

/*
 * Recieve function with error handeling
 *
 */
PetscErrorCode PETSC_DLLEXPORT Recv(int fd, void *buf, size_t len, int flags, unsigned int *size){
  ssize_t recvLen;

  PetscFunctionBegin;
  recvLen = recv(fd, buf, len, flags);
  if(recvLen < 0) {SETERRQ(PETSC_ERR_ARG_CORRUPT,"Could not complete recv");}
  *size = (unsigned int) recvLen;
  PetscFunctionReturn(0);
}   

/*
 * Write function with error handeling
 *
 */
PetscErrorCode PETSC_DLLEXPORT Write(int fd, void *buf, size_t len, unsigned int *size){
  ssize_t sendLen;

  PetscFunctionBegin;
  sendLen = write(fd, buf, len);
  if(sendLen < 0) {SETERRQ(PETSC_ERR_ARG_CORRUPT, "Could not complete write: ");}
  *size = (unsigned int) sendLen;
  PetscFunctionReturn(0);
}  

/*
 * Opens a listening socket to allow for the PETSc remote to set options
 *
 */
PetscErrorCode PETSC_DLLEXPORT PetscSocketListen(char * hostname, int portnum, int *listenfd){
    int MAX_BUF = 256;
    int MAX_PENDING = 1;
    struct sockaddr_in sin;
    int optval = 1;
    char iname[MAX_BUF];
    char value[MAX_BUF];
    unsigned int len = 0;
    unsigned int len2 = 0;
    int newfd;
    typedef struct sockaddr SA;
    socklen_t sin_size, sout_size;
    int PETSC_LISTEN_CHECK;
    PetscErrorCode ierr;
    
    PetscFunctionBegin;
    sin.sin_family = AF_INET;
    sin.sin_addr.s_addr = INADDR_ANY;
    sin.sin_port = htons(portnum);
    /* passive open */
    if((*listenfd = socket(PF_INET, SOCK_STREAM, 0)) < 0){
        SETERRQ(PETSC_ERR_ARG_CORRUPT, "could not make a new socket for server");}
    /* Allow for non-blocking on the socket */
    int flags;
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
          Recv(newfd, iname, MAX_BUF, 0, &len);
	  iname[len] = '\0';
	  printf("len = %d iname = %s\n",len, iname);
	  Write(newfd, iname, MAX_BUF, &len2);
          Recv(newfd, value, MAX_BUF, 0, &len);
	  value[len] = '\0';
	  printf("len = %d value = %s\n", len, value);
	  ierr = PetscOptionsSetValue(iname, value); CHKERRQ(ierr);
	  close(newfd);
	  exit(0);}
	close(newfd);
	if(!PETSC_LISTEN_CHECK) exit(0);}
      exit(0);}
    PetscFunctionReturn(0);}
