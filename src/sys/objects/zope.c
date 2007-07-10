#include "zope.h"

/*
 * Opens a socket to the PETSc remote server
 * returns a file descriptor for the socket
 *
 */
int PetscOpenSocket(char * hostname, int portnum){
    struct sockaddr_in sin;
    int clientfd;
    typedef struct sockaddr SA;
    struct hostent *host;

    //retrieve information of host
    host = gethostbyname(hostname);
    if(!host){
        fprintf(stderr, "unknown host %s\n", hostname);
        exit(1);}
    //build address data structure
    bzero((char *)&sin, sizeof(sin));
    sin.sin_family = AF_INET;
    bcopy(host->h_addr, (char *)&sin.sin_addr, host->h_length);
    sin.sin_port = htons(portnum);
    //active open
    if((clientfd = socket(AF_INET, SOCK_STREAM, 0)) < 0 ){
        perror("could not create new socket for client");
        exit(1);}
    if(connect(clientfd, (SA*)&sin, sizeof(sin)) < 0){
        perror("could not create new connection for client");
        close(clientfd);
        exit(1);}
    return clientfd;
}

/*
 * Recieve function with error handeling
 *
 */
ssize_t Recv(int fd, void *buf, size_t len, int flags){
    int size;
    if((size = recv(fd, buf, len, flags)) < 0){
        perror("Could not complete recv: ");
        exit(1);}
    return size;
}   

/*
 * Write function with error handeling
 *
 */
ssize_t Write(int fd, void *buf, size_t len){
    int size;
    if((size = write(fd, buf, len)) < 0){
        perror("Could not complete write: ");
        exit(1);}
    return size;
}  

/*
 * Opens a listening socket to allow for the PETSc remote to set options
 *
 */
int PetscSocketListen(char * hostname, int portnum){
    int MAX_BUF = 256;
    int MAX_PENDING = 1;
    struct sockaddr_in sin;
    int optval = 1;
    char iname[MAX_BUF];
    char value[MAX_BUF];
    unsigned int len = 0;
    int listenfd, newfd;
    typedef struct sockaddr SA;
    socklen_t sin_size;
    extern int errno;
	extern int PETSC_LISTEN_CHECK;
    //build the address struct
    bzero((char *)&sin, sizeof(sin));
    sin.sin_family = AF_INET;
    sin.sin_addr.s_addr = INADDR_ANY;
    sin.sin_port = htons(portnum);
    //passive open
    if((listenfd = socket(PF_INET, SOCK_STREAM, 0)) < 0){
        perror("could not make a new socket for server\n");
        exit(1);}
    //Allow for non-blocking on the socket
    int flags;
    if(!(flags = fcntl(listenfd, F_GETFL, NULL))) perror("flags error ");
    flags = 0 | O_NONBLOCK;
    if(fcntl(listenfd, F_SETFL, 0 | O_NONBLOCK)) perror("set error ");
    //so it is possible to reuse port numbers
    if(setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR,
                (const void *)&optval, sizeof(int)) < 0){
        perror("could not complete setsockopt: ");
        exit(1);}
    if((bind(listenfd, (SA*)&sin, sizeof(sin))) < 0){
        perror("could not complete bind\n");
        exit(1);}
    listen(listenfd, MAX_PENDING);
    //Non-blocking busy loop waiting for a connnection
    if(fork() == 0){
      //PETSC_LISTEN_CHECK is set in PetscInitilize and PetscFinalize to
      //tell the forked process when the program has finished
      while(PETSC_LISTEN_CHECK){
	sin_size = sizeof(struct sockaddr_in);
	//non-blocking listen
        if((newfd = accept(listenfd, (SA*)&sin, &len)) < 0){
	  if(errno == EAGAIN) break;
	  perror("error with accept ");
          exit(1);}
	//If a connection is found, fork off process to handle the connection
        if(fork() == 0){
          close(listenfd);
          len = Recv(newfd, iname, MAX_BUF, 0);
	  iname[len] = '\0';
	  printf("len = %d iname = %s\n",len, iname);
	  Write(newfd, iname, MAX_BUF);
          len = Recv(newfd, value, MAX_BUF, 0);
	  value[len] = '\0';
	  printf("len = %d value = %s\n", len, value);
	  PetscOptionsSetValue(iname, value);
	  close(newfd);
	  exit(0);}
	close(newfd);
	if(!PETSC_LISTEN_CHECK) exit(0);}
      exit(0);}
    return listenfd;}
