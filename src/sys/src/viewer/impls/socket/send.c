#ifndef lint
static char vcid[] = "$Id: send.c,v 1.5 1995/03/17 04:57:42 bsmith Exp bsmith $";
#endif
/* This is part of the MatlabSockettool package. 
 
        Written by Barry Smith, bsmith@mcs.anl.gov 4/14/92
*/
#if defined(PARCH_rs6000)
/* include files are all messed up on rs6000, IBM likes to 
pretend they conform to all standards like ANSI C, POSIX, X Open,
etc. but they do a half-assed job of organizing their include files */
typedef unsigned char   u_char;
typedef unsigned short  u_short;
typedef unsigned short  ushort;
typedef unsigned int    u_int;
typedef unsigned long   u_long;
#endif
#include <stdio.h>
#include <errno.h> 
#include <sys/types.h>
#if defined(PARCH_rs6000)
#include <ctype.h>
#endif
#include <sys/socket.h>
#include <sys/wait.h>
#include <netinet/in.h>
#include <netdb.h>
#include <fcntl.h>
#if !defined(PARCH_freebsd) && !defined(PARCH_rs6000)  && !defined(PARCH_NeXT)
#include <stropts.h>
#endif

#include "matlab.h"

#if defined(PARCH_paragon) || defined(PARCH_alpha) || defined(PARCH_freebsd)
int byteswapint(int *,int),byteswapdouble(double*,int);
#define BYTESWAPINT(buff,n)    byteswapint(buff,n)
#define BYTESWAPDOUBLE(buff,n) byteswapdouble(buff,n)
#else
#define BYTESWAPINT(buff,n)
#define BYTESWAPDOUBLE(buff,n)
#endif

typedef struct { int onoff; int time; } Linger;
static int MatlabDestroy(PetscObject obj)
{
  Linger linger;
  Viewer viewer = (Viewer) obj; 
  linger.onoff = 1;
  linger.time  = 0;

  if (setsockopt(viewer->port,SOL_SOCKET,SO_LINGER,&linger,sizeof(Linger))) 
    SETERR(1,"Setting linger");
  if (close(viewer->port)) SETERR(1,"closing socket");
#if !defined(PARCH_IRIX) 
  usleep((unsigned) 100);
#endif
  FREE(viewer);
  return 0;
}

/*-----------------------------------------------------------------*/
int write_int(int t,int *buff,int n)
{
  int err;
  BYTESWAPINT(buff,n);
  err = write_data(t,(void *) buff,n*sizeof(int));
  BYTESWAPINT(buff,n);
  return err;
}
/*-----------------------------------------------------------------*/
int write_double(int t,double *buff,int n)
{
  int err;
  BYTESWAPDOUBLE((double*)buff,n);
  err = write_data(t,(void *) buff,n*sizeof(double)); 
  BYTESWAPDOUBLE((double*)buff,n);
  return err; 
}
/*-----------------------------------------------------------------*/
int write_data(int t,void *buff,int n)
{
  if ( n <= 0 ) return 0;
  if ( write(t,(char *)buff,n) < 0 ) {
    SETERR(1,"SEND: error writing "); 
  }
  return 0; 
}
/*--------------------------------------------------------------*/
int call_socket(char *hostname,int portnum)
{
  struct sockaddr_in sa;
  struct hostent     *hp;
  int                s = 0,flag = 1;
  
  if ( (hp=gethostbyname(hostname)) == NULL ) {
    perror("SEND: error gethostbyname: ");   
    SETERR(1,0);
  }
  bzero(&sa,sizeof(sa));
  bcopy(hp->h_addr,(char*)&sa.sin_addr,hp->h_length);
  sa.sin_family = hp->h_addrtype;
  sa.sin_port = htons((u_short) portnum);
  while (flag) {
    if ( (s=socket(hp->h_addrtype,SOCK_STREAM,0)) < 0 ) {
      perror("SEND: error socket");  SETERR(-1,0);
    }
#if defined(PARCH_freebsd)
    if ( connect(s,(struct sockaddr *)&sa,sizeof(sa)) < 0 ) {
#else
    if ( connect(s,&sa,sizeof(sa)) < 0 ) {
#endif
      if ( errno == EALREADY ) {
        fprintf(stderr,"SEND: socket is non-blocking \n");
      }
      else if ( errno == EADDRINUSE ) {
        fprintf(stderr,"SEND: address is in use\n");
      }
      else if ( errno == ECONNREFUSED ) {
        /* fprintf(stderr,"SEND: forcefully rejected\n"); */
#if !defined(PARCH_IRIX)
        usleep((unsigned) 1000);
#endif
      }
      else if ( errno == EISCONN ) {
        fprintf(stderr,"SEND: socket already connected\n"); 
        sleep((unsigned) 1);
      }
      else {
        perror(NULL); SETERR(-1,0);
      }
      flag = 1; close(s);
    } 
    else flag = 0;
  }
  return(s);
}
/*  ------------------- BYTE SWAPPING ROUTINES ---------------------*/
#if defined(PARCH_paragon) || defined(PARCH_alpha) || defined(PARCH_freebsd)
int byteswapint(int *buff,int n)
{
  int  i,j,tmp;
  char *ptr1,*ptr2 = (char *) &tmp;
  for ( j=0; j<n; j++ ) { 
    ptr1 = (char *) (&buff[j]);                              
    for (i=0; i<sizeof(int); i++) {                        
      ptr2[i] = ptr1[sizeof(int)-1-i];             
    } 
    buff[j] = tmp;                                          
  }
  return 0;
}
int byteswapdouble(double *buff,int n)
{
  int    i,j;
  double tmp,*ptr3;
  char   *ptr1,*ptr2 = (char *) &tmp;
  for ( j=0; j<n; j++ ) { 
    ptr3 = &buff[j];
    ptr1 = (char *) ptr3;                              
    for (i=0; i<sizeof(double); i++) {                        
      ptr2[i] = ptr1[sizeof(double)-1-i];             
    } 
    buff[j] = tmp;                                          
  }
  return 0;
}
#endif


/*@
     ViewerMatlabOpen - Opens a connection to a Matlab server.

  Input Parameters:
.   machine - the machine the server is running on
.   port - the port to connect to, use -1 for default. 

  Output Parameter:
.   lab - a context to use when communicating with the server.
@*/
int ViewerMatlabOpen(char *machine,int port,Viewer *lab)
{
  Viewer v;
  int    t;
  if (port <= 0) port = DEFAULTPORT;
  t = call_socket(machine,port);
  PETSCHEADERCREATE(v,_Viewer,VIEWER_COOKIE,MATLAB_VIEWER,MPI_COMM_SELF);
  PLogObjectCreate(v);
  v->port        = t;
  v->destroy     = MatlabDestroy;
  *lab           = v;
  return 0;
}
