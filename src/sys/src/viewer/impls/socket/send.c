#ifndef lint
static char vcid[] = "$Id: send.c,v 1.20 1995/08/15 20:29:10 bsmith Exp bsmith $";
#endif
/* 
 
        Written by Barry Smith, bsmith@mcs.anl.gov 4/14/92
*/
#if defined(PARCH_rs6000) || defined(PARCH_paragon) ||  defined(PARCH_alpha)
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
#if defined(HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#include <sys/types.h>
#if defined(PARCH_rs6000)
#include <ctype.h>
#endif
#if defined(PARCH_alpha)
#include <machine/endian.h>
#endif
#include <sys/socket.h>
#include <sys/wait.h>
#include <netinet/in.h>
#include <netdb.h>
#include <fcntl.h>
#if defined(HAVE_STROPTS_H)
#include <stropts.h>
#endif

#include "matlab.h"
#include "pinclude/petscfix.h"

/*
     Many machines don't prototype many of the socket functions?
*/
#if defined(PARCH_sun4) || defined(PARCH_rs6000) || defined(PARCH_freebsd) \
    || defined(PARCH_hpux) || defined(PARCH_alpha) || defined(PARCH_solaris)
#if defined(__cplusplus)
extern "C" {
#endif
#if !defined(PARCH_rs6000) && !defined(PARCH_freebsd) && !defined(PARCH_hpux) \
    && !defined(PARCH_alpha) && !defined(PARCH_solaris)
extern int setsockopt(int,int,int,char*,int);
#endif
extern int write(int,char*,int);
extern int close(int);
#if !defined(PARCH_freebsd) && !defined(PARCH_solaris)
extern int socket(int,int,int);
#if !defined(PARCH_hpux) && !defined(PARCH_alpha) && !defined(PARCH_solaris)
extern int connect(int,struct sockaddr *,int);
#endif
#endif
#if !defined(PARCH_alpha)
extern int sleep(unsigned);
extern int usleep(unsigned);
#endif
#if defined(__cplusplus)
};
#endif
#endif

#if defined(PARCH_IRIX) && defined(__cplusplus)
extern "C" {
extern int sleep(unsigned);
extern int write(int,char*,int);
extern int close(int);
};
#endif

/*
      Byte swapping for certain Machines.
*/
#if defined(PARCH_paragon) || defined(PARCH_alpha) || defined(PARCH_freebsd)
static int byteswapint(int *,int),byteswapdouble(double*,int);
#define BYTESWAPINT(buff,n)    byteswapint(buff,n)
#define BYTESWAPDOUBLE(buff,n) byteswapdouble(buff,n)
#else
#define BYTESWAPINT(buff,n)
#define BYTESWAPDOUBLE(buff,n)
#endif

typedef struct { int onoff; int time; } Linger;
static int ViewerDestroy_Matlab(PetscObject obj)
{
  Linger linger;
  Viewer viewer = (Viewer) obj; 
  linger.onoff = 1;
  linger.time  = 0;

  if (setsockopt(viewer->port,SOL_SOCKET,SO_LINGER,(char*)&linger,sizeof(Linger))) 
    SETERRQ(1,"ViewerDestroy_Matlab: System error setting linger");
  if (close(viewer->port)) 
    SETERRQ(1,"ViewerDestroy_Matlab: System error closing socket");
#if !defined(PARCH_IRIX) && !defined(PARCH_hpux) && !defined(PARCH_solaris) \
    && !defined(PARCH_t3d)
  usleep((unsigned) 100);
#endif
  PETSCHEADERDESTROY(viewer);
  return 0;
}

/*-----------------------------------------------------------------*/
int SOCKWriteInt_Private(int t,int *buff,int n)
{
  int err;
  BYTESWAPINT(buff,n);
  err = SOCKWrite_Private(t,(void *) buff,n*sizeof(int));
  BYTESWAPINT(buff,n);
  return err;
}
/*-----------------------------------------------------------------*/
int SOCKWriteDouble_Private(int t,double *buff,int n)
{
  int err;
  BYTESWAPDOUBLE((double*)buff,n);
  err = SOCKWrite_Private(t,(void *) buff,n*sizeof(double)); 
  BYTESWAPDOUBLE((double*)buff,n);
  return err; 
}
/*-----------------------------------------------------------------*/
int SOCKWrite_Private(int t,void *buff,int n)
{
  if ( n <= 0 ) return 0;
  if ( write(t,(char *)buff,n) < 0 ) {
    SETERRQ(1,"write_data: error writing "); 
  }
  return 0; 
}
/*--------------------------------------------------------------*/
int SOCKCall_Private(char *hostname,int portnum)
{
  struct sockaddr_in sa;
  struct hostent     *hp;
  int                s = 0,flag = 1;
  
  if ( (hp=gethostbyname(hostname)) == NULL ) {
    perror("SEND: error gethostbyname: ");   
    SETERRQ(1,"call_socket: system error open connection");
  }
  PETSCMEMSET(&sa,0,sizeof(sa));
  PETSCMEMCPY(&sa.sin_addr,hp->h_addr,hp->h_length);
  sa.sin_family = hp->h_addrtype;
  sa.sin_port = htons((u_short) portnum);
  while (flag) {
    if ( (s=socket(hp->h_addrtype,SOCK_STREAM,0)) < 0 ) {
      perror("SEND: error socket");  SETERRQ(-1,"call_socket: system error");
    }
    if ( connect(s,(struct sockaddr *)&sa,sizeof(sa)) < 0 ) {
      if ( errno == EALREADY ) {
        fprintf(stderr,"SEND: socket is non-blocking \n");
      }
      else if ( errno == EADDRINUSE ) {
        fprintf(stderr,"SEND: address is in use\n");
      }
      else if ( errno == ECONNREFUSED ) {
        /* fprintf(stderr,"SEND: forcefully rejected\n"); */
#if !defined(PARCH_IRIX) && !defined(PARCH_hpux) && !defined(PARCH_solaris) \
    && !defined(PARCH_t3d)
        usleep((unsigned) 1000);
#endif
      }
      else if ( errno == EISCONN ) {
        fprintf(stderr,"SEND: socket already connected\n"); 
        sleep((unsigned) 1);
      }
      else {
        perror(NULL); SETERRQ(-1,"call_socket:system error");
      }
      flag = 1; close(s);
    } 
    else flag = 0;
  }
  return(s);
}
/*  ------------------- BYTE SWAPPING ROUTINES ---------------------*/
#if defined(PARCH_paragon) || defined(PARCH_alpha) || defined(PARCH_freebsd)
static int byteswapint(int *buff,int n)
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
static int byteswapdouble(double *buff,int n)
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
.  machine - the machine the server is running on
.  port - the port to connect to, use -1 for the default

   Output Parameter:
.  lab - a context to use when communicating with the server

   Notes:
   Most users should employ the following commands to access the 
   Matlab viewers
$
$    ViewerMatlabOpen(char *machine,int port,Viewer &viewer)
$    MatView(Mat matrix,Viewer viewer)
$
$                or
$
$    ViewerMatlabOpen(char *machine,int port,Viewer &viewer)
$    VecView(Vec vector,Viewer viewer)

.keywords: Viewer, Matlab, open

.seealso: MatView(), VecView()
@*/
int ViewerMatlabOpen(char *machine,int port,Viewer *lab)
{
  Viewer v;
  int    t;
  if (port <= 0) port = DEFAULTPORT;
  t = SOCKCall_Private(machine,port);
  PETSCHEADERCREATE(v,_Viewer,VIEWER_COOKIE,MATLAB_VIEWER,MPI_COMM_SELF);
  PLogObjectCreate(v);
  v->port        = t;
  v->destroy     = ViewerDestroy_Matlab;
  *lab           = v;
  return 0;
}
