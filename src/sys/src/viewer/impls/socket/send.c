#ifndef lint
static char vcid[] = "$Id: send.c,v 1.34 1996/04/10 04:30:20 bsmith Exp curfman $";
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
    || defined(PARCH_hpux) || defined(PARCH_alpha) || defined(PARCH_solaris) \
    || defined(PARCH_linux)
#if defined(__cplusplus)
extern "C" {
#endif
#if !defined(PARCH_rs6000) && !defined(PARCH_freebsd) && !defined(PARCH_hpux) \
    && !defined(PARCH_alpha) && !defined(PARCH_solaris) && \
    !defined(PARCH_linux)
extern int setsockopt(int,int,int,char*,int);
#endif
extern int close(int);
#if !defined(PARCH_freebsd) && !defined(PARCH_linux) && !defined(PARCH_solaris)
extern int socket(int,int,int);
#if !defined(PARCH_hpux) && !defined(PARCH_alpha) && !defined(PARCH_solaris)
/*
    Some IBM rs6000 machines require the prototype of 
    extern int connect(int, const struct sockaddr *,int);
*/
extern int connect(int,struct sockaddr *,int);
#endif
#endif
#if !defined(PARCH_alpha)
/* 
   Some IBM rs6000 machines have the sleep prototype already declared
   in unistd.h, so just remove it below.
 */
extern int sleep(unsigned);
#endif
#if defined(__cplusplus)
};
#endif
#endif

#if defined(PARCH_IRIX) && defined(__cplusplus)
extern "C" {
extern int sleep(unsigned);
extern int close(int);
};
#endif


typedef struct { int onoff; int time; } Linger;
static int ViewerDestroy_Matlab(PetscObject obj)
{
  Linger linger;
  Viewer viewer = (Viewer) obj; 
  linger.onoff = 1;
  linger.time  = 0;

  if (setsockopt(viewer->port,SOL_SOCKET,SO_LINGER,(char*)&linger,sizeof(Linger))) 
    SETERRQ(1,"ViewerDestroy_Matlab:System error setting linger");
  if (close(viewer->port)) 
    SETERRQ(1,"ViewerDestroy_Matlab:System error closing socket");
  PetscHeaderDestroy(viewer);
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
    SETERRQ(1,"SOCKCall_Private:system error open connection");
  }
  PetscMemzero(&sa,sizeof(sa));
  PetscMemcpy(&sa.sin_addr,hp->h_addr,hp->h_length);
  sa.sin_family = hp->h_addrtype;
  sa.sin_port = htons((u_short) portnum);
  while (flag) {
    if ( (s=socket(hp->h_addrtype,SOCK_STREAM,0)) < 0 ) {
      perror("SEND: error socket");  SETERRQ(-1,"SOCKCall_Private:system error");
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
        sleep((unsigned) 1);
      }
      else if ( errno == EISCONN ) {
        fprintf(stderr,"SEND: socket already connected\n"); 
        sleep((unsigned) 1);
      }
      else {
        perror(NULL); SETERRQ(-1,"SOCKCall_Private:system error");
      }
      flag = 1; close(s);
    } 
    else flag = 0;
  }
  return(s);
}

/*@C
   ViewerMatlabOpen - Opens a connection to a Matlab server.

   Input Parameters:
.  comm - the MPI communicator
.  machine - the machine the server is running on
.  port - the port to connect to, use -1 for the default

   Output Parameter:
.  lab - a context to use when communicating with the server

   Notes:
   Most users should employ the following commands to access the 
   Matlab viewers
$
$    ViewerMatlabOpen(MPI_Comm comm, char *machine,int port,Viewer &viewer)
$    MatView(Mat matrix,Viewer viewer)
$
$                or
$
$    ViewerMatlabOpen(MPI_Comm comm, char *machine,int port,Viewer &viewer)
$    VecView(Vec vector,Viewer viewer)

.keywords: Viewer, Matlab, open

.seealso: MatView(), VecView()
@*/
int ViewerMatlabOpen(MPI_Comm comm,char *machine,int port,Viewer *lab)
{
  Viewer v;
  int    t,rank;

  if (port <= 0) port = DEFAULTPORT;
  PetscHeaderCreate(v,_Viewer,VIEWER_COOKIE,MATLAB_VIEWER,comm);
  PLogObjectCreate(v);
  MPI_Comm_rank(comm,&rank);
  if (!rank) {
    t = SOCKCall_Private(machine,port);
    v->port        = t;
  }
  v->destroy     = ViewerDestroy_Matlab;
  v->flush       = 0;
  *lab           = v;
  return 0;
}

