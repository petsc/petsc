#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: send.c,v 1.69 1998/04/13 17:54:03 bsmith Exp bsmith $";
#endif

#include "petsc.h"

#if defined(NEED_UTYPE_TYPEDEFS)
/* Some systems have inconsistent include files that use but don't
   ensure that the following definitions are made */
typedef unsigned char   u_char;
typedef unsigned short  u_short;
typedef unsigned short  ushort;
typedef unsigned int    u_int;
typedef unsigned long   u_long;
#endif

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
#if !defined(PARCH_nt)
#include <sys/socket.h>
#include <sys/wait.h>
#include <netinet/in.h>
#include <netdb.h>
#include <fcntl.h>
#if defined(HAVE_STROPTS_H)
#include <stropts.h>
#endif

#include "src/viewer/impls/matlab/matlab.h"
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
/*
    Some versions of the Gnu g++ compiler on the IBM RS6000 require the 
  prototype below.
*/
extern int setsockopt(int,int,int,char*,int);
#endif
extern int close(int);
#if !defined(PARCH_freebsd) && !defined(PARCH_linux) && !defined(PARCH_solaris)
extern int socket(int,int,int);
#if !defined(PARCH_hpux) && !defined(PARCH_alpha) && !defined(PARCH_solaris)
/*
    For some IBM rs6000 machines running AIX 3.2 uncomment the prototype 
   below for connect()
*/
/*
#if defined(PARCH_rs6000) && !defined(__cplusplus)
extern int connect(int,const struct sockaddr *,size_t);
#endif
*/

#if !defined(PARCH_rs6000)
extern int connect(int,struct sockaddr *,int);
#endif
#endif
#endif
#if !defined(PARCH_alpha)
/* 
   Some IBM rs6000 machines have the sleep prototype already declared
   in unistd.h, so just remove it below.
 */
#if defined(PARCH_rs6000)
extern unsigned int sleep(unsigned int);
#else
extern int sleep(unsigned);
#endif
#endif
#if defined(__cplusplus)
};
#endif
#endif

#if (defined(PARCH_IRIX)  || defined(PARCH_IRIX64) || defined(PARCH_IRIX5)) && defined(__cplusplus)
extern "C" {
extern int sleep(unsigned);
extern int close(int);
};
#endif


typedef struct { int onoff; int time; } Linger;
#undef __FUNC__  
#define __FUNC__ "ViewerDestroy_Matlab"
static int ViewerDestroy_Matlab(Viewer viewer)
{
  Linger locallinger;
  locallinger.onoff = 1;
  locallinger.time  = 0;

  PetscFunctionBegin;
  if (setsockopt(viewer->port,SOL_SOCKET,SO_LINGER,(char*)&locallinger,sizeof(Linger))) {
    SETERRQ(PETSC_ERR_LIB,0,"System error setting linger");
  }
  if (close(viewer->port)) {
    SETERRQ(PETSC_ERR_LIB,0,"System error closing socket");
  }
  PetscHeaderDestroy(viewer);
  PetscFunctionReturn(0);
}

/*--------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "SOCKCall_Private"
int SOCKCall_Private(char *hostname,int portnum)
{
  struct sockaddr_in sa;
  struct hostent     *hp;
  int                s = 0,flag = 1;
  
  PetscFunctionBegin;
  if ( (hp=gethostbyname(hostname)) == NULL ) {
    perror("SEND: error gethostbyname: ");   
    SETERRQ(PETSC_ERR_LIB,0,"system error open connection");
  }
  PetscMemzero(&sa,sizeof(sa));
  PetscMemcpy(&sa.sin_addr,hp->h_addr,hp->h_length);
  sa.sin_family = hp->h_addrtype;
  sa.sin_port = htons((u_short) portnum);
  while (flag) {
    if ( (s=socket(hp->h_addrtype,SOCK_STREAM,0)) < 0 ) {
      perror("SEND: error socket");  SETERRQ(PETSC_ERR_LIB,0,"system error");
    }
    if ( connect(s,(struct sockaddr *)&sa,sizeof(sa)) < 0 ) {
       if ( errno == EADDRINUSE ) {
        fprintf(stderr,"SEND: address is in use\n");
      }
#if !defined(PARCH_nt_gnu)
       else if ( errno == EALREADY ) {
        fprintf(stderr,"SEND: socket is non-blocking \n");
      }
      else if ( errno == EISCONN ) {
        fprintf(stderr,"SEND: socket already connected\n"); 
        sleep((unsigned) 1);
      }
#endif
      else if ( errno == ECONNREFUSED ) {
        /* fprintf(stderr,"SEND: forcefully rejected\n"); */
        sleep((unsigned) 1);
      } else {
        perror(NULL); SETERRQ(PETSC_ERR_LIB,0,"system error");
      }
      flag = 1; close(s);
    } 
    else flag = 0;
  }
  PetscFunctionReturn(s);
}

#undef __FUNC__  
#define __FUNC__ "ViewerMatlabOpen"
/*@C
   ViewerMatlabOpen - Opens a connection to a Matlab server.

   Input Parameters:
.  comm - the MPI communicator
.  machine - the machine the server is running on
.  port - the port to connect to, use -1 for the default

   Output Parameter:
.  lab - a context to use when communicating with the server

   Collective on MPI_Comm

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

   Options Database Keys:
   For use with the default Matlab viewer, VIEWER_MATLAB_WORLD
$    -viewer_matlab_machine <machine>
$    -viewer_matlab_port <port>

.keywords: Viewer, Matlab, open

.seealso: MatView(), VecView()
@*/
int ViewerMatlabOpen(MPI_Comm comm,char *machine,int port,Viewer *lab)
{
  Viewer v;
  int    t,rank;

  PetscFunctionBegin;
  if (port <= 0) port = DEFAULTPORT;
  PetscHeaderCreate(v,_p_Viewer,int,VIEWER_COOKIE,MATLAB_VIEWER,comm,ViewerDestroy,0);
  PLogObjectCreate(v);
  MPI_Comm_rank(comm,&rank);
  if (!rank) {
    t = SOCKCall_Private(machine,port);
    v->port        = t;
  }
  v->destroy     = ViewerDestroy_Matlab;
  v->flush       = 0;
  *lab           = v;
  PetscFunctionReturn(0);
}

Viewer VIEWER_MATLAB_WORLD_PRIVATE = 0;

#undef __FUNC__  
#define __FUNC__ "ViewerInitializeMatlabWorld_Private"
int ViewerInitializeMatlabWorld_Private(void)
{
  int  ierr,port = 5005,flag;
  char machine[128];

  PetscFunctionBegin;
  if (VIEWER_MATLAB_WORLD_PRIVATE) PetscFunctionReturn(0);
  ierr = OptionsGetString(PETSC_NULL,"-viewer_matlab_machine",machine,128,&flag);CHKERRQ(ierr);
  if (!flag) {
    ierr = PetscGetHostName(machine,128); CHKERRQ(ierr);
  }
  ierr = OptionsGetInt(PETSC_NULL,"-viewer_matlab_port",&port,&flag); CHKERRQ(ierr);
  ierr = ViewerMatlabOpen(PETSC_COMM_WORLD,machine,port,&VIEWER_MATLAB_WORLD_PRIVATE); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerDestroyMatlab_Private"
int ViewerDestroyMatlab_Private(void)
{
  int ierr;

  PetscFunctionBegin;
  if (VIEWER_MATLAB_WORLD_PRIVATE) {
    ierr = ViewerDestroy(VIEWER_MATLAB_WORLD_PRIVATE); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
#else /* defined (PARCH_nt) */
 
#include "viewer.h"
Viewer VIEWER_MATLAB_WORLD_PRIVATE = 0;

int ViewerInitializeMatlabWorld_Private(void)
{ 
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

int ViewerMatlabOpen(MPI_Comm comm,char *machine,int port,Viewer *lab)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
int ViewerDestroyMatlab_Private(void)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
#endif

