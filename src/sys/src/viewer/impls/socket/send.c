#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: send.c,v 1.94 1999/05/04 20:27:46 balay Exp bsmith $";
#endif

#include "petsc.h"
#include "sys.h"

#if defined(PETSC_NEEDS_UTYPE_TYPEDEFS)
/* Some systems have inconsistent include files that use but don't
   ensure that the following definitions are made */
typedef unsigned char   u_char;
typedef unsigned short  u_short;
typedef unsigned short  ushort;
typedef unsigned int    u_int;
typedef unsigned long   u_long;
#endif

#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#include <sys/types.h>
#if defined(PETSC_HAVE_CTYPE_H)
#include <ctype.h>
#endif
#if defined(PARCH_alpha)
#include <machine/endian.h>
#endif
#if defined(PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif
#if !defined(PARCH_win32)
#include <sys/socket.h>
#include <sys/wait.h>
#include <netinet/in.h>
#include <netdb.h>
#include <fcntl.h>
#if defined(PETSC_HAVE_STROPTS_H)
#include <stropts.h>
#endif

#include "src/sys/src/viewer/impls/socket/socket.h"
#include "pinclude/petscfix.h"

/*
     Many machines don't prototype many of the socket functions?
*/
#if defined(PARCH_sun4) || defined(PARCH_rs6000) || defined(PARCH_freebsd) \
    || defined(PARCH_hpux) || defined(PARCH_alpha) || defined(PARCH_solaris) \
    || defined(PARCH_linux) || defined(PARCH_ascired)
EXTERN_C_BEGIN
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
#elif !defined(PARCH_ascired)
extern int sleep(unsigned);
#endif
#endif
EXTERN_C_END
#endif

#if (defined(PARCH_IRIX)  || defined(PARCH_IRIX64) || defined(PARCH_IRIX5)) && defined(__cplusplus)
extern "C" {
extern int sleep(unsigned);
extern int close(int);
};
#endif

#undef __FUNC__  
#define __FUNC__ "ViewerDestroy_Socket"
static int ViewerDestroy_Socket(Viewer viewer)
{
  Viewer_Socket *vmatlab = (Viewer_Socket *) viewer->data;
  int           ierr;

  PetscFunctionBegin;
  if (vmatlab->port) {
    ierr = close(vmatlab->port);
    if (ierr) {
      SETERRQ(PETSC_ERR_LIB,0,"System error closing socket");
    }
  }
  PetscFree(vmatlab);
  PetscFunctionReturn(0);
}

/*--------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "SOCKCall_Private"
int SOCKCall_Private(char *hostname,int portnum,int *t)
{
  struct sockaddr_in sa;
  struct hostent     *hp;
  int                s = 0,flag = 1,ierr;
  
  PetscFunctionBegin;
  if ( (hp=gethostbyname(hostname)) == NULL ) {
    perror("SEND: error gethostbyname: ");   
    fprintf(stderr,"hostname tried %s\n",hostname);
    SETERRQ(PETSC_ERR_LIB,0,"system error open connection");
  }
  ierr = PetscMemzero(&sa,sizeof(sa));CHKERRQ(ierr);
  ierr = PetscMemcpy(&sa.sin_addr,hp->h_addr,hp->h_length);CHKERRQ(ierr);

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
#if !defined(PARCH_win32_gnu)
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
  *t = s;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerSocketOpen"
/*@C
   ViewerSocketOpen - Opens a connection to a Matlab or other socket
        based server.

   Collective on MPI_Comm

   Input Parameters:
+  comm - the MPI communicator
.  machine - the machine the server is running on
-  port - the port to connect to, use PETSC_DEFAULT for the default

   Output Parameter:
.  lab - a context to use when communicating with the server

   Level: intermediate

   Notes:
   Most users should employ the following commands to access the 
   Matlab viewers
$
$    ViewerSocketOpen(MPI_Comm comm, char *machine,int port,Viewer &viewer)
$    MatView(Mat matrix,Viewer viewer)
$
$                or
$
$    ViewerSocketOpen(MPI_Comm comm, char *machine,int port,Viewer &viewer)
$    VecView(Vec vector,Viewer viewer)

   Options Database Keys:
   For use with the default Matlab viewer, VIEWER_SOCKET_WORLD or if 
    PETSC_NULL is passed for machine or PETSC_DEFAULT is passed for port
$    -viewer_socket_machine <machine>
$    -viewer_socket_port <port>

   Environmental variables:
.   PETSC_VIEWER_SOCKET_PORT portnumber

     Currently the only socket client available is Matlab. See 
     src/dm/da/examples/tests/ex12.c and ex12.m for an example of usage.

.keywords: Viewer, Socket, open

.seealso: MatView(), VecView()
@*/
int ViewerSocketOpen(MPI_Comm comm,const char machine[],int port,Viewer *lab)
{
  int ierr;

  PetscFunctionBegin;
  ierr = ViewerCreate(comm,lab);CHKERRQ(ierr);
  ierr = ViewerSetType(*lab,SOCKET_VIEWER);CHKERRQ(ierr);
  ierr = ViewerSocketSetConnection(*lab,machine,port);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "ViewerCreate_Socket"
int ViewerCreate_Socket(Viewer v)
{
  int           ierr;
  Viewer_Socket *vmatlab;

  PetscFunctionBegin;

  vmatlab         = PetscNew(Viewer_Socket);CHKPTRQ(vmatlab);
  vmatlab->port   = 0;
  v->data         = (void *) vmatlab;
  v->ops->destroy = ViewerDestroy_Socket;
  v->ops->flush   = 0;
  v->type_name    = (char *)PetscMalloc((1+PetscStrlen(SOCKET_VIEWER))*sizeof(char));CHKPTRQ(v->type_name);
  ierr = PetscStrcpy(v->type_name,SOCKET_VIEWER);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNC__  
#define __FUNC__ "ViewerSocketSetConnection"
int ViewerSocketSetConnection(Viewer v,const char machine[],int port)
{
  int           ierr,rank,flag;
  char          mach[256];
  PetscTruth    tflag;
  Viewer_Socket *vmatlab = (Viewer_Socket *)v->data;

  PetscFunctionBegin;
   if (port <= 0) {
    ierr = OptionsGetInt(PETSC_NULL,"-viewer_socket_port",&port,&flag);CHKERRQ(ierr);
    if (!flag) {
      char portn[16];
      ierr = OptionsGetenv(v->comm,"PETSC_VIEWER_SOCKET_PORT",portn,16,&tflag);CHKERRQ(ierr);
      if (tflag) {
        port = OptionsAtoi(portn);
      } else {
        port = DEFAULTPORT;
      }
    }
  }
  if (!machine) {
    ierr = OptionsGetString(PETSC_NULL,"-viewer_socket_machine",mach,128,&flag);CHKERRQ(ierr);
    if (!flag) {
      ierr = PetscGetHostName(mach,256);CHKERRQ(ierr);
    }
  } else {
    ierr = PetscStrncpy(mach,machine,256);CHKERRQ(ierr);
  }

  ierr = MPI_Comm_rank(v->comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    PLogInfo(0,"ViewerSocketSetConnection:Connecting to socket process on port %d machine %s\n",port,mach);
    ierr          = SOCKCall_Private(mach,port,&vmatlab->port);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

Viewer VIEWER_SOCKET_WORLD_PRIVATE = 0;

#undef __FUNC__  
#define __FUNC__ "ViewerInitializeSocketWorld_Private"
int ViewerInitializeSocketWorld_Private(void)
{
  int  ierr;

  PetscFunctionBegin;
  if (VIEWER_SOCKET_WORLD_PRIVATE) PetscFunctionReturn(0);
  ierr = ViewerSocketOpen(PETSC_COMM_WORLD,0,0,&VIEWER_SOCKET_WORLD_PRIVATE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerDestroySocket_Private"
int ViewerDestroySocket_Private(void)
{
  int ierr;

  PetscFunctionBegin;
  if (VIEWER_SOCKET_WORLD_PRIVATE) {
    ierr = ViewerDestroy(VIEWER_SOCKET_WORLD_PRIVATE);CHKERRQ(ierr);
  }
  /*
      Free any viewers created with the VIEWER_DRAW_(MPI_Comm comm) trick.
  */
  ierr = VIEWER_SOCKET_Destroy(PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = VIEWER_SOCKET_Destroy(PETSC_COMM_SELF);CHKERRQ(ierr);
  ierr = VIEWER_SOCKET_Destroy(MPI_COMM_WORLD);CHKERRQ(ierr);
  ierr = VIEWER_SOCKET_Destroy(MPI_COMM_SELF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* ---------------------------------------------------------------------*/
/*
    The variable Petsc_Viewer_Socket_keyval is used to indicate an MPI attribute that
  is attached to a communicator, in this case the attribute is a Viewer.
*/
static int Petsc_Viewer_Socket_keyval = MPI_KEYVAL_INVALID;

#undef __FUNC__  
#define __FUNC__ "VIEWER_SOCKET_" 
/*@C
     VIEWER_SOCKET_ - Creates a socket viewer shared by all processors 
                     in a communicator.

     Collective on MPI_Comm

     Input Parameter:
.    comm - the MPI communicator to share the socket viewer

     Level: intermediate

     Notes:
     Unlike almost all other PETSc routines, VIEWER_SOCKET_ does not return 
     an error code.  The socket viewer is usually used in the form
$       XXXView(XXX object,VIEWER_SOCKET_(comm));

     Currently the only socket client available is Matlab. See 
     src/dm/da/examples/tests/ex12.c and ex12.m for an example of usage.

.seealso: VIEWER_SOCKET_WORLD, VIEWER_SOCKET_SELF, ViewerSocketOpen(), 
@*/
Viewer VIEWER_SOCKET_(MPI_Comm comm)
{
  int    ierr,flag;
  Viewer viewer;

  PetscFunctionBegin;
  if (Petsc_Viewer_Socket_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,MPI_NULL_DELETE_FN,&Petsc_Viewer_Socket_keyval,0);
    if (ierr) {PetscError(__LINE__,"VIEWER_SOCKET_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
  }
  ierr = MPI_Attr_get( comm, Petsc_Viewer_Socket_keyval, (void **)&viewer, &flag );
  if (ierr) {PetscError(__LINE__,"VIEWER_SOCKET_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
  if (!flag) { /* viewer not yet created */
    ierr = ViewerSocketOpen(comm,0,0,&viewer); 
    if (ierr) {PetscError(__LINE__,"VIEWER_SOCKET_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
    ierr = MPI_Attr_put( comm, Petsc_Viewer_Socket_keyval, (void *) viewer );
    if (ierr) {PetscError(__LINE__,"VIEWER_SOCKET_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
  } 
  PetscFunctionReturn(viewer);
}

/*
       If there is a Viewer associated with this communicator it is destroyed.
*/
int VIEWER_SOCKET_Destroy(MPI_Comm comm)
{
  int    ierr,flag;
  Viewer viewer;

  PetscFunctionBegin;
  if (Petsc_Viewer_Socket_keyval == MPI_KEYVAL_INVALID) {
    PetscFunctionReturn(0);
  }
  ierr = MPI_Attr_get( comm, Petsc_Viewer_Socket_keyval, (void **)&viewer, &flag );CHKERRQ(ierr);
  if (flag) { 
    ierr = ViewerDestroy(viewer);CHKERRQ(ierr);
    ierr = MPI_Attr_delete(comm,Petsc_Viewer_Socket_keyval);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#else /* defined (PARCH_win32) */
 
#include "viewer.h"
Viewer VIEWER_SOCKET_WORLD_PRIVATE = 0;

int ViewerInitializeSocketWorld_Private(void)
{ 
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

int ViewerSocketOpen(MPI_Comm comm,const char machine[],int port,Viewer *lab)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
int ViewerDestroySocket_Private(void)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
Viewer VIEWER_SOCKET_(MPI_Comm comm)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
EXTERN_C_BEGIN
int ViewerCreate_Socket(Viewer v)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
EXTERN_C_END
#endif

