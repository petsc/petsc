/* $Id: send.c,v 1.114 2000/09/22 20:41:42 bsmith Exp bsmith $ */

#include "petsc.h"
#include "petscsys.h"

#if defined(PETSC_NEEDS_UTYPE_TYPEDEFS)
/* Some systems have inconsistent include files that use but do not
   ensure that the following definitions are made */
typedef unsigned char   u_char;
typedef unsigned short  u_short;
typedef unsigned short  ushort;
typedef unsigned int    u_int;
typedef unsigned long   u_long;
#endif

#include <errno.h>
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#include <sys/types.h>
#include <ctype.h>
#if defined(PETSC_HAVE_ENDIAN_H)
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
#if defined (PETSC_HAVE_IO_H)
#include <io.h>
#endif

#include "src/sys/src/viewer/impls/socket/socket.h"
#include "petscfix.h"

EXTERN_C_BEGIN
#if defined(PETSC_NEED_SETSOCKETOPT_PROTO)
extern int setsockopt(int,int,int,char*,int);
#endif
#if defined(PETSC_NEED_CLOSE_PROTO)
extern int close(int);
#endif
#if defined(PETSC_NEED_SOCKET_PROTO)
extern int socket(int,int,int);
#endif
#if defined(PETSC_NEED_SLEEP_PROTO)
extern int sleep(unsigned);
#endif
#if defined(PETSC_NEED_CONNECT_PROTO)
extern int connect(int,struct sockaddr *,int);
#endif
EXTERN_C_END

/*--------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ /*<a name="ViewerDestroy_Socket"></a>*/"ViewerDestroy_Socket" 
static int ViewerDestroy_Socket(Viewer viewer)
{
  Viewer_Socket *vmatlab = (Viewer_Socket*)viewer->data;
  int           ierr;

  PetscFunctionBegin;
  if (vmatlab->port) {
    ierr = close(vmatlab->port);
    if (ierr) {
      SETERRQ(PETSC_ERR_LIB,"System error closing socket");
    }
  }
  ierr = PetscFree(vmatlab);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*--------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ /*<a name="SOCKCall_Private"></a>*/"SOCKCall_Private" 
int SOCKCall_Private(char *hostname,int portnum,int *t)
{
  struct sockaddr_in sa;
  struct hostent     *hp;
  int                s = 0,ierr;
  PetscTruth         flg = PETSC_TRUE;

  PetscFunctionBegin;
  if (!(hp=gethostbyname(hostname))) {
    perror("SEND: error gethostbyname: ");   
    SETERRQ1(PETSC_ERR_LIB,"system error open connection to %s",hostname);
  }
  ierr = PetscMemzero(&sa,sizeof(sa));CHKERRQ(ierr);
  ierr = PetscMemcpy(&sa.sin_addr,hp->h_addr,hp->h_length);CHKERRQ(ierr);

  sa.sin_family = hp->h_addrtype;
  sa.sin_port = htons((u_short) portnum);
  while (flg) {
    if ((s=socket(hp->h_addrtype,SOCK_STREAM,0)) < 0) {
      perror("SEND: error socket");  SETERRQ(PETSC_ERR_LIB,"system error");
    }
    if (connect(s,(struct sockaddr*)&sa,sizeof(sa)) < 0) {
       if (errno == EADDRINUSE) {
        (*PetscErrorPrintf)("SEND: address is in use\n");
      }
#if !defined(PARCH_win32_gnu)
       else if (errno == EALREADY) {
        (*PetscErrorPrintf)("SEND: socket is non-blocking \n");
      }
      else if (errno == EISCONN) {
        (*PetscErrorPrintf)("SEND: socket already connected\n"); 
        sleep((unsigned) 1);
      }
#endif
      else if (errno == ECONNREFUSED) {
        /* (*PetscErrorPrintf)("SEND: forcefully rejected\n"); */
        sleep((unsigned) 1);
      } else {
        perror(NULL); SETERRQ(PETSC_ERR_LIB,"system error");
      }
      flg = PETSC_TRUE; close(s);
    } 
    else flg = PETSC_FALSE;
  }
  *t = s;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="ViewerSocketOpen"></a>*/"ViewerSocketOpen" 
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
$    ViewerSocketOpen(MPI_Comm comm,char *machine,int port,Viewer &viewer)
$    VecView(Vec vector,Viewer viewer)

   Options Database Keys:
   For use with the default Matlab viewer, VIEWER_SOCKET_WORLD or if 
    PETSC_NULL is passed for machine or PETSC_DEFAULT is passed for port
$    -viewer_socket_machine <machine>
$    -viewer_socket_port <port>

   Environmental variables:
+   PETSC_VIEWER_SOCKET_PORT portnumber
-   PETSC_VIEWER_SOCKET_MACHINE machine name

     Currently the only socket client available is Matlab. See 
     src/dm/da/examples/tests/ex12.c and ex12.m for an example of usage.

   Concepts: Matlab^sending data
   Concepts: Sockets^sending data

.seealso: MatView(), VecView(), ViewerDestroy(), ViewerCreate(), ViewerSetType(),
          ViewerSocketSetConnection()
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

#undef __FUNC__  
#define __FUNC__ /*<a name="ViewerSetFromOptions_Socket"></a>*/"ViewerSetFromOptions_Socket" 
int ViewerSetFromOptions_Socket(Viewer v)
{
  int           ierr,def = -1;
  char          sdef[256];
  PetscTruth    tflg;

  PetscFunctionBegin;
  /*
       These options are not processed here, they are processed in ViewerSocketSetConnection(), they
    are listed here for the GUI to display
  */
  ierr = OptionsHead("Socket Viewer Options");CHKERRQ(ierr);
    ierr = OptionsGetenv(v->comm,"PETSC_VIEWER_SOCKET_PORT",sdef,16,&tflg);CHKERRQ(ierr);
    if (tflg) {
      ierr = OptionsAtoi(sdef,&def);CHKERRQ(ierr);
    } else {
      def = DEFAULTPORT;
    }
    ierr = OptionsInt("-viewer_socket_port","Port number to use for socket","ViewerSocketSetConnection",def,0,0);CHKERRQ(ierr);

    ierr = OptionsString("-viewer_socket_machine","Machine to use for socket","ViewerSocketSetConnection",sdef,0,0,0);CHKERRQ(ierr);
    ierr = OptionsGetenv(v->comm,"PETSC_VIEWER_SOCKET_MACHINE",sdef,256,&tflg);CHKERRQ(ierr);
    if (!tflg) {
      ierr = PetscGetHostName(sdef,256);CHKERRQ(ierr);
    }
  ierr = OptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name="ViewerCreate_Socket"></a>*/"ViewerCreate_Socket" 
int ViewerCreate_Socket(Viewer v)
{
  Viewer_Socket *vmatlab;

  PetscFunctionBegin;
  vmatlab                = PetscNew(Viewer_Socket);CHKPTRQ(vmatlab);
  vmatlab->port          = 0;
  v->data                = (void*)vmatlab;
  v->ops->destroy        = ViewerDestroy_Socket;
  v->ops->flush          = 0;
  v->ops->setfromoptions = ViewerSetFromOptions_Socket;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNC__  
#define __FUNC__ /*<a name="ViewerSocketSetConnection"></a>*/"ViewerSocketSetConnection" 
int ViewerSocketSetConnection(Viewer v,const char machine[],int port)
{
  int           ierr,rank;
  char          mach[256];
  PetscTruth    tflg;
  Viewer_Socket *vmatlab = (Viewer_Socket *)v->data;

  PetscFunctionBegin;
  if (port <= 0) {
    char portn[16];
    ierr = OptionsGetenv(v->comm,"PETSC_VIEWER_SOCKET_PORT",portn,16,&tflg);CHKERRQ(ierr);
    if (tflg) {
      ierr = OptionsAtoi(portn,&port);CHKERRQ(ierr);
    } else {
      port = DEFAULTPORT;
    }
  }
  if (!machine) {
    ierr = OptionsGetenv(v->comm,"PETSC_VIEWER_SOCKET_MACHINE",mach,256,&tflg);CHKERRQ(ierr);
    if (!tflg) {
      ierr = PetscGetHostName(mach,256);CHKERRQ(ierr);
    }
  } else {
    ierr = PetscStrncpy(mach,machine,256);CHKERRQ(ierr);
  }

  ierr = MPI_Comm_rank(v->comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    PLogInfo(0,"ViewerSocketSetConnection:Connecting to socket process on port %d machine %s\n",port,mach);
    ierr = SOCKCall_Private(mach,port,&vmatlab->port);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------*/
/*
    The variable Petsc_Viewer_Socket_keyval is used to indicate an MPI attribute that
  is attached to a communicator, in this case the attribute is a Viewer.
*/
static int Petsc_Viewer_Socket_keyval = MPI_KEYVAL_INVALID;

#undef __FUNC__  
#define __FUNC__ /*<a name="VIEWER_SOCKET_"></a>*/"VIEWER_SOCKET_"  
/*@C
     VIEWER_SOCKET_ - Creates a socket viewer shared by all processors 
                     in a communicator.

     Collective on MPI_Comm

     Input Parameter:
.    comm - the MPI communicator to share the socket viewer

     Level: intermediate

   Options Database Keys:
   For use with the default Matlab viewer, VIEWER_SOCKET_WORLD or if 
    PETSC_NULL is passed for machine or PETSC_DEFAULT is passed for port
$    -viewer_socket_machine <machine>
$    -viewer_socket_port <port>

   Environmental variables:
+   PETSC_VIEWER_SOCKET_PORT portnumber
-   PETSC_VIEWER_SOCKET_MACHINE machine name

     Notes:
     Unlike almost all other PETSc routines, VIEWER_SOCKET_ does not return 
     an error code.  The socket viewer is usually used in the form
$       XXXView(XXX object,VIEWER_SOCKET_(comm));

     Currently the only socket client available is Matlab. See 
     src/dm/da/examples/tests/ex12.c and ex12.m for an example of usage.

     Connects to a waiting socket and stays connected until ViewerDestroy() is called.

.seealso: VIEWER_SOCKET_WORLD, VIEWER_SOCKET_SELF, ViewerSocketOpen(), ViewerCreate(),
          ViewerSocketSetConnection(), ViewerDestroy()
@*/
Viewer VIEWER_SOCKET_(MPI_Comm comm)
{
  int        ierr;
  PetscTruth flg;
  Viewer     viewer;

  PetscFunctionBegin;
  if (Petsc_Viewer_Socket_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,MPI_NULL_DELETE_FN,&Petsc_Viewer_Socket_keyval,0);
    if (ierr) {PetscError(__LINE__,"VIEWER_SOCKET_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
  }
  ierr = MPI_Attr_get(comm,Petsc_Viewer_Socket_keyval,(void **)&viewer,(int *)&flg);
  if (ierr) {PetscError(__LINE__,"VIEWER_SOCKET_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
  if (!flg) { /* viewer not yet created */
    ierr = ViewerSocketOpen(comm,0,0,&viewer); 
    if (ierr) {PetscError(__LINE__,"VIEWER_SOCKET_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
    ierr = PetscObjectRegisterDestroy((PetscObject)viewer);
    if (ierr) {PetscError(__LINE__,"VIEWER_STDOUT_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
    ierr = MPI_Attr_put(comm,Petsc_Viewer_Socket_keyval,(void*)viewer);
    if (ierr) {PetscError(__LINE__,"VIEWER_SOCKET_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
  } 
  PetscFunctionReturn(viewer);
}

#else /* defined (PARCH_win32) */
 
#include "petscviewer.h"
#undef __FUNC__  
#define __FUNC__ /*<a name="ViewerSocketOpen"></a>*/"ViewerSocketOpen" 
int ViewerSocketOpen(MPI_Comm comm,const char machine[],int port,Viewer *lab)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
#undef __FUNC__  
#define __FUNC__ /*<a name="VIEWER_SOCKET_"></a>*/"VIEWER_SOCKET_" 
Viewer VIEWER_SOCKET_(MPI_Comm comm)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name="ViewerCreate_Socket"></a>*/"ViewerCreate_Socket" 
int ViewerCreate_Socket(Viewer v)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
EXTERN_C_END
#endif


















