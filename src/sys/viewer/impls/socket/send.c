
#include <petscsys.h>

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
#if defined(PETSC_HAVE_MACHINE_ENDIAN_H)
#include <machine/endian.h>
#endif
#if defined(PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif
#if defined(PETSC_HAVE_SYS_SOCKET_H)
#include <sys/socket.h>
#endif
#if defined(PETSC_HAVE_SYS_WAIT_H)
#include <sys/wait.h>
#endif
#if defined(PETSC_HAVE_NETINET_IN_H)
#include <netinet/in.h>
#endif
#if defined(PETSC_HAVE_NETDB_H)
#include <netdb.h>
#endif
#if defined(PETSC_HAVE_FCNTL_H)
#include <fcntl.h>
#endif
#if defined(PETSC_HAVE_IO_H)
#include <io.h>
#endif
#if defined(PETSC_HAVE_WINSOCK2_H)
#include <Winsock2.h>
#endif
#include <sys/stat.h>
#include <../src/sys/viewer/impls/socket/socket.h>

EXTERN_C_BEGIN
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
#undef __FUNCT__
#define __FUNCT__ "PetscViewerDestroy_Socket"
static PetscErrorCode PetscViewerDestroy_Socket(PetscViewer viewer)
{
  PetscViewer_Socket *vmatlab = (PetscViewer_Socket*)viewer->data;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (vmatlab->port) {
#if defined(PETSC_HAVE_CLOSESOCKET)
    ierr = closesocket(vmatlab->port);
#else
    ierr = close(vmatlab->port);
#endif
    if (ierr) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"System error closing socket");
  }
  ierr = PetscFree(vmatlab);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*--------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "PetscOpenSocket"
/*
    PetscSocketOpen - handles connected to an open port where someone is waiting.

.seealso:   PetscSocketListen(), PetscSocketEstablish()
*/
PetscErrorCode  PetscOpenSocket(char *hostname,int portnum,int *t)
{
  struct sockaddr_in sa;
  struct hostent     *hp;
  int                s = 0;
  PetscErrorCode     ierr;
  PetscBool          flg = PETSC_TRUE;

  PetscFunctionBegin;
  if (!(hp=gethostbyname(hostname))) {
    perror("SEND: error gethostbyname: ");
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SYS,"system error open connection to %s",hostname);
  }
  ierr = PetscMemzero(&sa,sizeof(sa));CHKERRQ(ierr);
  ierr = PetscMemcpy(&sa.sin_addr,hp->h_addr_list[0],hp->h_length);CHKERRQ(ierr);

  sa.sin_family = hp->h_addrtype;
  sa.sin_port = htons((u_short) portnum);
  while (flg) {
    if ((s=socket(hp->h_addrtype,SOCK_STREAM,0)) < 0) {
      perror("SEND: error socket");  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"system error");
    }
    if (connect(s,(struct sockaddr*)&sa,sizeof(sa)) < 0) {
#if defined(PETSC_HAVE_WSAGETLASTERROR)
      ierr = WSAGetLastError();
      if (ierr == WSAEADDRINUSE) {
        (*PetscErrorPrintf)("SEND: address is in use\n");
      } else if (ierr == WSAEALREADY) {
        (*PetscErrorPrintf)("SEND: socket is non-blocking \n");
      } else if (ierr == WSAEISCONN) {
        (*PetscErrorPrintf)("SEND: socket already connected\n");
        Sleep((unsigned) 1);
      } else if (ierr == WSAECONNREFUSED) {
        /* (*PetscErrorPrintf)("SEND: forcefully rejected\n"); */
        Sleep((unsigned) 1);
      } else {
        perror(NULL); SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"system error");
      }
#else
      if (errno == EADDRINUSE) {
        (*PetscErrorPrintf)("SEND: address is in use\n");
      } else if (errno == EALREADY) {
        (*PetscErrorPrintf)("SEND: socket is non-blocking \n");
      } else if (errno == EISCONN) {
        (*PetscErrorPrintf)("SEND: socket already connected\n");
        sleep((unsigned) 1);
      } else if (errno == ECONNREFUSED) {
        /* (*PetscErrorPrintf)("SEND: forcefully rejected\n"); */
        ierr = PetscInfo(0,"Connection refused in attaching socket, trying again");CHKERRQ(ierr);
        sleep((unsigned) 1);
      } else {
        perror(NULL); SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"system error");
      }
#endif
      flg = PETSC_TRUE;
#if defined(PETSC_HAVE_CLOSESOCKET)
      closesocket(s);
#else
      close(s);
#endif
    }
    else flg = PETSC_FALSE;
  }
  *t = s;
  PetscFunctionReturn(0);
}

#define MAXHOSTNAME 100
#undef __FUNCT__
#define __FUNCT__ "PetscSocketEstablish"
/*
   PetscSocketEstablish - starts a listener on a socket

.seealso:   PetscSocketListen()
*/
PetscErrorCode PetscSocketEstablish(int portnum,int *ss)
{
  char               myname[MAXHOSTNAME+1];
  int                s;
  PetscErrorCode     ierr;
  struct sockaddr_in sa;
  struct hostent     *hp;

  PetscFunctionBegin;
  ierr = PetscGetHostName(myname,MAXHOSTNAME);CHKERRQ(ierr);

  ierr = PetscMemzero(&sa,sizeof(struct sockaddr_in));CHKERRQ(ierr);

  hp = gethostbyname(myname);
  if (!hp) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"Unable to get hostent information from system");

  sa.sin_family = hp->h_addrtype;
  sa.sin_port = htons((u_short)portnum);

  if ((s = socket(AF_INET,SOCK_STREAM,0)) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"Error running socket() command");
#if defined(PETSC_HAVE_SO_REUSEADDR)
  {
    int optval = 1; /* Turn on the option */
    ierr = setsockopt(s,SOL_SOCKET,SO_REUSEADDR,(char *)&optval,sizeof(optval));CHKERRQ(ierr);
  }
#endif

  while (bind(s,(struct sockaddr*)&sa,sizeof(sa)) < 0) {
#if defined(PETSC_HAVE_WSAGETLASTERROR)
    ierr = WSAGetLastError();
    if (ierr != WSAEADDRINUSE) {
#else
    if (errno != EADDRINUSE) {
#endif
      close(s);
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"Error from bind()");
    }
  }
  listen(s,0);
  *ss = s;
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSocketListen"
/*
   PetscSocketListens - Listens at a socket created with PetscSocketEstablish()

.seealso:   PetscSocketEstablish()
*/
PetscErrorCode PetscSocketListen(int listenport,int *t)
{
  struct sockaddr_in isa;
#if defined(PETSC_HAVE_ACCEPT_SIZE_T)
  size_t             i;
#else
  int                i;
#endif

  PetscFunctionBegin;
  /* wait for someone to try to connect */
  i = sizeof(struct sockaddr_in);
  if ((*t = accept(listenport,(struct sockaddr *)&isa,(socklen_t *)&i)) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"error from accept()\n");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscViewerSocketOpen"
/*@C
   PetscViewerSocketOpen - Opens a connection to a MATLAB or other socket
        based server.

   Collective on MPI_Comm

   Input Parameters:
+  comm - the MPI communicator
.  machine - the machine the server is running on,, use PETSC_NULL for the local machine, use "server" to passively wait for
             a connection from elsewhere
-  port - the port to connect to, use PETSC_DEFAULT for the default

   Output Parameter:
.  lab - a context to use when communicating with the server

   Level: intermediate

   Notes:
   Most users should employ the following commands to access the
   MATLAB PetscViewers
$
$    PetscViewerSocketOpen(MPI_Comm comm, char *machine,int port,PetscViewer &viewer)
$    MatView(Mat matrix,PetscViewer viewer)
$
$                or
$
$    PetscViewerSocketOpen(MPI_Comm comm,char *machine,int port,PetscViewer &viewer)
$    VecView(Vec vector,PetscViewer viewer)

   Options Database Keys:
   For use with  PETSC_VIEWER_SOCKET_WORLD, PETSC_VIEWER_SOCKET_SELF,
   PETSC_VIEWER_SOCKET_() or if
    PETSC_NULL is passed for machine or PETSC_DEFAULT is passed for port
$    -viewer_socket_machine <machine>
$    -viewer_socket_port <port>

   Environmental variables:
+   PETSC_VIEWER_SOCKET_PORT portnumber
-   PETSC_VIEWER_SOCKET_MACHINE machine name

     Currently the only socket client available is MATLAB. See
     src/dm/da/examples/tests/ex12.c and ex12.m for an example of usage.

   Notes: The socket viewer is in some sense a subclass of the binary viewer, to read and write to the socket
          use PetscViewerBinaryRead/Write/GetDescriptor().

   Concepts: MATLAB^sending data
   Concepts: sockets^sending data

.seealso: MatView(), VecView(), PetscViewerDestroy(), PetscViewerCreate(), PetscViewerSetType(),
          PetscViewerSocketSetConnection(), PETSC_VIEWER_SOCKET_, PETSC_VIEWER_SOCKET_WORLD,
          PETSC_VIEWER_SOCKET_SELF, PetscViewerBinaryWrite(), PetscViewerBinaryRead(), PetscViewerBinaryWriteStringArray(),
          PetscBinaryViewerGetDescriptor()
@*/
PetscErrorCode  PetscViewerSocketOpen(MPI_Comm comm,const char machine[],int port,PetscViewer *lab)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerCreate(comm,lab);CHKERRQ(ierr);
  ierr = PetscViewerSetType(*lab,PETSCVIEWERSOCKET);CHKERRQ(ierr);
  ierr = PetscViewerSocketSetConnection(*lab,machine,port);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscViewerSetFromOptions_Socket"
PetscErrorCode PetscViewerSetFromOptions_Socket(PetscViewer v)
{
  PetscErrorCode ierr;
  PetscInt       def = -1;
  char           sdef[256];
  PetscBool      tflg;

  PetscFunctionBegin;
  /*
       These options are not processed here, they are processed in PetscViewerSocketSetConnection(), they
    are listed here for the GUI to display
  */
  ierr = PetscOptionsHead("Socket PetscViewer Options");CHKERRQ(ierr);
    ierr = PetscOptionsGetenv(((PetscObject)v)->comm,"PETSC_VIEWER_SOCKET_PORT",sdef,16,&tflg);CHKERRQ(ierr);
    if (tflg) {
      ierr = PetscOptionsStringToInt(sdef,&def);CHKERRQ(ierr);
    } else {
      def = PETSCSOCKETDEFAULTPORT;
    }
    ierr = PetscOptionsInt("-viewer_socket_port","Port number to use for socket","PetscViewerSocketSetConnection",def,0,0);CHKERRQ(ierr);

    ierr = PetscOptionsString("-viewer_socket_machine","Machine to use for socket","PetscViewerSocketSetConnection",sdef,0,0,0);CHKERRQ(ierr);
    ierr = PetscOptionsGetenv(((PetscObject)v)->comm,"PETSC_VIEWER_SOCKET_MACHINE",sdef,256,&tflg);CHKERRQ(ierr);
    if (!tflg) {
      ierr = PetscGetHostName(sdef,256);CHKERRQ(ierr);
    }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PetscViewerCreate_Socket"
PetscErrorCode  PetscViewerCreate_Socket(PetscViewer v)
{
  PetscViewer_Socket *vmatlab;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr                   = PetscNewLog(v,PetscViewer_Socket,&vmatlab);CHKERRQ(ierr);
  vmatlab->port          = 0;
  v->data                = (void*)vmatlab;
  v->ops->destroy        = PetscViewerDestroy_Socket;
  v->ops->flush          = 0;
  v->ops->setfromoptions = PetscViewerSetFromOptions_Socket;

  /* lie and say this is a binary viewer; then all the XXXView_Binary() methods will work correctly on it */
  ierr                   = PetscObjectChangeTypeName((PetscObject)v,PETSCVIEWERBINARY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "PetscViewerSocketSetConnection"
/*@C
      PetscViewerSocketSetConnection - Sets the machine and port that a PETSc socket
             viewer is to use

  Logically Collective on PetscViewer

  Input Parameters:
+   v - viewer to connect
.   machine - host to connect to, use PETSC_NULL for the local machine,use "server" to passively wait for
             a connection from elsewhere
-   port - the port on the machine one is connecting to, use PETSC_DEFAULT for default

    Level: advanced

.seealso: PetscViewerSocketOpen()
@*/
PetscErrorCode  PetscViewerSocketSetConnection(PetscViewer v,const char machine[],int port)
{
  PetscErrorCode     ierr;
  PetscMPIInt        rank;
  char               mach[256];
  PetscBool          tflg;
  PetscViewer_Socket *vmatlab = (PetscViewer_Socket *)v->data;

  PetscFunctionBegin;
  /* PetscValidLogicalCollectiveInt(v,port,3); not a PetscInt */
  if (port <= 0) {
    char portn[16];
    ierr = PetscOptionsGetenv(((PetscObject)v)->comm,"PETSC_VIEWER_SOCKET_PORT",portn,16,&tflg);CHKERRQ(ierr);
    if (tflg) {
      PetscInt pport;
      ierr = PetscOptionsStringToInt(portn,&pport);CHKERRQ(ierr);
      port = (int)pport;
    } else {
      port = PETSCSOCKETDEFAULTPORT;
    }
  }
  if (!machine) {
    ierr = PetscOptionsGetenv(((PetscObject)v)->comm,"PETSC_VIEWER_SOCKET_MACHINE",mach,256,&tflg);CHKERRQ(ierr);
    if (!tflg) {
      ierr = PetscGetHostName(mach,256);CHKERRQ(ierr);
    }
  } else {
    ierr = PetscStrncpy(mach,machine,256);CHKERRQ(ierr);
  }

  ierr = MPI_Comm_rank(((PetscObject)v)->comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscStrcmp(mach,"server",&tflg);CHKERRQ(ierr);
    if (tflg) {
      int listenport;
      ierr = PetscInfo1(v,"Waiting for connection from socket process on port %D\n",port);CHKERRQ(ierr);
      ierr = PetscSocketEstablish(port,&listenport);CHKERRQ(ierr);
      ierr = PetscSocketListen(listenport,&vmatlab->port);CHKERRQ(ierr);
      close(listenport);
    } else {
      ierr = PetscInfo2(v,"Connecting to socket process on port %D machine %s\n",port,mach);CHKERRQ(ierr);
      ierr = PetscOpenSocket(mach,port,&vmatlab->port);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------*/
/*
    The variable Petsc_Viewer_Socket_keyval is used to indicate an MPI attribute that
  is attached to a communicator, in this case the attribute is a PetscViewer.
*/
static PetscMPIInt Petsc_Viewer_Socket_keyval = MPI_KEYVAL_INVALID;


#undef __FUNCT__
#define __FUNCT__ "PETSC_VIEWER_SOCKET_"
/*@C
     PETSC_VIEWER_SOCKET_ - Creates a socket viewer shared by all processors in a communicator.

     Collective on MPI_Comm

     Input Parameter:
.    comm - the MPI communicator to share the socket PetscViewer

     Level: intermediate

   Options Database Keys:
   For use with the default PETSC_VIEWER_SOCKET_WORLD or if
    PETSC_NULL is passed for machine or PETSC_DEFAULT is passed for port
$    -viewer_socket_machine <machine>
$    -viewer_socket_port <port>

   Environmental variables:
+   PETSC_VIEWER_SOCKET_PORT portnumber
-   PETSC_VIEWER_SOCKET_MACHINE machine name

     Notes:
     Unlike almost all other PETSc routines, PetscViewer_SOCKET_ does not return
     an error code.  The socket PetscViewer is usually used in the form
$       XXXView(XXX object,PETSC_VIEWER_SOCKET_(comm));

     Currently the only socket client available is MATLAB. See
     src/dm/da/examples/tests/ex12.c and ex12.m for an example of usage.

     Connects to a waiting socket and stays connected until PetscViewerDestroy() is called.

     Use this for communicating with an interactive MATLAB session, see PETSC_VIEWER_MATLAB_() for communicating with the MATLAB engine.

.seealso: PETSC_VIEWER_SOCKET_WORLD, PETSC_VIEWER_SOCKET_SELF, PetscViewerSocketOpen(), PetscViewerCreate(),
          PetscViewerSocketSetConnection(), PetscViewerDestroy(), PETSC_VIEWER_SOCKET_(), PetscViewerBinaryWrite(), PetscViewerBinaryRead(),
          PetscViewerBinaryWriteStringArray(), PetscBinaryViewerGetDescriptor(), PETSC_VIEWER_MATLAB_()
@*/
PetscViewer  PETSC_VIEWER_SOCKET_(MPI_Comm comm)
{
  PetscErrorCode ierr;
  PetscBool      flg;
  PetscViewer    viewer;
  MPI_Comm       ncomm;

  PetscFunctionBegin;
  ierr = PetscCommDuplicate(comm,&ncomm,PETSC_NULL);if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_VIEWER_SOCKET_",__FILE__,__SDIR__,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL," ");PetscFunctionReturn(0);}
  if (Petsc_Viewer_Socket_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,MPI_NULL_DELETE_FN,&Petsc_Viewer_Socket_keyval,0);
    if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_VIEWER_SOCKET_",__FILE__,__SDIR__,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL," ");PetscFunctionReturn(0);}
  }
  ierr = MPI_Attr_get(ncomm,Petsc_Viewer_Socket_keyval,(void **)&viewer,(int*)&flg);
  if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_VIEWER_SOCKET_",__FILE__,__SDIR__,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL," ");PetscFunctionReturn(0);}
  if (!flg) { /* PetscViewer not yet created */
    ierr = PetscViewerSocketOpen(ncomm,0,0,&viewer);
    if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_VIEWER_SOCKET_",__FILE__,__SDIR__,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL," ");PetscFunctionReturn(0);}
    ierr = PetscObjectRegisterDestroy((PetscObject)viewer);
    if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_VIEWER_SOCKET_",__FILE__,__SDIR__,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL," ");PetscFunctionReturn(0);}
    ierr = MPI_Attr_put(ncomm,Petsc_Viewer_Socket_keyval,(void*)viewer);
    if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_VIEWER_SOCKET_",__FILE__,__SDIR__,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL," ");PetscFunctionReturn(0);}
  }
  ierr = PetscCommDestroy(&ncomm);
  if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_VIEWER_SOCKET_",__FILE__,__SDIR__,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL," ");PetscFunctionReturn(0);}
  PetscFunctionReturn(viewer);
}

#if defined(PETSC_USE_SERVER)

#include <pthread.h>
#include <time.h>
#define PROTOCOL   "HTTP/1.1"
#define RFC1123FMT "%a, %d %b %Y %H:%M:%S GMT"

#undef __FUNCT__
#define __FUNCT__ "PetscWebSendHeader"
PetscErrorCode PetscWebSendHeader(FILE *f, int status, const char *title, const char *extra, const char *mime, int length)
{
  time_t now;
  char   timebuf[128];

  PetscFunctionBegin;
  fprintf(f, "%s %d %s\r\n", PROTOCOL, status, title);
  fprintf(f, "Server: %s\r\n", "petscserver/1.0");
  now = time(NULL);
  strftime(timebuf, sizeof(timebuf), RFC1123FMT, gmtime(&now));
  fprintf(f, "Date: %s\r\n", timebuf);
  if (extra) fprintf(f, "%s\r\n", extra);
  if (mime) fprintf(f, "Content-Type: %s\r\n", mime);
  if (length >= 0) fprintf(f, "Content-Length: %d\r\n", length);
  fprintf(f, "Connection: close\r\n");
  fprintf(f, "\r\n");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscWebSendFooter"
PetscErrorCode PetscWebSendFooter(FILE *fd)
{
  PetscFunctionBegin;
  fprintf(fd, "</BODY></HTML>\r\n");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscWebSendError"
PetscErrorCode PetscWebSendError(FILE *f, int status, const char *title, const char *extra, const char *text)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscWebSendHeader(f, status, title, extra, "text/html", -1);CHKERRQ(ierr);
  fprintf(f, "<HTML><HEAD><TITLE>%d %s</TITLE></HEAD>\r\n", status, title);
  fprintf(f, "<BODY><H4>%d %s</H4>\r\n", status, title);
  fprintf(f, "%s\r\n", text);
  ierr = PetscWebSendFooter(f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_AMS)
#undef __FUNCT__
#define __FUNCT__ "PetscAMSDisplayList"
PetscErrorCode PetscAMSDisplayList(FILE *fd)
{
  PetscErrorCode     ierr;
  char               host[256],**comm_list,**mem_list,**fld_list;
  AMS_Comm           ams;
  PetscInt           i = 0,j;
  AMS_Memory_type    mtype;
  AMS_Data_type      dtype;
  AMS_Shared_type    stype;
  AMS_Reduction_type rtype;
  AMS_Memory         memory;
  int                len;
  void               *addr;

  ierr = PetscGetHostName(host,256);CHKERRQ(ierr);
  ierr = AMS_Connect(host, -1, &comm_list);CHKERRQ(ierr);
  ierr = PetscWebSendHeader(fd, 200, "OK", NULL, "text/html", -1);CHKERRQ(ierr);
  if (!comm_list || !comm_list[0]) {
    fprintf(fd, "AMS Communicator not running</p>\r\n");
  } else {
    ierr = AMS_Comm_attach(comm_list[0],&ams);CHKERRQ(ierr);
    ierr = AMS_Comm_get_memory_list(ams,&mem_list);CHKERRQ(ierr);
    if (!mem_list[0]) {
      fprintf(fd, "AMS Communicator %s has no published memories</p>\r\n",comm_list[0]);
    } else {
      fprintf(fd, "<HTML><HEAD><TITLE>Petsc Application Server</TITLE></HEAD>\r\n<BODY>");
      fprintf(fd,"<ul>\r\n");
      while (mem_list[i]) {
	fprintf(fd,"<li> %s</li>\r\n",mem_list[i]);
	ierr = AMS_Memory_attach(ams,mem_list[i],&memory,NULL);CHKERRQ(ierr);
	ierr = AMS_Memory_get_field_list(memory, &fld_list);CHKERRQ(ierr);
	j = 0;
	fprintf(fd,"<ul>\r\n");
	while (fld_list[j]) {
	  fprintf(fd,"<li> %s",fld_list[j]);
	  ierr = AMS_Memory_get_field_info(memory, fld_list[j], &addr, &len, &dtype, &mtype, &stype, &rtype);CHKERRQ(ierr);
	  if (len == 1) {
	    if (dtype == AMS_INT)        fprintf(fd," %d",*(int*)addr);
	    else if (dtype == AMS_STRING) fprintf(fd," %s",*(char**)addr);
	  }
	  fprintf(fd,"</li>\r\n");
	  j++;
	}
	fprintf(fd,"</ul>\r\n");
	i++;
      }
      fprintf(fd,"</ul>\r\n");
    }
  }
  ierr = PetscWebSendFooter(fd);CHKERRQ(ierr);
  ierr = AMS_Disconnect();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscAMSDisplayTree"
PetscErrorCode PetscAMSDisplayTree(FILE *fd)
{
  PetscErrorCode     ierr;
  char               host[256],**comm_list,**mem_list,**fld_list;
  AMS_Comm           ams;
  PetscInt           i = 0,j;
  AMS_Memory_type    mtype;
  AMS_Data_type      dtype;
  AMS_Shared_type    stype;
  AMS_Reduction_type rtype;
  AMS_Memory         memory;
  int                len;
  void               *addr2,*addr3,*addr,*addr4;

  ierr = PetscGetHostName(host,256);CHKERRQ(ierr);
  ierr = AMS_Connect(host, -1, &comm_list);CHKERRQ(ierr);
  ierr = PetscWebSendHeader(fd, 200, "OK", NULL, "text/html", -1);CHKERRQ(ierr);
  if (!comm_list || !comm_list[0]) {
    fprintf(fd, "AMS Communicator not running</p>\r\n");
  } else {
    ierr = AMS_Comm_attach(comm_list[0],&ams);CHKERRQ(ierr);
    ierr = AMS_Comm_get_memory_list(ams,&mem_list);CHKERRQ(ierr);
    if (!mem_list[0]) {
      fprintf(fd, "AMS Communicator %s has no published memories</p>\r\n",comm_list[0]);
    } else {
      PetscInt   Nlevels,*Level,*Levelcnt,*Idbylevel,*Column,*parentid,*Id,maxId = 0,maxCol = 0,*parentId,id,cnt,Nlevelcnt = 0;
      PetscBool  *mask;
      char       **classes,*clas,**subclasses,*sclas;

      /* get maximum number of objects */
      while (mem_list[i]) {
	ierr = AMS_Memory_attach(ams,mem_list[i],&memory,NULL);CHKERRQ(ierr);
	ierr = AMS_Memory_get_field_list(memory, &fld_list);CHKERRQ(ierr);
        ierr = AMS_Memory_get_field_info(memory, "Id", &addr2, &len, &dtype, &mtype, &stype, &rtype);CHKERRQ(ierr);
        Id = (int*) addr2;
        maxId = PetscMax(maxId,*Id);
	i++;
      }
      maxId++;

      /* Gets everyone's parent ID and which nodes are masked */
      ierr = PetscMalloc4(maxId,PetscInt,&parentid,maxId,PetscBool ,&mask,maxId,char**,&classes,maxId,char**,&subclasses);CHKERRQ(ierr);
      ierr = PetscMemzero(classes,maxId*sizeof(char*));CHKERRQ(ierr);
      ierr = PetscMemzero(subclasses,maxId*sizeof(char*));CHKERRQ(ierr);
      for (i=0; i<maxId; i++) mask[i] = PETSC_TRUE;
      i = 0;
      while (mem_list[i]) {
	ierr = AMS_Memory_attach(ams,mem_list[i],&memory,NULL);CHKERRQ(ierr);
	ierr = AMS_Memory_get_field_list(memory, &fld_list);CHKERRQ(ierr);
        ierr = AMS_Memory_get_field_info(memory, "Id", &addr2, &len, &dtype, &mtype, &stype, &rtype);CHKERRQ(ierr);
        Id = (int*) addr2;
	ierr = AMS_Memory_get_field_info(memory, "ParentId", &addr3, &len, &dtype, &mtype, &stype, &rtype);CHKERRQ(ierr);
	parentId = (int*) addr3;
	ierr = AMS_Memory_get_field_info(memory, "Class", &addr, &len, &dtype, &mtype, &stype, &rtype);CHKERRQ(ierr);
        clas = *(char**)addr;
	ierr = AMS_Memory_get_field_info(memory, "Type", &addr4, &len, &dtype, &mtype, &stype, &rtype);CHKERRQ(ierr);
        sclas = *(char**)addr4;
        parentid[*Id] = *parentId;
	mask[*Id]     = PETSC_FALSE;
        ierr = PetscStrallocpy(clas,classes+*Id);CHKERRQ(ierr);
        ierr = PetscStrallocpy(sclas,subclasses+*Id);CHKERRQ(ierr);
        i++;
      }

      /* if the parent is masked then relabel the parent as 0 since the true parent was deleted */
      for (i=0; i<maxId; i++) {
        if (!mask[i] && parentid[i] > 0 && mask[parentid[i]]) parentid[i] = 0;
      }

      ierr = PetscProcessTree(maxId,mask,parentid,&Nlevels,&Level,&Levelcnt,&Idbylevel,&Column);CHKERRQ(ierr);

      for (i=0; i<Nlevels; i++) {
        maxCol = PetscMax(maxCol,Levelcnt[i]);
      }
      for (i=0; i<Nlevels; i++) {
        Nlevelcnt = PetscMax(Nlevelcnt,Levelcnt[i]);
      }

      /* print all the top-level objects */
      fprintf(fd, "<HTML><HEAD><TITLE>Petsc Application Server</TITLE>\r\n");
      fprintf(fd, "<canvas width=800 height=600 id=\"tree\"></canvas>\r\n");
      fprintf(fd, "<script type=\"text/javascript\">\r\n");
      fprintf(fd, "  function draw(){\r\n");
      fprintf(fd, "  var example = document.getElementById('tree');\r\n");
      fprintf(fd, "  var context = example.getContext('2d');\r\n");
      /* adjust font size based on how big a tree is printed */
      if (Nlevels > 5 || Nlevelcnt > 10) {
        fprintf(fd, "  context.font         = \"normal 12px sans-serif\";\r\n");
      } else {
        fprintf(fd, "  context.font         = \"normal 24px sans-serif\";\r\n");
      }
      fprintf(fd, "  context.fillStyle = \"rgb(255,0,0)\";\r\n");
      fprintf(fd, "  context.textBaseline = \"top\";\r\n");
      fprintf(fd, "  var xspacep = 0;\r\n");
      fprintf(fd, "  var yspace = example.height/%d;\r\n",(Nlevels+1));
      /* estimate the height of a string as twice the width of a character */
      fprintf(fd, "  var wheight = context.measureText(\"K\");\r\n");
      fprintf(fd, "  var height = 1.6*wheight.width;\r\n");

      cnt = 0;
      for (i=0; i<Nlevels; i++) {
	fprintf(fd, "  var xspace = example.width/%d;\r\n",Levelcnt[i]+1);
	for (j=0; j<Levelcnt[i]; j++) {
          id   = Idbylevel[cnt++];
          clas  = classes[id];
          sclas = subclasses[id];
	  fprintf(fd, "  var width = context.measureText(\"%s\");\r\n",clas);
	  fprintf(fd, "  var swidth = context.measureText(\"%s\");\r\n",sclas);
	  fprintf(fd, "  context.fillStyle = \"rgb(255,0,0)\";\r\n");
	  fprintf(fd, "  context.fillRect((%d)*xspace-width.width/2, %d*yspace-height/2, width.width, height);\r\n",j+1,i+1);
	  fprintf(fd, "  context.fillRect((%d)*xspace-swidth.width/2, %d*yspace+height/2, swidth.width, height);\r\n",j+1,i+1);
	  fprintf(fd, "  context.fillStyle = \"rgb(0,0,0)\";\r\n");
	  fprintf(fd, "  context.fillText(\"%s\",(%d)*xspace-width.width/2, %d*yspace-height/2);\r\n",clas,j+1,i+1);
	  fprintf(fd, "  context.fillText(\"%s\",(%d)*xspace-swidth.width/2, %d*yspace+height/2);\r\n",sclas,j+1,i+1);
          if (parentid[id]) {
	    fprintf(fd, "  context.moveTo(%d*xspace,%d*yspace-height/2);\r\n",j+1,i+1);
	    fprintf(fd, "  context.lineTo(%d*xspacep,%d*yspace+3*height/2);\r\n",Column[parentid[id]]+1,i);
	    fprintf(fd, "  context.stroke();\r\n");
          }
	}
	fprintf(fd, "  xspacep = xspace;\r\n");
      }
      ierr = PetscFree(Level);CHKERRQ(ierr);
      ierr = PetscFree(Levelcnt);CHKERRQ(ierr);
      ierr = PetscFree(Idbylevel);CHKERRQ(ierr);
      ierr = PetscFree(Column);CHKERRQ(ierr);
      for (i=0; i<maxId; i++) {
        ierr = PetscFree(classes[i]);CHKERRQ(ierr);
        ierr = PetscFree(subclasses[i]);CHKERRQ(ierr);
      }
      ierr = PetscFree4(mask,parentid,classes,subclasses);CHKERRQ(ierr);

      ierr = AMS_Disconnect();CHKERRQ(ierr);
      fprintf(fd, "}\r\n");
      fprintf(fd, "</script>\r\n");
      fprintf(fd, "<body onload=\"draw();\">\r\n");
      fprintf(fd, "</body></html>\r\n");
    }
  }
  ierr = PetscWebSendFooter(fd);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
#endif

#if defined(PETSC_HAVE_YAML)

EXTERN_C_BEGIN
/*
    Toy function that returns all the arguments it is passed
*/
#undef __FUNCT__
#define __FUNCT__ "YAML_echo"
PetscErrorCode YAML_echo(PetscInt argc,char **args,PetscInt *argco,char ***argso)
{
  PetscErrorCode ierr;
  PetscInt       i;

  ierr = PetscPrintf(PETSC_COMM_SELF,"Number of arguments to function %d\n",argc);CHKERRQ(ierr);
  for (i=0; i<argc; i++) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"  %s\n",args[i]);CHKERRQ(ierr);
  }
  *argco = argc;
  ierr = PetscMalloc(argc*sizeof(char*),argso);CHKERRQ(ierr);
  for (i=0; i<argc; i++) {
    ierr = PetscStrallocpy(args[i],&(*argso)[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "YAML_AMS_Connect"
/*
      Connects to the local AMS and gets only the first communication name

   Input Parameters:
.     none

   Output Parameter:
.     oarg1 - the string name of the first communicator

*/
PetscErrorCode YAML_AMS_Connect(PetscInt argc,char **args,PetscInt *argco,char ***argso)
{
  PetscErrorCode ierr;
  char           **list = 0;

  PetscFunctionBegin;
  ierr = AMS_Connect(0,-1,&list);
  if (ierr) {ierr = PetscInfo1(PETSC_NULL,"AMS_Connect() error %d\n",ierr);CHKERRQ(ierr);}
  else if (!list) {ierr = PetscInfo(PETSC_NULL,"AMS_Connect() list empty, not running AMS server\n");CHKERRQ(ierr);}
  *argco = 1;
  ierr = PetscMalloc(sizeof(char*),argso);CHKERRQ(ierr);
  if (list){
    ierr = PetscStrallocpy(list[0],&(*argso)[0]);CHKERRQ(ierr);
  } else {
    ierr = PetscStrallocpy("No AMS publisher running",&(*argso)[0]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "YAML_AMS_Comm_attach"
/*
      Attaches to an AMS communicator

   Input Parameter:
.     arg1 - string name of the communicator

   Output Parameter:
.     oarg1 - the integer name of the communicator

*/
PetscErrorCode YAML_AMS_Comm_attach(PetscInt argc,char **args,PetscInt *argco,char ***argso)
{
  PetscErrorCode ierr;
  AMS_Comm       comm = -1;

  PetscFunctionBegin;
  ierr = AMS_Comm_attach(args[0],&comm);
  if (ierr) {ierr = PetscInfo1(PETSC_NULL,"AMS_Comm_attach() error %d\n",ierr);CHKERRQ(ierr);}
  *argco = 1;
  ierr = PetscMalloc(sizeof(char*),argso);CHKERRQ(ierr);
  ierr = PetscMalloc(3*sizeof(char*),&argso[0][0]);CHKERRQ(ierr);
  sprintf(argso[0][0],"%d",(int)comm);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "YAML_AMS_Comm_get_memory_list"
/*
      Gets the list of memories on an AMS Comm

   Input Parameter:
.     arg1 - integer name of the communicator

   Output Parameter:
.     oarg1 - the list of names

*/
PetscErrorCode YAML_AMS_Comm_get_memory_list(PetscInt argc,char **args,PetscInt *argco,char ***argso)
{
  PetscErrorCode ierr;
  char           **mem_list;
  AMS_Comm       comm;
  PetscInt       i,iargco = 0;

  PetscFunctionBegin;
  sscanf(args[0],"%d",&comm);
  ierr = AMS_Comm_get_memory_list(comm,&mem_list);
  if (ierr) {
    ierr = PetscInfo1(PETSC_NULL,"AMS_Comm_get_memory_list() error %d\n",ierr);CHKERRQ(ierr);
  } else {
    while (mem_list[iargco++]) ;
    iargco--;

    ierr = PetscMalloc((iargco)*sizeof(char*),argso);CHKERRQ(ierr);
    for (i=0; i<iargco; i++) {
      ierr = PetscStrallocpy(mem_list[i],(*argso)+i);CHKERRQ(ierr);
    }
  }
  *argco = iargco;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "YAML_AMS_Memory_attach"
/*
      Attaches to an AMS memory in a communicator

   Input Parameter:
.     arg1 - communicator
.     arg2 - string name of the memory

   Output Parameter:
.     oarg1 - the integer name of the memory
.     oarg2 - the integer step of the memory

*/
PetscErrorCode YAML_AMS_Memory_attach(PetscInt argc,char **args,PetscInt *argco,char ***argso)
{
  PetscErrorCode ierr;
  AMS_Comm       comm;
  AMS_Memory     mem;
  unsigned int   step;

  PetscFunctionBegin;
  sscanf(args[0],"%d",&comm);
  ierr = AMS_Memory_attach(comm,args[1],&mem,&step);
  if (ierr) {ierr = PetscInfo1(PETSC_NULL,"AMS_Memory_attach() error %d\n",ierr);CHKERRQ(ierr);}
  *argco = 2;
  ierr = PetscMalloc(2*sizeof(char*),argso);CHKERRQ(ierr);
  ierr = PetscMalloc(3*sizeof(char*),&argso[0][0]);CHKERRQ(ierr);
  sprintf(argso[0][0],"%d",(int)mem);
  ierr = PetscMalloc(3*sizeof(char*),&argso[0][1]);CHKERRQ(ierr);
  sprintf(argso[0][1],"%d",(int)step);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "YAML_AMS_Memory_get_field_list"
/*
      Gets the list of fields on an AMS Memory

   Input Parameter:
.     arg1 - integer name of the memory

   Output Parameter:
.     oarg1 - the list of names

*/
PetscErrorCode YAML_AMS_Memory_get_field_list(PetscInt argc,char **args,PetscInt *argco,char ***argso)
{
  PetscErrorCode ierr;
  char           **field_list;
  AMS_Memory     mem;
  PetscInt       i,iargco = 0;

  PetscFunctionBegin;
  sscanf(args[0],"%d",&mem);
  ierr = AMS_Memory_get_field_list(mem,&field_list);
  if (ierr) {
    ierr = PetscInfo1(PETSC_NULL,"AMS_Memory_get_field_list() error %d\n",ierr);CHKERRQ(ierr);
  } else {
    while (field_list[iargco++]) ;
    iargco--;

    ierr = PetscMalloc((iargco)*sizeof(char*),argso);CHKERRQ(ierr);
    for (i=0; i<iargco; i++) {
      ierr = PetscStrallocpy(field_list[i],(*argso)+i);CHKERRQ(ierr);
    }
  }
  *argco = iargco;
  PetscFunctionReturn(0);
}
EXTERN_C_END

const char *AMS_Data_types[] = {"AMS_DATA_UNDEF","AMS_BOOLEAN","AMS_INT","AMS_FLOAT","AMS_DOUBLE","AMS_STRING","AMS_Data_type","AMS_",0};
const char *AMS_Memory_types[] = {"AMS_MEMORY_UNDEF","AMS_READ","AMS_WRITE","AMS_Memory_type","AMS_",0};
const char *AMS_Shared_types[] = {"AMS_SHARED_UNDEF","AMS_COMMON","AMS_REDUCED","AMS_DISTRIBUTED","AMS_Shared_type","AMS_",0};
const char *AMS_Reduction_types[] = {"AMS_REDUCTION_WHY_NOT_UNDEF?","AMS_SUM","AMS_MAX","AMS_MIN","AMS_REDUCTION_UNDEF","AMS_Reduction_type","AMS_",0};

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "YAML_AMS_Memory_get_field_info"
/*
      Gets information about a field

   Input Parameter:
.     arg1 - memory
.     arg2 - string name of the field

   Output Parameter:

*/
PetscErrorCode YAML_AMS_Memory_get_field_info(PetscInt argc,char **args,PetscInt *argco,char ***argso)
{
  PetscErrorCode     ierr;
  AMS_Memory         mem;
  void               *addr;
  int                len;
  AMS_Data_type      dtype;
  AMS_Memory_type    mtype;
  AMS_Shared_type    stype;
  AMS_Reduction_type rtype;
  PetscInt           i;

  PetscFunctionBegin;
  sscanf(args[0],"%d",&mem);
  ierr = AMS_Memory_get_field_info(mem,args[1],&addr,&len,&dtype,&mtype,&stype,&rtype);
  if (ierr) {ierr = PetscInfo1(PETSC_NULL,"AMS_Memory_get_field_info() error %d\n",ierr);CHKERRQ(ierr);}
  *argco = 4 + len;
  ierr = PetscMalloc((*argco)*sizeof(char*),argso);CHKERRQ(ierr);
  ierr = PetscStrallocpy(AMS_Data_types[dtype],&argso[0][0]);CHKERRQ(ierr);
  ierr = PetscStrallocpy(AMS_Memory_types[mtype],&argso[0][1]);CHKERRQ(ierr);
  ierr = PetscStrallocpy(AMS_Shared_types[stype],&argso[0][2]);CHKERRQ(ierr);
  ierr = PetscStrallocpy(AMS_Reduction_types[rtype],&argso[0][3]);CHKERRQ(ierr);
  for (i=0; i<len; i++) {
    if (dtype == AMS_STRING) {
      ierr = PetscStrallocpy(*(const char **)addr,&argso[0][4+i]);CHKERRQ(ierr);
    } else if (dtype == AMS_DOUBLE) {
      ierr = PetscMalloc(20*sizeof(char),&argso[0][4+i]);CHKERRQ(ierr);
      sprintf(argso[0][4+i],"%18.16e",*(double*)addr);
    } else if (dtype == AMS_INT) {
      ierr = PetscMalloc(10*sizeof(char),&argso[0][4+i]);CHKERRQ(ierr);
      sprintf(argso[0][4+i],"%d",*(int*)addr);
    } else if (dtype == AMS_BOOLEAN) {
      if (*(int*)addr) {
        ierr = PetscStrallocpy("true",&argso[0][4+i]);CHKERRQ(ierr);
      } else {
        ierr = PetscStrallocpy("false",&argso[0][4+i]);CHKERRQ(ierr);
      }
    } else {
      ierr = PetscStrallocpy("Not yet done",&argso[0][4+i]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

#include "yaml.h"
#undef __FUNCT__
#define __FUNCT__ "PetscProcessYAMLRPC"
PetscErrorCode PetscProcessYAMLRPC(const char* request,char **result)
{
  yaml_parser_t  parser;
  yaml_event_t   event;
  int            done = 0;
  int            count = 0;
  size_t         len;
  PetscErrorCode ierr;
  PetscBool      method,params,id;
  char           *methodname,*idname,**args,**argso = 0;
  PetscInt       argc = 0,argco,i;
  PetscErrorCode (*fun)(PetscInt,char **,PetscInt*,char ***);

  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(char*),&args);CHKERRQ(ierr);
  yaml_parser_initialize(&parser);
  PetscStrlen(request,&len);
  yaml_parser_set_input_string(&parser, (unsigned char *)request, len);

  /* this is totally bogus; it only handles the simple JSON-RPC messages */
  while (!done) {
    if (!yaml_parser_parse(&parser, &event)) {
      ierr = PetscInfo(PETSC_NULL,"Found error in yaml/json\n");CHKERRQ(ierr);
      break;
    }
    done = (event.type == YAML_STREAM_END_EVENT);
    switch (event.type) {
    case YAML_STREAM_START_EVENT:
      ierr = PetscInfo(PETSC_NULL,"Stream start\n");CHKERRQ(ierr);
      break;
    case YAML_STREAM_END_EVENT:
      ierr = PetscInfo(PETSC_NULL,"Stream end\n");CHKERRQ(ierr);
      break;
    case YAML_DOCUMENT_START_EVENT:
      ierr = PetscInfo(PETSC_NULL,"Document start\n");CHKERRQ(ierr);
      break;
    case YAML_DOCUMENT_END_EVENT:
      ierr = PetscInfo(PETSC_NULL,"Document end\n");CHKERRQ(ierr);
      break;
    case YAML_MAPPING_START_EVENT:
      ierr = PetscInfo(PETSC_NULL,"Mapping start event\n");CHKERRQ(ierr);
      break;
    case YAML_MAPPING_END_EVENT:
      ierr = PetscInfo(PETSC_NULL,"Mapping end event \n");CHKERRQ(ierr);
      break;
    case YAML_ALIAS_EVENT:
      ierr = PetscInfo1(PETSC_NULL,"Alias event %s\n",event.data.alias.anchor);CHKERRQ(ierr);
      break;
    case YAML_SCALAR_EVENT:
      ierr = PetscInfo1(PETSC_NULL,"Scalar event %s\n",event.data.scalar.value);CHKERRQ(ierr);
      ierr = PetscStrcmp((char*)event.data.scalar.value,"method",&method);CHKERRQ(ierr);
      ierr = PetscStrcmp((char*)event.data.scalar.value,"params",&params);CHKERRQ(ierr);
      ierr = PetscStrcmp((char*)event.data.scalar.value,"id",&id);CHKERRQ(ierr);
      if (method) {
        yaml_event_delete(&event);
        ierr = yaml_parser_parse(&parser, &event);CHKERRQ(!ierr);
        ierr = PetscInfo1(PETSC_NULL,"Method %s\n",event.data.scalar.value);CHKERRQ(ierr);
        ierr = PetscStrallocpy((char*)event.data.scalar.value,&methodname);CHKERRQ(ierr);
      } else if (id) {
        yaml_event_delete(&event);
        ierr = yaml_parser_parse(&parser, &event);CHKERRQ(!ierr);
        ierr = PetscInfo1(PETSC_NULL,"Id %s\n",event.data.scalar.value);CHKERRQ(ierr);
        ierr = PetscStrallocpy((char*)event.data.scalar.value,&idname);CHKERRQ(ierr);
      } else if (params) {
        yaml_event_delete(&event);
        ierr = yaml_parser_parse(&parser, &event);CHKERRQ(!ierr);
        yaml_event_delete(&event);
        ierr = yaml_parser_parse(&parser, &event);CHKERRQ(!ierr);
        while (event.type != YAML_SEQUENCE_END_EVENT) {
          ierr = PetscInfo1(PETSC_NULL,"  Parameter %s\n",event.data.scalar.value);CHKERRQ(ierr);
          ierr = PetscStrallocpy((char*)event.data.scalar.value,&args[argc++]);CHKERRQ(ierr);
          yaml_event_delete(&event);
          ierr = yaml_parser_parse(&parser, &event);CHKERRQ(!ierr);
        }
      } else { /* ignore all the other variables in the mapping */
        yaml_event_delete(&event);
        ierr = yaml_parser_parse(&parser, &event);CHKERRQ(!ierr);
      }
      break;
    case YAML_SEQUENCE_START_EVENT:
      ierr = PetscInfo(PETSC_NULL,"Sequence start event \n");CHKERRQ(ierr);
      break;
    case YAML_SEQUENCE_END_EVENT:
      ierr = PetscInfo(PETSC_NULL,"Sequence end event \n");CHKERRQ(ierr);
      break;
    default:
      /* It couldn't really happen. */
      break;
    }

    yaml_event_delete(&event);
    count ++;
  }
  yaml_parser_delete(&parser);

  ierr = PetscDLLibrarySym(PETSC_COMM_SELF,PETSC_NULL,PETSC_NULL,methodname,(void**)&fun);CHKERRQ(ierr);
  if (fun) {
    ierr = PetscInfo1(PETSC_NULL,"Located function %s and running it\n",methodname);CHKERRQ(ierr);
    ierr = (*fun)(argc,args,&argco,&argso);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo1(PETSC_NULL,"Did not locate function %s skipping it\n",methodname);CHKERRQ(ierr);
  }

  for (i=0; i<argc; i++) {
    ierr = PetscFree(args[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(args);CHKERRQ(ierr);
  ierr = PetscFree(methodname);CHKERRQ(ierr);

  /* convert the result back to YAML; should use YAML encoder, does not handle zero return arguments */
  ierr = PetscMalloc(1024,result);CHKERRQ(ierr);
  ierr = PetscStrcpy(*result,"{\"error\": null, \"id\": \"");CHKERRQ(ierr);
  ierr = PetscStrcat(*result,idname);CHKERRQ(ierr);
  ierr = PetscStrcat(*result,"\", \"result\" : ");CHKERRQ(ierr);
  if (argco > 1) {ierr = PetscStrcat(*result,"[");CHKERRQ(ierr);}
  for (i=0; i<argco; i++) {
    ierr = PetscStrcat(*result,"\"");CHKERRQ(ierr);
    ierr = PetscStrcat(*result,argso[i]);CHKERRQ(ierr);
    ierr = PetscStrcat(*result,"\"");CHKERRQ(ierr);
    if (i < argco-1) {ierr = PetscStrcat(*result,",");CHKERRQ(ierr);}
  }
  if (argco > 1) {ierr = PetscStrcat(*result,"]");CHKERRQ(ierr);}
  ierr = PetscStrcat(*result,"}");CHKERRQ(ierr);
  ierr = PetscInfo1(PETSC_NULL,"YAML result of function %s\n",*result);CHKERRQ(ierr);

  /* free work space */
  ierr = PetscFree(idname);CHKERRQ(ierr);
  for (i=0; i<argco; i++) {
    ierr = PetscFree(argso[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(argso);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__
#define __FUNCT__ "PetscWebServeRequest"
/*@C
      PetscWebServeRequest - serves a single web request

    Not collective

  Input Parameters:
.   port - the port

    Level: developer

.seealso: PetscWebServe()
@*/
PetscErrorCode  PetscWebServeRequest(int port)
{
  PetscErrorCode ierr;
  FILE           *fd,*fdo;
  char           buf[4096],fullpath[PETSC_MAX_PATH_LEN],truefullpath[PETSC_MAX_PATH_LEN];
  char           *method, *path, *protocol,*result;
  const char*    type;
  PetscBool      flg;
  PetscToken     tok;
  PetscInt       cnt = 8;

  PetscFunctionBegin;
  fd = fdopen(port, "r+");

  ierr = PetscInfo(PETSC_NULL,"Processing web request\n");CHKERRQ(ierr);
  if (!fgets(buf, sizeof(buf), fd)) {
    ierr = PetscInfo(PETSC_NULL,"Cannot read web request, giving up\n");CHKERRQ(ierr);
    goto theend;
  }
  ierr = PetscInfo1(PETSC_NULL,"Processing web request %s",buf);CHKERRQ(ierr);

  ierr = PetscTokenCreate(buf,' ',&tok);CHKERRQ(ierr);
  ierr = PetscTokenFind(tok,&method);CHKERRQ(ierr);
  ierr = PetscTokenFind(tok,&path);CHKERRQ(ierr);
  ierr = PetscTokenFind(tok,&protocol);CHKERRQ(ierr);

  if (!method || !path || !protocol) {
    ierr = PetscInfo(PETSC_NULL,"Web request not well formatted, giving up\n");CHKERRQ(ierr);
    goto theend;
  }

  ierr = PetscStrcmp(method,"GET",&flg);
  if (!flg) {
#if defined(PETSC_HAVE_YAML)
    ierr = PetscStrcmp(method,"POST",&flg);
    /*
          Start to handle support for POSTs based on json-rpc
    */
    if (flg) {
      int    len;
      size_t elen;
      char   *fnd;
      while (cnt--) {

        if (!fgets(buf, sizeof(buf), fd)) {
          ierr = PetscInfo(PETSC_NULL,"Cannot read POST data, giving up\n");CHKERRQ(ierr);
          goto theend;
        }
        ierr = PetscInfo1(PETSC_NULL,"POSTED data %s",buf);CHKERRQ(ierr);
        ierr = PetscStrstr(buf,"Content-Type:",&fnd);CHKERRQ(ierr);
        if (fnd) {
          ierr = PetscStrstr(buf,"application/json-rpc",&fnd);CHKERRQ(ierr);
          if (!fnd) {
            ierr = PetscInfo(PETSC_NULL,"POST content is not json-rpc, skipping post\n");CHKERRQ(ierr);
            goto theend;
          }
        }
      }
      if (!fgets(buf, sizeof(buf), fd)) {
        ierr = PetscInfo(PETSC_NULL,"Cannot read POST length data, giving up\n");CHKERRQ(ierr);
        goto theend;
      }
      ierr = PetscInfo1(PETSC_NULL,"POSTED length data %s",buf);CHKERRQ(ierr);
      sscanf(buf,"Content-Length: %d\n",&len);
      ierr = PetscInfo1(PETSC_NULL,"Length of POSTED data %d\n",len);CHKERRQ(ierr);
      if (!fgets(buf, sizeof(buf), fd)) {
        ierr = PetscInfo(PETSC_NULL,"Cannot read POST data, giving up\n");CHKERRQ(ierr);
        goto theend;
      }
      ierr = PetscInfo1(PETSC_NULL,"POSTED data %s",buf);CHKERRQ(ierr);
      if (!fgets(buf, sizeof(buf), fd)) {
        ierr = PetscInfo(PETSC_NULL,"Cannot read POST data, giving up\n");CHKERRQ(ierr);
        goto theend;
      }
      ierr = PetscInfo1(PETSC_NULL,"POSTED data %s",buf);CHKERRQ(ierr);
      if (!fgets(buf, len+1, fd)) { /* why is this len + 1? */
        ierr = PetscInfo(PETSC_NULL,"Cannot read POST data, giving up\n");CHKERRQ(ierr);
        goto theend;
      }
      ierr = PetscInfo1(PETSC_NULL,"POSTED data %s\n",buf);CHKERRQ(ierr);
      fseek(fd, 0, SEEK_CUR); /* Force change of stream direction */
      ierr = PetscProcessYAMLRPC(buf,&result);CHKERRQ(ierr);
      ierr = PetscStrlen(result,&elen);CHKERRQ(ierr);
      ierr = PetscWebSendHeader(fd, 200, "OK", NULL, "application/json-rpc",(int)elen);CHKERRQ(ierr);
      fprintf(fd, "%s",result);
      goto theend;
    } else {
#endif
      ierr = PetscWebSendError(fd, 501, "Not supported", NULL, "Method is not supported.");CHKERRQ(ierr);
      ierr = PetscInfo(PETSC_NULL,"Web request not a GET or POST, giving up\n");CHKERRQ(ierr);
#if defined(PETSC_HAVE_YAML)
    }
#endif
  } else {
    fseek(fd, 0, SEEK_CUR); /* Force change of stream direction */

    ierr = PetscStrcmp(path,"/favicon.ico",&flg);CHKERRQ(ierr);
    if (flg) {
      /* should have cool PETSc icon */;
      goto theend;
    }
    ierr = PetscStrcmp(path,"/",&flg);CHKERRQ(ierr);
    if (flg) {
      char        program[128];
      PetscMPIInt size;
      PetscViewer viewer;

      ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
      ierr = PetscGetProgramName(program,128);CHKERRQ(ierr);
      ierr = PetscWebSendHeader(fd, 200, "OK", NULL, "text/html", -1);CHKERRQ(ierr);
      fprintf(fd, "<HTML><HEAD><TITLE>Petsc Application Server</TITLE></HEAD>\r\n<BODY>");
      fprintf(fd, "<H4>Serving PETSc application code %s </H4>\r\n\n",program);
      fprintf(fd, "Number of processes %d\r\n\n",size);
      fprintf(fd, "<HR>\r\n");
      ierr = PetscViewerASCIIOpenWithFILE(PETSC_COMM_WORLD,fd,&viewer);CHKERRQ(ierr);
      ierr = PetscOptionsView(viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
      fprintf(fd, "<HR>\r\n");
#if defined(PETSC_HAVE_AMS)
      if (PetscAMSPublishAll) {
	fprintf(fd, "<a href=\"./ams-tree\">Connect to Memory Snooper--Tree Display</a></p>\r\n\r\n");
        fprintf(fd, "<a href=\"./ams-list\">Connect to Memory Snooper--List Display</a></p>\r\n\r\n");
      }
#endif
      fprintf(fd, "<a href=\"./AMSJavascript.html\">Connect to Memory Snooper--Interactive Javascript</a></p>\r\n\r\n");
      ierr = PetscWebSendFooter(fd);CHKERRQ(ierr);
      goto theend;
    }

#if defined(PETSC_HAVE_AMS)
    ierr = PetscStrcmp(path,"/ams-list",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscAMSDisplayList(fd);CHKERRQ(ierr);
      goto theend;
    }
    ierr = PetscInfo1(PETSC_NULL,"Browser path %s\n",path);
    ierr = PetscStrcmp(path,"/ams-tree",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscAMSDisplayTree(fd);CHKERRQ(ierr);
      goto theend;
    }
#endif
    ierr = PetscStrcpy(fullpath,"${PETSC_DIR}/include/web");CHKERRQ(ierr);
    ierr = PetscStrcat(fullpath,path);CHKERRQ(ierr);
    ierr = PetscInfo1(PETSC_NULL,"Checking for file %s\n",fullpath);CHKERRQ(ierr);
    ierr = PetscStrreplace(PETSC_COMM_SELF,fullpath,truefullpath,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
    fdo  = fopen(truefullpath,"r");
    if (fdo) {
      PetscInt    length,index;
      char        data[4096];
      struct stat statbuf;
      int         n;
      const char  *suffixes[] = {".html",".js",".gif",0}, *mimes[] = {"text/html","text/javascript","image/gif","text/unknown"};

      ierr = PetscStrendswithwhich(fullpath,suffixes,&index);CHKERRQ(ierr);
      type = mimes[index];
      if (!stat(truefullpath, &statbuf)) length = -1;
      else length = S_ISREG(statbuf.st_mode) ? statbuf.st_size : -1;
      ierr = PetscWebSendHeader(fd, 200, "OK", NULL, type, length);CHKERRQ(ierr);
      while ((n = fread(data, 1, sizeof(data), fdo)) > 0) fwrite(data, 1, n, fd);
      fclose(fdo);
      ierr = PetscInfo2(PETSC_NULL,"Sent file %s to browser using format %s\n",fullpath,type);CHKERRQ(ierr);
      goto theend;
    }
    ierr = PetscWebSendError(fd, 501, "Not supported", NULL, "Unknown request.");CHKERRQ(ierr);
  }
  theend:
  ierr = PetscTokenDestroy(&tok);CHKERRQ(ierr);
  fclose(fd);
  ierr = PetscInfo(PETSC_NULL,"Finished processing request\n");CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscWebServeWait"
/*@C
      PetscWebServeWait - waits for requests on a thread

    Not collective

  Input Parameter:
.   port - port to listen on

    Level: developer

.seealso: PetscViewerSocketOpen(), PetscWebServe()
@*/
void  *PetscWebServeWait(int *port)
{
  PetscErrorCode ierr;
  int            iport,listenport,tport = *port;

  ierr = PetscInfo1(PETSC_NULL,"Starting webserver at port %d\n",tport);if (ierr) return 0;
  ierr = PetscFree(port);if (ierr) return 0;
  ierr = PetscSocketEstablish(tport,&listenport);if (ierr) return 0;
  while (1) {
    ierr = PetscSocketListen(listenport,&iport);if (ierr) return 0;
    ierr = PetscWebServeRequest(iport);if (ierr) return 0;
    close(iport);
  }
  close(listenport);
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "PetscWebServe"
/*@C
      PetscWebServe - start up the PETSc web server and respond to requests

    Not collective - only does something on process zero of the communicator

  Input Parameters:
+   comm - the MPI communicator
-   port - port to listen on

  Options Database Key:
+  -server <port> - start PETSc webserver (default port is 8080)
-  -ams_publish_objects


   Notes: Point your browser to http://hostname:8080   to access the PETSc web server, where hostname is the name of your machine.
      If you are running PETSc on your local machine you can use http://localhost:8080

      If the PETSc program completes before you connect with the browser you will not be able to connect to the PETSc webserver.

      Read the top of $PETSC_DIR/include/web/AMSJavascript.py before running.

    Level: developer

.seealso: PetscViewerSocketOpen()
@*/
PetscErrorCode  PetscWebServe(MPI_Comm comm,int port)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  pthread_t      thread;
  int            *trueport;

  PetscFunctionBegin;
  if (port < 1 && port != PETSC_DEFAULT && port != PETSC_DECIDE) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Cannot use negative port number %d",port);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (rank) PetscFunctionReturn(0);

  if (port == PETSC_DECIDE || port == PETSC_DEFAULT) port = 8080;
  ierr = PetscMalloc(1*sizeof(int),&trueport);CHKERRQ(ierr); /* malloc this so it still exists in thread */
  *trueport = port;
  ierr = pthread_create(&thread, NULL, (void *(*)(void *))PetscWebServeWait, trueport);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif














