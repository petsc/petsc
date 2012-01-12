/* 
  Usage: A = sopen(portnumber);  [ 5000 < portnumber < 5010 ]
 
        Written by Barry Smith, bsmith@mcs.anl.gov 4/14/92
	 Updated by Richard Katz, katz@ldeo.columbia.edu 9/28/03
	 Updated by Barry Smith, bsmith@mcs.anl.gov 8/11/06

 Similar to MATLAB's sopen() only does not take file name, instead optional
 port to listen at.

 Only compiles as C code.
*/

#include <petscsys.h>

#if defined(PETSC_NEEDS_UTYPE_TYPEDEFS)
/* Some systems have inconsistent include files that use but don't
   ensure that the following definitions are made */
typedef unsigned char   u_char;
typedef unsigned short  u_short;
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
#if defined (PETSC_HAVE_IO_H)
#include <io.h>
#endif
#if defined(PETSC_HAVE_SYS_UTSNAME_H)
#include <sys/utsname.h>
#endif
#if defined(PETSC_HAVE_STRINGS_H)
#include <strings.h>
#endif
#if defined(PETSC_HAVE_STRING_H)
#include <string.h>
#endif
#if defined(PETSC_HAVE_WINSOCK2_H)
#include <Winsock2.h>
#endif
#if defined(PETSC_HAVE_WS2TCPIP_H)
#include <Ws2tcpip.h>
#endif
#include <../src/sys/viewer/impls/socket/socket.h>
#include <mex.h>

#define PETSC_MEX_ERROR(a) {mexErrMsgTxt(a); return ;}
#define PETSC_MEX_ERRORQ(a) {mexErrMsgTxt(a); return -1;}

/*-----------------------------------------------------------------*/
/* The listenport variable is an ugly hack. If the user hits a         */
/* control c while we are listening then we stop listening         */
/* but do not close the listen. Therefore if we try to bind again  */
/* and get an address in use, close the listen which was left      */
/* hanging; the problem is if the user uses several portnumbers    */
/* and control c we may not be able to close the correct listener. */
static int listenport;
/*-----------------------------------------------------------------*/
extern int establish(u_short);
#undef __FUNCT__  
#define __FUNCT__ "SOCKConnect_Private"
int SOCKConnect_Private(int portnumber)
{
  struct sockaddr_in isa; 
#if defined(PETSC_HAVE_ACCEPT_SIZE_T)
  size_t             i;
#else
  int                i;
#endif
  int                t;

/* open port*/
  listenport = establish((u_short) portnumber);
  if (listenport == -1) {
    PETSC_MEX_ERRORQ("RECEIVE: unable to establish port\n");
  }

/* wait for someone to try to connect */
  i = sizeof(struct sockaddr_in);
  if ((t = accept(listenport,(struct sockaddr *)&isa,(socklen_t *)&i)) < 0) {
    PETSC_MEX_ERRORQ("RECEIVE: error from accept\n");
  }
  close(listenport);  
  return(t);
}
/*-----------------------------------------------------------------*/
#define MAXHOSTNAME 100
#undef __FUNCT__  
#define __FUNCT__ "establish"
int establish(u_short portnum)
{
  char               myname[MAXHOSTNAME+1];
  int                s;
  PetscErrorCode     ierr;
  struct sockaddr_in sa;  
  struct hostent     *hp;
#if defined(PETSC_HAVE_UNAME)
  struct utsname     utname;
#elif defined(PETSC_HAVE_GETCOMPUTERNAME)
  int                namelen=MAXHOSTNAME;
#endif

  /* Note we do not use gethostname since that is not POSIX */
#if defined(PETSC_HAVE_GETCOMPUTERNAME)
  GetComputerName((LPTSTR)myname,(LPDWORD)&namelen);
#elif defined(PETSC_HAVE_UNAME)
  uname(&utname);
  strncpy(myname,utname.nodename,MAXHOSTNAME);
#endif
#if defined(PETSC_HAVE_BZERO)
  bzero(&sa,sizeof(struct sockaddr_in));
#else
  memset(&sa,0,sizeof(struct sockaddr_in));
#endif
  hp = gethostbyname(myname);
  if (!hp) {
    PETSC_MEX_ERRORQ("RECEIVE: error from gethostbyname\n");
  }

  sa.sin_family = hp->h_addrtype; 
  sa.sin_port = htons(portnum); 

  if ((s = socket(AF_INET,SOCK_STREAM,0)) < 0) {
    PETSC_MEX_ERRORQ("RECEIVE: error from socket\n");
  }
  {
  int optval = 1; /* Turn on the option */
  ierr = setsockopt(s,SOL_SOCKET,SO_REUSEADDR,(char *)&optval,sizeof(optval));
  }

  while (bind(s,(struct sockaddr*)&sa,sizeof(sa)) < 0) {
#if defined(PETSC_HAVE_WSAGETLASTERROR)
    ierr = WSAGetLastError();
    if (ierr != WSAEADDRINUSE) {
#else
    if (errno != EADDRINUSE) { 
#endif
      close(s);
      PETSC_MEX_ERRORQ("RECEIVE: error from bind\n");
      return(-1);
    }
    close(listenport); 
  }
  listen(s,0);
  return(s);
}

/*-----------------------------------------------------------------*/
/*                                                                 */
/*-----------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "mexFunction"
void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
  int        t,portnumber;

  /* check output parameters */
  if (nlhs != 1) PETSC_MEX_ERROR("Open requires one output argument.");

  /* figure out portnumber user wants to use; default to 5005 */
  if (!nrhs) {
    char *str;
    str = getenv("PETSC_VIEWER_SOCKET_PORT");
    if (str) portnumber = atoi(str);
    else portnumber = PETSCSOCKETDEFAULTPORT;  
  } else {
    portnumber = (int)*mxGetPr(prhs[0]);
  }

  /* open connection */
  t = SOCKConnect_Private(portnumber); if (t == -1)  PETSC_MEX_ERROR("opening socket");

  plhs[0]  = mxCreateDoubleMatrix(1,1,mxREAL);
 
  *mxGetPr(plhs[0]) = t;
  return;
}
