/*$Id: openport.c,v 1.24 2001/03/23 23:19:53 balay Exp $*/
/* 
  Usage: A = openport(portnumber);  [ 5000 < portnumber < 5010 ]
 
        Written by Barry Smith, bsmith@mcs.anl.gov 4/14/92
	 Updated by Ridhard Katz, katz@ldeo.columbia.edu 9/28/03

   This code has not been tested on all machines, the function prototypes may not
exist for certain systems. Only compiles as C code.
*/

#include "petsc.h"
#include "petscsys.h"

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
#if defined(PETSC_HAVE_SYS_UTSNAME_H)
#include <sys/utsname.h>
#endif
#if defined(PETSC_HAVE_STRINGS_H)
#include <strings.h>
#endif

#include "src/sys/src/viewer/impls/socket/socket.h"
#include "petscfix.h"
#include "mex.h"

EXTERN int SOCKConnect_Private(int);
#define ERROR(a) {fprintf(stdout,"OPENPORT: %s \n",a); return ;}
/*-----------------------------------------------------------------*/
/*                                                                 */
/*-----------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "mexFunction"
void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
  int t,portnumber;

  /* check output parameters */
  if (nlhs != 1) ERROR("Open requires one output argument.");

  /* figure out portnumber user wants to use; default to 5005 */
  if (!nrhs) {
    char *str;
    str = getenv("PETSC_VIEWER_SOCKET_PORT");
    if (str) portnumber = atoi(str);
    else portnumber = DEFAULTPORT;  
  } else {
    portnumber = (int)*mxGetPr(prhs[0]);
  }

  /* open connection */
  t = SOCKConnect_Private(portnumber); if (t == -1)  ERROR("opening socket");

  plhs[0]  = mxCreateDoubleMatrix(1,1,mxREAL);
 
  *mxGetPr(plhs[0]) = t;
  return;
}

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
       fprintf(stdout,"RECEIVE: unable to establish port\n");
       return -1;
  }

/* wait for someone to try to connect */
  i = sizeof(struct sockaddr_in);
  if ((t = accept(listenport,(struct sockaddr *)&isa,(socklen_t *)&i)) < 0) {
     fprintf(stdout,"RECEIVE: error from accept\n");
     return(-1);
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
  int                s,ierr;
  struct sockaddr_in sa;  
  struct hostent     *hp;
  struct utsname     utname;

  /* Note we do not use gethostname since that is not POSIX */
  uname(&utname); strncpy(myname,utname.nodename,MAXHOSTNAME);
  bzero(&sa,sizeof(struct sockaddr_in));
  hp = gethostbyname(myname);
  if (hp == NULL) {
     fprintf(stdout,"RECEIVE: error from gethostbyname\n");
     return(-1);
  }

  sa.sin_family = hp->h_addrtype; 
  sa.sin_port = htons(portnum); 

  if ((s = socket(AF_INET,SOCK_STREAM,0)) < 0) {
     fprintf(stdout,"RECEIVE: error from socket\n");
     return(-1);
  }
  {
  int optval = 1; /* Turn on the option */
  ierr = setsockopt(s,SOL_SOCKET,SO_REUSEADDR,(char *)&optval,sizeof(optval));
  }

  while (bind(s,(struct sockaddr*)&sa,sizeof(sa)) < 0) {
     if (errno != EADDRINUSE) { 
        close(s);
        fprintf(stdout,"RECEIVE: error from bind\n");
        return(-1);
     }
     close(listenport); 
  }
  listen(s,0);
  return(s);
}
#endif    
 
