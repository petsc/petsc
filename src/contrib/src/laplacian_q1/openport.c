#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: openport.c,v 1.6 1998/04/03 21:00:37 bsmith Exp bsmith $";
#endif
/* 
  Usage: A = openport(portnumber);  [ 5000 < portnumber < 5010 ]
 
        Written by Barry Smith, bsmith@mcs.anl.gov 4/14/92
*/

#if defined(PARCH_rs6000)
/* 
   Had trouble locating the right include file on IBM for these definitions
*/
typedef unsigned char   u_char;
typedef unsigned short  u_short;
typedef unsigned short  ushort;
typedef unsigned int    u_int;
typedef unsigned long   u_long;
#endif
#include <stdio.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <netinet/in.h>
#include <netdb.h>
#include <fcntl.h>
#include <stropts.h>
#include <sys/utsname.h>
#include <math.h>
#include "src/viewer/impls/matlab/matlab.h"
#include "mex.h"

extern int SOCKConnect_Private(int);
#define ERROR(a) {fprintf(stderr,"OPENPORT: %s \n",a); return ;}
/*-----------------------------------------------------------------*/
/*                                                                 */
/*-----------------------------------------------------------------*/
#undef __FUNCTION__  
#define __FUNCTION__ "mexFunction"
void mexFunction(int nlhs, Matrix *plhs[], int nrhs, Matrix *prhs[])
{
  int t, portnumber;

  /* check output parameters */
  if (nlhs != 1) ERROR("Open requires one output argument.");

  /* figure out portnumber user wants to use; default to 5005 */
  if (nrhs == 0) portnumber = DEFAULTPORT;  
  else portnumber = (int) *mxGetPr(prhs[0]);

  /* open connection */
  t = SOCKConnect_Private(portnumber); if (t == -1)  ERROR("opening socket");

  plhs[0]  = mxCreateFull(1, 1, 0);
 
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
int establish(u_short);
#undef __FUNCTION__  
#define __FUNCTION__ "SOCKConnect_Private"
int SOCKConnect_Private(int portnumber)
{
  struct sockaddr_in isa;
  int                i,t;

/* open port*/
  listenport = establish( (u_short) portnumber);
  if ( listenport == -1 ) {
       fprintf(stderr,"RECEIVE: unable to establish port\n");
       return -1;
  }

/* wait for someone to try to connect */
  i = sizeof(struct sockaddr_in);
  if ( (t = accept(listenport,&isa,&i)) < 0 ) {
     fprintf(stderr,"RECEIVE: error from accept\n");
     return(-1);
  }
  close(listenport);  
  return(t);
}
/*-----------------------------------------------------------------*/
#define MAXHOSTNAME 100
#undef __FUNCTION__  
#define __FUNCTION__ "establish"
int establish(u_short portnum)
{
  char               myname[MAXHOSTNAME+1];
  int                s;
  struct sockaddr_in sa;
  struct hostent     *hp;
  struct utsname utname;

  /* Note we do not use gethostname since that is not POSIX */
  uname(&utname); strncpy(myname,utname.nodename,MAXHOSTNAME);
  bzero(&sa,sizeof(struct sockaddr_in));
  hp = gethostbyname(myname);
  if ( hp == NULL ) {
     fprintf(stderr,"RECEIVE: error from gethostbyname\n");
     return(-1);
  }
  sa.sin_family = hp->h_addrtype;
  sa.sin_port = htons(portnum);
  if ( (s = socket(AF_INET,SOCK_STREAM,0)) < 0 ) {
     fprintf(stderr,"RECEIVE: error from socket\n");
     return(-1);
  }
  while ( bind(s,&sa,sizeof(sa) ) < 0 ) {
     if ( errno != EADDRINUSE ) { 
        close(s);
        fprintf(stderr,"RECEIVE: error from bind\n");
        return(-1);
     }
     close(listenport); 
  }
  listen(s,0);
  return(s);
}
    
 
