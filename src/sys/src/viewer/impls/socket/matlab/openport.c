/* 
  Usage: A = openport(portnumber);  [ 5000 < portnumber < 5010 ]
 
        Written by Barry Smith, bsmith@mcs.anl.gov 4/14/92
*/

#include <stdio.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <netinet/in.h>
#include <netdb.h>
#include <fcntl.h>
#include <stropts.h>
#include <math.h>
#include "mex.h"
#include "matlab.h"
#define ERROR(a) {fprintf(stderr,"RECEIVE: %s \n",a); return ;}
typedef struct { int onoff; int time; } Linger;
/*-----------------------------------------------------------------*/
/*                                                                 */
/*-----------------------------------------------------------------*/
void mexFunction(int nlhs, Matrix *plhs[], int nrhs, Matrix *prhs[])
{
  int    type;
  static int t = 0;
  int    portnumber;
  Linger linger;
  linger.onoff = 1;
  linger.time  = 0;

  /* check output parameters */
  if (nlhs != 1) ERROR("Receive requires one output argument.");

  /* figure out portnumber user wants to use; default to 5001 */
  if (nrhs == 0) portnumber = 5001;  
  else portnumber = (int) *mxGetPr(prhs[0]);

   
  /* open connection */
  t = get_connection(portnumber); if (t == -1)  ERROR("opening socket");

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
int get_connection(int portnumber)
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
int establish(u_short portnum)
{
  char               myname[MAXHOSTNAME+1];
  int                s;
  struct sockaddr_in sa;
  struct hostent     *hp;

  bzero(&sa,sizeof(struct sockaddr_in));
  gethostname(myname,MAXHOSTNAME);
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
#if !defined(alliant)
     if ( errno != EADDRINUSE ) { 
        close(s);
        fprintf(stderr,"RECEIVE: error from bind\n");
        return(-1);
     }
     close(listenport); 
#endif
  }
  listen(s,0);
  return(s);
}
    
 
