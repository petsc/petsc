/* This is part of the MatlabSockettool package. 
 
        Written by Barry Smith, bsmith@mcs.anl.gov 4/14/92
*/

#if defined(PARCH_rs6000)
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
  int t = 0;
  Linger linger;
  linger.onoff = 1;
  linger.time  = 0;

  if (nrhs == 0) ERROR("Needs one argument, the port");
  t = (int) *mxGetPr(prhs[0]);

  if (setsockopt(t,SOL_SOCKET,SO_LINGER,&linger,sizeof(Linger))) 
    ERROR("Setting linger");
  if (close(t)) ERROR("closing socket");
#if !defined(PARCH_IRIX) 
  usleep((unsigned) 100);
#endif
  return;
}
