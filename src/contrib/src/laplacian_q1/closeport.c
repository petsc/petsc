#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: closeport.c,v 1.8 1997/09/09 15:15:15 curfman Exp $";
#endif
/* This is part of the MatlabSockettool package. 
 
        Written by Barry Smith, bsmith@mcs.anl.gov 4/14/92
*/

#if defined(NEED_UTYPE_TYPEDEFS)
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
#include "src/viewer/impls/matlab/matlab.h"
#include "mex.h"
#define ERROR(a) {fprintf(stderr,"CLOSEPORT: %s \n",a); return ;}
typedef struct { int onoff; int time; } Linger;
/*-----------------------------------------------------------------*/
/*                                                                 */
/*-----------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "mexFunction"
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
  return;
}
