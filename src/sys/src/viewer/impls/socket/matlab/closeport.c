/* This was part of the MatlabSockettool package. 
 
        Written by Barry Smith, bsmith@mcs.anl.gov 4/14/92
	 Updated by Ridhard Katz, katz@ldeo.columbia.edu 9/28/03
*/

#include "petsc.h"
#if defined(PETSC_NEEDS_UTYPE_TYPEDEFS)
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
#include "src/sys/src/viewer/impls/socket/socket.h"
#include "mex.h"
#define PETSC_MEX_ERROR(a) {fprintf(stdout,"CLOSEPORT: %s \n",a); return ;}
typedef struct { int onoff; int time; } Linger;
/*-----------------------------------------------------------------*/
/*                                                                 */
/*-----------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "mexFunction"
void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
  int    t = 0;
  Linger linger;

  linger.onoff = 1;
  linger.time  = 0; 

  if (!nrhs) PETSC_MEX_ERROR("Needs one argument, the port");
  t = (int)*mxGetPr(prhs[0]);

  if (setsockopt(t,SOL_SOCKET,SO_LINGER,(char*)&linger,sizeof(Linger))) 
    PETSC_MEX_ERROR("Setting linger");
  if (close(t)) PETSC_MEX_ERROR("closing socket");
  return;
}
