/*

        Written by Barry Smith, bsmith@mcs.anl.gov 4/14/92
        Updated by Richard Katz, katz@ldeo.columbia.edu 9/28/03
*/

#include <petscsys.h>
#include <../src/sys/classes/viewer/impls/socket/socket.h>

#include <errno.h>
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

#if defined(PETSC_NEED_CLOSE_PROTO)
PETSC_EXTERN int close(int);
#endif

#include <mex.h>
#define PETSC_MEX_ERROR(a) {mexErrMsgTxt(a); return;}
typedef struct { int onoff; int time; } Linger;
/*-----------------------------------------------------------------*/
/*                                                                 */
/*-----------------------------------------------------------------*/
PETSC_EXTERN void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
  int    t = 0;
  Linger linger;

  linger.onoff = 1;
  linger.time  = 0;

  if (!nrhs) PETSC_MEX_ERROR("Needs one argument, the port");
  t = (int)*mxGetPr(prhs[0]);

  if (setsockopt(t,SOL_SOCKET,SO_LINGER,(char*)&linger,sizeof(Linger))) PETSC_MEX_ERROR("Setting linger");
  if (close(t)) PETSC_MEX_ERROR("closing socket");
  return;
}

int main(int argc, char **argv)
{
  return 0;
}

