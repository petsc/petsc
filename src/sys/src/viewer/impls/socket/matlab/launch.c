#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: launch.c,v 1.6 1999/01/12 23:17:16 bsmith Exp bsmith $";
#endif
/* 
  Usage: A = launch(programname,number processors); 
 
*/

#include <stdio.h>
#include <errno.h>
extern int fork();
extern int system(const char *);
#include "src/sys/src/viewer/impls/socket/socket.h"
#include "mex.h"
#define ERROR(a) {fprintf(stderr,"LAUNCH: %s \n",a); return ;}
/*-----------------------------------------------------------------*/
/*                                                                 */
/*-----------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "mexFunction"
void mexFunction(int nlhs, Matrix *plhs[], int nrhs, Matrix *prhs[])
{
  int  np, child;
  char program[1024],executable[1024];

  if (nlhs == 1) {
    plhs[0]  = mxCreateFull(1, 1, 0);
    *mxGetPr(plhs[0]) = 1;
  }

  /* check output parameters */
  if (nlhs > 1) ERROR("Open requires at most one output argument.");
  if (nrhs == 0) ERROR("Open requires at least one input argument.");
  if (!mxIsString(prhs[0])) ERROR("First arg must be string.");

  if (nrhs == 1) np = 1;  
  else np = (int) *mxGetPr(prhs[1]);

  /* attempt a fork */
  child = fork();
  if (child < 0) ERROR("Unable to fork.");

  if (child) {
    ; /* I am parent, simply return */
  } else {
    mxGetString(prhs[0],program,1000);
    sprintf(executable,"mpirun -np %d %s",np,program);
    printf("About to execute %s\n",executable);
    system(executable);
    printf("Completed subprocess\n");
    fflush(stdout);
    exit(0);
  }
    

 
  if (nlhs == 1) {
    *mxGetPr(plhs[0]) = 0;
  }
  return;
}

    
 

