/* 
  Usage: A = launch(programname,number processors); 
  Modified Sept 28, 2003 RFK: updated obsolete mx functions.
*/

#include <stdio.h>
#include <errno.h>
extern int fork();
extern int system(const char *);
#include "src/sys/src/viewer/impls/socket/socket.h"
#include "mex.h"
#define PETSC_MEX_ERROR(a) {fprintf(stdout,"LAUNCH: %s \n",a); return ;}
/*-----------------------------------------------------------------*/
/*                                                                 */
/*-----------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "mexFunction"
void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
  int  np,child;
  char program[PETSC_MAX_PATH_LEN],executable[PETSC_MAX_PATH_LEN];

  if (nlhs == 1) {
    plhs[0]  = mxCreateDoubleMatrix(1,1,mxREAL);
    *mxGetPr(plhs[0]) = 1;
  }

  /* check output parameters */
  if (nlhs > 1) PETSC_MEX_ERROR("Open requires at most one output argument.");
  if (!nrhs) PETSC_MEX_ERROR("Open requires at least one input argument.");
  if (!mxIsChar(prhs[0])) PETSC_MEX_ERROR("First arg must be string.");

  if (nrhs == 1) np = 1;  
  else           np = (int)*mxGetPr(prhs[1]);

  /* attempt a fork */
  child = fork();
  if (child < 0) {
    PETSC_MEX_ERROR("Unable to fork.");
  } else if (!child) {  /* I am child, start up MPI program */
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

    
 

