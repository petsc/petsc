/*
 
    This is the equivalent of Matlab's fwrite() only on sockets instead of
   binary files.
*/

#include <stdio.h>
#include "petscsys.h"
#include "src/sys/viewer/impls/socket/socket.h"
#include "mex.h"

PetscErrorCode PetscBinaryWrite(int,void *p,int,PetscDataType,PetscTruth);

#define PETSC_MEX_ERROR(a) {fprintf(stdout,"sread: %s \n",a); return ;}
/*-----------------------------------------------------------------*/
/*                                                                 */
/*-----------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "mexFunction"
void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
  int            fd,cnt,dt;
  PetscErrorCode ierr;

  /* check output parameters */
  if (nrhs != 3) PETSC_MEX_ERROR("Receive requires three input arguments.");
  fd  = (int) mxGetScalar(prhs[0]);
  cnt = mxGetNumberOfElements(prhs[1]);
  dt  = (PetscDataType) mxGetScalar(prhs[2]);

int *t = mxGetPr(prhs[1]);
 printf("cnt %d dt %d values %d %d %d %d\n",cnt,dt,t[0],t[1],t[2],t[3]);
  if (dt == PETSC_DOUBLE) {
    ierr = PetscBinaryWrite(fd,mxGetPr(prhs[1]),cnt,dt,PETSC_FALSE);if (ierr) PETSC_MEX_ERROR("Unable to send double items.");
  } else if (dt == PETSC_INT) {
    ierr = PetscBinaryWrite(fd,mxGetPr(prhs[1]),cnt,dt,PETSC_FALSE);if (ierr) PETSC_MEX_ERROR("Unable to send int items.");
  } else if (dt == PETSC_CHAR) {
    char *tmp = (char*) mxMalloc((cnt+5)*sizeof(char));
    mxGetNChars(prhs[1],tmp,cnt+1);
    ierr = PetscBinaryWrite(fd,tmp,cnt,dt,PETSC_FALSE);if (ierr) PETSC_MEX_ERROR("Unable to send char items.");
    mxFree(tmp);
  } else {
    PETSC_MEX_ERROR("Unknown datatype.");
  }
  return;
}















