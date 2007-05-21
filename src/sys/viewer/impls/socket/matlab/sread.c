/*
 
    This is the equivalent of Matlab's fread() only on sockets instead of
   binary files.
*/

#include <stdio.h>
#include "petscsys.h"
#include "src/sys/viewer/impls/socket/socket.h"
#include "mex.h"

PetscErrorCode PetscBinaryRead(int,void *p,int,PetscDataType);

#define PETSC_MEX_ERROR(a) {fprintf(stdout,"sread: %s \n",a); return ;}
/*-----------------------------------------------------------------*/
/*                                                                 */
/*-----------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "mexFunction"
void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
  int  fd,cnt,dt;

  /* check output parameters */
  if (nlhs != 1) PETSC_MEX_ERROR("Receive requires one output argument.");

  if (nrhs != 3) PETSC_MEX_ERROR("Receive requires three input arguments.");
  fd  = *(int*)mxGetPr(prhs[0]);
  cnt = *(int*)mxGetPr(prhs[1]);
  dt  = *(PetscDataType*)mxGetPr(prhs[2]);

  plhs[0]  = mxCreateDoubleMatrix(1,cnt,mxREAL);
  ierr = PetscBinaryRead(fd,mxGetPr(plhs[0]),cnt,dt);if (ierr) PETSC_MEX_ERROR("Unable to receive %d items.",cnt);
  return;
}















