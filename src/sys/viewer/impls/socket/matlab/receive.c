/*
 
  This is a MATLAB Mex program which waits at a particular 
  portnumber until a matrix arrives,it then returns to 
  matlab with that matrix.

  Usage: A = receive(portnumber);  portnumber obtained with openport();
 
        Written by Barry Smith, bsmith@mcs.anl.gov 4/14/92
	 Updated by Ridhard Katz, katz@ldeo.columbia.edu 9/28/03

  Since this is called from Matlab it cannot be compiled with C++.
*/

#include <stdio.h>
#include "petscsys.h"
#include "src/sys/viewer/impls/socket/socket.h"
#include "mex.h"
EXTERN PetscErrorCode ReceiveSparseMatrix(mxArray **,int);
EXTERN PetscErrorCode ReceiveDenseIntMatrix(mxArray **,int);
EXTERN PetscErrorCode ReceiveDenseMatrix(mxArray **,int);

#define PETSC_MEX_ERROR(a) {fprintf(stdout,"RECEIVE: %s \n",a); return ;}
/*-----------------------------------------------------------------*/
/*                                                                 */
/*-----------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "mexFunction"
void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
 int    type,t;

  /* check output parameters */
  if (nlhs != 1) PETSC_MEX_ERROR("Receive requires one output argument.");

  if (!nrhs) PETSC_MEX_ERROR("Receive requires one input argument.");
  t = (int)*mxGetPr(prhs[0]);

  /* get type of matrix */
  if (PetscBinaryRead(t,&type,1,PETSC_INT))   PETSC_MEX_ERROR("reading type"); 

  if (type == DENSEREAL) ReceiveDenseMatrix(plhs,t);
  if (type == DENSEINT) ReceiveDenseIntMatrix(plhs,t);
  if (type == DENSECHARACTER) {
    if (ReceiveDenseMatrix(plhs,t)) return;
    /* mxSetDispMode(plhs[0],1); */
  }
  if (type == SPARSEREAL) ReceiveSparseMatrix(plhs,t); 
  return;
}















