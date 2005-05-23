/* 
   This is part of the MatlabSockettool Package. It is called by 
 the receive.mex4 Matlab program. 
  
        Written by Barry Smith, bsmith@mcs.anl.gov 4/14/92
	 Updated by Ridhard Katz, katz@ldeo.columbia.edu 9/28/03

  Since this is called from Matlab it cannot be compiled with C++.
*/

#undef __SDIR__
#define __SDIR__ "src/sys/viewer/impls/socket/matlab"

#include <stdio.h>
#include "petscsys.h"
#include "mex.h"
#define PETSC_MEX_ERROR(a) {fprintf(stdout,"RECEIVE %s \n",a); return -1;}
/*-----------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "ReceiveDenseMatrix"
PetscErrorCode ReceiveDenseMatrix(mxArray *plhs[],int t)
{
  int    m,n,compx = 0,i;
  
  /* get size of matrix */
  if (PetscBinaryRead(t,&m,1,PETSC_INT))   PETSC_MEX_ERROR("reading number columns"); 
  if (PetscBinaryRead(t,&n,1,PETSC_INT))   PETSC_MEX_ERROR("reading number rows"); 
  if (PetscBinaryRead(t,&compx,1,PETSC_INT))   PETSC_MEX_ERROR("reading if complex"); 
  
  /*allocate matrix */
  plhs[0]  = mxCreateDoubleMatrix(m,n,compx);
  /* read in matrix */
  if (!compx) {
    if (PetscBinaryRead(t,mxGetPr(plhs[0]),n*m,PETSC_DOUBLE)) PETSC_MEX_ERROR("read dense matrix");
  } else {
    for (i=0; i<n*m; i++) {
      if (PetscBinaryRead(t,mxGetPr(plhs[0])+i,1,PETSC_DOUBLE))PETSC_MEX_ERROR("read dense matrix");
      if (PetscBinaryRead(t,mxGetPi(plhs[0])+i,1,PETSC_DOUBLE))PETSC_MEX_ERROR("read dense matrix");
    }
  }
  return 0;
}

#undef __FUNCT__  
#define __FUNCT__ "ReceiveIntDenseMatrix"
PetscErrorCode ReceiveDenseIntMatrix(mxArray *plhs[],int t)
{
  int            m,compx = 0,i,*array;
  double         *values;
  PetscErrorCode ierr;
  
  /* get size of matrix */
  ierr = PetscBinaryRead(t,&m,1,PETSC_INT); if (ierr) PETSC_MEX_ERROR("reading number columns"); 
  
  /*allocate matrix */
  plhs[0] = mxCreateDoubleMatrix(m,1,mxREAL);

  /* read in matrix */
  array = (int*) malloc(m*sizeof(int)); if (!array) PETSC_MEX_ERROR("reading allocating space");
  ierr = PetscBinaryRead(t,array,m,PETSC_INT); if (ierr) PETSC_MEX_ERROR("read dense matrix");

  values = mxGetPr(plhs[0]);
  for (i =0; i<m; i++) {
    values[i] = array[i];
  }
  free(array);

  return 0;
}
    
 
