/*$Id: receivesparse.c,v 1.16 2001/03/23 23:19:53 balay Exp $*/
/*
    Part of the MatlabSockettool Package. Receive a sparse matrix
  at a socket address, called by the receive.mex4 Matlab program.

        Written by Barry Smith, bsmith@mcs.anl.gov 4/14/92

   Since this is called from Matlab it cannot be compiled with C++.
*/
#include <stdio.h>
#include "petscsys.h"
#include "mex.h"

#define ERROR(a) {fprintf(stdout,"RECEIVE: %s \n",a); return -1;}
#undef __FUNCT__  
#define __FUNCT__ "ReceiveSparseMatrix"
int ReceiveSparseMatrix(mxArray *plhs[],int t)
{
  int    *tr,*tc,compx = 0;
  int    *r,*c;
  int    i,j,m,n,nnz,lnnz,jstart,jend,off = 0;
  double *tv,*v,*diag,*vi;

  /* get size of matrix */
  if (PetscBinaryRead(t,&m,1,PETSC_INT))   ERROR("reading number columns"); 
  if (PetscBinaryRead(t,&n,1,PETSC_INT))   ERROR("reading number rows"); 
  /* get number of nonzeros */
  if (PetscBinaryRead(t,&nnz,1,PETSC_INT))   ERROR("reading nnz"); 
  if (PetscBinaryRead(t,&compx,1,PETSC_INT))   ERROR("reading row lengths"); 
  /* Create a matrix for Matlab */
  /* since Matlab stores by columns not rows we actually will 
     create transpose of desired matrix */
  plhs[0] = mxCreateSparse(n,m,nnz,compx);
  r = mxGetIr(plhs[0]);
  c = mxGetJc(plhs[0]);
  v = mxGetPr(plhs[0]);
  /* Matlab sparse matrix pointers start at 0 not 1 */
  if (!compx) {
    if (PetscBinaryRead(t,v,nnz,PETSC_DOUBLE)) ERROR("reading values");
  } else {
    for (i=0; i<nnz; i++) {
      vi = mxGetPi(plhs[0]);
      if (PetscBinaryRead(t,v+i,1,PETSC_DOUBLE)) ERROR("reading values");
      if (PetscBinaryRead(t,vi+i,1,PETSC_DOUBLE)) ERROR("reading values");
    }
  }
  if (PetscBinaryRead(t,c,m+1,PETSC_INT)) ERROR("reading column pointers");
  if (PetscBinaryRead(t,r,nnz,PETSC_INT)) ERROR("reading row pointers");
  return 0;
}







