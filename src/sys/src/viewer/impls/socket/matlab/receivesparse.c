/*
    Part of the MatlabSockettool Package. Receive a sparse matrix
  at a socket address, called by the receive.mex4 Matlab program.

        Written by Barry Smith, bsmith@mcs.anl.gov 4/14/92
*/
#include <stdio.h>
#include <math.h>
#include "mex.h"


#define ERROR(a) {fprintf(stderr,"RECEIVE: %s \n",a); return -1;}
int ReceiveSparseMatrix(Matrix *plhs[],int t)
{
  int    *tr,*tc;
  int    *r, *c;
  int    i,j,m,n, nnz, lnnz, jstart,jend,off = 0;
  double *tv, *v, *diag;

  /* get size of matrix */
  if (read_int(t,&m,1))   ERROR("reading number columns"); 
  if (read_int(t,&n,1))   ERROR("reading number rows"); 
  /* get number of nonzeros */
  if (read_int(t,&nnz,1))   ERROR("reading nnz"); 
  /* Create a matrix for Matlab */
  /* since Matlab stores by columns not rows we actually will 
     create transpose of desired matrix */
  plhs[0] = mxCreateSparse(n,m, nnz, REAL);
  r = mxGetIr(plhs[0]);
  c = mxGetJc(plhs[0]);
  v = mxGetPr(plhs[0]);
  if (read_double(t,v,nnz)) ERROR("reading offdiag");
  if (read_int(t,c,m+1)) ERROR("reading column pointers");
  if (read_int(t,r,nnz)) ERROR("reading row pointers");
  /* pointers start at 0 not 1 */
  for ( i=0; i<m+1; i++ ) {c[i]--;}
  for ( i=0; i<nnz; i++ ) {r[i]--;}
  return 0;
}

