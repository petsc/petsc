#ifndef lint
static char vcid[] = "$Id: sendsparse.c,v 1.6 1995/04/26 19:30:51 curfman Exp bsmith $";
#endif
/* This is part of the MatlabSockettool package. Here are the routines
   to send a sparse matrix to Matlab.


        Written by Barry Smith, bsmith@mcs.anl.gov 4/14/92
*/
#include <stdio.h>
#include "matlab.h"

/*--------------------------------------------------------------*/
/*
   ViewerMatlabPutSparse_Private - Passes a sparse matrix in AIJ format
             to a Matlab viewer. 

   Input Parameters:
.  viewer - obtained from ViewerMatlabOpen()
.  m, n - number of rows and columns of matrix
.  nnz - number of nonzeros in matrix
.  v - the nonzero entries
.  r - the row pointers (m + 1 of them)
.  c - the column pointers (nnz of them)

   Notes:
   Most users should not call this routine, but instead should employ
$     MatView(Mat matrix,Viewer viewer)

   Notes for Advanced Users:
   ViewerMatlabPutSparse_Private() actually passes the matrix transpose, since 
   Matlab prefers column oriented storage.

.keywords: Viewer, Matlab, put, sparse, AIJ

.seealso: ViewerMatlabOpen(), MatView()
*/
int ViewerMatlabPutSparse_Private(Viewer viewer,int m,int n,int nnz,Scalar *v,int *r,
                        int *c)
{
  int t = viewer->port,type = SPARSEREAL,one = 1,zero = 0;
  if (write_int(t,&type,1))       SETERR(1,"writing type");
  if (write_int(t,&m,1))          SETERR(1,"writing number rows");
  if (write_int(t,&n,1))          SETERR(1,"writing number columns");
  if (write_int(t,&nnz,1))        SETERR(1,"writing number nonzeros");
#if !defined(PETSC_COMPLEX)
  if (write_int(t,&zero,1))        SETERR(1,"writing complex        ");
  if (write_double(t,v,nnz))      SETERR(1,"writing elements");
#else
  if (write_int(t,&one,1))        SETERR(1,"writing complex        ");  
  if (write_double(t,(double*)v,2*nnz))      SETERR(1,"writing elements");
#endif
  if (write_int(t,r,m+1))         SETERR(1,"writing column pointers");
  if (write_int(t,c,nnz))         SETERR(1,"writing row pointers");
  return 0;
}

