#ifndef lint
static char vcid[] = "$Id: sendsparse.c,v 1.10 1995/07/17 20:42:41 bsmith Exp bsmith $";
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
  int t = viewer->port,type = SPARSEREAL,value;
  if (SOCKWriteInt_Private(t,&type,1)) 
                                 SETERRQ(1,"ViewerMatlabPutSparse_Private");
  if (SOCKWriteInt_Private(t,&m,1))
                                  SETERRQ(1,"ViewerMatlabPutSparse_Private");
  if (SOCKWriteInt_Private(t,&n,1)) 
                                  SETERRQ(1,"ViewerMatlabPutSparse_Private");
  if (SOCKWriteInt_Private(t,&nnz,1))
                                SETERRQ(1,"ViewerMatlabPutSparse_Private");
#if !defined(PETSC_COMPLEX)
  value = 0;
  if (SOCKWriteInt_Private(t,&value,1))
                                SETERRQ(1,"ViewerMatlabPutSparse_Private");
  if (SOCKWriteDouble_Private(t,v,nnz)) 
                              SETERRQ(1,"ViewerMatlabPutSparse_Private");
#else
  value = 1;
  if (SOCKWriteInt_Private(t,&value,1)) 
                                SETERRQ(1,"ViewerMatlabPutSparse_Private");  
  if (SOCKWriteDouble_Private(t,(double*)v,2*nnz)) 
                             SETERRQ(1,"ViewerMatlabPutSparse_Private");
#endif
  if (SOCKWriteInt_Private(t,r,m+1))
                                 SETERRQ(1,"ViewerMatlabPutSparse_Private");
  if (SOCKWriteInt_Private(t,c,nnz))   
                                   SETERRQ(1,"ViewerMatlabPutSparse_Private");
  return 0;
}

