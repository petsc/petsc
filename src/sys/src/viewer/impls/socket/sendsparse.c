#ifndef lint
static char vcid[] = "$Id: sendsparse.c,v 1.11 1995/07/20 04:00:08 bsmith Exp bsmith $";
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
.  vw - obtained from ViewerMatlabOpen()
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
int ViewerMatlabPutSparse_Private(Viewer vw,int m,int n,int nnz,Scalar *v,int *r,int *c)
{
  int ierr,t = vw->port,type = SPARSEREAL,value;
  ierr = SYWrite(t,&type,1,SYINT,0); CHKERRQ(ierr);
  ierr = SYWrite(t,&m,1,SYINT,0); CHKERRQ(ierr);
  ierr = SYWrite(t,&n,1,SYINT,0); CHKERRQ(ierr);
  ierr = SYWrite(t,&nnz,1,SYINT,0); CHKERRQ(ierr);
#if !defined(PETSC_COMPLEX)
  value = 0;
#else
  value = 1;
#endif
  ierr = SYWrite(t,&value,1,SYINT,0); CHKERRQ(ierr);
  ierr = SYWrite(t,v,nnz,SYDOUBLE,0); CHKERRQ(ierr);
  ierr = SYWrite(t,r,m+1,SYINT,0); CHKERRQ(ierr);
  ierr = SYWrite(t,c,nnz,SYINT,0); CHKERRQ(ierr);
  return 0;
}

