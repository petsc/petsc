#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: sendsparse.c,v 1.21 1997/10/19 03:29:04 bsmith Exp bsmith $";
#endif

#include "src/viewer/impls/matlab/matlab.h"

/*--------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "ViewerMatlabPutSparse_Private"
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
  Viewer_Matlab *vmatlab = (Viewer_Matlab *) vw->data;
  int           ierr,t = vmatlab->port,type = SPARSEREAL,value;

  PetscFunctionBegin;
  ierr = PetscBinaryWrite(t,&type,1,PETSC_INT,0); CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,&m,1,PETSC_INT,0); CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,&n,1,PETSC_INT,0); CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,&nnz,1,PETSC_INT,0); CHKERRQ(ierr);
#if !defined(USE_PETSC_COMPLEX)
  value = 0;
#else
  value = 1;
#endif
  ierr = PetscBinaryWrite(t,&value,1,PETSC_INT,0); CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,v,nnz,PETSC_DOUBLE,0); CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,r,m+1,PETSC_INT,0); CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,c,nnz,PETSC_INT,0); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}







