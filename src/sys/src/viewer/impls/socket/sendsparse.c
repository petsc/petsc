#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: sendsparse.c,v 1.25 1999/05/04 20:27:46 balay Exp bsmith $";
#endif

#include "src/sys/src/viewer/impls/socket/socket.h"

/*--------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "ViewerSocketPutSparse_Private"
/*
   ViewerSocketPutSparse_Private - Passes a sparse matrix in AIJ format
             to a Socket viewer. 

   Input Parameters:
.  vw - obtained from ViewerSocketOpen()
.  m, n - number of rows and columns of matrix
.  nnz - number of nonzeros in matrix
.  v - the nonzero entries
.  r - the row pointers (m + 1 of them)
.  c - the column pointers (nnz of them)

   Notes:
   Most users should not call this routine, but instead should employ
$     MatView(Mat matrix,Viewer viewer)

   Notes for Advanced Users:
   ViewerSocketPutSparse_Private() actually passes the matrix transpose, since 
   Matlab prefers column oriented storage.

.keywords: Viewer, Socket, put, sparse, AIJ

.seealso: ViewerSocketOpen(), MatView()
*/
int ViewerSocketPutSparse_Private(Viewer vw,int m,int n,int nnz,Scalar *v,int *r,int *c)
{
  Viewer_Socket *vmatlab = (Viewer_Socket *) vw->data;
  int           ierr,t = vmatlab->port,type = SPARSEREAL,value;

  PetscFunctionBegin;
  ierr = PetscBinaryWrite(t,&type,1,PETSC_INT,0);CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,&m,1,PETSC_INT,0);CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,&n,1,PETSC_INT,0);CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,&nnz,1,PETSC_INT,0);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  value = 0;
#else
  value = 1;
#endif
  ierr = PetscBinaryWrite(t,&value,1,PETSC_INT,0);CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,v,nnz,PETSC_DOUBLE,0);CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,r,m+1,PETSC_INT,0);CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,c,nnz,PETSC_INT,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}







