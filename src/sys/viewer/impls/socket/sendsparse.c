#define PETSC_DLL

#include "src/sys/src/viewer/impls/socket/socket.h"

/*--------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerSocketPutSparse_Private" 
/*
   PetscViewerSocketPutSparse_Private - Passes a sparse matrix in AIJ format
             to a Socket PetscViewer. 

   Input Parameters:
+  vw - obtained from PetscViewerSocketOpen()
.  m - number of rows of matrix
.  m - number of columns of matrix
.  nnz - number of nonzeros in matrix
.  v - the nonzero entries
.  r - the row pointers (m + 1 of them)
-  c - the column pointers (nnz of them)

    Level: developer

   Notes:
   Most users should not call this routine, but instead should employ
$     MatView(Mat matrix,PetscViewer viewer)

   Notes for Advanced Users:
   PetscViewerSocketPutSparse_Private() actually passes the matrix transpose, since 
   Matlab prefers column oriented storage.

   Concepts: Matlab^sending data, sparse matrices
   Concepts: Sockets^sending data, sparse matrices

.seealso: PetscViewerSocketOpen(), MatView()
*/
PetscErrorCode PetscViewerSocketPutSparse_Private(PetscViewer vw,PetscInt m,PetscInt n,PetscInt nnz,PetscScalar *v,PetscInt *r,PetscInt *c)
{
  PetscViewer_Socket *vmatlab = (PetscViewer_Socket*)vw->data;
  PetscErrorCode     ierr;
  int                t = vmatlab->port,type = SPARSEREAL,value;

  PetscFunctionBegin;
  ierr = PetscBinaryWrite(t,&type,1,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,&m,1,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,&n,1,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,&nnz,1,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  value = 0;
#else
  value = 1;
#endif
  ierr = PetscBinaryWrite(t,&value,1,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,v,nnz,PETSC_REAL,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,r,m+1,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,c,nnz,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}







