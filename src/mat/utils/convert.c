#ifndef lint
static char vcid[] = "$Id: convert.c,v 1.52 1996/08/15 12:48:42 bsmith Exp bsmith $";
#endif

#include "src/mat/impls/aij/mpi/mpiaij.h"
#include "src/mat/impls/bdiag/mpi/mpibdiag.h"

/* This file contains a generic conversion routine and implementation specific
   versions for increased efficiency. */

/* 
  MatConvert_Basic - Converts from any input format to another format. For
  parallel formats, the new matrix distribution is determined by PETSc.
 */
int MatConvert_Basic(Mat mat,MatType newtype,Mat *M)
{
  Scalar *vwork;
  int    ierr, i, nz, m, n, *cwork, rstart, rend,flg;

  ierr = MatGetSize(mat,&m,&n); CHKERRQ(ierr);
  if (newtype == MATSAME) newtype = (MatType)mat->type;
  switch (newtype) {
    case MATSEQAIJ:
      ierr = MatCreateSeqAIJ(mat->comm,m,n,0,PETSC_NULL,M); CHKERRQ(ierr); 
      break;
    case MATMPIROWBS:
      if (m != n) SETERRQ(1,"MatConvert:MATMPIROWBS matrix must be square");
      ierr = MatCreateMPIRowbs(mat->comm,PETSC_DECIDE,m,0,PETSC_NULL,
             PETSC_NULL,M); CHKERRQ(ierr);
      break;
    case MATMPIAIJ:
      ierr = MatCreateMPIAIJ(mat->comm,PETSC_DECIDE,PETSC_DECIDE,
             m,n,0,PETSC_NULL,0,PETSC_NULL,M); CHKERRQ(ierr);
      break;
    case MATSEQDENSE:
      ierr = MatCreateSeqDense(mat->comm,m,n,PETSC_NULL,M); CHKERRQ(ierr);
      break;
    case MATMPIDENSE:
      ierr = MatCreateMPIDense(mat->comm,PETSC_DECIDE,PETSC_DECIDE,
             m,n,PETSC_NULL,M); CHKERRQ(ierr);
      break;
    case MATSEQBDIAG:
      {
      int bs = 1; /* Default block size = 1 */ 
      ierr = OptionsGetInt(PETSC_NULL,"-mat_block_size",&bs,&flg);CHKERRQ(ierr);    
      ierr = MatCreateSeqBDiag(mat->comm,m,n,0,bs,PETSC_NULL,PETSC_NULL,M);CHKERRQ(ierr); 
      break;
      }
    case MATMPIBDIAG:
      {
      int bs = 1; /* Default block size = 1 */ 
      ierr = OptionsGetInt(PETSC_NULL,"-mat_block_size",&bs,&flg);CHKERRQ(ierr);   
      ierr = MatCreateMPIBDiag(mat->comm,PETSC_DECIDE,m,n,0,bs,PETSC_NULL,
             PETSC_NULL,M); CHKERRQ(ierr); 
      break;
      }
    default:
      SETERRQ(1,"MatConvert:Matrix type is not currently supported");
  }
  ierr = MatGetOwnershipRange(*M,&rstart,&rend); CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    ierr = MatGetRow(mat,i,&nz,&cwork,&vwork); CHKERRQ(ierr);
    ierr = MatSetValues(*M,1,&i,nz,cwork,vwork,INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatRestoreRow(mat,i,&nz,&cwork,&vwork); CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*M,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*M,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}
/* -------------------------------------------------------------- */
/* 
  MatConvert_SeqAIJ - Converts from MATSEQAIJ format to another format. For
  parallel formats, the new matrix distribution is determined by PETSc.
 */
int MatConvert_SeqAIJ(Mat A, MatType newtype, Mat *B)
{ 
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  Scalar     *vwork;
  int        i, ierr, nz, m = a->m, n = a->n, *cwork, rstart, rend,flg;

  switch (newtype) {
    case MATMPIROWBS:
      if (m != n) SETERRQ(1,"MatConvert_SeqAIJ:MATMPIROWBS matrix must be square");
      ierr = MatCreateMPIRowbs(A->comm,PETSC_DECIDE,m,0,PETSC_NULL,PETSC_NULL,B);
             CHKERRQ(ierr);
      break;
    case MATMPIAIJ:
      ierr = MatCreateMPIAIJ(A->comm,PETSC_DECIDE,PETSC_DECIDE,
             m,n,0,PETSC_NULL,0,PETSC_NULL,B); CHKERRQ(ierr);
      break;
    case MATSEQDENSE:
      ierr = MatCreateSeqDense(A->comm,m,n,PETSC_NULL,B); CHKERRQ(ierr);
      break;
    case MATMPIDENSE:
      ierr = MatCreateMPIDense(A->comm,PETSC_DECIDE,PETSC_DECIDE,
             m,n,PETSC_NULL,B); CHKERRQ(ierr);
      break;
    case MATSEQBDIAG:
      {
      int bs = 1; /* Default block size = 1 */ 
      ierr = OptionsGetInt(PETSC_NULL,"-mat_block_size",&bs,&flg);CHKERRQ(ierr);
      ierr = MatCreateSeqBDiag(A->comm,m,n,0,bs,PETSC_NULL,PETSC_NULL,B);
             CHKERRQ(ierr); 
      break;
      }
    case MATMPIBDIAG:
      {
      int bs = 1; /* Default block size = 1 */ 
      ierr = OptionsGetInt(PETSC_NULL,"-mat_block_size",&bs,&flg);CHKERRQ(ierr);
             CHKERRQ(ierr);     
      ierr = MatCreateMPIBDiag(A->comm,PETSC_DECIDE,m,n,0,bs,PETSC_NULL,
             PETSC_NULL,B); CHKERRQ(ierr); 
      break;
      }
    default:
      SETERRQ(1,"MatConvert_SeqAIJ:Matrix type is not currently supported");
  }
  ierr = MatGetOwnershipRange(*B,&rstart,&rend); CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    ierr = MatGetRow(A,i,&nz,&cwork,&vwork); CHKERRQ(ierr);
    ierr = MatSetValues(*B,1,&i,nz,cwork,vwork,INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatRestoreRow(A,i,&nz,&cwork,&vwork); CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}
/* ------------------------------------------------------------------ */
/* 
  MatConvert_MPIAIJ - Converts from MATMPIAIJ format to another
  parallel format.
 */
int MatConvert_MPIAIJ(Mat A, MatType newtype, Mat *B)
{
  SETERRQ(1,"MatConvert_MPIAIJ:Not currently suported");

  /* Each processor converts its local rows */
/* ----------------------------------------------------
  Mat_MPIAIJ *a = (Mat_MPIAIJ *) A->data;
  int        ierr, nz, i, ig, rstart = a->rstart, m = a->m, *cwork;
  Scalar     *vwork;

  for (i=0; i<m; i++) {
    ig   = i + rstart;
    ierr = MatGetRow(A,ig,&nz,&cwork,&vwork); CHKERRQ(ierr);
    ierr = MatSetValues(*B,1,&ig,nz,cwork,vwork,INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatRestoreRow(A,ig,&nz,&cwork,&vwork); CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
---------------------------------------------------------*/
}
/* ------------------------------------------------------------------ */
/* 
  MatConvert_SeqBDiag - Converts from MATSEQBDiag format to another format. For
  parallel formats, the new matrix distribution is determined by PETSc.
 */
int MatConvert_SeqBDiag(Mat A, MatType newtype, Mat *B)
{ 
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  Scalar       *vwork, *vw2;
  int          i, ierr, nz, m = a->m, n = a->n, *cwork, rstart, rend;
  int          j, *cw2, ict;

  /* rough over-estimate; could refine for individual rows */
  nz = PetscMin(n,a->nd*a->bs); 
  switch (newtype) {
    case MATSEQAIJ:
      ierr = MatCreateSeqAIJ(A->comm,m,n,nz,PETSC_NULL,B); CHKERRQ(ierr); 
      break;
    case MATMPIROWBS:
      if (m != n) SETERRQ(1,"MatConvert_SeqBDiag:MATMPIROWBS matrix must be square");
      ierr = MatCreateMPIRowbs(A->comm,PETSC_DECIDE,m,0,PETSC_NULL,PETSC_NULL,
             B); CHKERRQ(ierr);
      break;
    case MATMPIAIJ:
      ierr = MatCreateMPIAIJ(A->comm,PETSC_DECIDE,PETSC_DECIDE,
                             m,n,0,PETSC_NULL,0,PETSC_NULL,B); CHKERRQ(ierr);
      break;
    case MATSEQDENSE:
      ierr = MatCreateSeqDense(A->comm,m,n,PETSC_NULL,B); CHKERRQ(ierr);
      break;
    case MATMPIDENSE:
      ierr = MatCreateMPIDense(A->comm,PETSC_DECIDE,PETSC_DECIDE,
                               m,n,PETSC_NULL,B); CHKERRQ(ierr);
      break;
    case MATMPIBDIAG:
      {
      ierr = MatCreateMPIBDiag(A->comm,PETSC_DECIDE,m,n,a->nd,a->bs,
             PETSC_NULL,PETSC_NULL,B); CHKERRQ(ierr); 
      break;
      }
    default:
      SETERRQ(1,"MatConvert_SeqBDiag:Matrix type is not currently supported");
  }
  ierr = MatGetOwnershipRange(*B,&rstart,&rend); CHKERRQ(ierr);

  cw2 = (int *)PetscMalloc( n * sizeof(int) ); CHKPTRQ(cw2);
  vw2 = (Scalar *)PetscMalloc( n * sizeof(Scalar) ); CHKPTRQ(vw2);
  for (i=rstart; i<rend; i++) {
   ierr = MatGetRow(A,i,&nz,&cwork,&vwork); CHKERRQ(ierr);
   ict = 0; /* strip out the zero elements ... is this what we really want? */
   for (j=0; j<nz; j++) {
     if (vwork[j] != 0) {vw2[ict] = vwork[j]; cw2[ict] = cwork[j]; ict++;}
   }
   if (ict) {
     ierr = MatSetValues(*B,1,&i,ict,cw2,vw2,INSERT_VALUES); CHKERRQ(ierr);
   }
   ierr = MatRestoreRow(A,i,&nz,&cwork,&vwork); CHKERRQ(ierr);
  }
  PetscFree(cw2); PetscFree(vw2);
  ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}
