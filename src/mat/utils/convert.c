#ifndef lint
static char vcid[] = "$Id: convert.c,v 1.33 1995/10/27 13:56:30 curfman Exp curfman $";
#endif

#include "mpiaij.h"
#include "mpibdiag.h"

/* This file contains a generic conversion routine and implementation specific
   versions for increased efficiency. */

/* 
  MatConvert_Basic - Converts from any input format to another format. For
  parallel formats, the new matrix distribution is determined by PETSc.
 */
int MatConvert_Basic(Mat mat,MatType newtype,Mat *M)
{
  Scalar *vwork;
  int    ierr, i, nz, m, n, *cwork, rstart, rend;
  ierr = MatGetSize(mat,&m,&n); CHKERRQ(ierr);
  if (newtype == MATSAME) newtype = mat->type;
  switch (newtype) {
    case MATSEQAIJ:
      ierr = MatCreateSeqAIJ(mat->comm,m,n,0,0,M); CHKERRQ(ierr); 
      break;
    case MATSEQROW:
      ierr = MatCreateSeqRow(mat->comm,m,n,0,0,M); CHKERRQ(ierr); 
      break;
    case MATMPIROW:
      ierr = MatCreateMPIRow(MPI_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,
                             m,n,0,0,0,0,M); CHKERRQ(ierr);
      break;
    case MATMPIROWBS:
      if (m != n) SETERRQ(1,"MatConvert:MATMPIROWBS matrix must be square");
      ierr = MatCreateMPIRowbs(MPI_COMM_WORLD,PETSC_DECIDE,m,0,0,0,M);
             CHKERRQ(ierr);
      break;
    case MATMPIAIJ:
      ierr = MatCreateMPIAIJ(MPI_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,
                             m,n,0,0,0,0,M); CHKERRQ(ierr);
      break;
    case MATSEQDENSE:
      ierr = MatCreateSeqDense(mat->comm,m,n,M); CHKERRQ(ierr);
      break;
    case MATMPIDENSE:
      ierr = MatCreateMPIDense(MPI_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,
                               m,n,M); CHKERRQ(ierr);
      break;
    case MATSEQBDIAG:
      {
      int nb = 1; /* Default block size = 1 */ 
      OptionsGetInt(0,"-mat_bdiag_bsize",&nb);     
      ierr = MatCreateSeqBDiag(mat->comm,m,n,0,nb,0,0,M); CHKERRQ(ierr); 
      break;
      }
    case MATMPIBDIAG:
      {
      int nb = 1; /* Default block size = 1 */ 
      OptionsGetInt(0,"-mat_bdiag_bsize",&nb);     
      ierr = MatCreateMPIBDiag(MPI_COMM_WORLD,PETSC_DECIDE,m,n,0,nb,0,0,M); 
      CHKERRQ(ierr); 
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
  ierr = MatAssemblyBegin(*M,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*M,FINAL_ASSEMBLY); CHKERRQ(ierr);
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
  int        i, ierr, nz, m = a->m, n = a->n, *cwork, rstart, rend;

  switch (newtype) {
    case MATSEQROW:
      ierr = MatCreateSeqRow(A->comm,m,n,0,a->ilen,B); CHKERRQ(ierr); 
      break;
    case MATMPIROW:
      ierr = MatCreateMPIRow(MPI_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,
                             m,n,0,0,0,0,B); CHKERRQ(ierr);
      break;
    case MATMPIROWBS:
      if (m != n) SETERRQ(1,"MatConvert_SeqAIJ:MATMPIROWBS matrix must be square");
      ierr = MatCreateMPIRowbs(MPI_COMM_WORLD,PETSC_DECIDE,m,0,0,0,B); CHKERRQ(ierr);
      break;
    case MATMPIAIJ:
      ierr = MatCreateMPIAIJ(MPI_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,
                             m,n,0,0,0,0,B); CHKERRQ(ierr);
      break;
    case MATSEQDENSE:
      ierr = MatCreateSeqDense(A->comm,m,n,B); CHKERRQ(ierr);
      break;
    case MATMPIDENSE:
      ierr = MatCreateMPIDense(MPI_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,
                               m,n,B); CHKERRQ(ierr);
      break;
    case MATSEQBDIAG:
      {
      int nb = 1; /* Default block size = 1 */ 
      OptionsGetInt(0,"-mat_bdiag_bsize",&nb);     
      ierr = MatCreateSeqBDiag(A->comm,m,n,0,nb,0,0,B); CHKERRQ(ierr); 
      break;
      }
    case MATMPIBDIAG:
      {
      int nb = 1; /* Default block size = 1 */ 
      OptionsGetInt(0,"-mat_bdiag_bsize",&nb);     
      ierr = MatCreateMPIBDiag(MPI_COMM_WORLD,PETSC_DECIDE,m,n,0,nb,0,0,B); 
      CHKERRQ(ierr); 
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
  ierr = MatAssemblyBegin(*B,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*B,FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}
/* ------------------------------------------------------------------ */
/* 
  MatConvert_MPIAIJ - Converts from MATMPIAIJ format to another
  parallel format.
 */
int MatConvert_MPIAIJ(Mat A, MatType newtype, Mat *B)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *) A->data;
  Mat_SeqAIJ *Ad = (Mat_SeqAIJ *)(a->A->data), *Bd = (Mat_SeqAIJ *)(a->B->data);
  int        ierr, nz, i, ig, rstart = a->rstart, m = a->m, *cwork;
  Scalar     *vwork;

  switch (newtype) {
    case MATMPIROW:
      ierr = MatCreateMPIRow(A->comm,m,a->n,a->M,a->N,0,Ad->ilen,
			0,Bd->ilen,B); CHKERRQ(ierr);
      break;
    default:
      SETERRQ(1,"MatConvert_MPIAIJ:Only MATMPIROW is currently suported");
  }
  /* Each processor converts its local rows */
  for (i=0; i<m; i++) {
    ig   = i + rstart;
    ierr = MatGetRow(A,ig,&nz,&cwork,&vwork);	CHKERRQ(ierr);
    ierr = MatSetValues(*B,1,&ig,nz,cwork,vwork,INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatRestoreRow(A,ig,&nz,&cwork,&vwork); CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*B,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*B,FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
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
  nz = PETSCMIN(n,a->nd*a->nb); 
  switch (newtype) {
    case MATSEQAIJ:
      ierr = MatCreateSeqAIJ(A->comm,m,n,nz,0,B); CHKERRQ(ierr); 
      break;
    case MATSEQROW:
      ierr = MatCreateSeqRow(A->comm,m,n,nz,0,B); CHKERRQ(ierr); 
      break;
    case MATMPIROW:
      ierr = MatCreateMPIRow(MPI_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,
                             m,n,0,0,0,0,B); CHKERRQ(ierr);
      break;
    case MATMPIROWBS:
      if (m != n) SETERRQ(1,"MatConvert_SeqBDiag:MATMPIROWBS matrix must be square");
      ierr = MatCreateMPIRowbs(MPI_COMM_WORLD,PETSC_DECIDE,m,0,0,0,B); CHKERRQ(ierr);
      break;
    case MATMPIAIJ:
      ierr = MatCreateMPIAIJ(MPI_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,
                             m,n,0,0,0,0,B); CHKERRQ(ierr);
      break;
    case MATSEQDENSE:
      ierr = MatCreateSeqDense(A->comm,m,n,B); CHKERRQ(ierr);
      break;
    case MATMPIDENSE:
      ierr = MatCreateMPIDense(MPI_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,
                               m,n,B); CHKERRQ(ierr);
      break;
    case MATMPIBDIAG:
      {
      ierr = MatCreateMPIBDiag(MPI_COMM_WORLD,PETSC_DECIDE,m,n,a->nd,a->nb,0,0,B); 
      CHKERRQ(ierr); 
      break;
      }
    default:
      SETERRQ(1,"MatConvert_SeqBDiag:Matrix type is not currently supported");
  }
  ierr = MatGetOwnershipRange(*B,&rstart,&rend); CHKERRQ(ierr);

  cw2 = (int *)PETSCMALLOC( n * sizeof(int) ); CHKPTRQ(cw2);
  vw2 = (Scalar *)PETSCMALLOC( n * sizeof(Scalar) ); CHKPTRQ(vw2);
  for (i=rstart; i<rend; i++) {
   ierr = MatGetRow(A,i,&nz,&cwork,&vwork); CHKERRQ(ierr);
   ict = 0; /* strip out the zero elements ... is this what we really want? */
   for (j=0; j<nz; j++) {
     if (vwork[j] != 0) {vw2[ict] = vwork[j]; cw2[ict] = cwork[j]; ict++;}
   }
   if (ict) 
     {ierr = MatSetValues(*B,1,&i,ict,cw2,vw2,INSERT_VALUES); CHKERRQ(ierr);}
   ierr = MatRestoreRow(A,i,&nz,&cwork,&vwork); CHKERRQ(ierr);
  }
  PETSCFREE(cw2); PETSCFREE(vw2);
  ierr = MatAssemblyBegin(*B,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*B,FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}
