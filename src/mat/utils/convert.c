#ifndef lint
static char vcid[] = "$Id: convert.c,v 1.31 1995/10/27 01:11:01 curfman Exp curfman $";
#endif

/* This file contains implementation-specific matrix conversion routines.
   For now, this has been implemented only for AIJ.  See MatConvert() for
   generic conversion code. */

#include "mpiaij.h"

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

#include "mpibdiag.h"

/* 
  MatConvert_SeqBDiag - Converts from MATSEQBDiag format to another format. For
  parallel formats, the new matrix distribution is determined by PETSc.
 */
int MatConvert_SeqBDiag(Mat A, MatType newtype, Mat *B)
{ 
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  Scalar       *vwork;
  int          i, ierr, nz, m = a->m, n = a->n, *cwork, rstart, rend;

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
  for (i=rstart; i<rend; i++) {
    ierr = MatGetRow(A,i,&nz,&cwork,&vwork); CHKERRQ(ierr);
    ierr = MatSetValues(*B,1,&i,nz,cwork,vwork,INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatRestoreRow(A,i,&nz,&cwork,&vwork); CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*B,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*B,FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}
