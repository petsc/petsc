#ifndef lint
static char vcid[] = "$Id: convert.c,v 1.28 1995/10/04 03:58:40 bsmith Exp curfman $";
#endif

/* Matrix conversion routines.  For now, this supports only conversion from AIJ */

#include "mpiaij.h"

/* 
  MatConvert_SeqAIJ - Converts from MATSEQAIJ format to another format. For
  parallel formats, the new matrix distribution is determined by PETSc.
 */
int MatConvert_SeqAIJ(Mat mat, MatType newtype, Mat *newmat)
{ 
  Mat_SeqAIJ *aij = (Mat_SeqAIJ *) mat->data;
  Scalar     *vwork;
  int        i, ierr, nz, m = aij->m, n = aij->n, *cwork, rstart, rend;

  switch (newtype) {
    case MATSEQROW:
      ierr = MatCreateSeqRow(mat->comm,m,n,0,aij->ilen,newmat);CHKERRQ(ierr); 
      break;
    case MATMPIROW:
      if (m != n) SETERRQ(1,"MatConvert_SeqAIJ: MPIRowbs matrix must be square");
      ierr = MatCreateMPIRow(MPI_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,
                             m,n,0,0,0,0,newmat); CHKERRQ(ierr);
      break;
    case MATMPIROWBS:
      ierr = MatCreateMPIRowbs(MPI_COMM_WORLD,PETSC_DECIDE,
                               m,0,0,0,newmat);CHKERRQ(ierr);
      break;
    case MATMPIAIJ:
      ierr = MatCreateMPIAIJ(MPI_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,
                             m,n,0,0,0,0,newmat);CHKERRQ(ierr);
      break;
    case MATSEQDENSE:
      ierr = MatCreateSeqDense(mat->comm,m,n,newmat); CHKERRQ(ierr);
      break;
    case MATSEQBDIAG:
      {
      int nb = 1; /* Default block size = 1 */ 
      OptionsGetInt(0,"-mat_bdiag_bsize",&nb);     
      ierr = MatCreateSeqBDiag(mat->comm,m,n,0,nb,0,0,newmat); CHKERRQ(ierr); 
      break;
      }
    case MATMPIBDIAG:
      {
      int nb = 1; /* Default block size = 1 */ 
      OptionsGetInt(0,"-mat_bdiag_bsize",&nb);     
      ierr = MatCreateMPIBDiag(MPI_COMM_WORLD,PETSC_DECIDE,m,n,0,nb,0,0,newmat); 
      CHKERRQ(ierr); 
      break;
      }
    default:
      SETERRQ(1,"MatConvert_SeqAIJ:Matrix type is not currently supported");
  }
  ierr = MatGetOwnershipRange(*newmat,&rstart,&rend); CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    ierr = MatGetRow(mat,i,&nz,&cwork,&vwork); CHKERRQ(ierr);
    ierr = MatSetValues(*newmat,1,&i,nz,cwork,vwork,INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatRestoreRow(mat,i,&nz,&cwork,&vwork); CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*newmat,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*newmat,FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}
/* ------------------------------------------------------------------ */
/* 
  MatConvert_MPIAIJ - Converts from MATMPIAIJ format to another
  parallel format.
 */
int MatConvert_MPIAIJ(Mat mat, MatType newtype, Mat *newmat)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) mat->data;
  Mat_SeqAIJ *Ad = (Mat_SeqAIJ *)(aij->A->data), *Bd = (Mat_SeqAIJ *)(aij->B->data);
  int        ierr, nz, i, ig,rstart = aij->rstart, m = aij->m, *cwork;
  Scalar     *vwork;

  switch (newtype) {
    case MATMPIROW:
      ierr = MatCreateMPIRow(mat->comm,m,aij->n,aij->M,aij->N,0,Ad->ilen,
			0,Bd->ilen,newmat); CHKERRQ(ierr);
      break;
    default:
      SETERRQ(1,"MatConvert_MPIAIJ:Only MATMPIROW is currently suported");
  }
  /* Each processor converts its local rows */
  for (i=0; i<m; i++) {
    ig   = i + rstart;
    ierr = MatGetRow(mat,ig,&nz,&cwork,&vwork);	CHKERRQ(ierr);
    ierr = MatSetValues(*newmat,1,&ig,nz,cwork,vwork,INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatRestoreRow(mat,ig,&nz,&cwork,&vwork); CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*newmat,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*newmat,FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}


