#ifndef lint
static char vcid[] = "$Id: convert.c,v 1.2 1995/03/23 22:05:59 curfman Exp curfman $";
#endif

/* Matrix conversion routines.  For now, this supports only AIJ */

#include "mpiaij.h"

/* 
  MatConvert_AIJ - Converts from MATAIJ format to another sequential format.
 */
int MatConvert_AIJ(Mat mat, MATTYPE newtype, Mat *newmat)
{ 
  Mat_AIJ *aij = (Mat_AIJ *) mat->data;
  Scalar  *vwork;
  int     i, ierr, nz, m = aij->m, n = aij->n, *cwork;

  if (mat->type != MATAIJ) SETERR(1,"Input matrix must be MATAIJ.");
  switch (newtype) {
    case MATROW:
      ierr = MatCreateSequentialRow(m,n,0,aij->ilen,newmat);
      CHKERR(ierr); break;
    case MATDENSE:
      ierr = MatCreateSequentialDense(m,n,newmat);
      CHKERR(ierr); break;
    default:
      SETERR(1,"Only MATROW and MATDENSE are currently suported.");
  }
  for (i=0; i<m; i++) {
    ierr = MatGetRow(mat,i,&nz,&cwork,&vwork);		CHKERR(ierr);
    ierr = MatSetValues(*newmat,1,&i,nz,cwork,vwork,InsertValues); CHKERR(ierr);
    ierr = MatRestoreRow(mat,i,&nz,&cwork,&vwork);	CHKERR(ierr);
  }
  ierr = MatBeginAssembly(*newmat);			CHKERR(ierr);
  ierr = MatEndAssembly(*newmat);			CHKERR(ierr);
  return 0;
}
/* ------------------------------------------------------------------ */
/* 
  MatConvert_MPIAIJ - Converts from MATMPIAIJ format to another
  parallel format.
 */
int MatConvert_MPIAIJ(Mat mat, MATTYPE newtype, Mat *newmat)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) mat->data;
  Mat_AIJ    *Ad = (Mat_AIJ *)(aij->A->data), *Bd = (Mat_AIJ *)(aij->B->data);
  int        ierr, nz, i, ig,rstart = aij->rstart, m = aij->m, *cwork;
  Scalar     *vwork;

  if (mat->type != MATMPIAIJ) SETERR(1,"Input matrix must be MATMPIAIJ.");
  switch (newtype) {
    case MATMPIROW:
      for (i=0; i<m; i++)
      ierr = MatCreateMPIRow(mat->comm,m,aij->n,aij->M,aij->N,0,Ad->ilen,
			0,Bd->ilen,newmat); CHKERR(ierr); 
      break;
    default:
      SETERR(1,"Only MATMPIROW is currently suported.");
  }
  /* Each processor converts its local rows */
  for (i=0; i<m; i++) {
    ig   = i + rstart;
    ierr = MatGetRow(mat,ig,&nz,&cwork,&vwork);	CHKERR(ierr);
    ierr = MatSetValues(*newmat,1,&ig,nz,cwork,vwork,
		InsertValues);				CHKERR(ierr);
    ierr = MatRestoreRow(mat,ig,&nz,&cwork,&vwork);	CHKERR(ierr);
  }
  ierr = MatBeginAssembly(*newmat);			CHKERR(ierr);
  ierr = MatEndAssembly(*newmat);			CHKERR(ierr);
  return 0;
}
