#ifndef lint
static char vcid[] = "$Id: convert.c,v 1.1 1995/03/23 05:01:38 bsmith Exp bsmith $";
#endif

/* Matrix conversion routines.  For now, this supports only AIJ */

#include "mpiaij.h"

/* 
  MatiAIJSeqConvert - Converts from MATAIJSEQ format to another
  sequential format.
 */
int MatiAIJSeqConvert(Mat mat, MATTYPE newtype, Mat *newmat)
{ 
  Matiaij *aij = (Matiaij *) mat->data;
  Scalar  *vwork;
  int     i, ierr, nz, m = aij->m, n = aij->n, *cwork;

  if (mat->type != MATAIJSEQ) SETERR(1,"Input matrix must be MATAIJSEQ.");
  switch (newtype) {
    case MATROWSEQ:
      ierr = MatCreateSequentialRow(m,n,0,aij->ilen,newmat);
      CHKERR(ierr); break;
    case MATDENSESEQ:
      ierr = MatCreateSequentialDense(m,n,newmat);
      CHKERR(ierr); break;
    default:
      SETERR(1,"Only MATROWSEQ and MATDENSESEQ are currently suported.");
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
  MatiAIJMPIConvert - Converts from MATAIJMPI format to another
  parallel format.
 */
int MatiAIJMPIConvert(Mat mat, MATTYPE newtype, Mat *newmat)
{
  Matimpiaij *aij = (Matimpiaij *) mat->data;
  Mat        A = aij->A, B = aij->B, OldM;
  Matiaij    *Ad = (Matiaij *)(A->data), *Bd = (Matiaij *)(B->data);
  Scalar     *vwork;
  int        ierr, m = aij->m, n = aij->n, M = aij->M, N = aij->N, *cwork;
  int        nz,i,j,ig,submat, rstart = aij->rstart, cstart = aij->cstart;

  if (mat->type != MATAIJMPI) SETERR(1,"Input matrix must be MATAIJMPI.");
  switch (newtype) {
    case MATROWMPI:
      for (i=0; i<m; i++)
           printf("row=%d, A-nz=%d, B-nz=%d\n",i,Ad->ilen[i],Bd->ilen[i]);
      ierr = MatCreateMPIRow(mat->comm,m,n,M,N,0,Ad->ilen,0,Bd->ilen,newmat);
      CHKERR(ierr); break;
    default:
      SETERR(1,"Only MATROWMPI is currently suported.");
  }
  for (submat = 0; submat<2; submat++) {
    if (submat == 0) OldM = A; 
    else OldM = B; 
    for (i=0; i<m; i++) {
      ig   = i + rstart;
      ierr = MatGetRow(OldM,i,&nz,&cwork,&vwork);	CHKERR(ierr);
      if (submat == 0) {
        for (j=0; j<nz; j++) cwork[j] += cstart;
      } else {
        /* This part is incorrect! */
        for (j=0; j<nz; j++) cwork[j] += cstart;
      }
      ierr = MatSetValues(*newmat,1,&ig,nz,cwork,vwork,
		InsertValues);				CHKERR(ierr);
      ierr = MatRestoreRow(OldM,ig,&nz,&cwork,&vwork);	CHKERR(ierr);
    }
  }
  ierr = MatBeginAssembly(*newmat);			CHKERR(ierr);
  ierr = MatEndAssembly(*newmat);			CHKERR(ierr);
  return 0;
}
