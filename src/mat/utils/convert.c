#ifndef lint
static char vcid[] = "$Id: convert.c,v 1.23 1995/08/23 17:17:23 curfman Exp bsmith $";
#endif

/* Matrix conversion routines.  For now, this supports only AIJ */

#include "mpiaij.h"

/* Determines the block diagonals within a subset of a matrix */
/* For now this is just sequential -- not parallel */

/*
   MatDetermineDiagonals_Private - Determines the diagonal structure 
   of a matrix.

   Input Parameters:
.  mat - the matrix
.  nb - block size
.  irows - rows to use
.  icols - columns to use

   Output Parameters:
.  ndiag - number of diagonals
.  diagonals - the diagonal numbers

   Note:  The user must free the diagonals array.
 */

int MatDetermineDiagonals_Private(Mat mat,int nb,int newr,int newc,
            int *rowrange, int *colrange,int *ndiag, int **diagonals)
{
  int    nd, clast, cfirst, ierr, nnc, maxd, nz, *col, *cwork, *diag;
  int    i, j, k, jdiag, cshift, row, dnew, temp;
  Scalar *v;

  PETSCVALIDHEADERSPECIFIC(mat,MAT_COOKIE);
  if ((newr%nb) || (newc%nb))
    SETERRQ(1,"MatDetermineDiagonals_Private:Invalid block size");
  cfirst = colrange[0];
  clast  = colrange[newc-1];
  nnc    = clast - cfirst + 1;
  cwork  = (int *) PETSCMALLOC( nnc * sizeof(int) );	CHKPTRQ(cwork);
  for (i=0; i<nnc; i++)  cwork[i] = -1;
  for (i=0; i<newc; i++) cwork[colrange[i]-cfirst] = i;

  /* Determine which diagonals exist:  compute nd, diag[]: */
  /* Temporarily ssume diag[0] = 0 (main diagonal) */
  maxd = newr + newc - 1;	/* maximum possible diagonals */
  diag = (int *)PETSCMALLOC( maxd * sizeof(int) );	CHKPTRQ(diag);
  nd = 1;
  for (i=0; i<maxd; i++) diag[i] = 0; 
  for (i=0; i<newr; i++) {
    ierr = MatGetRow( mat, rowrange[i], &nz, &col, &v ); CHKERRQ(ierr);
    row = i;
    j   = 0;
    /* Skip values until we reach the first column */
    while (j < nz && col[j] < cfirst) j++;
    while (j < nz) {
      if (clast < col[j]) break;
      cshift = cwork[col[j] - cfirst];
      if (cshift >= 0) {
        /* Determine if diagonal block already exits for valid colum */
        dnew = 1;
        jdiag = row/nb - cshift/nb;
        for (k=0; k<nd; k++) {
          if (diag[k] == jdiag) {	/* diagonal exists */
            dnew = 0;	break;
          }
        }
        if (dnew) {
	  diag[nd] = jdiag;
	  nd++;
          if (PETSCABS(jdiag) > newr/nb) 
             { printf("ERROR jdiag\n"); }
        }
      }
      j++;
    }
    ierr = MatRestoreRow( mat, rowrange[i], &nz, &col, &v ); CHKERRQ(ierr);
  }
  /* Sort diagonals in decreasing order. */
  for (k=0; k<nd; k++) {
    for (j=k+1; j<nd; j++) {
      if (diag[k] < diag[j]) {
        temp = diag[k];
        diag[k] = diag[j];
        diag[j] = temp;
      }
    }
  }
  PETSCFREE( cwork );  
  *ndiag = nd;
  *diagonals = diag;
  return 0;
}

/* 
  MatConvert_AIJ - Converts from MATAIJ format to another format. For
  parallel formats, the new matrix distribution is determined by PETSc.
 */
int MatConvert_AIJ(Mat mat, MatType newtype, Mat *newmat)
{ 
  Mat_AIJ *aij = (Mat_AIJ *) mat->data;
  Scalar  *vwork;
  int     i, ierr, nz, m = aij->m, n = aij->n, *cwork, rstart, rend;

  switch (newtype) {
    case MATROW:
      ierr = MatCreateSeqRow(mat->comm,m,n,0,aij->ilen,newmat);
      CHKERRQ(ierr); break;
    case MATMPIROW:
      ierr = MatCreateMPIRow(MPI_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,
             m,n,0,0,0,0,newmat); /* Could do smarter memory allocation */
      CHKERRQ(ierr); break;
    case MATMPIAIJ:
      ierr = MatCreateMPIAIJ(MPI_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,
             m,n,0,0,0,0,newmat); /* Could do smarter memory allocation */
      CHKERRQ(ierr); break;
    case MATDENSE:
      ierr = MatCreateSeqDense(mat->comm,m,n,newmat);
      CHKERRQ(ierr); break;
    case MATBDIAG:
    { int nb = 1; /* Default block size = 1 */
      int ndiag, *diag, *rr, *cr;
      rr = (int *) PETSCMALLOC( (m+n) * sizeof(int) ); CHKPTRQ(rr);
      cr = rr + m;
      for (i=0; i<m; i++) rr[i] = i;
      for (i=0; i<n; i++) cr[i] = i;
      OptionsGetInt(0,"-mat_bdiag_bsize",&nb);     
      ierr = MatDetermineDiagonals_Private(mat,nb,m,n,rr,cr,&ndiag,&diag);
      CHKERRQ(ierr); 
      ierr = MatCreateSeqBDiag(mat->comm,m,n,ndiag,nb,diag,0,newmat);
      CHKERRQ(ierr); 
      PETSCFREE(rr), PETSCFREE(diag);
      break;
    }
    case MATMPIBDIAG:
    { int nb = 1; /* Default block size = 1 */
      int ndiag, *diag, *rr, *cr;
      rr = (int *) PETSCMALLOC( (m+n) * sizeof(int) ); CHKPTRQ(rr);
      cr = rr + m;
      for (i=0; i<m; i++) rr[i] = i;
      for (i=0; i<n; i++) cr[i] = i;
      OptionsGetInt(0,"-mat_bdiag_bsize",&nb);     
      ierr = MatDetermineDiagonals_Private(mat,nb,m,n,rr,cr,&ndiag,&diag);
      CHKERRQ(ierr); 
      ierr = MatCreateMPIBDiag(MPI_COMM_WORLD,PETSC_DECIDE,m,n,ndiag,nb,
             diag,0,newmat); CHKERRQ(ierr); 
      PETSCFREE(rr), PETSCFREE(diag);
      CHKERRQ(ierr); break;
    }
    default:
      SETERRQ(1,"MatConvert_AIJ:Matrix type is not currently supported");
  }
  ierr = MatGetOwnershipRange(*newmat,&rstart,&rend); CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    ierr = MatGetRow(mat,i,&nz,&cwork,&vwork); CHKERRQ(ierr);
    ierr = MatSetValues(*newmat,1,&i,nz,cwork,vwork,INSERTVALUES); 
           CHKERRQ(ierr);
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
  Mat_AIJ    *Ad = (Mat_AIJ *)(aij->A->data), *Bd = (Mat_AIJ *)(aij->B->data);
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
    ierr = MatSetValues(*newmat,1,&ig,nz,cwork,vwork,
		INSERTVALUES); CHKERRQ(ierr);
    ierr = MatRestoreRow(mat,ig,&nz,&cwork,&vwork); CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*newmat,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*newmat,FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}
