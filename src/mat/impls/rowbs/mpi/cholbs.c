#ifndef lint
static char vcid[] = "$Id: cholbs.c,v 1.4 1995/04/18 02:17:15 curfman Exp curfman $";
#endif

#if defined(HAVE_BLOCKSOLVE) && !defined(PETSC_COMPLEX)
#include "src/mat/matimpl.h"
#include "mpirowbs.h"
#include "BSsparse.h"
#include "BSprivate.h"

extern int MatCreateShellMPIRowbs(MPI_Comm,int,int,int,int*,Mat*);

int MatIncompleteCholeskyFactorSymbolic_MPIRowbs( Mat mat,IS perm,
                                                int fill,Mat *newfact )
{
  Mat_MPIRowbs *mbs = (Mat_MPIRowbs *) mat->data, *fbs;
  Mat          fact;
  int          ierr;

  VALIDHEADER(mat,MAT_COOKIE);
  /* Form empty factor matrix context; just set pointers from mat */
  ierr = MatCreateShellMPIRowbs(mat->comm,mbs->m,mbs->M,
                         0,0,&fact); CHKERR(ierr);

  /* Copy permuted matrix */
  mbs->fpA = BScopy_par_mat(mbs->pA); CHKERRBS(0);

  /* Set up the communication for factorization */
  mbs->comm_fpA = BSsetup_factor(mbs->fpA,mbs->procinfo); CHKERRBS(0);

  fbs = (Mat_MPIRowbs *) fact->data;
  fbs->procinfo = mbs->procinfo;
  fbs->bsmap    = mbs->bsmap;
  fbs->A        = mbs->A;
  fbs->pA       = mbs->pA;
  fbs->fpA      = mbs->fpA;
  fbs->comm_pA  = mbs->comm_pA;
  fbs->comm_fpA = mbs->comm_fpA;
  fbs->diag     = mbs->diag;
  fbs->xwork    = mbs->xwork;

  *newfact = fact;
  return 0;
}
/* ----------------------------------------------------------------- */
/* 
   MatCholeskyFactorNumeric_MPIRowbs - Performs numeric factorization 
   of a symmetric parallel matrix, using BlockSolve.
 */
int MatCholeskyFactorNumeric_MPIRowbs(Mat mat,Mat *factp)
{
  Mat           fact = *factp;
  Mat_MPIRowbs  *mbs = (Mat_MPIRowbs *) mat->data;
  Mat_MPIRowbs  *fbs = (Mat_MPIRowbs *) fact->data;
  int           i, ierr, ldim, low, iwork;
  Scalar        v;

  VALIDHEADER(mat,MAT_COOKIE); VALIDHEADER(fact,MAT_COOKIE);
  /* Do prep work if same nonzero structure as previously factored matrix */
  if (fact->factor == FACTOR_CHOLESKY) {
    if (!mbs->nonew) SETERR(1,
      "Must call MatSetOption(mat,NO_NEW_NONZERO_LOCATIONS) for re-solve.");
    /* Copy only the nonzeros */
    BScopy_nz(mbs->pA,mbs->fpA); CHKERRBS(0);
  }
  /* Form incomplete Cholesky factor */
  mbs->ierr = 0; mbs->failures = 0; mbs->alpha = 1.0;
  while ((mbs->ierr = BSfactor(mbs->fpA,mbs->comm_fpA,mbs->procinfo))) {
    CHKERRBS(0); mbs->failures++;
    /* Copy only the nonzeros */
    BScopy_nz(mbs->pA,mbs->fpA); CHKERRBS(0);
    /* Increment the diagonal shift */
    mbs->alpha += 0.1;
    BSset_diag(mbs->fpA,mbs->alpha,mbs->procinfo); CHKERRBS(0);
    PLogInfo((PetscObject)fact,
     "BlockSolve error: %d failed factors, err=%d, alpha=%g\n",mbs->failures, mbs->ierr, mbs->alpha ); 
  }
  fact->factor = FACTOR_CHOLESKY;
  return 0;
}
/* ------------------------------------------------------------------- */
/*
   This routine assumes that the factored matrix has been produced by
   the ICC factorization of BlockSolve.  In particular, this routine
   assumes that the input/output vectors are permuted according to the
   BlockSolve coloring scheme.
 */

int MatSolve_MPIRowbs(Mat mat,Vec x,Vec y)
{
  Mat_MPIRowbs *mbs = (Mat_MPIRowbs *) mat->data;
  int          ierr;
  Scalar       *ya, *xa, *xworka;

  if (!mbs->vecs_permuted) {
    ierr = VecGetArray(x,&xa); CHKERR(ierr);
    ierr = VecGetArray(mbs->xwork,&xworka); CHKERR(ierr);
    BSperm_dvec(xa,xworka,mbs->pA->perm); CHKERRBS(0);
    ierr = VecCopy(mbs->xwork,x); CHKERR(ierr);
  }

  /* Apply diagonal scaling to vector, where D^{-1/2} is stored */
  ierr = VecPMult( mbs->diag, x, y ); CHKERR(ierr);
  VecGetArray(y,&ya);  

#if defined(PETSC_DEBUG)
  MLOG_ELM(mbs->procinfo->procset);
#endif
  if (mbs->procinfo->single)
      /* Use BlockSolve routine for no cliques/inodes */
      BSfor_solve1( mbs->fpA, ya, mbs->comm_pA, mbs->procinfo );
  else
      BSfor_solve( mbs->fpA, ya, mbs->comm_pA, mbs->procinfo );
  CHKERRBS(0);
#if defined(PETSC_DEBUG)
  MLOG_ACC(MS_FORWARD);
  MLOG_ELM(mbs->procinfo->procset);
#endif
  if (mbs->procinfo->single)
      /* Use BlockSolve routine for no cliques/inodes */
      BSback_solve1( mbs->fpA, ya, mbs->comm_pA, mbs->procinfo );
  else
      BSback_solve( mbs->fpA, ya, mbs->comm_pA, mbs->procinfo );
  CHKERRBS(0);
#if defined(PETSC_DEBUG)
  MLOG_ACC(MS_BACKWARD);
#endif

  /* Apply diagonal scaling to vector, where D^{-1/2} is stored */
  ierr = VecPMult( y, mbs->diag, y );  CHKERR(ierr);

  if (!mbs->vecs_permuted) {
    BSiperm_dvec(xa,xworka,mbs->pA->perm); CHKERRBS(0);
    ierr = VecCopy(mbs->xwork,x); CHKERR(ierr);
    BSiperm_dvec(ya,xworka,mbs->pA->perm); CHKERRBS(0);
    ierr = VecCopy(mbs->xwork,y); CHKERR(ierr);
  }

  return 0;
}
#else
static int MatNull_MPIRowbs()
{return 0;}
#endif
