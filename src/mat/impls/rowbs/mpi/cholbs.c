#ifndef lint
static char vcid[] = "$Id: cholbs.c,v 1.6 1995/04/20 16:03:39 curfman Exp curfman $";
#endif

#if defined(HAVE_BLOCKSOLVE) && !defined(PETSC_COMPLEX)
#include "matimpl.h"
#include "src/pc/pcimpl.h"
#include "mpirowbs.h"
#include "BSsparse.h"
#include "BSprivate.h"

extern int MatCreateShellMPIRowbs(MPI_Comm,int,int,int,int*,Mat*);

int MatIncompleteCholeskyFactorSymbolic_MPIRowbs( Mat mat,IS perm,
                                                int fill,Mat *newfact )
{
  Mat_MPIRowbs *mbs = (Mat_MPIRowbs *) mat->data;

  VALIDHEADER(mat,MAT_COOKIE);
  /* Copy permuted matrix */
  mbs->fpA = BScopy_par_mat(mbs->pA); CHKERRBS(0);

  /* Set up the communication for factorization */
  mbs->comm_fpA = BSsetup_factor(mbs->fpA,mbs->procinfo); CHKERRBS(0);

  mbs->fact_clone = 1;
  *newfact = mat; 
  return 0; 
}
/*  ----------------------------------------------------------------- */
/* MatCholeskyFactorNumeric_MPIRowbs - Performs numeric
   factorization of a symmetric parallel matrix, using BlockSolve.  
 */
int MatCholeskyFactorNumeric_MPIRowbs(Mat mat,Mat *factp) 
{
  Mat_MPIRowbs *mbs = (Mat_MPIRowbs *) mat->data;

  VALIDHEADER(mat,MAT_COOKIE);
  if (mat != *factp) SETERR(1,"factored matrix must be same context as mat.");

  /* Do prep work if same nonzero structure as previously factored matrix */
  if (mat->factor == FACTOR_CHOLESKY) {
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
    PLogInfo((PetscObject)mat,
     "BlockSolve error: %d failed factors, err=%d, alpha=%g\n",mbs->failures, mbs->ierr, mbs->alpha ); 
  }
  mat->factor = FACTOR_CHOLESKY;
  return 0;
}
/* ------------------------------------------------------------------- */
int MatSolve_MPIRowbs(Mat mat,Vec x,Vec y)
{
  Mat_MPIRowbs *mbs = (Mat_MPIRowbs *) mat->data;
  int          ierr;
  Scalar       *ya, *xa, *xworka;

  /* Permute and apply diagonal scaling to vector, where D^{-1/2} is stored */
  if (!mbs->vecs_permscale) {
    ierr = VecGetArray(x,&xa); CHKERR(ierr);
    ierr = VecGetArray(mbs->xwork,&xworka); CHKERR(ierr);
    BSperm_dvec(xa,xworka,mbs->pA->perm); CHKERRBS(0);
    ierr = VecPMult( mbs->diag, mbs->xwork, y ); CHKERR(ierr);
  } else {
    ierr = VecCopy( x, y ); CHKERR(ierr);
  }
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

  /* Apply diagonal scaling and unpermute, where D^{-1/2} is stored */
  if (!mbs->vecs_permscale) {
    ierr = VecPMult( y, mbs->diag, mbs->xwork );  CHKERR(ierr);
    BSiperm_dvec(xworka,ya,mbs->pA->perm); CHKERRBS(0);
  }
  return 0;
}

#else
static int MatNullMPIRowbs()
{return 0;}
#endif
