#ifndef lint
static char vcid[] = "$Id: cholbs.c,v 1.19 1995/12/01 22:30:06 curfman Exp $";
#endif

#if defined(HAVE_BLOCKSOLVE) && !defined(__cplusplus)
#include "matimpl.h"
#include "src/pc/pcimpl.h"
#include "mpirowbs.h"
#include "BSprivate.h"

extern int MatCreateShellMPIRowbs(MPI_Comm,int,int,int,int*,Mat*);

int MatIncompleteCholeskyFactorSymbolic_MPIRowbs( Mat mat,IS perm,
                                      double f,int fill,
Mat *newfact )
{
  /* Note:  f is not currently used in BlockSolve */
  Mat_MPIRowbs *mbs = (Mat_MPIRowbs *) mat->data;

  PETSCVALIDHEADERSPECIFIC(mat,MAT_COOKIE);
  if (!mbs->mat_is_symmetric) 
    SETERRQ(1,"MatIncompleteCholeskySymbolic_MPIRowbs:Must declare symmetric matrix");

  /* Copy permuted matrix */
  mbs->fpA = BScopy_par_mat(mbs->pA); CHKERRBS(0);

  /* Set up the communication for factorization */
  mbs->comm_fpA = BSsetup_factor(mbs->fpA,mbs->procinfo); CHKERRBS(0);

  mbs->fact_clone = 1;
  *newfact = mat; 
  return 0; 
}
int MatILUFactorSymbolic_MPIRowbs( Mat mat,IS perm,IS cperm,
                                      int fill,double f,Mat *newfact )
{
  /* Note:  f is not currently used in BlockSolve */
  Mat_MPIRowbs *mbs = (Mat_MPIRowbs *) mat->data;

  PETSCVALIDHEADERSPECIFIC(mat,MAT_COOKIE);
  if (mbs->mat_is_symmetric) 
    SETERRQ(1,"MatILUFactorSymbolic_MPIRowbs:Not for declared symmetric matrix");

  /* Copy permuted matrix */
  mbs->fpA = BScopy_par_mat(mbs->pA); CHKERRBS(0); 

  /* Set up the communication for factorization */
  mbs->comm_fpA = BSsetup_factor(mbs->fpA,mbs->procinfo); CHKERRBS(0);

  mbs->fact_clone = 1;
  *newfact = mat; 
  return 0; 
}
int MatCholeskyFactorNumeric_MPIRowbs(Mat mat,Mat *factp) 
{
  Mat_MPIRowbs *mbs = (Mat_MPIRowbs *) mat->data;

  PETSCVALIDHEADERSPECIFIC(mat,MAT_COOKIE);
  if (mat != *factp) SETERRQ(1,"MatCholeskyFactorNumeric_MPIRowbs:factored\
                                 matrix must be same context as mat");

  /* Do prep work if same nonzero structure as previously factored matrix */
  if (mat->factor == FACTOR_CHOLESKY) {
    if (!mbs->nonew) SETERRQ(1,"MatCholeskyFactorNumeric_MPIRowbs:\
      Must call MatSetOption(mat,NO_NEW_NONZERO_LOCATIONS) for re-solve.");
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
                     "BlockSolve: %d failed factors, err=%d, alpha=%g\n",
                                       mbs->failures, mbs->ierr, mbs->alpha ); 
  }

  mat->factor = FACTOR_CHOLESKY;
  return 0;
}
int MatLUFactorNumeric_MPIRowbs(Mat mat,Mat *factp) 
{
  Mat_MPIRowbs *mbs = (Mat_MPIRowbs *) mat->data;

  PETSCVALIDHEADERSPECIFIC(mat,MAT_COOKIE);
  if (mat != *factp) SETERRQ(1,"MatCholeskyFactorNumeric_MPIRowbs:factored\
                                 matrix must be same context as mat");

  /* Do prep work if same nonzero structure as previously factored matrix */
  if (mat->factor == FACTOR_LU) {
    if (!mbs->nonew) SETERRQ(1,"MatCholeskyFactorNumeric_MPIRowbs:\
      Must call MatSetOption(mat,NO_NEW_NONZERO_LOCATIONS) for re-solve.");
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
    PLogInfo((PetscObject)mat,"BlockSolve: %d failed factors, err=%d, alpha=%g\n",
                                       mbs->failures, mbs->ierr, mbs->alpha ); 
  }
  mat->factor = FACTOR_LU;
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
    ierr = VecGetArray(x,&xa); CHKERRQ(ierr);
    ierr = VecGetArray(mbs->xwork,&xworka); CHKERRQ(ierr);
    BSperm_dvec(xa,xworka,mbs->pA->perm); CHKERRBS(0);
    ierr = VecPMult( mbs->diag, mbs->xwork, y ); CHKERRQ(ierr);
  } else {
    ierr = VecCopy( x, y ); CHKERRQ(ierr);
  }
  ierr = VecGetArray(y,&ya); CHKERRQ(ierr);

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

#if defined(PETSC_LOG)
  MLOG_ACC(MS_BACKWARD);
#endif

  /* Apply diagonal scaling and unpermute, where D^{-1/2} is stored */
  if (!mbs->vecs_permscale) {
    ierr = VecPMult( y, mbs->diag, mbs->xwork );  CHKERRQ(ierr);
    BSiperm_dvec(xworka,ya,mbs->pA->perm); CHKERRBS(0);
    ierr = VecRestoreArray(x,&xa); CHKERRQ(ierr);
    ierr = VecRestoreArray(mbs->xwork,&xworka); CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(y,&ya); CHKERRQ(ierr);
  PLogFlops(2*mbs->nz - mbs->m);
  return 0;
}
/* ------------------------------------------------------------------- */
int MatForwardSolve_MPIRowbs(Mat mat,Vec x,Vec y)
{
  Mat_MPIRowbs *mbs = (Mat_MPIRowbs *) mat->data;
  int          ierr;
  Scalar       *ya, *xa, *xworka;

  /* Permute and apply diagonal scaling to vector, where D^{-1/2} is stored */
  if (!mbs->vecs_permscale) {
    ierr = VecGetArray(x,&xa); CHKERRQ(ierr);
    ierr = VecGetArray(mbs->xwork,&xworka); CHKERRQ(ierr);
    BSperm_dvec(xa,xworka,mbs->pA->perm); CHKERRBS(0);
    ierr = VecPMult( mbs->diag, mbs->xwork, y ); CHKERRQ(ierr);
    ierr = VecRestoreArray(x,&xa); CHKERRQ(ierr);
    ierr = VecRestoreArray(mbs->xwork,&xworka); CHKERRQ(ierr);
  } else {
    ierr = VecCopy( x, y ); CHKERRQ(ierr);
  }
  ierr = VecGetArray(y,&ya); CHKERRQ(ierr);

#if defined(PETSC_LOG)
  MLOG_ELM(mbs->procinfo->procset);
#endif
  if (mbs->procinfo->single)
    /* Use BlockSolve routine for no cliques/inodes */
    BSfor_solve1( mbs->fpA, ya, mbs->comm_pA, mbs->procinfo );
  else
    BSfor_solve( mbs->fpA, ya, mbs->comm_pA, mbs->procinfo );
  CHKERRBS(0);
#if defined(PETSC_LOG)
  MLOG_ACC(MS_FORWARD);
  MLOG_ELM(mbs->procinfo->procset);
#endif
  ierr = VecRestoreArray(y,&ya); CHKERRQ(ierr);

  return 0;
}
/* ------------------------------------------------------------------- */
int MatBackwardSolve_MPIRowbs(Mat mat,Vec x,Vec y)
{
  Mat_MPIRowbs *mbs = (Mat_MPIRowbs *) mat->data;
  int          ierr;
  Scalar       *ya, *xworka;

  ierr = VecCopy( x, y ); CHKERRQ(ierr);
  ierr = VecGetArray( y, &ya );   CHKERRQ(ierr);
  ierr = VecGetArray( mbs->xwork, &xworka ); CHKERRQ(ierr);
#if defined(PETSC_LOG)
  MLOG_ELM(mbs->procinfo->procset);
#endif
  if (mbs->procinfo->single)
    /* Use BlockSolve routine for no cliques/inodes */
    BSback_solve1( mbs->fpA, ya, mbs->comm_pA, mbs->procinfo );
  else
    BSback_solve( mbs->fpA, ya, mbs->comm_pA, mbs->procinfo );
  CHKERRBS(0);
#if defined(PETSC_LOG)
  MLOG_ACC(MS_BACKWARD);
#endif

  /* Apply diagonal scaling and unpermute, where D^{-1/2} is stored */
  if (!mbs->vecs_permscale) {
    ierr = VecPMult( y, mbs->diag, mbs->xwork );  CHKERRQ(ierr);
    BSiperm_dvec(xworka,ya,mbs->pA->perm); CHKERRBS(0);
  }
  ierr = VecRestoreArray( y, &ya );   CHKERRQ(ierr);
  ierr = VecRestoreArray( mbs->xwork, &xworka ); CHKERRQ(ierr);
  return 0;
}

#else
int MatNullMPIRowbs()
{return 0;}
#endif
