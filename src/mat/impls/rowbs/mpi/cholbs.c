#ifndef lint
static char vcid[] = "$Id: cholbs.c,v 1.37 1996/08/12 03:41:39 bsmith Exp bsmith $";
#endif

#if defined(HAVE_BLOCKSOLVE) && !defined(PETSC_COMPLEX)

/* We must define both BSMAINLOG and MLOG for BlockSolve logging */ 
#if defined(PETSC_LOG)
#define MLOG
#endif

#include "src/pc/pcimpl.h"
#include "src/mat/impls/rowbs/mpi/mpirowbs.h"



int MatCholeskyFactorNumeric_MPIRowbs(Mat mat,Mat *factp) 
{
  Mat_MPIRowbs *mbs = (Mat_MPIRowbs *) mat->data;

#if defined(PETSC_LOG)
  double flop1 = BSlocal_flops();
#endif

  /* Do prep work if same nonzero structure as previously factored matrix */
  if (mbs->factor == FACTOR_CHOLESKY) {
    /* Copy the nonzeros */
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
    PLogInfo(mat,"BlockSolve95: %d failed factor(s), err=%d, alpha=%g\n",
                                 mbs->failures,mbs->ierr,mbs->alpha); 
  }
#if defined(PETSC_LOG)
  PLogFlops((int)(BSlocal_flops()-flop1));
#endif

  mbs->factor = FACTOR_CHOLESKY;
  return 0;
}

int MatLUFactorNumeric_MPIRowbs(Mat mat,Mat *factp) 
{
  Mat_MPIRowbs *mbs = (Mat_MPIRowbs *) mat->data;

  /* Do prep work if same nonzero structure as previously factored matrix */
  if (mbs->factor == FACTOR_LU) {
    /* Copy the nonzeros */
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
    PLogInfo(mat,"BlockSolve95: %d failed factor(s), err=%d, alpha=%g\n",
                                       mbs->failures,mbs->ierr,mbs->alpha); 
  }
  mbs->factor = FACTOR_LU;
  return 0;
}
/* ------------------------------------------------------------------- */
int MatSolve_MPIRowbs(Mat mat,Vec x,Vec y)
{
  Mat          submat = (Mat) mat->data;
  Mat_MPIRowbs *mbs = (Mat_MPIRowbs *) submat->data;
  int          ierr;
  Scalar       *ya, *xa, *xworka;

#if defined(PETSC_LOG)
  double flop1 = BSlocal_flops();
#endif
  /* Permute and apply diagonal scaling to vector, where D^{-1/2} is stored */
  if (!mbs->vecs_permscale) {
    ierr = VecGetArray(x,&xa); CHKERRQ(ierr);
    ierr = VecGetArray(mbs->xwork,&xworka); CHKERRQ(ierr);
    BSperm_dvec(xa,xworka,mbs->pA->perm); CHKERRBS(0);
    ierr = VecPointwiseMult(mbs->diag,mbs->xwork,y); CHKERRQ(ierr);
  } else {
    ierr = VecCopy(x,y); CHKERRQ(ierr);
  }
  ierr = VecGetArray(y,&ya); CHKERRQ(ierr);

  if (mbs->procinfo->single)
    /* Use BlockSolve routine for no cliques/inodes */
    BSfor_solve1(mbs->fpA,ya,mbs->comm_pA,mbs->procinfo);
  else
    BSfor_solve(mbs->fpA,ya,mbs->comm_pA,mbs->procinfo);
  CHKERRBS(0);

  if (mbs->procinfo->single)
    /* Use BlockSolve routine for no cliques/inodes */
    BSback_solve1(mbs->fpA,ya,mbs->comm_pA,mbs->procinfo);
  else
    BSback_solve(mbs->fpA,ya,mbs->comm_pA,mbs->procinfo);
  CHKERRBS(0);

  /* Apply diagonal scaling and unpermute, where D^{-1/2} is stored */
  if (!mbs->vecs_permscale) {
    ierr = VecPointwiseMult(y,mbs->diag,mbs->xwork);  CHKERRQ(ierr);
    BSiperm_dvec(xworka,ya,mbs->pA->perm); CHKERRBS(0);
    ierr = VecRestoreArray(x,&xa); CHKERRQ(ierr);
    ierr = VecRestoreArray(mbs->xwork,&xworka); CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(y,&ya); CHKERRQ(ierr);
#if defined(PETSC_LOG)
  PLogFlops((int)(BSlocal_flops()-flop1));
#endif
  return 0;
}

/* ------------------------------------------------------------------- */
int MatForwardSolve_MPIRowbs(Mat mat,Vec x,Vec y)
{
  Mat          submat = (Mat) mat->data;
  Mat_MPIRowbs *mbs = (Mat_MPIRowbs *) submat->data;
  int          ierr;
  Scalar       *ya, *xa, *xworka;

#if defined(PETSC_LOG)
  double flop1 = BSlocal_flops();
#endif
  /* Permute and apply diagonal scaling to vector, where D^{-1/2} is stored */
  if (!mbs->vecs_permscale) {
    ierr = VecGetArray(x,&xa); CHKERRQ(ierr);
    ierr = VecGetArray(mbs->xwork,&xworka); CHKERRQ(ierr);
    BSperm_dvec(xa,xworka,mbs->pA->perm); CHKERRBS(0);
    ierr = VecPointwiseMult(mbs->diag,mbs->xwork,y); CHKERRQ(ierr);
    ierr = VecRestoreArray(x,&xa); CHKERRQ(ierr);
    ierr = VecRestoreArray(mbs->xwork,&xworka); CHKERRQ(ierr);
  } else {
    ierr = VecCopy(x,y); CHKERRQ(ierr);
  }
  ierr = VecGetArray(y,&ya); CHKERRQ(ierr);

  if (mbs->procinfo->single)
    /* Use BlockSolve routine for no cliques/inodes */
    BSfor_solve1(mbs->fpA,ya,mbs->comm_pA,mbs->procinfo);
  else
    BSfor_solve(mbs->fpA,ya,mbs->comm_pA,mbs->procinfo);
  CHKERRBS(0);
  ierr = VecRestoreArray(y,&ya); CHKERRQ(ierr);
#if defined(PETSC_LOG)
  PLogFlops((int)(BSlocal_flops()-flop1));
#endif

  return 0;
}

/* ------------------------------------------------------------------- */
int MatBackwardSolve_MPIRowbs(Mat mat,Vec x,Vec y)
{
  Mat          submat = (Mat) mat->data;
  Mat_MPIRowbs *mbs = (Mat_MPIRowbs *) submat->data;
  int          ierr;
  Scalar       *ya, *xworka;

#if defined (PETSC_LOG)
  double flop1 = BSlocal_flops();
#endif
  ierr = VecCopy(x,y); CHKERRQ(ierr);
  ierr = VecGetArray(y,&ya);   CHKERRQ(ierr);
  ierr = VecGetArray(mbs->xwork,&xworka); CHKERRQ(ierr);

  if (mbs->procinfo->single)
    /* Use BlockSolve routine for no cliques/inodes */
    BSback_solve1(mbs->fpA,ya,mbs->comm_pA,mbs->procinfo);
  else
    BSback_solve(mbs->fpA,ya,mbs->comm_pA,mbs->procinfo);
  CHKERRBS(0);

  /* Apply diagonal scaling and unpermute, where D^{-1/2} is stored */
  if (!mbs->vecs_permscale) {
    ierr = VecPointwiseMult(y,mbs->diag,mbs->xwork);  CHKERRQ(ierr);
    BSiperm_dvec(xworka,ya,mbs->pA->perm); CHKERRBS(0);
  }
  ierr = VecRestoreArray(y,&ya);   CHKERRQ(ierr);
  ierr = VecRestoreArray(mbs->xwork,&xworka); CHKERRQ(ierr);
#if defined (PETSC_LOG)
  PLogFlops((int)(BSlocal_flops()-flop1));
#endif
  return 0;
}

#else
int MatNullMPIRowbs()
{return 0;}
#endif

