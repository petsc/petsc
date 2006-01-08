#define PETSCMAT_DLL

#include "petsc.h"

/* We must define MLOG for BlockSolve logging */ 
#if defined(PETSC_USE_LOG)
#define MLOG
#endif

#include "src/mat/impls/rowbs/mpi/mpirowbs.h"

#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorNumeric_MPIRowbs"
PetscErrorCode MatCholeskyFactorNumeric_MPIRowbs(Mat mat,MatFactorInfo *info,Mat *factp) 
{
  Mat_MPIRowbs *mbs = (Mat_MPIRowbs*)mat->data;
  PetscErrorCode ierr;
#if defined(PETSC_USE_LOG)
  PetscReal flop1 = BSlocal_flops();
#endif

  PetscFunctionBegin;

  if (!mbs->blocksolveassembly) {
    ierr = MatAssemblyEnd_MPIRowbs_ForBlockSolve(mat);CHKERRQ(ierr);
  }

  /* Do prep work if same nonzero structure as previously factored matrix */
  if (mbs->factor == FACTOR_CHOLESKY) {
    /* Copy the nonzeros */
    BScopy_nz(mbs->pA,mbs->fpA);CHKERRBS(0);
  }
  /* Form incomplete Cholesky factor */
  mbs->ierr = 0; mbs->failures = 0; mbs->alpha = 1.0;
  while ((mbs->ierr = BSfactor(mbs->fpA,mbs->comm_fpA,mbs->procinfo))) {
    CHKERRBS(0); mbs->failures++;
    /* Copy only the nonzeros */
    BScopy_nz(mbs->pA,mbs->fpA);CHKERRBS(0);
    /* Increment the diagonal shift */
    mbs->alpha += 0.1;
    BSset_diag(mbs->fpA,mbs->alpha,mbs->procinfo);CHKERRBS(0);
    ierr = PetscInfo((mat,"MatCholeskyFactorNumeric_MPIRowbs:BlockSolve95: %d failed factor(s), err=%d, alpha=%g\n",
                                 mbs->failures,mbs->ierr,mbs->alpha));CHKERRQ(ierr);
  }
#if defined(PETSC_USE_LOG)
  ierr = PetscLogFlops((int)(BSlocal_flops()-flop1));CHKERRQ(ierr);
#endif

  mbs->factor = FACTOR_CHOLESKY;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorNumeric_MPIRowbs"
PetscErrorCode MatLUFactorNumeric_MPIRowbs(Mat mat,MatFactorInfo *info,Mat *factp) 
{
  Mat_MPIRowbs   *mbs = (Mat_MPIRowbs*)mat->data;

#if defined(PETSC_USE_LOG)
  PetscErrorCode ierr;
  PetscReal      flop1 = BSlocal_flops();
#endif

  PetscFunctionBegin;
  if (!mbs->blocksolveassembly) {
    ierr = MatAssemblyEnd_MPIRowbs_ForBlockSolve(mat);CHKERRQ(ierr);
  }

  /* Do prep work if same nonzero structure as previously factored matrix */
  if (mbs->factor == FACTOR_LU) {
    /* Copy the nonzeros */
    BScopy_nz(mbs->pA,mbs->fpA);CHKERRBS(0);
  }
  /* Form incomplete Cholesky factor */
  mbs->ierr = 0; mbs->failures = 0; mbs->alpha = 1.0;
  while ((mbs->ierr = BSfactor(mbs->fpA,mbs->comm_fpA,mbs->procinfo))) {
    CHKERRBS(0); mbs->failures++;
    /* Copy only the nonzeros */
    BScopy_nz(mbs->pA,mbs->fpA);CHKERRBS(0);
    /* Increment the diagonal shift */
    mbs->alpha += 0.1;
    BSset_diag(mbs->fpA,mbs->alpha,mbs->procinfo);CHKERRBS(0);
    ierr = PetscInfo((mat,"MatLUFactorNumeric_MPIRowbs:BlockSolve95: %d failed factor(s), err=%d, alpha=%g\n",
                                       mbs->failures,mbs->ierr,mbs->alpha));CHKERRQ(ierr);
  }
  mbs->factor = FACTOR_LU;
  (*factp)->assembled = PETSC_TRUE;
#if defined(PETSC_USE_LOG)
  ierr = PetscLogFlops((int)(BSlocal_flops()-flop1));CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "MatSolve_MPIRowbs"
PetscErrorCode MatSolve_MPIRowbs(Mat mat,Vec x,Vec y)
{
  Mat          submat = (Mat) mat->data;
  Mat_MPIRowbs *mbs = (Mat_MPIRowbs*)submat->data;
  PetscErrorCode ierr;
  PetscScalar  *ya,*xa,*xworka;

#if defined(PETSC_USE_LOG)
  PetscReal flop1 = BSlocal_flops();
#endif

  PetscFunctionBegin;
  /* Permute and apply diagonal scaling to vector, where D^{-1/2} is stored */
  if (!mbs->vecs_permscale) {
    ierr = VecGetArray(x,&xa);CHKERRQ(ierr);
    ierr = VecGetArray(mbs->xwork,&xworka);CHKERRQ(ierr);
    BSperm_dvec(xa,xworka,mbs->pA->perm);CHKERRBS(0);
    ierr = VecRestoreArray(x,&xa);CHKERRQ(ierr);
    ierr = VecRestoreArray(mbs->xwork,&xworka);CHKERRQ(ierr);
    ierr = VecPointwiseMult(y,mbs->diag,mbs->xwork);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(x,y);CHKERRQ(ierr);
  }

  ierr = VecGetArray(y,&ya);CHKERRQ(ierr);
  if (mbs->procinfo->single) {
    /* Use BlockSolve routine for no cliques/inodes */
    BSfor_solve1(mbs->fpA,ya,mbs->comm_pA,mbs->procinfo);CHKERRBS(0);
    BSback_solve1(mbs->fpA,ya,mbs->comm_pA,mbs->procinfo);CHKERRBS(0);
  } else {
    BSfor_solve(mbs->fpA,ya,mbs->comm_pA,mbs->procinfo);CHKERRBS(0);
    BSback_solve(mbs->fpA,ya,mbs->comm_pA,mbs->procinfo);CHKERRBS(0);
  }
  ierr = VecRestoreArray(y,&ya);CHKERRQ(ierr);

  /* Apply diagonal scaling and unpermute, where D^{-1/2} is stored */
  if (!mbs->vecs_permscale) {
    ierr = VecPointwiseMult(mbs->xwork,y,mbs->diag);CHKERRQ(ierr);
    ierr = VecGetArray(y,&ya);CHKERRQ(ierr);
    ierr = VecGetArray(mbs->xwork,&xworka);CHKERRQ(ierr);
    BSiperm_dvec(xworka,ya,mbs->pA->perm);CHKERRBS(0);
    ierr = VecRestoreArray(y,&ya);CHKERRQ(ierr);
    ierr = VecRestoreArray(mbs->xwork,&xworka);CHKERRQ(ierr);
  }
#if defined(PETSC_USE_LOG)
  ierr = PetscLogFlops((int)(BSlocal_flops()-flop1));CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "MatForwardSolve_MPIRowbs"
PetscErrorCode MatForwardSolve_MPIRowbs(Mat mat,Vec x,Vec y)
{
  Mat          submat = (Mat) mat->data;
  Mat_MPIRowbs *mbs = (Mat_MPIRowbs*)submat->data;
  PetscErrorCode ierr;
  PetscScalar  *ya,*xa,*xworka;

#if defined(PETSC_USE_LOG)
  PetscReal flop1 = BSlocal_flops();
#endif

  PetscFunctionBegin;
  /* Permute and apply diagonal scaling to vector, where D^{-1/2} is stored */
  if (!mbs->vecs_permscale) {
    ierr = VecGetArray(x,&xa);CHKERRQ(ierr);
    ierr = VecGetArray(mbs->xwork,&xworka);CHKERRQ(ierr);
    BSperm_dvec(xa,xworka,mbs->pA->perm);CHKERRBS(0);
    ierr = VecRestoreArray(x,&xa);CHKERRQ(ierr);
    ierr = VecRestoreArray(mbs->xwork,&xworka);CHKERRQ(ierr);
    ierr = VecPointwiseMult(y,mbs->diag,mbs->xwork);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(x,y);CHKERRQ(ierr);
  }

  ierr = VecGetArray(y,&ya);CHKERRQ(ierr);
  if (mbs->procinfo->single){
    /* Use BlockSolve routine for no cliques/inodes */
    BSfor_solve1(mbs->fpA,ya,mbs->comm_pA,mbs->procinfo);CHKERRBS(0);
  } else {
    BSfor_solve(mbs->fpA,ya,mbs->comm_pA,mbs->procinfo);CHKERRBS(0);
  }
  ierr = VecRestoreArray(y,&ya);CHKERRQ(ierr);

#if defined(PETSC_USE_LOG)
  ierr = PetscLogFlops((int)(BSlocal_flops()-flop1));CHKERRQ(ierr);
#endif

  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "MatBackwardSolve_MPIRowbs"
PetscErrorCode MatBackwardSolve_MPIRowbs(Mat mat,Vec x,Vec y)
{
  Mat          submat = (Mat) mat->data;
  Mat_MPIRowbs *mbs = (Mat_MPIRowbs*)submat->data;
  PetscErrorCode ierr;
  PetscScalar  *ya,*xworka;

#if defined (PETSC_USE_LOG)
  PetscReal flop1 = BSlocal_flops();
#endif

  PetscFunctionBegin;  
  ierr = VecCopy(x,y);CHKERRQ(ierr);

  ierr = VecGetArray(y,&ya);CHKERRQ(ierr);
  if (mbs->procinfo->single) {
    /* Use BlockSolve routine for no cliques/inodes */
    BSback_solve1(mbs->fpA,ya,mbs->comm_pA,mbs->procinfo);CHKERRBS(0);
  } else {
    BSback_solve(mbs->fpA,ya,mbs->comm_pA,mbs->procinfo);CHKERRBS(0);
  }
  ierr = VecRestoreArray(y,&ya);CHKERRQ(ierr);

  /* Apply diagonal scaling and unpermute, where D^{-1/2} is stored */
  if (!mbs->vecs_permscale) {
    ierr = VecPointwiseMult(mbs->xwork,y,mbs->diag);CHKERRQ(ierr);
    ierr = VecGetArray(y,&ya);CHKERRQ(ierr);
    ierr = VecGetArray(mbs->xwork,&xworka);CHKERRQ(ierr);
    BSiperm_dvec(xworka,ya,mbs->pA->perm);CHKERRBS(0);
    ierr = VecRestoreArray(y,&ya);CHKERRQ(ierr);
    ierr = VecRestoreArray(mbs->xwork,&xworka);CHKERRQ(ierr);
  }
#if defined (PETSC_USE_LOG)
  ierr = PetscLogFlops((int)(BSlocal_flops()-flop1));CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}


/* 
    The logging variables required by BlockSolve, 

    This is an ugly hack that allows PETSc to run properly with BlockSolve regardless
  of whether PETSc or BlockSolve is compiled with logging turned on. 

    It is bad because it relys on BlockSolve's internals not changing related to 
  logging but we have no choice, plus it is unlikely BlockSolve will be developed
  in the near future anyways.
*/
PETSCMAT_DLLEXPORT double MLOG_flops;
PETSCMAT_DLLEXPORT double MLOG_event_flops;
PETSCMAT_DLLEXPORT double MLOG_time_stamp;
PETSCMAT_DLLEXPORT PetscErrorCode    MLOG_sequence_num;
#if defined (MLOG_MAX_EVNTS) 
PETSCMAT_DLLEXPORT MLOG_log_type MLOG_event_log[MLOG_MAX_EVNTS];
PETSCMAT_DLLEXPORT MLOG_log_type MLOG_accum_log[MLOG_MAX_ACCUM];
#else
typedef struct __MLOG_log_type {
	double	time_stamp;
	double	total_time;
	double  flops;
	int	event_num;
} MLOG_log_type;
#define	MLOG_MAX_EVNTS 1300
#define	MLOG_MAX_ACCUM 75
PETSCMAT_DLLEXPORT MLOG_log_type MLOG_event_log[MLOG_MAX_EVNTS];
PETSCMAT_DLLEXPORT MLOG_log_type MLOG_accum_log[MLOG_MAX_ACCUM];
#endif

