#include "petscksp.h"
#include "../src/ksp/ksp/impls/lsqr/lsqr.h"
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPLSQRGetArnorm(KSP,PetscReal*,PetscReal*,PetscReal*);

PetscErrorCode KSPMonitorLSQR(KSP solksp, PetscInt iter, PetscReal rnorm, void *ctx)
{
  PetscInt         mxiter;    /* Maximum number of iterations */
  PetscReal        arnorm;    /* The norm of the vector A.r */
  PetscReal        atol;      /* Absolute convergence tolerance */
  PetscReal        dtol;      /* Divergence tolerance */
  PetscReal        rtol;      /* Relative convergence tolerance */
  Vec              x_sol;
  PetscReal        rdum;
  PetscReal        xnorm;
  PetscErrorCode   ierr;
  MPI_Comm         comm;      
  
  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)solksp,&comm);CHKERRQ(ierr);
  ierr = KSPGetTolerances( solksp, &rtol, &atol, &dtol, &mxiter );CHKERRQ(ierr);
  ierr = KSPLSQRGetArnorm( solksp, &arnorm, &rdum, &rdum);CHKERRQ(ierr);
  ierr = KSPGetSolution( solksp, &x_sol );CHKERRQ(ierr);
  ierr = VecNorm( x_sol, NORM_2, &xnorm ); CHKERRQ(ierr);

  if (iter % 100 == 0){
    ierr = PetscPrintf(comm, "Iteration  Res norm      Grad norm     Upd norm\n");CHKERRQ(ierr);
  }
  if (iter <= 10 || iter >= mxiter - 10 || iter % 10 == 0){
    ierr = PetscPrintf(comm, "%10d %10.7e %10.7e %10.7e\n", iter, rnorm , arnorm, xnorm );CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
