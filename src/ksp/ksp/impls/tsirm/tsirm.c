
#include <petsc/private/kspimpl.h>      /*I "petscksp.h" I*/

typedef struct {
  PetscReal tol_ls;
  PetscInt  size_ls,maxiter_ls,cgls,size,Istart,Iend;
  Mat       A,S;
  Vec       Alpha,r;
} KSP_TSIRM;

static PetscErrorCode KSPSetUp_TSIRM(KSP ksp)
{
  KSP_TSIRM      *tsirm = (KSP_TSIRM*)ksp->data;

  PetscFunctionBegin;
  /* Initialization */
#if defined(PETSC_USE_REAL_SINGLE)
  tsirm->tol_ls     = 1e-25;
#else
  tsirm->tol_ls     = 1e-50;
#endif
  tsirm->size_ls    = 12;
  tsirm->maxiter_ls = 15;
  tsirm->cgls       = 0;

  /* Matrix of the system */
  PetscCall(KSPGetOperators(ksp,&tsirm->A,NULL));    /* Matrix of the system   */
  PetscCall(MatGetSize(tsirm->A,&tsirm->size,NULL)); /* Size of the system     */
  PetscCall(MatGetOwnershipRange(tsirm->A,&tsirm->Istart,&tsirm->Iend));

  /* Matrix S of residuals */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&tsirm->S));
  PetscCall(MatSetSizes(tsirm->S,tsirm->Iend-tsirm->Istart,PETSC_DECIDE,tsirm->size,tsirm->size_ls));
  PetscCall(MatSetType(tsirm->S,MATDENSE));
  PetscCall(MatSetUp(tsirm->S));

  /* Residual and vector Alpha computed in the minimization step */
  PetscCall(MatCreateVecs(tsirm->S,&tsirm->Alpha,&tsirm->r));
  PetscFunctionReturn(0);
}

PetscErrorCode KSPSolve_TSIRM(KSP ksp)
{
  KSP_TSIRM      *tsirm = (KSP_TSIRM*)ksp->data;
  KSP            sub_ksp;
  PC             pc;
  Mat            AS = NULL;
  Vec            x,b;
  PetscScalar    *array;
  PetscReal      norm = 20;
  PetscInt       i,*ind_row,first_iteration = 1,its = 0,total = 0,col = 0;
  PetscInt       restart = 30;
  KSP            ksp_min;  /* KSP for minimization */
  PC             pc_min;    /* PC for minimization */
  PetscBool      isksp;

  PetscFunctionBegin;
  x = ksp->vec_sol; /* Solution vector        */
  b = ksp->vec_rhs; /* Right-hand side vector */

  /* Row indexes (these indexes are global) */
  PetscCall(PetscMalloc1(tsirm->Iend-tsirm->Istart,&ind_row));
  for (i=0;i<tsirm->Iend-tsirm->Istart;i++) ind_row[i] = i+tsirm->Istart;

  /* Inner solver */
  PetscCall(KSPGetPC(ksp,&pc));
  PetscCall(PetscObjectTypeCompare((PetscObject)pc,PCKSP,&isksp));
  PetscCheck(isksp,PetscObjectComm((PetscObject)pc),PETSC_ERR_USER,"PC must be of type PCKSP");
  PetscCall(PCKSPGetKSP(pc,&sub_ksp));
  PetscCall(KSPSetTolerances(sub_ksp,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,restart));

  /* previously it seemed good but with SNES it seems not good... */
  PetscCall(KSP_MatMult(sub_ksp,tsirm->A,x,tsirm->r));
  PetscCall(VecAXPY(tsirm->r,-1,b));
  PetscCall(VecNorm(tsirm->r,NORM_2,&norm));
  KSPCheckNorm(ksp,norm);
  ksp->its = 0;
  PetscCall(KSPConvergedDefault(ksp,ksp->its,norm,&ksp->reason,ksp->cnvP));
  PetscCall(KSPSetInitialGuessNonzero(sub_ksp,PETSC_TRUE));
  do {
    for (col=0;col<tsirm->size_ls && ksp->reason==0;col++) {
      /* Solve (inner iteration) */
      PetscCall(KSPSolve(sub_ksp,b,x));
      PetscCall(KSPGetIterationNumber(sub_ksp,&its));
      total += its;

      /* Build S^T */
      PetscCall(VecGetArray(x,&array));
      PetscCall(MatSetValues(tsirm->S,tsirm->Iend-tsirm->Istart,ind_row,1,&col,array,INSERT_VALUES));
      PetscCall(VecRestoreArray(x,&array));

      PetscCall(KSPGetResidualNorm(sub_ksp,&norm));
      ksp->rnorm = norm;
      ksp->its ++;
      PetscCall(KSPConvergedDefault(ksp,ksp->its,norm,&ksp->reason,ksp->cnvP));
      PetscCall(KSPMonitor(ksp,ksp->its,norm));
    }

    /* Minimization step */
    if (!ksp->reason) {
      PetscCall(MatAssemblyBegin(tsirm->S,MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(tsirm->S,MAT_FINAL_ASSEMBLY));
      if (first_iteration) {
        PetscCall(MatMatMult(tsirm->A,tsirm->S,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&AS));
        first_iteration = 0;
      } else {
        PetscCall(MatMatMult(tsirm->A,tsirm->S,MAT_REUSE_MATRIX,PETSC_DEFAULT,&AS));
      }

      /* CGLS or LSQR method to minimize the residuals*/

      PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp_min));
      if (tsirm->cgls) {
        PetscCall(KSPSetType(ksp_min,KSPCGLS));
      } else {
        PetscCall(KSPSetType(ksp_min,KSPLSQR));
      }
      PetscCall(KSPSetOperators(ksp_min,AS,AS));
      PetscCall(KSPSetTolerances(ksp_min,tsirm->tol_ls,PETSC_DEFAULT,PETSC_DEFAULT,tsirm->maxiter_ls));
      PetscCall(KSPGetPC(ksp_min,&pc_min));
      PetscCall(PCSetType(pc_min,PCNONE));
      PetscCall(KSPSolve(ksp_min,b,tsirm->Alpha));    /* Find Alpha such that ||AS Alpha = b|| */
      PetscCall(KSPDestroy(&ksp_min));
      /* Apply minimization */
      PetscCall(MatMult(tsirm->S,tsirm->Alpha,x)); /* x = S * Alpha */
    }
  } while (ksp->its<ksp->max_it && !ksp->reason);
  PetscCall(MatDestroy(&AS));
  PetscCall(PetscFree(ind_row));
  ksp->its = total;
  PetscFunctionReturn(0);
}

PetscErrorCode KSPSetFromOptions_TSIRM(PetscOptionItems *PetscOptionsObject,KSP ksp)
{
  KSP_TSIRM      *tsirm = (KSP_TSIRM*)ksp->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject,"KSP TSIRM options");
  PetscCall(PetscOptionsInt("-ksp_tsirm_cgls","Method used for the minimization step","",tsirm->cgls,&tsirm->cgls,NULL)); /*0:LSQR, 1:CGLS*/
  PetscCall(PetscOptionsReal("-ksp_tsirm_tol_ls","Tolerance threshold for the minimization step","",tsirm->tol_ls,&tsirm->tol_ls,NULL));
  PetscCall(PetscOptionsInt("-ksp_tsirm_max_it_ls","Maximum number of iterations for the minimization step","",tsirm->maxiter_ls,&tsirm->maxiter_ls,NULL));
  PetscCall(PetscOptionsInt("-ksp_tsirm_size_ls","Number of residuals for minimization","",tsirm->size_ls,&tsirm->size_ls,NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

PetscErrorCode KSPDestroy_TSIRM(KSP ksp)
{
  KSP_TSIRM       *tsirm = (KSP_TSIRM*)ksp->data;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&tsirm->S));
  PetscCall(VecDestroy(&tsirm->Alpha));
  PetscCall(VecDestroy(&tsirm->r));
  PetscCall(PetscFree(ksp->data));
  PetscFunctionReturn(0);
}

/*MC
     KSPTSIRM - Implements the two-stage iteration with least-squares residual minimization method.

   Options Database Keys:
+  -ksp_ksp_type <solver> -         the type of the inner solver (GMRES or any of its variants for instance)
.  -ksp_pc_type <preconditioner> - the type of the preconditioner applied to the inner solver
.  -ksp_ksp_max_it <maxits> -      the maximum number of inner iterations (iterations of the inner solver)
.  -ksp_ksp_rtol <tol> -           sets the relative convergence tolerance of the inner solver
.  -ksp_tsirm_cgls <number> -      if 1 use CGLS solver in the minimization step, otherwise use LSQR solver
.  -ksp_tsirm_max_it_ls <maxits> - the maximum number of iterations for the least-squares minimization solver
.  -ksp_tsirm_tol_ls <tol> -       sets the convergence tolerance of the least-squares minimization solver
-  -ksp_tsirm_size_ls <size> -     the number of residuals for the least-squares minimization step

   Level: advanced

   Notes:
    TSIRM is a new two-stage iteration method for solving large sparse linear systems of the form Ax=b. The main idea behind this new
          method is the use a least-squares residual minimization to improve the convergence of Krylov based iterative methods, typically those of GMRES variants.
          The principle of TSIRM algorithm  is to build an outer iteration over a Krylov method, called inner solver, and to frequently store the current residual
          computed by the given Krylov method in a matrix of residuals S. After a few outer iterations, a least-squares minimization step is applied on the matrix
          composed by the saved residuals, in order to compute a better solution and to make new iterations if required. The GMRES method , or any of its variants,
          can potentially be used as inner solver. The minimization step consists in solving the least-squares problem min||b-ASa|| to find 'a' which minimizes the
          residuals (b-AS). The minimization step is performed using two solvers of linear least-squares problems: CGLS  or LSQR. A new solution x with
          a minimal residual is computed with x=Sa.

   References:
.  * - R. Couturier, L. Ziane Khodja, and C. Guyeux. TSIRM: A Two-Stage Iteration with least-squares Residual Minimization algorithm to solve large sparse linear systems. In PDSEC 2015, 16th IEEE Int. Workshop on Parallel and Distributed Scientific and Engineering Computing (in conjunction with IPDPS 2015), Hyderabad, India, 2015.

   Contributed by: Lilia Ziane Khodja

.seealso:  KSPCreate(), KSPSetType(), KSPType, KSP, KSPFGMRES, KSPLGMRES,
           KSPGMRESSetRestart(), KSPGMRESSetHapTol(), KSPGMRESSetPreAllocateVectors(), KSPGMRESSetOrthogonalization(), KSPGMRESGetOrthogonalization(),
           KSPGMRESClassicalGramSchmidtOrthogonalization(), KSPGMRESModifiedGramSchmidtOrthogonalization(),
           KSPGMRESCGSRefinementType, KSPGMRESSetCGSRefinementType(), KSPGMRESGetCGSRefinementType(), KSPGMRESMonitorKrylov(), KSPSetPCSide()

M*/
PETSC_EXTERN PetscErrorCode KSPCreate_TSIRM(KSP ksp)
{
  KSP_TSIRM      *tsirm;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(ksp,&tsirm));
  ksp->data                = (void*)tsirm;
  PetscCall(KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,2));
  PetscCall(KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_RIGHT,1));
  ksp->ops->setup          = KSPSetUp_TSIRM;
  ksp->ops->solve          = KSPSolve_TSIRM;
  ksp->ops->destroy        = KSPDestroy_TSIRM;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  ksp->ops->setfromoptions = KSPSetFromOptions_TSIRM;
  ksp->ops->view           = NULL;
#if defined(PETSC_USE_COMPLEX)
  SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"This is not supported for complex numbers");
#else
  PetscFunctionReturn(0);
#endif
}
