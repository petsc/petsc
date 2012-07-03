
#include <petsc-private/kspimpl.h>

typedef struct {
  KSP kspest;                   /* KSP capable of estimating eigenvalues */
  KSP kspcheap;                 /* Cheap smoother (should have few dot products) */
  PC  pcnone;                   /* Dummy PC to drop in so PCSetFromOptions doesn't get called extra times */
  PetscReal min,max;            /* Singular value estimates */
  PetscReal radius;             /* Spectral radius of 1-B where B is the preconditioned operator */
  PetscBool current;            /* Eigenvalue estimates are current */
  PetscReal minfactor,maxfactor;
  PetscReal richfactor;
} KSP_SpecEst;

#undef __FUNCT__  
#define __FUNCT__ "KSPSetUp_SpecEst"
static PetscErrorCode KSPSetUp_SpecEst(KSP ksp)
{
  KSP_SpecEst    *spec = (KSP_SpecEst*)ksp->data;
  PetscErrorCode ierr;
  PetscBool      nonzero;

  PetscFunctionBegin;
  ierr = KSPSetPC(spec->kspest,ksp->pc);CHKERRQ(ierr);
  ierr = KSPSetPC(spec->kspcheap,ksp->pc);CHKERRQ(ierr);
  ierr = KSPGetInitialGuessNonzero(ksp,&nonzero);CHKERRQ(ierr);
  ierr = KSPSetInitialGuessNonzero(spec->kspest,nonzero);CHKERRQ(ierr);
  ierr = KSPSetInitialGuessNonzero(spec->kspcheap,nonzero);CHKERRQ(ierr);
  ierr = KSPSetComputeSingularValues(spec->kspest,PETSC_TRUE);CHKERRQ(ierr);
  ierr = KSPSetUp(spec->kspest);CHKERRQ(ierr);
  spec->current    = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSpecEstPropagateUp"
static PetscErrorCode KSPSpecEstPropagateUp(KSP ksp,KSP subksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPGetConvergedReason(subksp,&ksp->reason);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(subksp,&ksp->its);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSolve_SpecEst"
static PetscErrorCode  KSPSolve_SpecEst(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_SpecEst    *spec = (KSP_SpecEst*)ksp->data;

  PetscFunctionBegin;
  if (spec->current) {
    ierr = KSPSolve(spec->kspcheap,ksp->vec_rhs,ksp->vec_sol);CHKERRQ(ierr);
    ierr = KSPSpecEstPropagateUp(ksp,spec->kspcheap);CHKERRQ(ierr);
  } else {
    PetscInt  i,its,neig;
    PetscReal *real,*imag,rad = 0;
    ierr = KSPSolve(spec->kspest,ksp->vec_rhs,ksp->vec_sol);CHKERRQ(ierr);
    ierr = KSPSpecEstPropagateUp(ksp,spec->kspest);CHKERRQ(ierr);
    ierr = KSPComputeExtremeSingularValues(spec->kspest,&spec->max,&spec->min);CHKERRQ(ierr);

    ierr = KSPGetIterationNumber(spec->kspest,&its);CHKERRQ(ierr);
    ierr = PetscMalloc2(its,PetscReal,&real,its,PetscReal,&imag);CHKERRQ(ierr);
    ierr = KSPComputeEigenvalues(spec->kspest,its,real,imag,&neig);CHKERRQ(ierr);
    for (i=0; i<neig; i++) {
      /* We would really like to compute w (nominally 1/radius) to minimize |1-wB|.  Empirically it
         is better to compute rad = |1-B| than rad = |B|.  There must be a cheap way to do better. */
      rad = PetscMax(rad,PetscRealPart(PetscSqrtScalar((PetscScalar)(PetscSqr(real[i]-1.) + PetscSqr(imag[i])))));
    }
    ierr = PetscFree2(real,imag);CHKERRQ(ierr);
    spec->radius = rad;

    ierr = KSPChebyshevSetEigenvalues(spec->kspcheap,spec->max*spec->maxfactor,spec->min*spec->minfactor);CHKERRQ(ierr);
    ierr = KSPRichardsonSetScale(spec->kspcheap,spec->richfactor/spec->radius);
    ierr = PetscInfo3(ksp,"Estimated singular value min=%G max=%G, spectral radius=%G",spec->min,spec->max,spec->radius);CHKERRQ(ierr);
    spec->current = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPView_SpecEst"
static PetscErrorCode KSPView_SpecEst(KSP ksp,PetscViewer viewer)
{
  KSP_SpecEst *spec = (KSP_SpecEst*)ksp->data;
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  SpecEst: last singular value estimate min=%G max=%G rad=%G\n",spec->min,spec->max,spec->radius);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Using scaling factors min=%G max=%G rich=%G\n",spec->minfactor,spec->maxfactor,spec->richfactor);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Sub KSP used for estimating spectrum:\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = KSPView(spec->kspest,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Sub KSP used for subsequent smoothing steps:\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = KSPView(spec->kspcheap,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"Viewer type %s not supported for KSP cg",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSetFromOptions_SpecEst"
static PetscErrorCode KSPSetFromOptions_SpecEst(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_SpecEst    *spec = (KSP_SpecEst*)ksp->data;
  char           prefix[256];

  PetscFunctionBegin;
  ierr = PetscOptionsHead("KSP SpecEst Options");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ksp_specest_minfactor","Multiplier on the minimum eigen/singular value","None",spec->minfactor,&spec->minfactor,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ksp_specest_maxfactor","Multiplier on the maximum eigen/singular value","None",spec->maxfactor,&spec->maxfactor,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ksp_specest_richfactor","Multiplier on the richimum eigen/singular value","None",spec->richfactor,&spec->richfactor,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);

  /* Mask the PC so that PCSetFromOptions does not do anything */
  ierr = KSPSetPC(spec->kspest,spec->pcnone);CHKERRQ(ierr);
  ierr = KSPSetPC(spec->kspcheap,spec->pcnone);CHKERRQ(ierr);

  ierr = PetscSNPrintf(prefix,sizeof prefix,"%sspecest_",((PetscObject)ksp)->prefix?((PetscObject)ksp)->prefix:"");CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(spec->kspest,prefix);CHKERRQ(ierr);
  ierr = PetscSNPrintf(prefix,sizeof prefix,"%sspeccheap_",((PetscObject)ksp)->prefix?((PetscObject)ksp)->prefix:"");CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(spec->kspcheap,prefix);CHKERRQ(ierr);

  if (!((PetscObject)spec->kspest)->type_name) {
    ierr = KSPSetType(spec->kspest,KSPGMRES);CHKERRQ(ierr);
  }
  if (!((PetscObject)spec->kspcheap)->type_name) {
    ierr = KSPSetType(spec->kspcheap,KSPCHEBYSHEV);CHKERRQ(ierr);
  }
  ierr = KSPSetFromOptions(spec->kspest);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(spec->kspcheap);CHKERRQ(ierr);

  /* Unmask the PC */
  ierr = KSPSetPC(spec->kspest,ksp->pc);CHKERRQ(ierr);
  ierr = KSPSetPC(spec->kspcheap,ksp->pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "KSPDestroy_SpecEst"
static PetscErrorCode KSPDestroy_SpecEst(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_SpecEst    *spec = (KSP_SpecEst*)ksp->data;

  PetscFunctionBegin;
  ierr = KSPDestroy(&spec->kspest);CHKERRQ(ierr);
  ierr = KSPDestroy(&spec->kspcheap);CHKERRQ(ierr);
  ierr = PCDestroy(&spec->pcnone);CHKERRQ(ierr);
  ierr = PetscFree(ksp->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPCreate_SpecEst"
/*MC
     KSPSPECEST - Estimate the spectrum on the first KSPSolve, then use cheaper smoother for subsequent solves.

   Options Database Keys:
+  -ksp_specest_minfactor <0.9> - Multiplier on the minimum eigen/singular value
.  -ksp_specest_maxfactor <1.1> - Multiplier on the maximum eigen/singular value
.  -ksp_specest_richfactor <1>  - Multiplier on the richimum eigen/singular value
.  -specest_ksp_type <type>     - KSP used to estimate the spectrum (usually CG or GMRES)
.  -speccheap_ksp_type <type>   - KSP used as a cheap smoother once the spectrum has been estimated (usually Chebyshev or Richardson)
-   see KSPSolve() for more

   Notes:
    This KSP estimates the extremal singular values on the first pass, then uses them to configure a smoother that
    uses fewer dot products.  It is intended for use on the levels of multigrid, especially at high process counts,
    where dot products are very expensive.

    The same PC is used for both the estimator and the cheap smoother, it is only set up once.  There are no options
    keys for -specest_pc_ or speccheap_pc_ since it is the same object as -pc_.

   Level: intermediate

.seealso: KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPGMRES, KSPCG, KSPCHEBYSHEV, KSPRICHARDSON
M*/
PetscErrorCode  KSPCreate_SpecEst(KSP ksp)
{
  KSP_SpecEst    *spec;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_RIGHT,1);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,1);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_RIGHT,1);CHKERRQ(ierr);

  ierr = PetscNewLog(ksp,KSP_SpecEst,&spec);CHKERRQ(ierr);

  ksp->data                      = (void*)spec;
  ksp->ops->setup                = KSPSetUp_SpecEst;
  ksp->ops->solve                = KSPSolve_SpecEst;
  ksp->ops->destroy              = KSPDestroy_SpecEst;
  ksp->ops->buildsolution        = KSPDefaultBuildSolution;
  ksp->ops->buildresidual        = KSPDefaultBuildResidual;
  ksp->ops->setfromoptions       = KSPSetFromOptions_SpecEst;
  ksp->ops->view                 = KSPView_SpecEst;

  spec->minfactor = 0.9;
  spec->maxfactor = 1.1;
  spec->richfactor = 1.0;

  ierr = KSPCreate(((PetscObject)ksp)->comm,&spec->kspest);CHKERRQ(ierr);
  ierr = KSPCreate(((PetscObject)ksp)->comm,&spec->kspcheap);CHKERRQ(ierr);

  /* Hold an empty PC */
  ierr = KSPGetPC(spec->kspest,&spec->pcnone);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)spec->pcnone);CHKERRQ(ierr);
  ierr = PCSetType(spec->pcnone,PCNONE);CHKERRQ(ierr);
  ierr = KSPSetPC(spec->kspcheap,spec->pcnone);CHKERRQ(ierr);

  ierr = KSPSetTolerances(spec->kspest,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,5);CHKERRQ(ierr);

  /* Make the "cheap" preconditioner cheap by default */
  ierr = KSPSetConvergenceTest(spec->kspcheap,KSPSkipConverged,0,0);CHKERRQ(ierr);
  ierr = KSPSetNormType(spec->kspcheap,KSP_NORM_NONE);CHKERRQ(ierr);
  ierr = KSPSetTolerances(spec->kspcheap,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,5);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
