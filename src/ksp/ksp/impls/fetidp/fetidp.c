#include <petsc/private/kspimpl.h> /*I <petscksp.h> I*/
#include <../src/ksp/pc/impls/bddc/bddc.h>

/*
    This file implements the FETI-DP method in PETSc as part of KSP.
*/
typedef struct {
  KSP parentksp;
} KSP_FETIDPMon;

typedef struct {
  KSP           innerksp;        /* the KSP for the Lagrange multipliers */
  PC            innerbddc;       /* the inner BDDC object */
  PetscBool     fully_redundant; /* true for using a fully redundant set of multipliers */
  PetscBool     userbddc;        /* true if the user provided the PCBDDC object */
  KSP_FETIDPMon *monctx;         /* monitor context, used to pass user defined monitors for the physical */
} KSP_FETIDP;

#undef __FUNCT__
#define __FUNCT__ "KSPFETIDPGetInnerKSP_FETIDP"
static PetscErrorCode KSPFETIDPGetInnerKSP_FETIDP(KSP ksp,KSP* innerksp)
{
  KSP_FETIDP     *fetidp = (KSP_FETIDP*)ksp->data;

  PetscFunctionBegin;
  *innerksp = fetidp->innerksp;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPFETIDPGetInnerKSP"
/*@
 KSPFETIDPGetInnerKSP - Get the KSP object for the Lagrange multipliers

   Input Parameters:
+  ksp - the FETI-DP KSP
-  innerksp - the KSP for the multipliers

   Level: advanced

   Notes:

.seealso: MATIS, PCBDDC, KSPFETIDPSetInnerBDDC, KSPFETIDPGetInnerBDDC
@*/
PetscErrorCode KSPFETIDPGetInnerKSP(KSP ksp,KSP* innerksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidPointer(innerksp,2);
  ierr = PetscUseMethod(ksp,"KSPFETIDPGetInnerKSP_C",(KSP,KSP*),(ksp,innerksp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPFETIDPGetInnerBDDC_FETIDP"
static PetscErrorCode KSPFETIDPGetInnerBDDC_FETIDP(KSP ksp,PC* pc)
{
  KSP_FETIDP     *fetidp = (KSP_FETIDP*)ksp->data;

  PetscFunctionBegin;
  *pc = fetidp->innerbddc;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPFETIDPGetInnerBDDC"
/*@
 KSPFETIDPGetInnerBDDC - Get the PCBDDC object used to setup the FETI-DP matrix for the Lagrange multipliers

   Input Parameters:
+  ksp - the FETI-DP KSP
-  pc - the PCBDDC object

   Level: advanced

   Notes:

.seealso: MATIS, PCBDDC, KSPFETIDPSetInnerBDDC, KSPFETIDPGetInnerKSP
@*/
PetscErrorCode KSPFETIDPGetInnerBDDC(KSP ksp,PC* pc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidPointer(pc,2);
  ierr = PetscUseMethod(ksp,"KSPFETIDPGetInnerBDDC_C",(KSP,PC*),(ksp,pc));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPFETIDPSetInnerBDDC_FETIDP"
static PetscErrorCode KSPFETIDPSetInnerBDDC_FETIDP(KSP ksp,PC pc)
{
  KSP_FETIDP     *fetidp = (KSP_FETIDP*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectReference((PetscObject)pc);CHKERRQ(ierr);
  ierr = PCDestroy(&fetidp->innerbddc);CHKERRQ(ierr);
  fetidp->innerbddc = pc;
  fetidp->userbddc = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPFETIDPSetInnerBDDC"
/*@
 KSPFETIDPSetInnerBDDC - Set the PCBDDC object used to setup the FETI-DP matrix for the Lagrange multipliers

   Collective on KSP

   Input Parameters:
+  ksp - the FETI-DP KSP
-  pc - the PCBDDC object

   Level: advanced

   Notes:

.seealso: MATIS, PCBDDC, KSPFETIDPGetInnerBDDC, KSPFETIDPGetInnerKSP
@*/
PetscErrorCode KSPFETIDPSetInnerBDDC(KSP ksp,PC pc)
{
  PetscBool      isbddc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidHeaderSpecific(pc,PC_CLASSID,2);
  ierr = PetscObjectTypeCompare((PetscObject)pc,PCBDDC,&isbddc);CHKERRQ(ierr);
  if (!isbddc) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_WRONG,"KSPFETIDPSetInnerBDDC need a PCBDDC preconditioner");
  ierr = PetscTryMethod(ksp,"KSPFETIDPSetInnerBDDC_C",(KSP,PC),(ksp,pc));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPBuildSolution_FETIDP"
static PetscErrorCode KSPBuildSolution_FETIDP(KSP ksp,Vec v,Vec *V)
{
  KSP_FETIDP     *fetidp = (KSP_FETIDP*)ksp->data;
  Mat            F;
  Vec            Xl;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPGetOperators(fetidp->innerksp,&F,NULL);CHKERRQ(ierr);
  ierr = KSPBuildSolution(fetidp->innerksp,NULL,&Xl);CHKERRQ(ierr);
  if (v) {
    ierr = PCBDDCMatFETIDPGetSolution(F,Xl,v);CHKERRQ(ierr);
    *V = v;
  } else {
    ierr = PCBDDCMatFETIDPGetSolution(F,Xl,*V);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPMonitor_FETIDP"
static PetscErrorCode KSPMonitor_FETIDP(KSP ksp,PetscInt it,PetscReal rnorm,void* ctx)
{
  KSP_FETIDPMon  *monctx = (KSP_FETIDPMon*)ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPMonitor(monctx->parentksp,it,rnorm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPComputeEigenvalues_FETIDP"
static PetscErrorCode KSPComputeEigenvalues_FETIDP(KSP ksp,PetscInt nmax,PetscReal *r,PetscReal *c,PetscInt *neig)
{
  KSP_FETIDP     *fetidp = (KSP_FETIDP*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPComputeEigenvalues(fetidp->innerksp,nmax,r,c,neig);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPComputeExtremeSingularValues_FETIDP"
static PetscErrorCode KSPComputeExtremeSingularValues_FETIDP(KSP ksp,PetscReal *emax,PetscReal *emin)
{
  KSP_FETIDP     *fetidp = (KSP_FETIDP*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPComputeExtremeSingularValues(fetidp->innerksp,emax,emin);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSetUp_FETIDP"
static PetscErrorCode KSPSetUp_FETIDP(KSP ksp)
{
  KSP_FETIDP     *fetidp = (KSP_FETIDP*)ksp->data;
  PC_BDDC        *pcbddc = (PC_BDDC*)fetidp->innerbddc->data;
  Mat            A,Ap;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* set up BDDC */
  ierr = KSPGetOperators(ksp,&A,&Ap);CHKERRQ(ierr);
  ierr = PCSetOperators(fetidp->innerbddc,A,Ap);CHKERRQ(ierr);
  ierr = PCSetUp(fetidp->innerbddc);CHKERRQ(ierr);

  /* FETI-DP as it is implemented needs an exact coarse solver */
  if (pcbddc->coarse_ksp) {
    ierr = KSPSetTolerances(pcbddc->coarse_ksp,PETSC_SMALL,PETSC_SMALL,PETSC_DEFAULT,1000);CHKERRQ(ierr);
    ierr = KSPSetNormType(pcbddc->coarse_ksp,KSP_NORM_DEFAULT);CHKERRQ(ierr);
  }
  /* FETI-DP as it is implemented needs exact local solvers */
  ierr = KSPSetTolerances(pcbddc->ksp_R,PETSC_SMALL,PETSC_SMALL,PETSC_DEFAULT,1000);CHKERRQ(ierr);
  ierr = KSPSetNormType(pcbddc->ksp_R,KSP_NORM_DEFAULT);CHKERRQ(ierr);
  /* if the primal space is changed, setup F */
  if (pcbddc->new_primal_space) {
    Mat F; /* the FETI-DP matrix */
    PC  D; /* the FETI-DP preconditioner */
    ierr = KSPReset(fetidp->innerksp);CHKERRQ(ierr);
    ierr = PCBDDCCreateFETIDPOperators(fetidp->innerbddc,fetidp->fully_redundant,&F,&D);CHKERRQ(ierr);
    ierr = KSPSetOperators(fetidp->innerksp,F,F);CHKERRQ(ierr);
    ierr = KSPSetTolerances(fetidp->innerksp,ksp->rtol,ksp->abstol,ksp->divtol,ksp->max_it);CHKERRQ(ierr);
    ierr = KSPSetPC(fetidp->innerksp,D);CHKERRQ(ierr);
    ierr = MatCreateVecs(F,&(fetidp->innerksp)->vec_rhs,&(fetidp->innerksp)->vec_sol);CHKERRQ(ierr);
    ierr = MatDestroy(&F);CHKERRQ(ierr);
    ierr = PCDestroy(&D);CHKERRQ(ierr);
  }
  /* propagate settings to inner solve */
  ierr = KSPGetComputeSingularValues(ksp,&flg);CHKERRQ(ierr);
  ierr = KSPSetComputeSingularValues(fetidp->innerksp,flg);CHKERRQ(ierr);
  if (ksp->res_hist) {
    ierr = KSPSetResidualHistory(fetidp->innerksp,ksp->res_hist,ksp->res_hist_max,ksp->res_hist_reset);CHKERRQ(ierr);
  }
  ierr = KSPSetUp(fetidp->innerksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSolve_FETIDP"
static PetscErrorCode KSPSolve_FETIDP(KSP ksp)
{
  PetscErrorCode ierr;
  Mat            F;
  Vec            X,B,Xl,Bl;
  KSP_FETIDP     *fetidp = (KSP_FETIDP*)ksp->data;

  PetscFunctionBegin;
  ierr = KSPGetRhs(ksp,&B);CHKERRQ(ierr);
  ierr = KSPGetSolution(ksp,&X);CHKERRQ(ierr);
  ierr = KSPGetOperators(fetidp->innerksp,&F,NULL);CHKERRQ(ierr);
  ierr = KSPGetRhs(fetidp->innerksp,&Bl);CHKERRQ(ierr);
  ierr = KSPGetSolution(fetidp->innerksp,&Xl);CHKERRQ(ierr);
  ierr = PCBDDCMatFETIDPGetRHS(F,B,Bl);CHKERRQ(ierr);
  if (ksp->transpose_solve) {
    ierr = KSPSolveTranspose(fetidp->innerksp,Bl,Xl);CHKERRQ(ierr);
  } else {
    ierr = KSPSolve(fetidp->innerksp,Bl,Xl);CHKERRQ(ierr);
  }
  ierr = PCBDDCMatFETIDPGetSolution(F,Xl,X);CHKERRQ(ierr);
  /* update ksp with stats from inner ksp */
  ierr = KSPGetConvergedReason(fetidp->innerksp,&ksp->reason);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(fetidp->innerksp,&ksp->its);CHKERRQ(ierr);
  ksp->totalits += ksp->its;
  ierr = KSPGetResidualHistory(fetidp->innerksp,NULL,&ksp->res_hist_len);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPDestroy_FETIDP"
static PetscErrorCode KSPDestroy_FETIDP(KSP ksp)
{
  KSP_FETIDP     *fetidp = (KSP_FETIDP*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCDestroy(&fetidp->innerbddc);CHKERRQ(ierr);
  ierr = KSPDestroy(&fetidp->innerksp);CHKERRQ(ierr);
  ierr = PetscFree(fetidp->monctx);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPFETIDPSetInnerBDDC_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPFETIDPGetInnerBDDC_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPFETIDPGetInnerKSP_C",NULL);CHKERRQ(ierr);
  ierr = PetscFree(ksp->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPView_FETIDP"
static PetscErrorCode KSPView_FETIDP(KSP ksp,PetscViewer viewer)
{
  KSP_FETIDP     *fetidp = (KSP_FETIDP*)ksp->data;
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  FETI_DP: fully redundant: %d\n",fetidp->fully_redundant);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  FETI_DP: inner solver details\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIAddTab(viewer,2);CHKERRQ(ierr);
  }
  ierr = KSPView(fetidp->innerksp,viewer);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIISubtractTab(viewer,2);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  FETI_DP: BDDC solver details\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIAddTab(viewer,2);CHKERRQ(ierr);
  }
  ierr = PCView(fetidp->innerbddc,viewer);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIISubtractTab(viewer,2);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSetFromOptions_FETIDP"
static PetscErrorCode KSPSetFromOptions_FETIDP(PetscOptionItems *PetscOptionsObject,KSP ksp)
{
  KSP_FETIDP     *fetidp = (KSP_FETIDP*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* set options prefixes for the inner objects, since the parent prefix will be valid at this point */
  ierr = PetscObjectSetOptionsPrefix((PetscObject)fetidp->innerksp,((PetscObject)ksp)->prefix);CHKERRQ(ierr);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)fetidp->innerksp,"fetidp_");CHKERRQ(ierr);
  if (!fetidp->userbddc) {
    ierr = PetscObjectSetOptionsPrefix((PetscObject)fetidp->innerbddc,((PetscObject)ksp)->prefix);CHKERRQ(ierr);
    ierr = PetscObjectAppendOptionsPrefix((PetscObject)fetidp->innerbddc,"fetidp_bddc_");CHKERRQ(ierr);
  }
  ierr = PetscOptionsHead(PetscOptionsObject,"KSP FETIDP options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ksp_fetidp_fullyredundant","Use fully redundant multipliers","none",fetidp->fully_redundant,&fetidp->fully_redundant,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  ierr = PCSetFromOptions(fetidp->innerbddc);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(fetidp->innerksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
     KSPFETIDP - The FETI-DP method

   This class implements the FETI-DP method [1].
   The preconditioning matrix for the KSP must be of type MATIS.
   The FETI-DP linear system (automatically generated constructing an internal PCBDDC object) is solved using an inner KSP object.

   Options Database Keys:
.   -ksp_fetidp_fullyredundant <false> : use a fully redundant set of Lagrange multipliers

   Level: Advanced

   Notes: Options for the inner KSP and for the customization of the PCBDDC object can be specified at command line by using the prefixes -fetidp_ and -fetidp_bddc_. E.g.,
.vb
      -fetidp_ksp_type gmres -fetidp_bddc_pc_bddc_symmetric false
.ve
   will use GMRES for the solution of the linear system on the Lagrange multipliers, generated using a non-symmetric PCBDDC.

   References:
.  [1] - C. Farhat, M. Lesoinne, P. LeTallec, K. Pierson, and D. Rixen, FETI-DP: a dual-primal unified FETI method. I. A faster alternative to the two-level FETI method, Internat. J. Numer. Methods Engrg., 50 (2001), pp. 1523--1544

.seealso: MATIS, PCBDDC, KSPFETIDPSetInnerBDDC, KSPFETIDPGetInnerBDDC, KSPFETIDPGetInnerKSP
M*/
#undef __FUNCT__
#define __FUNCT__ "KSPCreate_FETIDP"
PETSC_EXTERN PetscErrorCode KSPCreate_FETIDP(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_FETIDP     *fetidp;
  KSP_FETIDPMon  *monctx;
  PC_BDDC        *pcbddc;
  PC             pc;

  PetscFunctionBegin;
  ierr = PetscNewLog(ksp,&fetidp);CHKERRQ(ierr);
  ksp->data = (void*)fetidp;
  ksp->ops->setup                        = KSPSetUp_FETIDP;
  ksp->ops->solve                        = KSPSolve_FETIDP;
  ksp->ops->destroy                      = KSPDestroy_FETIDP;
  ksp->ops->computeeigenvalues           = KSPComputeEigenvalues_FETIDP;
  ksp->ops->computeextremesingularvalues = KSPComputeExtremeSingularValues_FETIDP;
  ksp->ops->view                         = KSPView_FETIDP;
  ksp->ops->setfromoptions               = KSPSetFromOptions_FETIDP;
  ksp->ops->buildsolution                = KSPBuildSolution_FETIDP;
  ksp->ops->buildresidual                = KSPBuildResidualDefault;
  /* create the inner KSP for the Lagrange multipliers */
  ierr = KSPCreate(PetscObjectComm((PetscObject)ksp),&fetidp->innerksp);CHKERRQ(ierr);
  ierr = KSPGetPC(fetidp->innerksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)ksp,(PetscObject)fetidp->innerksp);CHKERRQ(ierr);
  /* monitor */
  ierr = PetscNew(&monctx);CHKERRQ(ierr);
  monctx->parentksp = ksp;
  fetidp->monctx = monctx;
  ierr = KSPMonitorSet(fetidp->innerksp,KSPMonitor_FETIDP,fetidp->monctx,NULL);CHKERRQ(ierr);
  /* create the inner BDDC */
  ierr = PCCreate(PetscObjectComm((PetscObject)ksp),&fetidp->innerbddc);CHKERRQ(ierr);
  ierr = PCSetType(fetidp->innerbddc,PCBDDC);CHKERRQ(ierr);
  /* make sure we always obtain a consistent FETI-DP matrix
     for symmetric problems, the user can always customize it through the command line */
  pcbddc = (PC_BDDC*)fetidp->innerbddc->data;
  pcbddc->symmetric_primal = PETSC_FALSE;
  ierr = PetscLogObjectParent((PetscObject)ksp,(PetscObject)fetidp->innerbddc);CHKERRQ(ierr);
  /* composed functions */
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPFETIDPSetInnerBDDC_C",KSPFETIDPSetInnerBDDC_FETIDP);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPFETIDPGetInnerBDDC_C",KSPFETIDPGetInnerBDDC_FETIDP);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPFETIDPGetInnerKSP_C",KSPFETIDPGetInnerKSP_FETIDP);CHKERRQ(ierr);
  /* need to call KSPSetUp_FETIDP even with KSP_SETUP_NEWMATRIX */
  ksp->setupnewmatrix = PETSC_TRUE;
  PetscFunctionReturn(0);
}
