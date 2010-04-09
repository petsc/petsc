#define PETSCKSP_DLL

/*
     The basic KSP routines, Create, View etc. are here.
*/
#include "private/kspimpl.h"      /*I "petscksp.h" I*/

/* Logging support */
PetscCookie PETSCKSP_DLLEXPORT KSP_COOKIE;
PetscLogEvent  KSP_GMRESOrthogonalization, KSP_SetUp, KSP_Solve;

/*
   Contains the list of registered KSP routines
*/
PetscFList KSPList = 0;
PetscTruth KSPRegisterAllCalled = PETSC_FALSE;

#undef __FUNCT__  
#define __FUNCT__ "KSPView"
/*@C 
   KSPView - Prints the KSP data structure.

   Collective on KSP

   Input Parameters:
+  ksp - the Krylov space context
-  viewer - visualization context

   Options Database Keys:
.  -ksp_view - print the ksp data structure at the end of a KSPSolve call

   Note:
   The available visualization contexts include
+     PETSC_VIEWER_STDOUT_SELF - standard output (default)
-     PETSC_VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their 
         data to the first processor to print. 

   The user can open an alternative visualization context with
   PetscViewerASCIIOpen() - output to a specified file.

   Level: beginner

.keywords: KSP, view

.seealso: PCView(), PetscViewerASCIIOpen()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPView(KSP ksp,PetscViewer viewer)
{
  const KSPType  type;
  PetscErrorCode ierr;
  PetscTruth     iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(((PetscObject)ksp)->comm);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE,2);
  PetscCheckSameComm(ksp,1,viewer,2);

  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = KSPGetType(ksp,&type);CHKERRQ(ierr);
    if (((PetscObject)ksp)->prefix) {
      ierr = PetscViewerASCIIPrintf(viewer,"KSP Object:(%s)\n",((PetscObject)ksp)->prefix);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"KSP Object:\n");CHKERRQ(ierr);
    }
    if (type) {
      ierr = PetscViewerASCIIPrintf(viewer,"  type: %s\n",type);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  type: not yet set\n");CHKERRQ(ierr);
    }
    if (ksp->ops->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*ksp->ops->view)(ksp,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    if (ksp->guess_zero) {ierr = PetscViewerASCIIPrintf(viewer,"  maximum iterations=%D, initial guess is zero\n",ksp->max_it);CHKERRQ(ierr);}
    else                 {ierr = PetscViewerASCIIPrintf(viewer,"  maximum iterations=%D\n", ksp->max_it);CHKERRQ(ierr);}
    if (ksp->guess_knoll) {ierr = PetscViewerASCIIPrintf(viewer,"  using preconditioner applied to right hand side for initial guess\n");CHKERRQ(ierr);}
    ierr = PetscViewerASCIIPrintf(viewer,"  tolerances:  relative=%G, absolute=%G, divergence=%G\n",ksp->rtol,ksp->abstol,ksp->divtol);CHKERRQ(ierr);
    if (ksp->pc_side == PC_RIGHT)          {ierr = PetscViewerASCIIPrintf(viewer,"  right preconditioning\n");CHKERRQ(ierr);}
    else if (ksp->pc_side == PC_SYMMETRIC) {ierr = PetscViewerASCIIPrintf(viewer,"  symmetric preconditioning\n");CHKERRQ(ierr);}
    else                                   {ierr = PetscViewerASCIIPrintf(viewer,"  left preconditioning\n");CHKERRQ(ierr);}
    if (ksp->guess) {ierr = PetscViewerASCIIPrintf(viewer,"  using Fischers initial guess method %D with size %D\n",ksp->guess->method,ksp->guess->maxl);CHKERRQ(ierr);}
    if (ksp->dscale) {ierr = PetscViewerASCIIPrintf(viewer,"  diagonally scaled system\n");CHKERRQ(ierr);}
    if (ksp->nullsp) {ierr = PetscViewerASCIIPrintf(viewer,"  has attached null space\n");CHKERRQ(ierr);}
    if (!ksp->guess_zero) {ierr = PetscViewerASCIIPrintf(viewer,"  using nonzero initial guess\n");CHKERRQ(ierr);}
    ierr = PetscViewerASCIIPrintf(viewer,"  using %s norm type for convergence test\n",KSPNormTypes[ksp->normtype]);CHKERRQ(ierr);
  } else {
    if (ksp->ops->view) {
      ierr = (*ksp->ops->view)(ksp,viewer);CHKERRQ(ierr);
    }
  }
  if (!ksp->pc) {ierr = KSPGetPC(ksp,&ksp->pc);CHKERRQ(ierr);}
  ierr = PCView(ksp->pc,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "KSPSetNormType"
/*@
   KSPSetNormType - Sets the norm that is used for convergence testing.

   Collective on KSP

   Input Parameter:
+  ksp - Krylov solver context
-  normtype - one of 
$   KSP_NORM_NO - skips computing the norm, this should only be used if you are using
$                 the Krylov method as a smoother with a fixed small number of iterations.
$                 Implicitly sets KSPSkipConverged as KSP convergence test.
$                 Supported only by CG, Richardson, Bi-CG-stab, CR, and CGS methods.
$   KSP_NORM_PRECONDITIONED - the default for left preconditioned solves, uses the l2 norm
$                 of the preconditioned residual
$   KSP_NORM_UNPRECONDITIONED - uses the l2 norm of the true b - Ax residual, supported only by
$                 CG, CHEBYCHEV, and RICHARDSON, automatically true for right (see KSPSetPreconditionerSide()) 
$                 preconditioning..
$   KSP_NORM_NATURAL - supported  by KSPCG, KSPCR, KSPCGNE, KSPCGS


   Options Database Key:
.   -ksp_norm_type <none,preconditioned,unpreconditioned,natural>

   Notes: 
   Currently only works with the CG, Richardson, Bi-CG-stab, CR, and CGS methods.

   Level: advanced

.keywords: KSP, create, context, norms

.seealso: KSPSetUp(), KSPSolve(), KSPDestroy(), KSPSkipConverged()                               
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPSetNormType(KSP ksp,KSPNormType normtype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  ksp->normtype = normtype;
  if (normtype == KSP_NORM_NO) {
    ierr = KSPSetConvergenceTest(ksp,KSPSkipConverged,0,0);CHKERRQ(ierr);
    ierr = PetscInfo(ksp,"Warning: setting KSPNormType to skip computing the norm\n\
 KSP convergence test is implicitly set to KSPSkipConverged\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSetCheckNormIteration"
/*@
   KSPSetCheckNormIteration - Sets the first iteration at which the norm of the residual will be 
     computed and used in the convergence test. 

   Collective on KSP

   Input Parameter:
+  ksp - Krylov solver context
-  it  - use -1 to check at all iterations

   Notes: 
   Currently only works with KSPCG, KSPBCGS and KSPIBCGS

   Use KSPSetNormType(ksp,KSP_NORM_NO) to never check the norm

   On steps where the norm is not computed, the previous norm is still in the variable, so if you run with, for example,
    -ksp_monitor the residual norm will appear to be unchanged for several iterations (though it is not really unchanged).
   Level: advanced

.keywords: KSP, create, context, norms

.seealso: KSPSetUp(), KSPSolve(), KSPDestroy(), KSPSkipConverged(), KSPSetNormType()                               
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPSetCheckNormIteration(KSP ksp,PetscInt it)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  ksp->chknorm = it;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSetLagNorm"
/*@
   KSPSetLagNorm - Lags the residual norm calculation so that it is computed as part of the MPI_Allreduce() for 
   computing the inner products for the next iteration.  This can reduce communication costs at the expense of doing 
   one additional iteration.


   Collective on KSP

   Input Parameter:
+  ksp - Krylov solver context
-  flg - PETSC_TRUE or PETSC_FALSE

   Options Database Keys:
.  -ksp_lag_norm - lag the calculated residual norm

   Notes: 
   Currently only works with KSPIBCGS.

   Use KSPSetNormType(ksp,KSP_NORM_NO) to never check the norm

   If you lag the norm and run with, for example, -ksp_monitor, the residual norm reported will be the lagged one.
   Level: advanced

.keywords: KSP, create, context, norms

.seealso: KSPSetUp(), KSPSolve(), KSPDestroy(), KSPSkipConverged(), KSPSetNormType()                               
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPSetLagNorm(KSP ksp,PetscTruth flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  ksp->lagnorm = flg;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPGetNormType"
/*@
   KSPGetNormType - Gets the norm that is used for convergence testing.

   Not Collective

   Input Parameter:
.  ksp - Krylov solver context

   Output Parameter:
.  normtype - norm that is used for convergence testing

   Level: advanced

.keywords: KSP, create, context, norms

.seealso: KSPNormType, KSPSetNormType(), KSPSkipConverged()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPGetNormType(KSP ksp, KSPNormType *normtype) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  PetscValidPointer(normtype, 2);
  *normtype = ksp->normtype;
  PetscFunctionReturn(0);
}

#if 0
#undef __FUNCT__  
#define __FUNCT__ "KSPPublish_Petsc"
static PetscErrorCode KSPPublish_Petsc(PetscObject obj)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__  
#define __FUNCT__ "KSPSetOperators"
/*@
   KSPSetOperators - Sets the matrix associated with the linear system
   and a (possibly) different one associated with the preconditioner. 

   Collective on KSP and Mat

   Input Parameters:
+  ksp - the KSP context
.  Amat - the matrix associated with the linear system
.  Pmat - the matrix to be used in constructing the preconditioner, usually the
          same as Amat. 
-  flag - flag indicating information about the preconditioner matrix structure
   during successive linear solves.  This flag is ignored the first time a
   linear system is solved, and thus is irrelevant when solving just one linear
   system.

   Notes: 
   The flag can be used to eliminate unnecessary work in the preconditioner 
   during the repeated solution of linear systems of the same size.  The
   available options are
$    SAME_PRECONDITIONER -
$      Pmat is identical during successive linear solves.
$      This option is intended for folks who are using
$      different Amat and Pmat matrices and want to reuse the
$      same preconditioner matrix.  For example, this option
$      saves work by not recomputing incomplete factorization
$      for ILU/ICC preconditioners.
$    SAME_NONZERO_PATTERN -
$      Pmat has the same nonzero structure during
$      successive linear solves. 
$    DIFFERENT_NONZERO_PATTERN -
$      Pmat does not have the same nonzero structure.

    Passing a PETSC_NULL for Amat or Pmat removes the matrix that is currently used.

    If you wish to replace either Amat or Pmat but leave the other one untouched then
    first call KSPGetOperators() to get the one you wish to keep, call PetscObjectReference()
    on it and then pass it back in in your call to KSPSetOperators().

    Caution:
    If you specify SAME_NONZERO_PATTERN, PETSc believes your assertion
    and does not check the structure of the matrix.  If you erroneously
    claim that the structure is the same when it actually is not, the new
    preconditioner will not function correctly.  Thus, use this optimization
    feature carefully!

    If in doubt about whether your preconditioner matrix has changed
    structure or not, use the flag DIFFERENT_NONZERO_PATTERN.

    Level: beginner

   Alternative usage: If the operators have NOT been set with KSP/PCSetOperators() then the operators
      are created in PC and returned to the user. In this case, if both operators
      mat and pmat are requested, two DIFFERENT operators will be returned. If
      only one is requested both operators in the PC will be the same (i.e. as
      if one had called KSP/PCSetOperators() with the same argument for both Mats).
      The user must set the sizes of the returned matrices and their type etc just
      as if the user created them with MatCreate(). For example,

$         KSP/PCGetOperators(ksp/pc,&mat,PETSC_NULL,PETSC_NULL); is equivalent to
$           set size, type, etc of mat

$         MatCreate(comm,&mat);
$         KSP/PCSetOperators(ksp/pc,mat,mat,SAME_NONZERO_PATTERN);
$         PetscObjectDereference((PetscObject)mat);
$           set size, type, etc of mat

     and

$         KSP/PCGetOperators(ksp/pc,&mat,&pmat,PETSC_NULL); is equivalent to
$           set size, type, etc of mat and pmat

$         MatCreate(comm,&mat);
$         MatCreate(comm,&pmat);
$         KSP/PCSetOperators(ksp/pc,mat,pmat,SAME_NONZERO_PATTERN);
$         PetscObjectDereference((PetscObject)mat);
$         PetscObjectDereference((PetscObject)pmat);
$           set size, type, etc of mat and pmat

    The rational for this support is so that when creating a TS, SNES, or KSP the hierarchy
    of underlying objects (i.e. SNES, KSP, PC, Mat) and their livespans can be completely 
    managed by the top most level object (i.e. the TS, SNES, or KSP). Another way to look
    at this is when you create a SNES you do not NEED to create a KSP and attach it to 
    the SNES object (the SNES object manages it for you). Similarly when you create a KSP
    you do not need to attach a PC to it (the KSP object manages the PC object for you).
    Thus, why should YOU have to create the Mat and attach it to the SNES/KSP/PC, when
    it can be created for you?

.keywords: KSP, set, operators, matrix, preconditioner, linear system

.seealso: KSPSolve(), KSPGetPC(), PCGetOperators(), PCSetOperators(), KSPGetOperators()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPSetOperators(KSP ksp,Mat Amat,Mat Pmat,MatStructure flag)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  if (Amat) PetscValidHeaderSpecific(Amat,MAT_COOKIE,2);
  if (Pmat) PetscValidHeaderSpecific(Pmat,MAT_COOKIE,3);
  if (Amat) PetscCheckSameComm(ksp,1,Amat,2);
  if (Pmat) PetscCheckSameComm(ksp,1,Pmat,3);
  if (!ksp->pc) {ierr = KSPGetPC(ksp,&ksp->pc);CHKERRQ(ierr);}
  ierr = PCSetOperators(ksp->pc,Amat,Pmat,flag);CHKERRQ(ierr);
  if (ksp->setupcalled > 1) ksp->setupcalled = 1;  /* so that next solve call will call setup */
  if (ksp->guess) {
    ierr = KSPFischerGuessReset(ksp->guess);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPGetOperators"
/*@
   KSPGetOperators - Gets the matrix associated with the linear system
   and a (possibly) different one associated with the preconditioner. 

   Collective on KSP and Mat

   Input Parameter:
.  ksp - the KSP context

   Output Parameters:
+  Amat - the matrix associated with the linear system
.  Pmat - the matrix to be used in constructing the preconditioner, usually the
          same as Amat. 
-  flag - flag indicating information about the preconditioner matrix structure
   during successive linear solves.  This flag is ignored the first time a
   linear system is solved, and thus is irrelevant when solving just one linear
   system.

    Level: intermediate

.keywords: KSP, set, get, operators, matrix, preconditioner, linear system

.seealso: KSPSolve(), KSPGetPC(), PCGetOperators(), PCSetOperators(), KSPSetOperators(), KSPGetOperatorsSet()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPGetOperators(KSP ksp,Mat *Amat,Mat *Pmat,MatStructure *flag)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  if (!ksp->pc) {ierr = KSPGetPC(ksp,&ksp->pc);CHKERRQ(ierr);}
  ierr = PCGetOperators(ksp->pc,Amat,Pmat,flag);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPGetOperatorsSet"
/*@C
   KSPGetOperatorsSet - Determines if the matrix associated with the linear system and
   possibly a different one associated with the preconditioner have been set in the KSP.

   Not collective, though the results on all processes should be the same

   Input Parameter:
.  pc - the preconditioner context

   Output Parameters:
+  mat - the matrix associated with the linear system was set
-  pmat - matrix associated with the preconditioner was set, usually the same

   Level: intermediate

.keywords: KSP, get, operators, matrix, linear system

.seealso: PCSetOperators(), KSPGetOperators(), KSPSetOperators(), PCGetOperators(), PCGetOperatorsSet()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPGetOperatorsSet(KSP ksp,PetscTruth *mat,PetscTruth *pmat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  if (!ksp->pc) {ierr = KSPGetPC(ksp,&ksp->pc);CHKERRQ(ierr);}
  ierr = PCGetOperatorsSet(ksp->pc,mat,pmat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPCreate"
/*@
   KSPCreate - Creates the default KSP context.

   Collective on MPI_Comm

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  ksp - location to put the KSP context

   Notes:
   The default KSP type is GMRES with a restart of 30, using modified Gram-Schmidt
   orthogonalization.

   Level: beginner

.keywords: KSP, create, context

.seealso: KSPSetUp(), KSPSolve(), KSPDestroy(), KSP
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPCreate(MPI_Comm comm,KSP *inksp)
{
  KSP            ksp;
  PetscErrorCode ierr;
  void           *ctx;

  PetscFunctionBegin;
  PetscValidPointer(inksp,2);
  *inksp = 0;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = KSPInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ierr = PetscHeaderCreate(ksp,_p_KSP,struct _KSPOps,KSP_COOKIE,-1,"KSP",comm,KSPDestroy,KSPView);CHKERRQ(ierr);

  ksp->max_it        = 10000;
  ksp->pc_side       = PC_LEFT;
  ksp->rtol          = 1.e-5;
  ksp->abstol        = 1.e-50;
  ksp->divtol        = 1.e4;
  
  ksp->chknorm             = -1;
  ksp->normtype            = KSP_NORM_PRECONDITIONED;
  ksp->rnorm               = 0.0;
  ksp->its                 = 0;
  ksp->guess_zero          = PETSC_TRUE;
  ksp->calc_sings          = PETSC_FALSE;
  ksp->res_hist            = PETSC_NULL;
  ksp->res_hist_alloc      = PETSC_NULL;
  ksp->res_hist_len        = 0;
  ksp->res_hist_max        = 0;
  ksp->res_hist_reset      = PETSC_TRUE;
  ksp->numbermonitors      = 0;

  ierr = KSPDefaultConvergedCreate(&ctx);CHKERRQ(ierr);
  ierr = KSPSetConvergenceTest(ksp,KSPDefaultConverged,ctx,KSPDefaultConvergedDestroy);CHKERRQ(ierr);
  ksp->ops->buildsolution  = KSPDefaultBuildSolution;
  ksp->ops->buildresidual  = KSPDefaultBuildResidual;

  ksp->vec_sol         = 0;
  ksp->vec_rhs         = 0;
  ksp->pc              = 0;
  ksp->data            = 0;
  ksp->nwork           = 0;
  ksp->work            = 0;
  ksp->reason          = KSP_CONVERGED_ITERATING;
  ksp->setupcalled     = 0;

  ierr = PetscPublishAll(ksp);CHKERRQ(ierr);
  *inksp = ksp;
  PetscFunctionReturn(0);
}
 
#undef __FUNCT__  
#define __FUNCT__ "KSPSetType"
/*@C
   KSPSetType - Builds KSP for a particular solver. 

   Collective on KSP

   Input Parameters:
+  ksp      - the Krylov space context
-  type - a known method

   Options Database Key:
.  -ksp_type  <method> - Sets the method; use -help for a list 
    of available methods (for instance, cg or gmres)

   Notes:  
   See "petsc/include/petscksp.h" for available methods (for instance,
   KSPCG or KSPGMRES).

  Normally, it is best to use the KSPSetFromOptions() command and
  then set the KSP type from the options database rather than by using
  this routine.  Using the options database provides the user with
  maximum flexibility in evaluating the many different Krylov methods.
  The KSPSetType() routine is provided for those situations where it
  is necessary to set the iterative solver independently of the command
  line or options database.  This might be the case, for example, when
  the choice of iterative solver changes during the execution of the
  program, and the user's application is taking responsibility for
  choosing the appropriate method.  In other words, this routine is
  not for beginners.

  Level: intermediate

.keywords: KSP, set, method

.seealso: PCSetType(), KSPType

@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPSetType(KSP ksp, const KSPType type)
{
  PetscErrorCode ierr,(*r)(KSP);
  PetscTruth     match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  PetscValidCharPointer(type,2);

  ierr = PetscTypeCompare((PetscObject)ksp,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr =  PetscFListFind(KSPList,((PetscObject)ksp)->comm,type,(void (**)(void)) &r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested KSP type %s",type);
  /* Destroy the previous private KSP context */
  if (ksp->ops->destroy) { ierr = (*ksp->ops->destroy)(ksp);CHKERRQ(ierr); }
  /* Reinitialize function pointers in KSPOps structure */
  ierr = PetscMemzero(ksp->ops,sizeof(struct _KSPOps));CHKERRQ(ierr);
  ksp->ops->buildsolution = KSPDefaultBuildSolution;
  ksp->ops->buildresidual = KSPDefaultBuildResidual;
  /* Call the KSPCreate_XXX routine for this particular Krylov solver */
  ksp->setupcalled = 0;
  ierr = (*r)(ksp);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)ksp,type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPRegisterDestroy"
/*@
   KSPRegisterDestroy - Frees the list of KSP methods that were
   registered by KSPRegisterDynamic().

   Not Collective

   Level: advanced

.keywords: KSP, register, destroy

.seealso: KSPRegisterDynamic(), KSPRegisterAll()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPRegisterDestroy(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFListDestroy(&KSPList);CHKERRQ(ierr);
  KSPRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPGetType"
/*@C
   KSPGetType - Gets the KSP type as a string from the KSP object.

   Not Collective

   Input Parameter:
.  ksp - Krylov context 

   Output Parameter:
.  name - name of KSP method 

   Level: intermediate

.keywords: KSP, get, method, name

.seealso: KSPSetType()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPGetType(KSP ksp,const KSPType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)ksp)->type_name;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPRegister"
/*@C
  KSPRegister - See KSPRegisterDynamic()

  Level: advanced
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPRegister(const char sname[],const char path[],const char name[],PetscErrorCode (*function)(KSP))
{
  PetscErrorCode ierr;
  char           fullname[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  ierr = PetscFListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = PetscFListAdd(&KSPList,sname,fullname,(void (*)(void))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSetNullSpace"
/*@
  KSPSetNullSpace - Sets the null space of the operator

  Collective on KSP

  Input Parameters:
+  ksp - the Krylov space object
-  nullsp - the null space of the operator

  Level: advanced

.seealso: KSPSetOperators(), MatNullSpaceCreate(), KSPGetNullSpace()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPSetNullSpace(KSP ksp,MatNullSpace nullsp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  PetscValidHeaderSpecific(nullsp,MAT_NULLSPACE_COOKIE,2);
  ierr = PetscObjectReference((PetscObject)nullsp);CHKERRQ(ierr);
  if (ksp->nullsp) { ierr = MatNullSpaceDestroy(ksp->nullsp);CHKERRQ(ierr); }
  ksp->nullsp = nullsp;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPGetNullSpace"
/*@
  KSPGetNullSpace - Gets the null space of the operator

  Collective on KSP

  Input Parameters:
+  ksp - the Krylov space object
-  nullsp - the null space of the operator

  Level: advanced

.seealso: KSPSetOperators(), MatNullSpaceCreate(), KSPSetNullSpace()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPGetNullSpace(KSP ksp,MatNullSpace *nullsp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  PetscValidPointer(nullsp,2);
  *nullsp = ksp->nullsp;
  PetscFunctionReturn(0);
}

