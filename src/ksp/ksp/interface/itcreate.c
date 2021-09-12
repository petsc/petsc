/*
     The basic KSP routines, Create, View etc. are here.
*/
#include <petsc/private/kspimpl.h>      /*I "petscksp.h" I*/

/* Logging support */
PetscClassId  KSP_CLASSID;
PetscClassId  DMKSP_CLASSID;
PetscClassId  KSPGUESS_CLASSID;
PetscLogEvent KSP_GMRESOrthogonalization, KSP_SetUp, KSP_Solve, KSP_SolveTranspose, KSP_MatSolve;

/*
   Contains the list of registered KSP routines
*/
PetscFunctionList KSPList              = NULL;
PetscBool         KSPRegisterAllCalled = PETSC_FALSE;

/*
   Contains the list of registered KSP monitors
*/
PetscFunctionList KSPMonitorList              = NULL;
PetscFunctionList KSPMonitorCreateList        = NULL;
PetscFunctionList KSPMonitorDestroyList       = NULL;
PetscBool         KSPMonitorRegisterAllCalled = PETSC_FALSE;

/*@C
  KSPLoad - Loads a KSP that has been stored in binary  with KSPView().

  Collective on viewer

  Input Parameters:
+ newdm - the newly loaded KSP, this needs to have been created with KSPCreate() or
           some related function before a call to KSPLoad().
- viewer - binary file viewer, obtained from PetscViewerBinaryOpen()

   Level: intermediate

  Notes:
   The type is determined by the data in the file, any type set into the KSP before this call is ignored.

  Notes for advanced users:
  Most users should not need to know the details of the binary storage
  format, since KSPLoad() and KSPView() completely hide these details.
  But for anyone who's interested, the standard binary matrix storage
  format is
.vb
     has not yet been determined
.ve

.seealso: PetscViewerBinaryOpen(), KSPView(), MatLoad(), VecLoad()
@*/
PetscErrorCode  KSPLoad(KSP newdm, PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isbinary;
  PetscInt       classid;
  char           type[256];
  PC             pc;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(newdm,KSP_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (!isbinary) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid viewer; open viewer with PetscViewerBinaryOpen()");

  ierr = PetscViewerBinaryRead(viewer,&classid,1,NULL,PETSC_INT);CHKERRQ(ierr);
  if (classid != KSP_FILE_CLASSID) SETERRQ(PetscObjectComm((PetscObject)newdm),PETSC_ERR_ARG_WRONG,"Not KSP next in file");
  ierr = PetscViewerBinaryRead(viewer,type,256,NULL,PETSC_CHAR);CHKERRQ(ierr);
  ierr = KSPSetType(newdm, type);CHKERRQ(ierr);
  if (newdm->ops->load) {
    ierr = (*newdm->ops->load)(newdm,viewer);CHKERRQ(ierr);
  }
  ierr = KSPGetPC(newdm,&pc);CHKERRQ(ierr);
  ierr = PCLoad(pc,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <petscdraw.h>
#if defined(PETSC_HAVE_SAWS)
#include <petscviewersaws.h>
#endif
/*@C
   KSPView - Prints the KSP data structure.

   Collective on ksp

   Input Parameters:
+  ksp - the Krylov space context
-  viewer - visualization context

   Options Database Keys:
.  -ksp_view - print the KSP data structure at the end of a KSPSolve call

   Note:
   The available visualization contexts include
+     PETSC_VIEWER_STDOUT_SELF - standard output (default)
-     PETSC_VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their
         data to the first processor to print.

   The available formats include
+     PETSC_VIEWER_DEFAULT - standard output (default)
-     PETSC_VIEWER_ASCII_INFO_DETAIL - more verbose output for PCBJACOBI and PCASM

   The user can open an alternative visualization context with
   PetscViewerASCIIOpen() - output to a specified file.

  In the debugger you can do "call KSPView(ksp,0)" to display the KSP. (The same holds for any PETSc object viewer).

   Level: beginner

.seealso: PCView(), PetscViewerASCIIOpen()
@*/
PetscErrorCode  KSPView(KSP ksp,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      iascii,isbinary,isdraw,isstring;
#if defined(PETSC_HAVE_SAWS)
  PetscBool      issaws;
#endif

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)ksp),&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(ksp,1,viewer,2);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring);CHKERRQ(ierr);
#if defined(PETSC_HAVE_SAWS)
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSAWS,&issaws);CHKERRQ(ierr);
#endif
  if (iascii) {
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)ksp,viewer);CHKERRQ(ierr);
    if (ksp->ops->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*ksp->ops->view)(ksp,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    if (ksp->guess_zero) {
      ierr = PetscViewerASCIIPrintf(viewer,"  maximum iterations=%D, initial guess is zero\n",ksp->max_it);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  maximum iterations=%D, nonzero initial guess\n", ksp->max_it);CHKERRQ(ierr);
    }
    if (ksp->guess_knoll) {ierr = PetscViewerASCIIPrintf(viewer,"  using preconditioner applied to right hand side for initial guess\n");CHKERRQ(ierr);}
    ierr = PetscViewerASCIIPrintf(viewer,"  tolerances:  relative=%g, absolute=%g, divergence=%g\n",(double)ksp->rtol,(double)ksp->abstol,(double)ksp->divtol);CHKERRQ(ierr);
    if (ksp->pc_side == PC_RIGHT) {
      ierr = PetscViewerASCIIPrintf(viewer,"  right preconditioning\n");CHKERRQ(ierr);
    } else if (ksp->pc_side == PC_SYMMETRIC) {
      ierr = PetscViewerASCIIPrintf(viewer,"  symmetric preconditioning\n");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  left preconditioning\n");CHKERRQ(ierr);
    }
    if (ksp->guess) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = KSPGuessView(ksp->guess,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    if (ksp->dscale) {ierr = PetscViewerASCIIPrintf(viewer,"  diagonally scaled system\n");CHKERRQ(ierr);}
    ierr = PetscViewerASCIIPrintf(viewer,"  using %s norm type for convergence test\n",KSPNormTypes[ksp->normtype]);CHKERRQ(ierr);
  } else if (isbinary) {
    PetscInt    classid = KSP_FILE_CLASSID;
    MPI_Comm    comm;
    PetscMPIInt rank;
    char        type[256];

    ierr = PetscObjectGetComm((PetscObject)ksp,&comm);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
    if (!rank) {
      ierr = PetscViewerBinaryWrite(viewer,&classid,1,PETSC_INT);CHKERRQ(ierr);
      ierr = PetscStrncpy(type,((PetscObject)ksp)->type_name,256);CHKERRQ(ierr);
      ierr = PetscViewerBinaryWrite(viewer,type,256,PETSC_CHAR);CHKERRQ(ierr);
    }
    if (ksp->ops->view) {
      ierr = (*ksp->ops->view)(ksp,viewer);CHKERRQ(ierr);
    }
  } else if (isstring) {
    const char *type;
    ierr = KSPGetType(ksp,&type);CHKERRQ(ierr);
    ierr = PetscViewerStringSPrintf(viewer," KSPType: %-7.7s",type);CHKERRQ(ierr);
    if (ksp->ops->view) {ierr = (*ksp->ops->view)(ksp,viewer);CHKERRQ(ierr);}
  } else if (isdraw) {
    PetscDraw draw;
    char      str[36];
    PetscReal x,y,bottom,h;
    PetscBool flg;

    ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
    ierr = PetscDrawGetCurrentPoint(draw,&x,&y);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)ksp,KSPPREONLY,&flg);CHKERRQ(ierr);
    if (!flg) {
      ierr   = PetscStrncpy(str,"KSP: ",sizeof(str));CHKERRQ(ierr);
      ierr   = PetscStrlcat(str,((PetscObject)ksp)->type_name,sizeof(str));CHKERRQ(ierr);
      ierr   = PetscDrawStringBoxed(draw,x,y,PETSC_DRAW_RED,PETSC_DRAW_BLACK,str,NULL,&h);CHKERRQ(ierr);
      bottom = y - h;
    } else {
      bottom = y;
    }
    ierr = PetscDrawPushCurrentPoint(draw,x,bottom);CHKERRQ(ierr);
#if defined(PETSC_HAVE_SAWS)
  } else if (issaws) {
    PetscMPIInt rank;
    const char  *name;

    ierr = PetscObjectGetName((PetscObject)ksp,&name);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
    if (!((PetscObject)ksp)->amsmem && !rank) {
      char       dir[1024];

      ierr = PetscObjectViewSAWs((PetscObject)ksp,viewer);CHKERRQ(ierr);
      ierr = PetscSNPrintf(dir,1024,"/PETSc/Objects/%s/its",name);CHKERRQ(ierr);
      PetscStackCallSAWs(SAWs_Register,(dir,&ksp->its,1,SAWs_READ,SAWs_INT));
      if (!ksp->res_hist) {
        ierr = KSPSetResidualHistory(ksp,NULL,PETSC_DECIDE,PETSC_TRUE);CHKERRQ(ierr);
      }
      ierr = PetscSNPrintf(dir,1024,"/PETSc/Objects/%s/res_hist",name);CHKERRQ(ierr);
      PetscStackCallSAWs(SAWs_Register,(dir,ksp->res_hist,10,SAWs_READ,SAWs_DOUBLE));
    }
#endif
  } else if (ksp->ops->view) {
    ierr = (*ksp->ops->view)(ksp,viewer);CHKERRQ(ierr);
  }
  if (ksp->pc) {
    ierr = PCView(ksp->pc,viewer);CHKERRQ(ierr);
  }
  if (isdraw) {
    PetscDraw draw;
    ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
    ierr = PetscDrawPopCurrentPoint(draw);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   KSPViewFromOptions - View from Options

   Collective on KSP

   Input Parameters:
+  A - Krylov solver context
.  obj - Optional object
-  name - command line option

   Level: intermediate
.seealso:  KSP, KSPView, PetscObjectViewFromOptions(), KSPCreate()
@*/
PetscErrorCode  KSPViewFromOptions(KSP A,PetscObject obj,const char name[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,KSP_CLASSID,1);
  ierr = PetscObjectViewFromOptions((PetscObject)A,obj,name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   KSPSetNormType - Sets the norm that is used for convergence testing.

   Logically Collective on ksp

   Input Parameters:
+  ksp - Krylov solver context
-  normtype - one of
$   KSP_NORM_NONE - skips computing the norm, this should generally only be used if you are using
$                 the Krylov method as a smoother with a fixed small number of iterations.
$                 Implicitly sets KSPConvergedSkip() as KSP convergence test.
$                 Note that certain algorithms such as KSPGMRES ALWAYS require the norm calculation,
$                 for these methods the norms are still computed, they are just not used in
$                 the convergence test.
$   KSP_NORM_PRECONDITIONED - the default for left preconditioned solves, uses the l2 norm
$                 of the preconditioned residual P^{-1}(b - A x)
$   KSP_NORM_UNPRECONDITIONED - uses the l2 norm of the true b - Ax residual.
$   KSP_NORM_NATURAL - supported  by KSPCG, KSPCR, KSPCGNE, KSPCGS

   Options Database Key:
.   -ksp_norm_type <none,preconditioned,unpreconditioned,natural>

   Notes:
   Not all combinations of preconditioner side (see KSPSetPCSide()) and norm type are supported by all Krylov methods.
   If only one is set, PETSc tries to automatically change the other to find a compatible pair.  If no such combination
   is supported, PETSc will generate an error.

   Developer Notes:
   Supported combinations of norm and preconditioner side are set using KSPSetSupportedNorm().

   Level: advanced

.seealso: KSPSetUp(), KSPSolve(), KSPDestroy(), KSPConvergedSkip(), KSPSetCheckNormIteration(), KSPSetPCSide(), KSPGetPCSide(), KSPNormType
@*/
PetscErrorCode  KSPSetNormType(KSP ksp,KSPNormType normtype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveEnum(ksp,normtype,2);
  ksp->normtype = ksp->normtype_set = normtype;
  PetscFunctionReturn(0);
}

/*@
   KSPSetCheckNormIteration - Sets the first iteration at which the norm of the residual will be
     computed and used in the convergence test.

   Logically Collective on ksp

   Input Parameters:
+  ksp - Krylov solver context
-  it  - use -1 to check at all iterations

   Notes:
   Currently only works with KSPCG, KSPBCGS and KSPIBCGS

   Use KSPSetNormType(ksp,KSP_NORM_NONE) to never check the norm

   On steps where the norm is not computed, the previous norm is still in the variable, so if you run with, for example,
    -ksp_monitor the residual norm will appear to be unchanged for several iterations (though it is not really unchanged).
   Level: advanced

.seealso: KSPSetUp(), KSPSolve(), KSPDestroy(), KSPConvergedSkip(), KSPSetNormType()
@*/
PetscErrorCode  KSPSetCheckNormIteration(KSP ksp,PetscInt it)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveInt(ksp,it,2);
  ksp->chknorm = it;
  PetscFunctionReturn(0);
}

/*@
   KSPSetLagNorm - Lags the residual norm calculation so that it is computed as part of the MPI_Allreduce() for
   computing the inner products for the next iteration.  This can reduce communication costs at the expense of doing
   one additional iteration.

   Logically Collective on ksp

   Input Parameters:
+  ksp - Krylov solver context
-  flg - PETSC_TRUE or PETSC_FALSE

   Options Database Keys:
.  -ksp_lag_norm - lag the calculated residual norm

   Notes:
   Currently only works with KSPIBCGS.

   Use KSPSetNormType(ksp,KSP_NORM_NONE) to never check the norm

   If you lag the norm and run with, for example, -ksp_monitor, the residual norm reported will be the lagged one.
   Level: advanced

.seealso: KSPSetUp(), KSPSolve(), KSPDestroy(), KSPConvergedSkip(), KSPSetNormType(), KSPSetCheckNormIteration()
@*/
PetscErrorCode  KSPSetLagNorm(KSP ksp,PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveBool(ksp,flg,2);
  ksp->lagnorm = flg;
  PetscFunctionReturn(0);
}

/*@
   KSPSetSupportedNorm - Sets a norm and preconditioner side supported by a KSP

   Logically Collective

   Input Arguments:
+  ksp - Krylov method
.  normtype - supported norm type
.  pcside - preconditioner side that can be used with this norm
-  priority - positive integer preference for this combination; larger values have higher priority

   Level: developer

   Notes:
   This function should be called from the implementation files KSPCreate_XXX() to declare
   which norms and preconditioner sides are supported. Users should not need to call this
   function.

.seealso: KSPSetNormType(), KSPSetPCSide()
@*/
PetscErrorCode KSPSetSupportedNorm(KSP ksp,KSPNormType normtype,PCSide pcside,PetscInt priority)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  ksp->normsupporttable[normtype][pcside] = priority;
  PetscFunctionReturn(0);
}

PetscErrorCode KSPNormSupportTableReset_Private(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMemzero(ksp->normsupporttable,sizeof(ksp->normsupporttable));CHKERRQ(ierr);
  ksp->pc_side  = ksp->pc_side_set;
  ksp->normtype = ksp->normtype_set;
  PetscFunctionReturn(0);
}

PetscErrorCode KSPSetUpNorms_Private(KSP ksp,PetscBool errorifnotsupported,KSPNormType *normtype,PCSide *pcside)
{
  PetscInt i,j,best,ibest = 0,jbest = 0;

  PetscFunctionBegin;
  best = 0;
  for (i=0; i<KSP_NORM_MAX; i++) {
    for (j=0; j<PC_SIDE_MAX; j++) {
      if ((ksp->normtype == KSP_NORM_DEFAULT || ksp->normtype == i) && (ksp->pc_side == PC_SIDE_DEFAULT || ksp->pc_side == j) && (ksp->normsupporttable[i][j] > best)) {
        best  = ksp->normsupporttable[i][j];
        ibest = i;
        jbest = j;
      }
    }
  }
  if (best < 1 && errorifnotsupported) {
    if (ksp->normtype == KSP_NORM_DEFAULT && ksp->pc_side == PC_SIDE_DEFAULT) SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_PLIB,"The %s KSP implementation did not call KSPSetSupportedNorm()",((PetscObject)ksp)->type_name);
    if (ksp->normtype == KSP_NORM_DEFAULT) SETERRQ2(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"KSP %s does not support %s",((PetscObject)ksp)->type_name,PCSides[ksp->pc_side]);
    if (ksp->pc_side == PC_SIDE_DEFAULT) SETERRQ2(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"KSP %s does not support %s",((PetscObject)ksp)->type_name,KSPNormTypes[ksp->normtype]);
    SETERRQ3(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"KSP %s does not support %s with %s",((PetscObject)ksp)->type_name,KSPNormTypes[ksp->normtype],PCSides[ksp->pc_side]);
  }
  if (normtype) *normtype = (KSPNormType)ibest;
  if (pcside)   *pcside   = (PCSide)jbest;
  PetscFunctionReturn(0);
}

/*@
   KSPGetNormType - Gets the norm that is used for convergence testing.

   Not Collective

   Input Parameter:
.  ksp - Krylov solver context

   Output Parameter:
.  normtype - norm that is used for convergence testing

   Level: advanced

.seealso: KSPNormType, KSPSetNormType(), KSPConvergedSkip()
@*/
PetscErrorCode  KSPGetNormType(KSP ksp, KSPNormType *normtype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidPointer(normtype,2);
  ierr      = KSPSetUpNorms_Private(ksp,PETSC_TRUE,&ksp->normtype,&ksp->pc_side);CHKERRQ(ierr);
  *normtype = ksp->normtype;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_SAWS)
#include <petscviewersaws.h>
#endif

/*@
   KSPSetOperators - Sets the matrix associated with the linear system
   and a (possibly) different one associated with the preconditioner.

   Collective on ksp

   Input Parameters:
+  ksp - the KSP context
.  Amat - the matrix that defines the linear system
-  Pmat - the matrix to be used in constructing the preconditioner, usually the same as Amat.

   Notes:

    If you know the operator Amat has a null space you can use MatSetNullSpace() and MatSetTransposeNullSpace() to supply the null
    space to Amat and the KSP solvers will automatically use that null space as needed during the solution process.

    All future calls to KSPSetOperators() must use the same size matrices!

    Passing a NULL for Amat or Pmat removes the matrix that is currently used.

    If you wish to replace either Amat or Pmat but leave the other one untouched then
    first call KSPGetOperators() to get the one you wish to keep, call PetscObjectReference()
    on it and then pass it back in in your call to KSPSetOperators().

    Level: beginner

   Alternative usage: If the operators have NOT been set with KSP/PCSetOperators() then the operators
      are created in PC and returned to the user. In this case, if both operators
      mat and pmat are requested, two DIFFERENT operators will be returned. If
      only one is requested both operators in the PC will be the same (i.e. as
      if one had called KSP/PCSetOperators() with the same argument for both Mats).
      The user must set the sizes of the returned matrices and their type etc just
      as if the user created them with MatCreate(). For example,

$         KSP/PCGetOperators(ksp/pc,&mat,NULL); is equivalent to
$           set size, type, etc of mat

$         MatCreate(comm,&mat);
$         KSP/PCSetOperators(ksp/pc,mat,mat);
$         PetscObjectDereference((PetscObject)mat);
$           set size, type, etc of mat

     and

$         KSP/PCGetOperators(ksp/pc,&mat,&pmat); is equivalent to
$           set size, type, etc of mat and pmat

$         MatCreate(comm,&mat);
$         MatCreate(comm,&pmat);
$         KSP/PCSetOperators(ksp/pc,mat,pmat);
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

.seealso: KSPSolve(), KSPGetPC(), PCGetOperators(), PCSetOperators(), KSPGetOperators(), KSPSetComputeOperators(), KSPSetComputeInitialGuess(), KSPSetComputeRHS()
@*/
PetscErrorCode  KSPSetOperators(KSP ksp,Mat Amat,Mat Pmat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (Amat) PetscValidHeaderSpecific(Amat,MAT_CLASSID,2);
  if (Pmat) PetscValidHeaderSpecific(Pmat,MAT_CLASSID,3);
  if (Amat) PetscCheckSameComm(ksp,1,Amat,2);
  if (Pmat) PetscCheckSameComm(ksp,1,Pmat,3);
  if (!ksp->pc) {ierr = KSPGetPC(ksp,&ksp->pc);CHKERRQ(ierr);}
  ierr = PCSetOperators(ksp->pc,Amat,Pmat);CHKERRQ(ierr);
  if (ksp->setupstage == KSP_SETUP_NEWRHS) ksp->setupstage = KSP_SETUP_NEWMATRIX;  /* so that next solve call will call PCSetUp() on new matrix */
  PetscFunctionReturn(0);
}

/*@
   KSPGetOperators - Gets the matrix associated with the linear system
   and a (possibly) different one associated with the preconditioner.

   Collective on ksp

   Input Parameter:
.  ksp - the KSP context

   Output Parameters:
+  Amat - the matrix that defines the linear system
-  Pmat - the matrix to be used in constructing the preconditioner, usually the same as Amat.

    Level: intermediate

   Notes:
    DOES NOT increase the reference counts of the matrix, so you should NOT destroy them.

.seealso: KSPSolve(), KSPGetPC(), PCGetOperators(), PCSetOperators(), KSPSetOperators(), KSPGetOperatorsSet()
@*/
PetscErrorCode  KSPGetOperators(KSP ksp,Mat *Amat,Mat *Pmat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (!ksp->pc) {ierr = KSPGetPC(ksp,&ksp->pc);CHKERRQ(ierr);}
  ierr = PCGetOperators(ksp->pc,Amat,Pmat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   KSPGetOperatorsSet - Determines if the matrix associated with the linear system and
   possibly a different one associated with the preconditioner have been set in the KSP.

   Not collective, though the results on all processes should be the same

   Input Parameter:
.  pc - the KSP context

   Output Parameters:
+  mat - the matrix associated with the linear system was set
-  pmat - matrix associated with the preconditioner was set, usually the same

   Level: intermediate

.seealso: PCSetOperators(), KSPGetOperators(), KSPSetOperators(), PCGetOperators(), PCGetOperatorsSet()
@*/
PetscErrorCode  KSPGetOperatorsSet(KSP ksp,PetscBool  *mat,PetscBool  *pmat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (!ksp->pc) {ierr = KSPGetPC(ksp,&ksp->pc);CHKERRQ(ierr);}
  ierr = PCGetOperatorsSet(ksp->pc,mat,pmat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   KSPSetPreSolve - Sets a function that is called before every KSPSolve() is started

   Logically Collective on ksp

   Input Parameters:
+   ksp - the solver object
.   presolve - the function to call before the solve
-   prectx - any context needed by the function

   Calling sequence of presolve:
$  func(KSP ksp,Vec rhs,Vec x,void *ctx)

+  ksp - the KSP context
.  rhs - the right-hand side vector
.  x - the solution vector
-  ctx - optional user-provided context

   Level: developer

.seealso: KSPSetUp(), KSPSolve(), KSPDestroy(), KSP, KSPSetPostSolve()
@*/
PetscErrorCode  KSPSetPreSolve(KSP ksp,PetscErrorCode (*presolve)(KSP,Vec,Vec,void*),void *prectx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  ksp->presolve = presolve;
  ksp->prectx   = prectx;
  PetscFunctionReturn(0);
}

/*@C
   KSPSetPostSolve - Sets a function that is called after every KSPSolve() completes (whether it converges or not)

   Logically Collective on ksp

   Input Parameters:
+   ksp - the solver object
.   postsolve - the function to call after the solve
-   postctx - any context needed by the function

   Level: developer

   Calling sequence of postsolve:
$  func(KSP ksp,Vec rhs,Vec x,void *ctx)

+  ksp - the KSP context
.  rhs - the right-hand side vector
.  x - the solution vector
-  ctx - optional user-provided context

.seealso: KSPSetUp(), KSPSolve(), KSPDestroy(), KSP, KSPSetPreSolve()
@*/
PetscErrorCode  KSPSetPostSolve(KSP ksp,PetscErrorCode (*postsolve)(KSP,Vec,Vec,void*),void *postctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  ksp->postsolve = postsolve;
  ksp->postctx   = postctx;
  PetscFunctionReturn(0);
}

/*@
   KSPCreate - Creates the default KSP context.

   Collective

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  ksp - location to put the KSP context

   Notes:
   The default KSP type is GMRES with a restart of 30, using modified Gram-Schmidt
   orthogonalization.

   Level: beginner

.seealso: KSPSetUp(), KSPSolve(), KSPDestroy(), KSP
@*/
PetscErrorCode  KSPCreate(MPI_Comm comm,KSP *inksp)
{
  KSP            ksp;
  PetscErrorCode ierr;
  void           *ctx;

  PetscFunctionBegin;
  PetscValidPointer(inksp,2);
  *inksp = NULL;
  ierr = KSPInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(ksp,KSP_CLASSID,"KSP","Krylov Method","KSP",comm,KSPDestroy,KSPView);CHKERRQ(ierr);

  ksp->max_it  = 10000;
  ksp->pc_side = ksp->pc_side_set = PC_SIDE_DEFAULT;
  ksp->rtol    = 1.e-5;
#if defined(PETSC_USE_REAL_SINGLE)
  ksp->abstol  = 1.e-25;
#else
  ksp->abstol  = 1.e-50;
#endif
  ksp->divtol  = 1.e4;

  ksp->chknorm        = -1;
  ksp->normtype       = ksp->normtype_set = KSP_NORM_DEFAULT;
  ksp->rnorm          = 0.0;
  ksp->its            = 0;
  ksp->guess_zero     = PETSC_TRUE;
  ksp->calc_sings     = PETSC_FALSE;
  ksp->res_hist       = NULL;
  ksp->res_hist_alloc = NULL;
  ksp->res_hist_len   = 0;
  ksp->res_hist_max   = 0;
  ksp->res_hist_reset = PETSC_TRUE;
  ksp->err_hist       = NULL;
  ksp->err_hist_alloc = NULL;
  ksp->err_hist_len   = 0;
  ksp->err_hist_max   = 0;
  ksp->err_hist_reset = PETSC_TRUE;
  ksp->numbermonitors = 0;
  ksp->numberreasonviews = 0;
  ksp->setfromoptionscalled = 0;
  ksp->nmax = PETSC_DECIDE;

  ierr                    = KSPConvergedDefaultCreate(&ctx);CHKERRQ(ierr);
  ierr                    = KSPSetConvergenceTest(ksp,KSPConvergedDefault,ctx,KSPConvergedDefaultDestroy);CHKERRQ(ierr);
  ksp->ops->buildsolution = KSPBuildSolutionDefault;
  ksp->ops->buildresidual = KSPBuildResidualDefault;

  ksp->vec_sol    = NULL;
  ksp->vec_rhs    = NULL;
  ksp->pc         = NULL;
  ksp->data       = NULL;
  ksp->nwork      = 0;
  ksp->work       = NULL;
  ksp->reason     = KSP_CONVERGED_ITERATING;
  ksp->setupstage = KSP_SETUP_NEW;

  ierr = KSPNormSupportTableReset_Private(ksp);CHKERRQ(ierr);

  *inksp = ksp;
  PetscFunctionReturn(0);
}

/*@C
   KSPSetType - Builds KSP for a particular solver.

   Logically Collective on ksp

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

  Developer Note: KSPRegister() is used to add Krylov types to KSPList from which they
  are accessed by KSPSetType().

.seealso: PCSetType(), KSPType, KSPRegister(), KSPCreate()

@*/
PetscErrorCode  KSPSetType(KSP ksp, KSPType type)
{
  PetscErrorCode ierr,(*r)(KSP);
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidCharPointer(type,2);

  ierr = PetscObjectTypeCompare((PetscObject)ksp,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr = PetscFunctionListFind(KSPList,type,&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested KSP type %s",type);
  /* Destroy the previous private KSP context */
  if (ksp->ops->destroy) {
    ierr              = (*ksp->ops->destroy)(ksp);CHKERRQ(ierr);
    ksp->ops->destroy = NULL;
  }
  /* Reinitialize function pointers in KSPOps structure */
  ierr                    = PetscMemzero(ksp->ops,sizeof(struct _KSPOps));CHKERRQ(ierr);
  ksp->ops->buildsolution = KSPBuildSolutionDefault;
  ksp->ops->buildresidual = KSPBuildResidualDefault;
  ierr                    = KSPNormSupportTableReset_Private(ksp);CHKERRQ(ierr);
  ksp->setupnewmatrix     = PETSC_FALSE; // restore default (setup not called in case of new matrix)
  /* Call the KSPCreate_XXX routine for this particular Krylov solver */
  ksp->setupstage = KSP_SETUP_NEW;
  ierr            = (*r)(ksp);CHKERRQ(ierr);
  ierr            = PetscObjectChangeTypeName((PetscObject)ksp,type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   KSPGetType - Gets the KSP type as a string from the KSP object.

   Not Collective

   Input Parameter:
.  ksp - Krylov context

   Output Parameter:
.  name - name of KSP method

   Level: intermediate

.seealso: KSPSetType()
@*/
PetscErrorCode  KSPGetType(KSP ksp,KSPType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)ksp)->type_name;
  PetscFunctionReturn(0);
}

/*@C
  KSPRegister -  Adds a method to the Krylov subspace solver package.

   Not Collective

   Input Parameters:
+  name_solver - name of a new user-defined solver
-  routine_create - routine to create method context

   Notes:
   KSPRegister() may be called multiple times to add several user-defined solvers.

   Sample usage:
.vb
   KSPRegister("my_solver",MySolverCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     KSPSetType(ksp,"my_solver")
   or at runtime via the option
$     -ksp_type my_solver

   Level: advanced

.seealso: KSPRegisterAll()
@*/
PetscErrorCode  KSPRegister(const char sname[],PetscErrorCode (*function)(KSP))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPInitializePackage();CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&KSPList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode KSPMonitorMakeKey_Internal(const char name[], PetscViewerType vtype, PetscViewerFormat format, char key[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrncpy(key, name, PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
  ierr = PetscStrlcat(key, ":", PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
  ierr = PetscStrlcat(key, vtype, PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
  ierr = PetscStrlcat(key, ":", PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
  ierr = PetscStrlcat(key, PetscViewerFormats[format], PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  KSPMonitorRegister -  Adds Krylov subspace solver monitor routine.

  Not Collective

  Input Parameters:
+ name    - name of a new monitor routine
. vtype   - A PetscViewerType for the output
. format  - A PetscViewerFormat for the output
. monitor - Monitor routine
. create  - Creation routine, or NULL
- destroy - Destruction routine, or NULL

  Notes:
  KSPMonitorRegister() may be called multiple times to add several user-defined monitors.

  Sample usage:
.vb
  KSPMonitorRegister("my_monitor",PETSCVIEWERASCII,PETSC_VIEWER_ASCII_INFO_DETAIL,MyMonitor,NULL,NULL);
.ve

  Then, your monitor can be chosen with the procedural interface via
$     KSPMonitorSetFromOptions(ksp,"-ksp_monitor_my_monitor","my_monitor",NULL)
  or at runtime via the option
$     -ksp_monitor_my_monitor

   Level: advanced

.seealso: KSPMonitorRegisterAll()
@*/
PetscErrorCode KSPMonitorRegister(const char name[], PetscViewerType vtype, PetscViewerFormat format,
                                  PetscErrorCode (*monitor)(KSP, PetscInt, PetscReal, PetscViewerAndFormat *),
                                  PetscErrorCode (*create)(PetscViewer, PetscViewerFormat, void *, PetscViewerAndFormat **),
                                  PetscErrorCode (*destroy)(PetscViewerAndFormat **))
{
  char           key[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPInitializePackage();CHKERRQ(ierr);
  ierr = KSPMonitorMakeKey_Internal(name, vtype, format, key);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&KSPMonitorList, key, monitor);CHKERRQ(ierr);
  if (create)  {ierr = PetscFunctionListAdd(&KSPMonitorCreateList,  key, create);CHKERRQ(ierr);}
  if (destroy) {ierr = PetscFunctionListAdd(&KSPMonitorDestroyList, key, destroy);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}
