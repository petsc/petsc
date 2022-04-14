#include <petsc/private/snesimpl.h>      /*I "petscsnes.h"  I*/
#include <petscdmshell.h>
#include <petscdraw.h>
#include <petscds.h>
#include <petscdmadaptor.h>
#include <petscconvest.h>

PetscBool         SNESRegisterAllCalled = PETSC_FALSE;
PetscFunctionList SNESList              = NULL;

/* Logging support */
PetscClassId  SNES_CLASSID, DMSNES_CLASSID;
PetscLogEvent SNES_Solve, SNES_Setup, SNES_FunctionEval, SNES_JacobianEval, SNES_NGSEval, SNES_NGSFuncEval, SNES_NPCSolve, SNES_ObjectiveEval;

/*@
   SNESSetErrorIfNotConverged - Causes SNESSolve() to generate an error if the solver has not converged.

   Logically Collective on SNES

   Input Parameters:
+  snes - iterative context obtained from SNESCreate()
-  flg - PETSC_TRUE indicates you want the error generated

   Options database keys:
.  -snes_error_if_not_converged <true,false> - cause an immediate error condition and stop the program if the solver does not converge

   Level: intermediate

   Notes:
    Normally PETSc continues if a linear solver fails to converge, you can call SNESGetConvergedReason() after a SNESSolve()
    to determine if it has converged.

.seealso: SNESGetErrorIfNotConverged(), KSPGetErrorIfNotConverged(), KSPSetErrorIfNotConverged()
@*/
PetscErrorCode  SNESSetErrorIfNotConverged(SNES snes,PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidLogicalCollectiveBool(snes,flg,2);
  snes->errorifnotconverged = flg;
  PetscFunctionReturn(0);
}

/*@
   SNESGetErrorIfNotConverged - Will SNESSolve() generate an error if the solver does not converge?

   Not Collective

   Input Parameter:
.  snes - iterative context obtained from SNESCreate()

   Output Parameter:
.  flag - PETSC_TRUE if it will generate an error, else PETSC_FALSE

   Level: intermediate

.seealso:  SNESSetErrorIfNotConverged(), KSPGetErrorIfNotConverged(), KSPSetErrorIfNotConverged()
@*/
PetscErrorCode  SNESGetErrorIfNotConverged(SNES snes,PetscBool  *flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidBoolPointer(flag,2);
  *flag = snes->errorifnotconverged;
  PetscFunctionReturn(0);
}

/*@
    SNESSetAlwaysComputesFinalResidual - does the SNES always compute the residual at the final solution?

   Logically Collective on SNES

    Input Parameters:
+   snes - the shell SNES
-   flg - is the residual computed?

   Level: advanced

.seealso: SNESGetAlwaysComputesFinalResidual()
@*/
PetscErrorCode  SNESSetAlwaysComputesFinalResidual(SNES snes, PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  snes->alwayscomputesfinalresidual = flg;
  PetscFunctionReturn(0);
}

/*@
    SNESGetAlwaysComputesFinalResidual - does the SNES always compute the residual at the final solution?

   Logically Collective on SNES

    Input Parameter:
.   snes - the shell SNES

    Output Parameter:
.   flg - is the residual computed?

   Level: advanced

.seealso: SNESSetAlwaysComputesFinalResidual()
@*/
PetscErrorCode  SNESGetAlwaysComputesFinalResidual(SNES snes, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  *flg = snes->alwayscomputesfinalresidual;
  PetscFunctionReturn(0);
}

/*@
   SNESSetFunctionDomainError - tells SNES that the input vector to your SNESFunction is not
     in the functions domain. For example, negative pressure.

   Logically Collective on SNES

   Input Parameters:
.  snes - the SNES context

   Level: advanced

.seealso: SNESCreate(), SNESSetFunction(), SNESFunction
@*/
PetscErrorCode  SNESSetFunctionDomainError(SNES snes)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscCheck(!snes->errorifnotconverged,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"User code indicates input vector is not in the function domain");
  snes->domainerror = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
   SNESSetJacobianDomainError - tells SNES that computeJacobian does not make sense any more. For example there is a negative element transformation.

   Logically Collective on SNES

   Input Parameters:
.  snes - the SNES context

   Level: advanced

.seealso: SNESCreate(), SNESSetFunction(), SNESFunction(), SNESSetFunctionDomainError()
@*/
PetscErrorCode SNESSetJacobianDomainError(SNES snes)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscCheck(!snes->errorifnotconverged,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"User code indicates computeJacobian does not make sense");
  snes->jacobiandomainerror = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
   SNESSetCheckJacobianDomainError - if or not to check jacobian domain error after each Jacobian evaluation. By default, we check Jacobian domain error
   in the debug mode, and do not check it in the optimized mode.

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
-  flg  - indicates if or not to check jacobian domain error after each Jacobian evaluation

   Level: advanced

.seealso: SNESCreate(), SNESSetFunction(), SNESFunction(), SNESSetFunctionDomainError(), SNESGetCheckJacobianDomainError()
@*/
PetscErrorCode SNESSetCheckJacobianDomainError(SNES snes, PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  snes->checkjacdomainerror = flg;
  PetscFunctionReturn(0);
}

/*@
   SNESGetCheckJacobianDomainError - Get an indicator whether or not we are checking Jacobian domain errors after each Jacobian evaluation.

   Logically Collective on SNES

   Input Parameters:
.  snes - the SNES context

   Output Parameters:
.  flg  - PETSC_FALSE indicates that we don't check jacobian domain errors after each Jacobian evaluation

   Level: advanced

.seealso: SNESCreate(), SNESSetFunction(), SNESFunction(), SNESSetFunctionDomainError(), SNESSetCheckJacobianDomainError()
@*/
PetscErrorCode SNESGetCheckJacobianDomainError(SNES snes, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidBoolPointer(flg,2);
  *flg = snes->checkjacdomainerror;
  PetscFunctionReturn(0);
}

/*@
   SNESGetFunctionDomainError - Gets the status of the domain error after a call to SNESComputeFunction;

   Logically Collective on SNES

   Input Parameters:
.  snes - the SNES context

   Output Parameters:
.  domainerror - Set to PETSC_TRUE if there's a domain error; PETSC_FALSE otherwise.

   Level: advanced

.seealso: SNESSetFunctionDomainError(), SNESComputeFunction()
@*/
PetscErrorCode  SNESGetFunctionDomainError(SNES snes, PetscBool *domainerror)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidBoolPointer(domainerror,2);
  *domainerror = snes->domainerror;
  PetscFunctionReturn(0);
}

/*@
   SNESGetJacobianDomainError - Gets the status of the Jacobian domain error after a call to SNESComputeJacobian;

   Logically Collective on SNES

   Input Parameters:
.  snes - the SNES context

   Output Parameters:
.  domainerror - Set to PETSC_TRUE if there's a jacobian domain error; PETSC_FALSE otherwise.

   Level: advanced

.seealso: SNESSetFunctionDomainError(), SNESComputeFunction(),SNESGetFunctionDomainError()
@*/
PetscErrorCode SNESGetJacobianDomainError(SNES snes, PetscBool *domainerror)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidBoolPointer(domainerror,2);
  *domainerror = snes->jacobiandomainerror;
  PetscFunctionReturn(0);
}

/*@C
  SNESLoad - Loads a SNES that has been stored in binary  with SNESView().

  Collective on PetscViewer

  Input Parameters:
+ newdm - the newly loaded SNES, this needs to have been created with SNESCreate() or
           some related function before a call to SNESLoad().
- viewer - binary file viewer, obtained from PetscViewerBinaryOpen()

   Level: intermediate

  Notes:
   The type is determined by the data in the file, any type set into the SNES before this call is ignored.

  Notes for advanced users:
  Most users should not need to know the details of the binary storage
  format, since SNESLoad() and TSView() completely hide these details.
  But for anyone who's interested, the standard binary matrix storage
  format is
.vb
     has not yet been determined
.ve

.seealso: PetscViewerBinaryOpen(), SNESView(), MatLoad(), VecLoad()
@*/
PetscErrorCode  SNESLoad(SNES snes, PetscViewer viewer)
{
  PetscBool      isbinary;
  PetscInt       classid;
  char           type[256];
  KSP            ksp;
  DM             dm;
  DMSNES         dmsnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary));
  PetscCheck(isbinary,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid viewer; open viewer with PetscViewerBinaryOpen()");

  PetscCall(PetscViewerBinaryRead(viewer,&classid,1,NULL,PETSC_INT));
  PetscCheck(classid == SNES_FILE_CLASSID,PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONG,"Not SNES next in file");
  PetscCall(PetscViewerBinaryRead(viewer,type,256,NULL,PETSC_CHAR));
  PetscCall(SNESSetType(snes, type));
  if (snes->ops->load) {
    PetscCall((*snes->ops->load)(snes,viewer));
  }
  PetscCall(SNESGetDM(snes,&dm));
  PetscCall(DMGetDMSNES(dm,&dmsnes));
  PetscCall(DMSNESLoad(dmsnes,viewer));
  PetscCall(SNESGetKSP(snes,&ksp));
  PetscCall(KSPLoad(ksp,viewer));
  PetscFunctionReturn(0);
}

#include <petscdraw.h>
#if defined(PETSC_HAVE_SAWS)
#include <petscviewersaws.h>
#endif

/*@C
   SNESViewFromOptions - View from Options

   Collective on SNES

   Input Parameters:
+  A - the application ordering context
.  obj - Optional object
-  name - command line option

   Level: intermediate
.seealso:  SNES, SNESView, PetscObjectViewFromOptions(), SNESCreate()
@*/
PetscErrorCode  SNESViewFromOptions(SNES A,PetscObject obj,const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,SNES_CLASSID,1);
  PetscCall(PetscObjectViewFromOptions((PetscObject)A,obj,name));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode SNESComputeJacobian_DMDA(SNES,Vec,Mat,Mat,void*);

/*@C
   SNESView - Prints the SNES data structure.

   Collective on SNES

   Input Parameters:
+  SNES - the SNES context
-  viewer - visualization context

   Options Database Key:
.  -snes_view - Calls SNESView() at end of SNESSolve()

   Notes:
   The available visualization contexts include
+     PETSC_VIEWER_STDOUT_SELF - standard output (default)
-     PETSC_VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their
         data to the first processor to print.

   The available formats include
+     PETSC_VIEWER_DEFAULT - standard output (default)
-     PETSC_VIEWER_ASCII_INFO_DETAIL - more verbose output for SNESNASM

   The user can open an alternative visualization context with
   PetscViewerASCIIOpen() - output to a specified file.

  In the debugger you can do "call SNESView(snes,0)" to display the SNES solver. (The same holds for any PETSc object viewer).

   Level: beginner

.seealso: PetscViewerASCIIOpen()
@*/
PetscErrorCode  SNESView(SNES snes,PetscViewer viewer)
{
  SNESKSPEW      *kctx;
  KSP            ksp;
  SNESLineSearch linesearch;
  PetscBool      iascii,isstring,isbinary,isdraw;
  DMSNES         dmsnes;
#if defined(PETSC_HAVE_SAWS)
  PetscBool      issaws;
#endif

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (!viewer) {
    PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)snes),&viewer));
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(snes,1,viewer,2);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw));
#if defined(PETSC_HAVE_SAWS)
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSAWS,&issaws));
#endif
  if (iascii) {
    SNESNormSchedule normschedule;
    DM               dm;
    PetscErrorCode   (*cJ)(SNES,Vec,Mat,Mat,void*);
    void             *ctx;
    const char       *pre = "";

    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)snes,viewer));
    if (!snes->setupcalled) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  SNES has not been set up so information may be incomplete\n"));
    }
    if (snes->ops->view) {
      PetscCall(PetscViewerASCIIPushTab(viewer));
      PetscCall((*snes->ops->view)(snes,viewer));
      PetscCall(PetscViewerASCIIPopTab(viewer));
    }
    PetscCall(PetscViewerASCIIPrintf(viewer,"  maximum iterations=%D, maximum function evaluations=%D\n",snes->max_its,snes->max_funcs));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  tolerances: relative=%g, absolute=%g, solution=%g\n",(double)snes->rtol,(double)snes->abstol,(double)snes->stol));
    if (snes->usesksp) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  total number of linear solver iterations=%D\n",snes->linear_its));
    }
    PetscCall(PetscViewerASCIIPrintf(viewer,"  total number of function evaluations=%D\n",snes->nfuncs));
    PetscCall(SNESGetNormSchedule(snes, &normschedule));
    if (normschedule > 0) PetscCall(PetscViewerASCIIPrintf(viewer,"  norm schedule %s\n",SNESNormSchedules[normschedule]));
    if (snes->gridsequence) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  total number of grid sequence refinements=%D\n",snes->gridsequence));
    }
    if (snes->ksp_ewconv) {
      kctx = (SNESKSPEW*)snes->kspconvctx;
      if (kctx) {
        PetscCall(PetscViewerASCIIPrintf(viewer,"  Eisenstat-Walker computation of KSP relative tolerance (version %D)\n",kctx->version));
        PetscCall(PetscViewerASCIIPrintf(viewer,"    rtol_0=%g, rtol_max=%g, threshold=%g\n",(double)kctx->rtol_0,(double)kctx->rtol_max,(double)kctx->threshold));
        PetscCall(PetscViewerASCIIPrintf(viewer,"    gamma=%g, alpha=%g, alpha2=%g\n",(double)kctx->gamma,(double)kctx->alpha,(double)kctx->alpha2));
      }
    }
    if (snes->lagpreconditioner == -1) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  Preconditioned is never rebuilt\n"));
    } else if (snes->lagpreconditioner > 1) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  Preconditioned is rebuilt every %D new Jacobians\n",snes->lagpreconditioner));
    }
    if (snes->lagjacobian == -1) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  Jacobian is never rebuilt\n"));
    } else if (snes->lagjacobian > 1) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  Jacobian is rebuilt every %D SNES iterations\n",snes->lagjacobian));
    }
    PetscCall(SNESGetDM(snes,&dm));
    PetscCall(DMSNESGetJacobian(dm,&cJ,&ctx));
    if (snes->mf_operator) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  Jacobian is applied matrix-free with differencing\n"));
      pre  = "Preconditioning ";
    }
    if (cJ == SNESComputeJacobianDefault) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  %sJacobian is built using finite differences one column at a time\n",pre));
    } else if (cJ == SNESComputeJacobianDefaultColor) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  %sJacobian is built using finite differences with coloring\n",pre));
    /* it slightly breaks data encapsulation for access the DMDA information directly */
    } else if (cJ == SNESComputeJacobian_DMDA) {
      MatFDColoring fdcoloring;
      PetscCall(PetscObjectQuery((PetscObject)dm,"DMDASNES_FDCOLORING",(PetscObject*)&fdcoloring));
      if (fdcoloring) {
        PetscCall(PetscViewerASCIIPrintf(viewer,"  %sJacobian is built using colored finite differences on a DMDA\n",pre));
      } else {
        PetscCall(PetscViewerASCIIPrintf(viewer,"  %sJacobian is built using a DMDA local Jacobian\n",pre));
      }
    } else if (snes->mf) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  Jacobian is applied matrix-free with differencing, no explicit Jacobian\n"));
    }
  } else if (isstring) {
    const char *type;
    PetscCall(SNESGetType(snes,&type));
    PetscCall(PetscViewerStringSPrintf(viewer," SNESType: %-7.7s",type));
    if (snes->ops->view) PetscCall((*snes->ops->view)(snes,viewer));
  } else if (isbinary) {
    PetscInt    classid = SNES_FILE_CLASSID;
    MPI_Comm    comm;
    PetscMPIInt rank;
    char        type[256];

    PetscCall(PetscObjectGetComm((PetscObject)snes,&comm));
    PetscCallMPI(MPI_Comm_rank(comm,&rank));
    if (rank == 0) {
      PetscCall(PetscViewerBinaryWrite(viewer,&classid,1,PETSC_INT));
      PetscCall(PetscStrncpy(type,((PetscObject)snes)->type_name,sizeof(type)));
      PetscCall(PetscViewerBinaryWrite(viewer,type,sizeof(type),PETSC_CHAR));
    }
    if (snes->ops->view) {
      PetscCall((*snes->ops->view)(snes,viewer));
    }
  } else if (isdraw) {
    PetscDraw draw;
    char      str[36];
    PetscReal x,y,bottom,h;

    PetscCall(PetscViewerDrawGetDraw(viewer,0,&draw));
    PetscCall(PetscDrawGetCurrentPoint(draw,&x,&y));
    PetscCall(PetscStrncpy(str,"SNES: ",sizeof(str)));
    PetscCall(PetscStrlcat(str,((PetscObject)snes)->type_name,sizeof(str)));
    PetscCall(PetscDrawStringBoxed(draw,x,y,PETSC_DRAW_BLUE,PETSC_DRAW_BLACK,str,NULL,&h));
    bottom = y - h;
    PetscCall(PetscDrawPushCurrentPoint(draw,x,bottom));
    if (snes->ops->view) {
      PetscCall((*snes->ops->view)(snes,viewer));
    }
#if defined(PETSC_HAVE_SAWS)
  } else if (issaws) {
    PetscMPIInt rank;
    const char *name;

    PetscCall(PetscObjectGetName((PetscObject)snes,&name));
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
    if (!((PetscObject)snes)->amsmem && rank == 0) {
      char       dir[1024];

      PetscCall(PetscObjectViewSAWs((PetscObject)snes,viewer));
      PetscCall(PetscSNPrintf(dir,1024,"/PETSc/Objects/%s/its",name));
      PetscStackCallSAWs(SAWs_Register,(dir,&snes->iter,1,SAWs_READ,SAWs_INT));
      if (!snes->conv_hist) {
        PetscCall(SNESSetConvergenceHistory(snes,NULL,NULL,PETSC_DECIDE,PETSC_TRUE));
      }
      PetscCall(PetscSNPrintf(dir,1024,"/PETSc/Objects/%s/conv_hist",name));
      PetscStackCallSAWs(SAWs_Register,(dir,snes->conv_hist,10,SAWs_READ,SAWs_DOUBLE));
    }
#endif
  }
  if (snes->linesearch) {
    PetscCall(SNESGetLineSearch(snes, &linesearch));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(SNESLineSearchView(linesearch, viewer));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  if (snes->npc && snes->usesnpc) {
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(SNESView(snes->npc, viewer));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(DMGetDMSNES(snes->dm,&dmsnes));
  PetscCall(DMSNESView(dmsnes, viewer));
  PetscCall(PetscViewerASCIIPopTab(viewer));
  if (snes->usesksp) {
    PetscCall(SNESGetKSP(snes,&ksp));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(KSPView(ksp,viewer));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  if (isdraw) {
    PetscDraw draw;
    PetscCall(PetscViewerDrawGetDraw(viewer,0,&draw));
    PetscCall(PetscDrawPopCurrentPoint(draw));
  }
  PetscFunctionReturn(0);
}

/*
  We retain a list of functions that also take SNES command
  line options. These are called at the end SNESSetFromOptions()
*/
#define MAXSETFROMOPTIONS 5
static PetscInt numberofsetfromoptions;
static PetscErrorCode (*othersetfromoptions[MAXSETFROMOPTIONS])(SNES);

/*@C
  SNESAddOptionsChecker - Adds an additional function to check for SNES options.

  Not Collective

  Input Parameter:
. snescheck - function that checks for options

  Level: developer

.seealso: SNESSetFromOptions()
@*/
PetscErrorCode  SNESAddOptionsChecker(PetscErrorCode (*snescheck)(SNES))
{
  PetscFunctionBegin;
  PetscCheck(numberofsetfromoptions < MAXSETFROMOPTIONS,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE, "Too many options checkers, only %D allowed", MAXSETFROMOPTIONS);
  othersetfromoptions[numberofsetfromoptions++] = snescheck;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode SNESDefaultMatrixFreeCreate2(SNES,Vec,Mat*);

static PetscErrorCode SNESSetUpMatrixFree_Private(SNES snes, PetscBool hasOperator, PetscInt version)
{
  Mat            J;
  MatNullSpace   nullsp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);

  if (!snes->vec_func && (snes->jacobian || snes->jacobian_pre)) {
    Mat A = snes->jacobian, B = snes->jacobian_pre;
    PetscCall(MatCreateVecs(A ? A : B, NULL,&snes->vec_func));
  }

  if (version == 1) {
    PetscCall(MatCreateSNESMF(snes,&J));
    PetscCall(MatMFFDSetOptionsPrefix(J,((PetscObject)snes)->prefix));
    PetscCall(MatSetFromOptions(J));
    /* TODO: the version 2 code should be merged into the MatCreateSNESMF() and MatCreateMFFD() infrastructure and then removed */
  } else if (version == 2) {
    PetscCheck(snes->vec_func,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"SNESSetFunction() must be called first");
#if !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_REAL_SINGLE) && !defined(PETSC_USE_REAL___FLOAT128) && !defined(PETSC_USE_REAL___FP16)
    PetscCall(SNESDefaultMatrixFreeCreate2(snes,snes->vec_func,&J));
#else
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP, "matrix-free operator routines (version 2)");
#endif
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE, "matrix-free operator routines, only version 1 and 2");

  /* attach any user provided null space that was on Amat to the newly created matrix free matrix */
  if (snes->jacobian) {
    PetscCall(MatGetNullSpace(snes->jacobian,&nullsp));
    if (nullsp) {
      PetscCall(MatSetNullSpace(J,nullsp));
    }
  }

  PetscCall(PetscInfo(snes,"Setting default matrix-free operator routines (version %D)\n", version));
  if (hasOperator) {

    /* This version replaces the user provided Jacobian matrix with a
       matrix-free version but still employs the user-provided preconditioner matrix. */
    PetscCall(SNESSetJacobian(snes,J,NULL,NULL,NULL));
  } else {
    /* This version replaces both the user-provided Jacobian and the user-
     provided preconditioner Jacobian with the default matrix free version. */
    if (snes->npcside == PC_LEFT && snes->npc) {
      if (!snes->jacobian) PetscCall(SNESSetJacobian(snes,J,NULL,NULL,NULL));
    } else {
      KSP       ksp;
      PC        pc;
      PetscBool match;

      PetscCall(SNESSetJacobian(snes,J,J,MatMFFDComputeJacobian,NULL));
      /* Force no preconditioner */
      PetscCall(SNESGetKSP(snes,&ksp));
      PetscCall(KSPGetPC(ksp,&pc));
      PetscCall(PetscObjectTypeCompare((PetscObject)pc,PCSHELL,&match));
      if (!match) {
        PetscCall(PetscInfo(snes,"Setting default matrix-free preconditioner routines\nThat is no preconditioner is being used\n"));
        PetscCall(PCSetType(pc,PCNONE));
      }
    }
  }
  PetscCall(MatDestroy(&J));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMRestrictHook_SNESVecSol(DM dmfine,Mat Restrict,Vec Rscale,Mat Inject,DM dmcoarse,void *ctx)
{
  SNES           snes = (SNES)ctx;
  Vec            Xfine,Xfine_named = NULL,Xcoarse;

  PetscFunctionBegin;
  if (PetscLogPrintInfo) {
    PetscInt finelevel,coarselevel,fineclevel,coarseclevel;
    PetscCall(DMGetRefineLevel(dmfine,&finelevel));
    PetscCall(DMGetCoarsenLevel(dmfine,&fineclevel));
    PetscCall(DMGetRefineLevel(dmcoarse,&coarselevel));
    PetscCall(DMGetCoarsenLevel(dmcoarse,&coarseclevel));
    PetscCall(PetscInfo(dmfine,"Restricting SNES solution vector from level %D-%D to level %D-%D\n",finelevel,fineclevel,coarselevel,coarseclevel));
  }
  if (dmfine == snes->dm) Xfine = snes->vec_sol;
  else {
    PetscCall(DMGetNamedGlobalVector(dmfine,"SNESVecSol",&Xfine_named));
    Xfine = Xfine_named;
  }
  PetscCall(DMGetNamedGlobalVector(dmcoarse,"SNESVecSol",&Xcoarse));
  if (Inject) {
    PetscCall(MatRestrict(Inject,Xfine,Xcoarse));
  } else {
    PetscCall(MatRestrict(Restrict,Xfine,Xcoarse));
    PetscCall(VecPointwiseMult(Xcoarse,Xcoarse,Rscale));
  }
  PetscCall(DMRestoreNamedGlobalVector(dmcoarse,"SNESVecSol",&Xcoarse));
  if (Xfine_named) PetscCall(DMRestoreNamedGlobalVector(dmfine,"SNESVecSol",&Xfine_named));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCoarsenHook_SNESVecSol(DM dm,DM dmc,void *ctx)
{
  PetscFunctionBegin;
  PetscCall(DMCoarsenHookAdd(dmc,DMCoarsenHook_SNESVecSol,DMRestrictHook_SNESVecSol,ctx));
  PetscFunctionReturn(0);
}

/* This may be called to rediscretize the operator on levels of linear multigrid. The DM shuffle is so the user can
 * safely call SNESGetDM() in their residual evaluation routine. */
static PetscErrorCode KSPComputeOperators_SNES(KSP ksp,Mat A,Mat B,void *ctx)
{
  SNES           snes = (SNES)ctx;
  Vec            X,Xnamed = NULL;
  DM             dmsave;
  void           *ctxsave;
  PetscErrorCode (*jac)(SNES,Vec,Mat,Mat,void*) = NULL;

  PetscFunctionBegin;
  dmsave = snes->dm;
  PetscCall(KSPGetDM(ksp,&snes->dm));
  if (dmsave == snes->dm) X = snes->vec_sol; /* We are on the finest level */
  else {                                     /* We are on a coarser level, this vec was initialized using a DM restrict hook */
    PetscCall(DMGetNamedGlobalVector(snes->dm,"SNESVecSol",&Xnamed));
    X    = Xnamed;
    PetscCall(SNESGetJacobian(snes,NULL,NULL,&jac,&ctxsave));
    /* If the DM's don't match up, the MatFDColoring context needed for the jacobian won't match up either -- fixit. */
    if (jac == SNESComputeJacobianDefaultColor) {
      PetscCall(SNESSetJacobian(snes,NULL,NULL,SNESComputeJacobianDefaultColor,NULL));
    }
  }
  /* Make sure KSP DM has the Jacobian computation routine */
  {
    DMSNES sdm;

    PetscCall(DMGetDMSNES(snes->dm, &sdm));
    if (!sdm->ops->computejacobian) {
      PetscCall(DMCopyDMSNES(dmsave, snes->dm));
    }
  }
  /* Compute the operators */
  PetscCall(SNESComputeJacobian(snes,X,A,B));
  /* Put the previous context back */
  if (snes->dm != dmsave && jac == SNESComputeJacobianDefaultColor) {
    PetscCall(SNESSetJacobian(snes,NULL,NULL,jac,ctxsave));
  }

  if (Xnamed) PetscCall(DMRestoreNamedGlobalVector(snes->dm,"SNESVecSol",&Xnamed));
  snes->dm = dmsave;
  PetscFunctionReturn(0);
}

/*@
   SNESSetUpMatrices - ensures that matrices are available for SNES, to be called by SNESSetUp_XXX()

   Collective

   Input Parameter:
.  snes - snes to configure

   Level: developer

.seealso: SNESSetUp()
@*/
PetscErrorCode SNESSetUpMatrices(SNES snes)
{
  DM             dm;
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscCall(SNESGetDM(snes,&dm));
  PetscCall(DMGetDMSNES(dm,&sdm));
  if (!snes->jacobian && snes->mf) {
    Mat  J;
    void *functx;
    PetscCall(MatCreateSNESMF(snes,&J));
    PetscCall(MatMFFDSetOptionsPrefix(J,((PetscObject)snes)->prefix));
    PetscCall(MatSetFromOptions(J));
    PetscCall(SNESGetFunction(snes,NULL,NULL,&functx));
    PetscCall(SNESSetJacobian(snes,J,J,NULL,NULL));
    PetscCall(MatDestroy(&J));
  } else if (snes->mf_operator && !snes->jacobian_pre && !snes->jacobian) {
    Mat J,B;
    PetscCall(MatCreateSNESMF(snes,&J));
    PetscCall(MatMFFDSetOptionsPrefix(J,((PetscObject)snes)->prefix));
    PetscCall(MatSetFromOptions(J));
    PetscCall(DMCreateMatrix(snes->dm,&B));
    /* sdm->computejacobian was already set to reach here */
    PetscCall(SNESSetJacobian(snes,J,B,NULL,NULL));
    PetscCall(MatDestroy(&J));
    PetscCall(MatDestroy(&B));
  } else if (!snes->jacobian_pre) {
    PetscDS   prob;
    Mat       J, B;
    PetscBool hasPrec   = PETSC_FALSE;

    J    = snes->jacobian;
    PetscCall(DMGetDS(dm, &prob));
    if (prob) PetscCall(PetscDSHasJacobianPreconditioner(prob, &hasPrec));
    if (J)            PetscCall(PetscObjectReference((PetscObject) J));
    else if (hasPrec) PetscCall(DMCreateMatrix(snes->dm, &J));
    PetscCall(DMCreateMatrix(snes->dm, &B));
    PetscCall(SNESSetJacobian(snes, J ? J : B, B, NULL, NULL));
    PetscCall(MatDestroy(&J));
    PetscCall(MatDestroy(&B));
  }
  {
    KSP ksp;
    PetscCall(SNESGetKSP(snes,&ksp));
    PetscCall(KSPSetComputeOperators(ksp,KSPComputeOperators_SNES,snes));
    PetscCall(DMCoarsenHookAdd(snes->dm,DMCoarsenHook_SNESVecSol,DMRestrictHook_SNESVecSol,snes));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESMonitorPauseFinal_Internal(SNES snes)
{
  PetscInt       i;

  PetscFunctionBegin;
  if (!snes->pauseFinal) PetscFunctionReturn(0);
  for (i = 0; i < snes->numbermonitors; ++i) {
    PetscViewerAndFormat *vf = (PetscViewerAndFormat *) snes->monitorcontext[i];
    PetscDraw             draw;
    PetscReal             lpause;

    if (!vf) continue;
    if (vf->lg) {
      if (!PetscCheckPointer(vf->lg, PETSC_OBJECT)) continue;
      if (((PetscObject) vf->lg)->classid != PETSC_DRAWLG_CLASSID) continue;
      PetscCall(PetscDrawLGGetDraw(vf->lg, &draw));
      PetscCall(PetscDrawGetPause(draw, &lpause));
      PetscCall(PetscDrawSetPause(draw, -1.0));
      PetscCall(PetscDrawPause(draw));
      PetscCall(PetscDrawSetPause(draw, lpause));
    } else {
      PetscBool isdraw;

      if (!PetscCheckPointer(vf->viewer, PETSC_OBJECT)) continue;
      if (((PetscObject) vf->viewer)->classid != PETSC_VIEWER_CLASSID) continue;
      PetscCall(PetscObjectTypeCompare((PetscObject) vf->viewer, PETSCVIEWERDRAW, &isdraw));
      if (!isdraw) continue;
      PetscCall(PetscViewerDrawGetDraw(vf->viewer, 0, &draw));
      PetscCall(PetscDrawGetPause(draw, &lpause));
      PetscCall(PetscDrawSetPause(draw, -1.0));
      PetscCall(PetscDrawPause(draw));
      PetscCall(PetscDrawSetPause(draw, lpause));
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   SNESMonitorSetFromOptions - Sets a monitor function and viewer appropriate for the type indicated by the user

   Collective on SNES

   Input Parameters:
+  snes - SNES object you wish to monitor
.  name - the monitor type one is seeking
.  help - message indicating what monitoring is done
.  manual - manual page for the monitor
.  monitor - the monitor function
-  monitorsetup - a function that is called once ONLY if the user selected this monitor that may set additional features of the SNES or PetscViewer objects

   Level: developer

.seealso: PetscOptionsGetViewer(), PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode  SNESMonitorSetFromOptions(SNES snes,const char name[],const char help[], const char manual[],PetscErrorCode (*monitor)(SNES,PetscInt,PetscReal,PetscViewerAndFormat*),PetscErrorCode (*monitorsetup)(SNES,PetscViewerAndFormat*))
{
  PetscViewer       viewer;
  PetscViewerFormat format;
  PetscBool         flg;

  PetscFunctionBegin;
  PetscCall(PetscOptionsGetViewer(PetscObjectComm((PetscObject)snes),((PetscObject) snes)->options,((PetscObject)snes)->prefix,name,&viewer,&format,&flg));
  if (flg) {
    PetscViewerAndFormat *vf;
    PetscCall(PetscViewerAndFormatCreate(viewer,format,&vf));
    PetscCall(PetscObjectDereference((PetscObject)viewer));
    if (monitorsetup) {
      PetscCall((*monitorsetup)(snes,vf));
    }
    PetscCall(SNESMonitorSet(snes,(PetscErrorCode (*)(SNES,PetscInt,PetscReal,void*))monitor,vf,(PetscErrorCode (*)(void**))PetscViewerAndFormatDestroy));
  }
  PetscFunctionReturn(0);
}

/*@
   SNESSetFromOptions - Sets various SNES and KSP parameters from user options.

   Collective on SNES

   Input Parameter:
.  snes - the SNES context

   Options Database Keys:
+  -snes_type <type> - newtonls, newtontr, ngmres, ncg, nrichardson, qn, vi, fas, SNESType for complete list
.  -snes_stol - convergence tolerance in terms of the norm
                of the change in the solution between steps
.  -snes_atol <abstol> - absolute tolerance of residual norm
.  -snes_rtol <rtol> - relative decrease in tolerance norm from initial
.  -snes_divergence_tolerance <divtol> - if the residual goes above divtol*rnorm0, exit with divergence
.  -snes_force_iteration <force> - force SNESSolve() to take at least one iteration
.  -snes_max_it <max_it> - maximum number of iterations
.  -snes_max_funcs <max_funcs> - maximum number of function evaluations
.  -snes_max_fail <max_fail> - maximum number of line search failures allowed before stopping, default is none
.  -snes_max_linear_solve_fail - number of linear solver failures before SNESSolve() stops
.  -snes_lag_preconditioner <lag> - how often preconditioner is rebuilt (use -1 to never rebuild)
.  -snes_lag_preconditioner_persists <true,false> - retains the -snes_lag_preconditioner information across multiple SNESSolve()
.  -snes_lag_jacobian <lag> - how often Jacobian is rebuilt (use -1 to never rebuild)
.  -snes_lag_jacobian_persists <true,false> - retains the -snes_lag_jacobian information across multiple SNESSolve()
.  -snes_trtol <trtol> - trust region tolerance
.  -snes_no_convergence_test - skip convergence test in nonlinear
                               solver; hence iterations will continue until max_it
                               or some other criterion is reached. Saves expense
                               of convergence test
.  -snes_monitor [ascii][:filename][:viewer format] - prints residual norm at each iteration. if no filename given prints to stdout
.  -snes_monitor_solution [ascii binary draw][:filename][:viewer format] - plots solution at each iteration
.  -snes_monitor_residual [ascii binary draw][:filename][:viewer format] - plots residual (not its norm) at each iteration
.  -snes_monitor_solution_update [ascii binary draw][:filename][:viewer format] - plots update to solution at each iteration
.  -snes_monitor_lg_residualnorm - plots residual norm at each iteration
.  -snes_monitor_lg_range - plots residual norm at each iteration
.  -snes_monitor_pause_final - Pauses all monitor drawing after the solver ends
.  -snes_fd - use finite differences to compute Jacobian; very slow, only for testing
.  -snes_fd_color - use finite differences with coloring to compute Jacobian
.  -snes_mf_ksp_monitor - if using matrix-free multiply then print h at each KSP iteration
.  -snes_converged_reason - print the reason for convergence/divergence after each solve
.  -npc_snes_type <type> - the SNES type to use as a nonlinear preconditioner
.   -snes_test_jacobian <optional threshold> - compare the user provided Jacobian with one computed via finite differences to check for errors.  If a threshold is given, display only those entries whose difference is greater than the threshold.
-   -snes_test_jacobian_view - display the user provided Jacobian, the finite difference Jacobian and the difference between them to help users detect the location of errors in the user provided Jacobian.

    Options Database for Eisenstat-Walker method:
+  -snes_ksp_ew - use Eisenstat-Walker method for determining linear system convergence
.  -snes_ksp_ew_version ver - version of  Eisenstat-Walker method
.  -snes_ksp_ew_rtol0 <rtol0> - Sets rtol0
.  -snes_ksp_ew_rtolmax <rtolmax> - Sets rtolmax
.  -snes_ksp_ew_gamma <gamma> - Sets gamma
.  -snes_ksp_ew_alpha <alpha> - Sets alpha
.  -snes_ksp_ew_alpha2 <alpha2> - Sets alpha2
-  -snes_ksp_ew_threshold <threshold> - Sets threshold

   Notes:
   To see all options, run your program with the -help option or consult the users manual

   Notes:
      SNES supports three approaches for computing (approximate) Jacobians: user provided via SNESSetJacobian(), matrix free, and computing explicitly with
      finite differences and coloring using MatFDColoring. It is also possible to use automatic differentiation and the MatFDColoring object.

   Level: beginner

.seealso: SNESSetOptionsPrefix(), SNESResetFromOptions(), SNES, SNESCreate()
@*/
PetscErrorCode  SNESSetFromOptions(SNES snes)
{
  PetscBool      flg,pcset,persist,set;
  PetscInt       i,indx,lag,grids;
  const char     *deft        = SNESNEWTONLS;
  const char     *convtests[] = {"default","skip","correct_pressure"};
  SNESKSPEW      *kctx        = NULL;
  char           type[256], monfilename[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;
  PCSide         pcside;
  const char     *optionsprefix;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscCall(SNESRegisterAll());
  ierr = PetscObjectOptionsBegin((PetscObject)snes);PetscCall(ierr);
  if (((PetscObject)snes)->type_name) deft = ((PetscObject)snes)->type_name;
  PetscCall(PetscOptionsFList("-snes_type","Nonlinear solver method","SNESSetType",SNESList,deft,type,256,&flg));
  if (flg) {
    PetscCall(SNESSetType(snes,type));
  } else if (!((PetscObject)snes)->type_name) {
    PetscCall(SNESSetType(snes,deft));
  }
  PetscCall(PetscOptionsReal("-snes_stol","Stop if step length less than","SNESSetTolerances",snes->stol,&snes->stol,NULL));
  PetscCall(PetscOptionsReal("-snes_atol","Stop if function norm less than","SNESSetTolerances",snes->abstol,&snes->abstol,NULL));

  PetscCall(PetscOptionsReal("-snes_rtol","Stop if decrease in function norm less than","SNESSetTolerances",snes->rtol,&snes->rtol,NULL));
  PetscCall(PetscOptionsReal("-snes_divergence_tolerance","Stop if residual norm increases by this factor","SNESSetDivergenceTolerance",snes->divtol,&snes->divtol,NULL));
  PetscCall(PetscOptionsInt("-snes_max_it","Maximum iterations","SNESSetTolerances",snes->max_its,&snes->max_its,NULL));
  PetscCall(PetscOptionsInt("-snes_max_funcs","Maximum function evaluations","SNESSetTolerances",snes->max_funcs,&snes->max_funcs,NULL));
  PetscCall(PetscOptionsInt("-snes_max_fail","Maximum nonlinear step failures","SNESSetMaxNonlinearStepFailures",snes->maxFailures,&snes->maxFailures,NULL));
  PetscCall(PetscOptionsInt("-snes_max_linear_solve_fail","Maximum failures in linear solves allowed","SNESSetMaxLinearSolveFailures",snes->maxLinearSolveFailures,&snes->maxLinearSolveFailures,NULL));
  PetscCall(PetscOptionsBool("-snes_error_if_not_converged","Generate error if solver does not converge","SNESSetErrorIfNotConverged",snes->errorifnotconverged,&snes->errorifnotconverged,NULL));
  PetscCall(PetscOptionsBool("-snes_force_iteration","Force SNESSolve() to take at least one iteration","SNESSetForceIteration",snes->forceiteration,&snes->forceiteration,NULL));
  PetscCall(PetscOptionsBool("-snes_check_jacobian_domain_error","Check Jacobian domain error after Jacobian evaluation","SNESCheckJacobianDomainError",snes->checkjacdomainerror,&snes->checkjacdomainerror,NULL));

  PetscCall(PetscOptionsInt("-snes_lag_preconditioner","How often to rebuild preconditioner","SNESSetLagPreconditioner",snes->lagpreconditioner,&lag,&flg));
  if (flg) {
    PetscCheck(lag != -1,PetscObjectComm((PetscObject)snes),PETSC_ERR_USER,"Cannot set the lag to -1 from the command line since the preconditioner must be built as least once, perhaps you mean -2");
    PetscCall(SNESSetLagPreconditioner(snes,lag));
  }
  PetscCall(PetscOptionsBool("-snes_lag_preconditioner_persists","Preconditioner lagging through multiple SNES solves","SNESSetLagPreconditionerPersists",snes->lagjac_persist,&persist,&flg));
  if (flg) {
    PetscCall(SNESSetLagPreconditionerPersists(snes,persist));
  }
  PetscCall(PetscOptionsInt("-snes_lag_jacobian","How often to rebuild Jacobian","SNESSetLagJacobian",snes->lagjacobian,&lag,&flg));
  if (flg) {
    PetscCheck(lag != -1,PetscObjectComm((PetscObject)snes),PETSC_ERR_USER,"Cannot set the lag to -1 from the command line since the Jacobian must be built as least once, perhaps you mean -2");
    PetscCall(SNESSetLagJacobian(snes,lag));
  }
  PetscCall(PetscOptionsBool("-snes_lag_jacobian_persists","Jacobian lagging through multiple SNES solves","SNESSetLagJacobianPersists",snes->lagjac_persist,&persist,&flg));
  if (flg) {
    PetscCall(SNESSetLagJacobianPersists(snes,persist));
  }

  PetscCall(PetscOptionsInt("-snes_grid_sequence","Use grid sequencing to generate initial guess","SNESSetGridSequence",snes->gridsequence,&grids,&flg));
  if (flg) {
    PetscCall(SNESSetGridSequence(snes,grids));
  }

  PetscCall(PetscOptionsEList("-snes_convergence_test","Convergence test","SNESSetConvergenceTest",convtests,sizeof(convtests)/sizeof(char*),"default",&indx,&flg));
  if (flg) {
    switch (indx) {
    case 0: PetscCall(SNESSetConvergenceTest(snes,SNESConvergedDefault,NULL,NULL)); break;
    case 1: PetscCall(SNESSetConvergenceTest(snes,SNESConvergedSkip,NULL,NULL)); break;
    case 2: PetscCall(SNESSetConvergenceTest(snes,SNESConvergedCorrectPressure,NULL,NULL)); break;
    }
  }

  PetscCall(PetscOptionsEList("-snes_norm_schedule","SNES Norm schedule","SNESSetNormSchedule",SNESNormSchedules,5,"function",&indx,&flg));
  if (flg) PetscCall(SNESSetNormSchedule(snes,(SNESNormSchedule)indx));

  PetscCall(PetscOptionsEList("-snes_function_type","SNES Norm schedule","SNESSetFunctionType",SNESFunctionTypes,2,"unpreconditioned",&indx,&flg));
  if (flg) PetscCall(SNESSetFunctionType(snes,(SNESFunctionType)indx));

  kctx = (SNESKSPEW*)snes->kspconvctx;

  PetscCall(PetscOptionsBool("-snes_ksp_ew","Use Eisentat-Walker linear system convergence test","SNESKSPSetUseEW",snes->ksp_ewconv,&snes->ksp_ewconv,NULL));

  PetscCall(PetscOptionsInt("-snes_ksp_ew_version","Version 1, 2 or 3","SNESKSPSetParametersEW",kctx->version,&kctx->version,NULL));
  PetscCall(PetscOptionsReal("-snes_ksp_ew_rtol0","0 <= rtol0 < 1","SNESKSPSetParametersEW",kctx->rtol_0,&kctx->rtol_0,NULL));
  PetscCall(PetscOptionsReal("-snes_ksp_ew_rtolmax","0 <= rtolmax < 1","SNESKSPSetParametersEW",kctx->rtol_max,&kctx->rtol_max,NULL));
  PetscCall(PetscOptionsReal("-snes_ksp_ew_gamma","0 <= gamma <= 1","SNESKSPSetParametersEW",kctx->gamma,&kctx->gamma,NULL));
  PetscCall(PetscOptionsReal("-snes_ksp_ew_alpha","1 < alpha <= 2","SNESKSPSetParametersEW",kctx->alpha,&kctx->alpha,NULL));
  PetscCall(PetscOptionsReal("-snes_ksp_ew_alpha2","alpha2","SNESKSPSetParametersEW",kctx->alpha2,&kctx->alpha2,NULL));
  PetscCall(PetscOptionsReal("-snes_ksp_ew_threshold","0 < threshold < 1","SNESKSPSetParametersEW",kctx->threshold,&kctx->threshold,NULL));

  flg  = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-snes_monitor_cancel","Remove all monitors","SNESMonitorCancel",flg,&flg,&set));
  if (set && flg) PetscCall(SNESMonitorCancel(snes));

  PetscCall(SNESMonitorSetFromOptions(snes,"-snes_monitor","Monitor norm of function","SNESMonitorDefault",SNESMonitorDefault,SNESMonitorDefaultSetUp));
  PetscCall(SNESMonitorSetFromOptions(snes,"-snes_monitor_short","Monitor norm of function with fewer digits","SNESMonitorDefaultShort",SNESMonitorDefaultShort,NULL));
  PetscCall(SNESMonitorSetFromOptions(snes,"-snes_monitor_range","Monitor range of elements of function","SNESMonitorRange",SNESMonitorRange,NULL));

  PetscCall(SNESMonitorSetFromOptions(snes,"-snes_monitor_ratio","Monitor ratios of the norm of function for consecutive steps","SNESMonitorRatio",SNESMonitorRatio,SNESMonitorRatioSetUp));
  PetscCall(SNESMonitorSetFromOptions(snes,"-snes_monitor_field","Monitor norm of function (split into fields)","SNESMonitorDefaultField",SNESMonitorDefaultField,NULL));
  PetscCall(SNESMonitorSetFromOptions(snes,"-snes_monitor_solution","View solution at each iteration","SNESMonitorSolution",SNESMonitorSolution,NULL));
  PetscCall(SNESMonitorSetFromOptions(snes,"-snes_monitor_solution_update","View correction at each iteration","SNESMonitorSolutionUpdate",SNESMonitorSolutionUpdate,NULL));
  PetscCall(SNESMonitorSetFromOptions(snes,"-snes_monitor_residual","View residual at each iteration","SNESMonitorResidual",SNESMonitorResidual,NULL));
  PetscCall(SNESMonitorSetFromOptions(snes,"-snes_monitor_jacupdate_spectrum","Print the change in the spectrum of the Jacobian","SNESMonitorJacUpdateSpectrum",SNESMonitorJacUpdateSpectrum,NULL));
  PetscCall(SNESMonitorSetFromOptions(snes,"-snes_monitor_fields","Monitor norm of function per field","SNESMonitorSet",SNESMonitorFields,NULL));
  PetscCall(PetscOptionsBool("-snes_monitor_pause_final", "Pauses all draw monitors at the final iterate", "SNESMonitorPauseFinal_Internal", PETSC_FALSE, &snes->pauseFinal, NULL));

  PetscCall(PetscOptionsString("-snes_monitor_python","Use Python function","SNESMonitorSet",NULL,monfilename,sizeof(monfilename),&flg));
  if (flg) PetscCall(PetscPythonMonitorSet((PetscObject)snes,monfilename));

  flg  = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-snes_monitor_lg_range","Plot function range at each iteration","SNESMonitorLGRange",flg,&flg,NULL));
  if (flg) {
    PetscViewer ctx;

    PetscCall(PetscViewerDrawOpen(PetscObjectComm((PetscObject)snes),NULL,NULL,PETSC_DECIDE,PETSC_DECIDE,400,300,&ctx));
    PetscCall(SNESMonitorSet(snes,SNESMonitorLGRange,ctx,(PetscErrorCode (*)(void**))PetscViewerDestroy));
  }

  flg  = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-snes_converged_reason_view_cancel","Remove all converged reason viewers","SNESConvergedReasonViewCancel",flg,&flg,&set));
  if (set && flg) PetscCall(SNESConvergedReasonViewCancel(snes));

  flg  = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-snes_fd","Use finite differences (slow) to compute Jacobian","SNESComputeJacobianDefault",flg,&flg,NULL));
  if (flg) {
    void    *functx;
    DM      dm;
    DMSNES  sdm;
    PetscCall(SNESGetDM(snes,&dm));
    PetscCall(DMGetDMSNES(dm,&sdm));
    sdm->jacobianctx = NULL;
    PetscCall(SNESGetFunction(snes,NULL,NULL,&functx));
    PetscCall(SNESSetJacobian(snes,snes->jacobian,snes->jacobian_pre,SNESComputeJacobianDefault,functx));
    PetscCall(PetscInfo(snes,"Setting default finite difference Jacobian matrix\n"));
  }

  flg  = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-snes_fd_function","Use finite differences (slow) to compute function from user objective","SNESObjectiveComputeFunctionDefaultFD",flg,&flg,NULL));
  if (flg) {
    PetscCall(SNESSetFunction(snes,NULL,SNESObjectiveComputeFunctionDefaultFD,NULL));
  }

  flg  = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-snes_fd_color","Use finite differences with coloring to compute Jacobian","SNESComputeJacobianDefaultColor",flg,&flg,NULL));
  if (flg) {
    DM             dm;
    DMSNES         sdm;
    PetscCall(SNESGetDM(snes,&dm));
    PetscCall(DMGetDMSNES(dm,&sdm));
    sdm->jacobianctx = NULL;
    PetscCall(SNESSetJacobian(snes,snes->jacobian,snes->jacobian_pre,SNESComputeJacobianDefaultColor,NULL));
    PetscCall(PetscInfo(snes,"Setting default finite difference coloring Jacobian matrix\n"));
  }

  flg  = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-snes_mf_operator","Use a Matrix-Free Jacobian with user-provided preconditioner matrix","SNESSetUseMatrixFree",PETSC_FALSE,&snes->mf_operator,&flg));
  if (flg && snes->mf_operator) {
    snes->mf_operator = PETSC_TRUE;
    snes->mf          = PETSC_TRUE;
  }
  flg  = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-snes_mf","Use a Matrix-Free Jacobian with no preconditioner matrix","SNESSetUseMatrixFree",PETSC_FALSE,&snes->mf,&flg));
  if (!flg && snes->mf_operator) snes->mf = PETSC_TRUE;
  PetscCall(PetscOptionsInt("-snes_mf_version","Matrix-Free routines version 1 or 2","None",snes->mf_version,&snes->mf_version,NULL));

  flg  = PETSC_FALSE;
  PetscCall(SNESGetNPCSide(snes,&pcside));
  PetscCall(PetscOptionsEnum("-snes_npc_side","SNES nonlinear preconditioner side","SNESSetNPCSide",PCSides,(PetscEnum)pcside,(PetscEnum*)&pcside,&flg));
  if (flg) PetscCall(SNESSetNPCSide(snes,pcside));

#if defined(PETSC_HAVE_SAWS)
  /*
    Publish convergence information using SAWs
  */
  flg  = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-snes_monitor_saws","Publish SNES progress using SAWs","SNESMonitorSet",flg,&flg,NULL));
  if (flg) {
    void *ctx;
    PetscCall(SNESMonitorSAWsCreate(snes,&ctx));
    PetscCall(SNESMonitorSet(snes,SNESMonitorSAWs,ctx,SNESMonitorSAWsDestroy));
  }
#endif
#if defined(PETSC_HAVE_SAWS)
  {
  PetscBool set;
  flg  = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-snes_saws_block","Block for SAWs at end of SNESSolve","PetscObjectSAWsBlock",((PetscObject)snes)->amspublishblock,&flg,&set));
  if (set) {
    PetscCall(PetscObjectSAWsSetBlock((PetscObject)snes,flg));
  }
  }
#endif

  for (i = 0; i < numberofsetfromoptions; i++) {
    PetscCall((*othersetfromoptions[i])(snes));
  }

  if (snes->ops->setfromoptions) {
    PetscCall((*snes->ops->setfromoptions)(PetscOptionsObject,snes));
  }

  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  PetscCall(PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject)snes));
  ierr = PetscOptionsEnd();PetscCall(ierr);

  if (snes->linesearch) {
    PetscCall(SNESGetLineSearch(snes, &snes->linesearch));
    PetscCall(SNESLineSearchSetFromOptions(snes->linesearch));
  }

  if (snes->usesksp) {
    if (!snes->ksp) PetscCall(SNESGetKSP(snes,&snes->ksp));
    PetscCall(KSPSetOperators(snes->ksp,snes->jacobian,snes->jacobian_pre));
    PetscCall(KSPSetFromOptions(snes->ksp));
  }

  /* if user has set the SNES NPC type via options database, create it. */
  PetscCall(SNESGetOptionsPrefix(snes, &optionsprefix));
  PetscCall(PetscOptionsHasName(((PetscObject)snes)->options,optionsprefix, "-npc_snes_type", &pcset));
  if (pcset && (!snes->npc)) {
    PetscCall(SNESGetNPC(snes, &snes->npc));
  }
  if (snes->npc) {
    PetscCall(SNESSetFromOptions(snes->npc));
  }
  snes->setfromoptionscalled++;
  PetscFunctionReturn(0);
}

/*@
   SNESResetFromOptions - Sets various SNES and KSP parameters from user options ONLY if the SNES was previously set from options

   Collective on SNES

   Input Parameter:
.  snes - the SNES context

   Level: beginner

.seealso: SNESSetFromOptions(), SNESSetOptionsPrefix()
@*/
PetscErrorCode SNESResetFromOptions(SNES snes)
{
  PetscFunctionBegin;
  if (snes->setfromoptionscalled) PetscCall(SNESSetFromOptions(snes));
  PetscFunctionReturn(0);
}

/*@C
   SNESSetComputeApplicationContext - Sets an optional function to compute a user-defined context for
   the nonlinear solvers.

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  compute - function to compute the context
-  destroy - function to destroy the context

   Level: intermediate

   Notes:
   This function is currently not available from Fortran.

.seealso: SNESGetApplicationContext(), SNESSetComputeApplicationContext(), SNESGetApplicationContext()
@*/
PetscErrorCode  SNESSetComputeApplicationContext(SNES snes,PetscErrorCode (*compute)(SNES,void**),PetscErrorCode (*destroy)(void**))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  snes->ops->usercompute = compute;
  snes->ops->userdestroy = destroy;
  PetscFunctionReturn(0);
}

/*@
   SNESSetApplicationContext - Sets the optional user-defined context for
   the nonlinear solvers.

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
-  usrP - optional user context

   Level: intermediate

   Fortran Notes:
    To use this from Fortran you must write a Fortran interface definition for this
    function that tells Fortran the Fortran derived data type that you are passing in as the ctx argument.

.seealso: SNESGetApplicationContext()
@*/
PetscErrorCode  SNESSetApplicationContext(SNES snes,void *usrP)
{
  KSP            ksp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscCall(SNESGetKSP(snes,&ksp));
  PetscCall(KSPSetApplicationContext(ksp,usrP));
  snes->user = usrP;
  PetscFunctionReturn(0);
}

/*@
   SNESGetApplicationContext - Gets the user-defined context for the
   nonlinear solvers.

   Not Collective

   Input Parameter:
.  snes - SNES context

   Output Parameter:
.  usrP - user context

   Fortran Notes:
    To use this from Fortran you must write a Fortran interface definition for this
    function that tells Fortran the Fortran derived data type that you are passing in as the ctx argument.

   Level: intermediate

.seealso: SNESSetApplicationContext()
@*/
PetscErrorCode  SNESGetApplicationContext(SNES snes,void *usrP)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  *(void**)usrP = snes->user;
  PetscFunctionReturn(0);
}

/*@
   SNESSetUseMatrixFree - indicates that SNES should use matrix free finite difference matrix vector products internally to apply the Jacobian.

   Collective on SNES

   Input Parameters:
+  snes - SNES context
.  mf_operator - use matrix-free only for the Amat used by SNESSetJacobian(), this means the user provided Pmat will continue to be used
-  mf - use matrix-free for both the Amat and Pmat used by SNESSetJacobian(), both the Amat and Pmat set in SNESSetJacobian() will be ignored

   Options Database:
+ -snes_mf - use matrix free for both the mat and pmat operator
. -snes_mf_operator - use matrix free only for the mat operator
. -snes_fd_color - compute the Jacobian via coloring and finite differences.
- -snes_fd - compute the Jacobian via finite differences (slow)

   Level: intermediate

   Notes:
      SNES supports three approaches for computing (approximate) Jacobians: user provided via SNESSetJacobian(), matrix free, and computing explicitly with
      finite differences and coloring using MatFDColoring. It is also possible to use automatic differentiation and the MatFDColoring object.

.seealso:   SNESGetUseMatrixFree(), MatCreateSNESMF(), SNESComputeJacobianDefaultColor()
@*/
PetscErrorCode  SNESSetUseMatrixFree(SNES snes,PetscBool mf_operator,PetscBool mf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidLogicalCollectiveBool(snes,mf_operator,2);
  PetscValidLogicalCollectiveBool(snes,mf,3);
  snes->mf          = mf_operator ? PETSC_TRUE : mf;
  snes->mf_operator = mf_operator;
  PetscFunctionReturn(0);
}

/*@
   SNESGetUseMatrixFree - indicates if the SNES uses matrix free finite difference matrix vector products to apply the Jacobian.

   Collective on SNES

   Input Parameter:
.  snes - SNES context

   Output Parameters:
+  mf_operator - use matrix-free only for the Amat used by SNESSetJacobian(), this means the user provided Pmat will continue to be used
-  mf - use matrix-free for both the Amat and Pmat used by SNESSetJacobian(), both the Amat and Pmat set in SNESSetJacobian() will be ignored

   Options Database:
+ -snes_mf - use matrix free for both the mat and pmat operator
- -snes_mf_operator - use matrix free only for the mat operator

   Level: intermediate

.seealso:   SNESSetUseMatrixFree(), MatCreateSNESMF()
@*/
PetscErrorCode  SNESGetUseMatrixFree(SNES snes,PetscBool *mf_operator,PetscBool *mf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (mf)          *mf          = snes->mf;
  if (mf_operator) *mf_operator = snes->mf_operator;
  PetscFunctionReturn(0);
}

/*@
   SNESGetIterationNumber - Gets the number of nonlinear iterations completed
   at this time.

   Not Collective

   Input Parameter:
.  snes - SNES context

   Output Parameter:
.  iter - iteration number

   Notes:
   For example, during the computation of iteration 2 this would return 1.

   This is useful for using lagged Jacobians (where one does not recompute the
   Jacobian at each SNES iteration). For example, the code
.vb
      ierr = SNESGetIterationNumber(snes,&it);
      if (!(it % 2)) {
        [compute Jacobian here]
      }
.ve
   can be used in your ComputeJacobian() function to cause the Jacobian to be
   recomputed every second SNES iteration.

   After the SNES solve is complete this will return the number of nonlinear iterations used.

   Level: intermediate

.seealso:   SNESGetLinearSolveIterations()
@*/
PetscErrorCode  SNESGetIterationNumber(SNES snes,PetscInt *iter)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidIntPointer(iter,2);
  *iter = snes->iter;
  PetscFunctionReturn(0);
}

/*@
   SNESSetIterationNumber - Sets the current iteration number.

   Not Collective

   Input Parameters:
+  snes - SNES context
-  iter - iteration number

   Level: developer

.seealso:   SNESGetLinearSolveIterations()
@*/
PetscErrorCode  SNESSetIterationNumber(SNES snes,PetscInt iter)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
  snes->iter = iter;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));
  PetscFunctionReturn(0);
}

/*@
   SNESGetNonlinearStepFailures - Gets the number of unsuccessful steps
   attempted by the nonlinear solver.

   Not Collective

   Input Parameter:
.  snes - SNES context

   Output Parameter:
.  nfails - number of unsuccessful steps attempted

   Notes:
   This counter is reset to zero for each successive call to SNESSolve().

   Level: intermediate

.seealso: SNESGetMaxLinearSolveFailures(), SNESGetLinearSolveIterations(), SNESSetMaxLinearSolveFailures(), SNESGetLinearSolveFailures(),
          SNESSetMaxNonlinearStepFailures(), SNESGetMaxNonlinearStepFailures()
@*/
PetscErrorCode  SNESGetNonlinearStepFailures(SNES snes,PetscInt *nfails)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidIntPointer(nfails,2);
  *nfails = snes->numFailures;
  PetscFunctionReturn(0);
}

/*@
   SNESSetMaxNonlinearStepFailures - Sets the maximum number of unsuccessful steps
   attempted by the nonlinear solver before it gives up.

   Not Collective

   Input Parameters:
+  snes     - SNES context
-  maxFails - maximum of unsuccessful steps

   Level: intermediate

.seealso: SNESGetMaxLinearSolveFailures(), SNESGetLinearSolveIterations(), SNESSetMaxLinearSolveFailures(), SNESGetLinearSolveFailures(),
          SNESGetMaxNonlinearStepFailures(), SNESGetNonlinearStepFailures()
@*/
PetscErrorCode  SNESSetMaxNonlinearStepFailures(SNES snes, PetscInt maxFails)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  snes->maxFailures = maxFails;
  PetscFunctionReturn(0);
}

/*@
   SNESGetMaxNonlinearStepFailures - Gets the maximum number of unsuccessful steps
   attempted by the nonlinear solver before it gives up.

   Not Collective

   Input Parameter:
.  snes     - SNES context

   Output Parameter:
.  maxFails - maximum of unsuccessful steps

   Level: intermediate

.seealso: SNESGetMaxLinearSolveFailures(), SNESGetLinearSolveIterations(), SNESSetMaxLinearSolveFailures(), SNESGetLinearSolveFailures(),
          SNESSetMaxNonlinearStepFailures(), SNESGetNonlinearStepFailures()

@*/
PetscErrorCode  SNESGetMaxNonlinearStepFailures(SNES snes, PetscInt *maxFails)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidIntPointer(maxFails,2);
  *maxFails = snes->maxFailures;
  PetscFunctionReturn(0);
}

/*@
   SNESGetNumberFunctionEvals - Gets the number of user provided function evaluations
     done by SNES.

   Not Collective

   Input Parameter:
.  snes     - SNES context

   Output Parameter:
.  nfuncs - number of evaluations

   Level: intermediate

   Notes:
    Reset every time SNESSolve is called unless SNESSetCountersReset() is used.

.seealso: SNESGetMaxLinearSolveFailures(), SNESGetLinearSolveIterations(), SNESSetMaxLinearSolveFailures(), SNESGetLinearSolveFailures(), SNESSetCountersReset()
@*/
PetscErrorCode  SNESGetNumberFunctionEvals(SNES snes, PetscInt *nfuncs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidIntPointer(nfuncs,2);
  *nfuncs = snes->nfuncs;
  PetscFunctionReturn(0);
}

/*@
   SNESGetLinearSolveFailures - Gets the number of failed (non-converged)
   linear solvers.

   Not Collective

   Input Parameter:
.  snes - SNES context

   Output Parameter:
.  nfails - number of failed solves

   Level: intermediate

   Options Database Keys:
. -snes_max_linear_solve_fail <num> - The number of failures before the solve is terminated

   Notes:
   This counter is reset to zero for each successive call to SNESSolve().

.seealso: SNESGetMaxLinearSolveFailures(), SNESGetLinearSolveIterations(), SNESSetMaxLinearSolveFailures()
@*/
PetscErrorCode  SNESGetLinearSolveFailures(SNES snes,PetscInt *nfails)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidIntPointer(nfails,2);
  *nfails = snes->numLinearSolveFailures;
  PetscFunctionReturn(0);
}

/*@
   SNESSetMaxLinearSolveFailures - the number of failed linear solve attempts
   allowed before SNES returns with a diverged reason of SNES_DIVERGED_LINEAR_SOLVE

   Logically Collective on SNES

   Input Parameters:
+  snes     - SNES context
-  maxFails - maximum allowed linear solve failures

   Level: intermediate

   Options Database Keys:
. -snes_max_linear_solve_fail <num> - The number of failures before the solve is terminated

   Notes:
    By default this is 0; that is SNES returns on the first failed linear solve

.seealso: SNESGetLinearSolveFailures(), SNESGetMaxLinearSolveFailures(), SNESGetLinearSolveIterations()
@*/
PetscErrorCode  SNESSetMaxLinearSolveFailures(SNES snes, PetscInt maxFails)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidLogicalCollectiveInt(snes,maxFails,2);
  snes->maxLinearSolveFailures = maxFails;
  PetscFunctionReturn(0);
}

/*@
   SNESGetMaxLinearSolveFailures - gets the maximum number of linear solve failures that
     are allowed before SNES terminates

   Not Collective

   Input Parameter:
.  snes     - SNES context

   Output Parameter:
.  maxFails - maximum of unsuccessful solves allowed

   Level: intermediate

   Notes:
    By default this is 1; that is SNES returns on the first failed linear solve

.seealso: SNESGetLinearSolveFailures(), SNESGetLinearSolveIterations(), SNESSetMaxLinearSolveFailures(),
@*/
PetscErrorCode  SNESGetMaxLinearSolveFailures(SNES snes, PetscInt *maxFails)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidIntPointer(maxFails,2);
  *maxFails = snes->maxLinearSolveFailures;
  PetscFunctionReturn(0);
}

/*@
   SNESGetLinearSolveIterations - Gets the total number of linear iterations
   used by the nonlinear solver.

   Not Collective

   Input Parameter:
.  snes - SNES context

   Output Parameter:
.  lits - number of linear iterations

   Notes:
   This counter is reset to zero for each successive call to SNESSolve() unless SNESSetCountersReset() is used.

   If the linear solver fails inside the SNESSolve() the iterations for that call to the linear solver are not included. If you wish to count them
   then call KSPGetIterationNumber() after the failed solve.

   Level: intermediate

.seealso:  SNESGetIterationNumber(), SNESGetLinearSolveFailures(), SNESGetMaxLinearSolveFailures(), SNESSetCountersReset()
@*/
PetscErrorCode  SNESGetLinearSolveIterations(SNES snes,PetscInt *lits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidIntPointer(lits,2);
  *lits = snes->linear_its;
  PetscFunctionReturn(0);
}

/*@
   SNESSetCountersReset - Sets whether or not the counters for linear iterations and function evaluations
   are reset every time SNESSolve() is called.

   Logically Collective on SNES

   Input Parameters:
+  snes - SNES context
-  reset - whether to reset the counters or not

   Notes:
   This defaults to PETSC_TRUE

   Level: developer

.seealso:  SNESGetNumberFunctionEvals(), SNESGetLinearSolveIterations(), SNESGetNPC()
@*/
PetscErrorCode  SNESSetCountersReset(SNES snes,PetscBool reset)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidLogicalCollectiveBool(snes,reset,2);
  snes->counters_reset = reset;
  PetscFunctionReturn(0);
}

/*@
   SNESSetKSP - Sets a KSP context for the SNES object to use

   Not Collective, but the SNES and KSP objects must live on the same MPI_Comm

   Input Parameters:
+  snes - the SNES context
-  ksp - the KSP context

   Notes:
   The SNES object already has its KSP object, you can obtain with SNESGetKSP()
   so this routine is rarely needed.

   The KSP object that is already in the SNES object has its reference count
   decreased by one.

   Level: developer

.seealso: KSPGetPC(), SNESCreate(), KSPCreate(), SNESSetKSP()
@*/
PetscErrorCode  SNESSetKSP(SNES snes,KSP ksp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,2);
  PetscCheckSameComm(snes,1,ksp,2);
  PetscCall(PetscObjectReference((PetscObject)ksp));
  if (snes->ksp) PetscCall(PetscObjectDereference((PetscObject)snes->ksp));
  snes->ksp = ksp;
  PetscFunctionReturn(0);
}

/* -----------------------------------------------------------*/
/*@
   SNESCreate - Creates a nonlinear solver context.

   Collective

   Input Parameters:
.  comm - MPI communicator

   Output Parameter:
.  outsnes - the new SNES context

   Options Database Keys:
+   -snes_mf - Activates default matrix-free Jacobian-vector products,
               and no preconditioning matrix
.   -snes_mf_operator - Activates default matrix-free Jacobian-vector
               products, and a user-provided preconditioning matrix
               as set by SNESSetJacobian()
-   -snes_fd - Uses (slow!) finite differences to compute Jacobian

   Level: beginner

   Developer Notes:
    SNES always creates a KSP object even though many SNES methods do not use it. This is
                    unfortunate and should be fixed at some point. The flag snes->usesksp indicates if the
                    particular method does use KSP and regulates if the information about the KSP is printed
                    in SNESView(). TSSetFromOptions() does call SNESSetFromOptions() which can lead to users being confused
                    by help messages about meaningless SNES options.

                    SNES always creates the snes->kspconvctx even though it is used by only one type. This should
                    be fixed.

.seealso: SNESSolve(), SNESDestroy(), SNES, SNESSetLagPreconditioner(), SNESSetLagJacobian()

@*/
PetscErrorCode  SNESCreate(MPI_Comm comm,SNES *outsnes)
{
  SNES           snes;
  SNESKSPEW      *kctx;

  PetscFunctionBegin;
  PetscValidPointer(outsnes,2);
  *outsnes = NULL;
  PetscCall(SNESInitializePackage());

  PetscCall(PetscHeaderCreate(snes,SNES_CLASSID,"SNES","Nonlinear solver","SNES",comm,SNESDestroy,SNESView));

  snes->ops->converged    = SNESConvergedDefault;
  snes->usesksp           = PETSC_TRUE;
  snes->tolerancesset     = PETSC_FALSE;
  snes->max_its           = 50;
  snes->max_funcs         = 10000;
  snes->norm              = 0.0;
  snes->xnorm             = 0.0;
  snes->ynorm             = 0.0;
  snes->normschedule      = SNES_NORM_ALWAYS;
  snes->functype          = SNES_FUNCTION_DEFAULT;
#if defined(PETSC_USE_REAL_SINGLE)
  snes->rtol              = 1.e-5;
#else
  snes->rtol              = 1.e-8;
#endif
  snes->ttol              = 0.0;
#if defined(PETSC_USE_REAL_SINGLE)
  snes->abstol            = 1.e-25;
#else
  snes->abstol            = 1.e-50;
#endif
#if defined(PETSC_USE_REAL_SINGLE)
  snes->stol              = 1.e-5;
#else
  snes->stol              = 1.e-8;
#endif
#if defined(PETSC_USE_REAL_SINGLE)
  snes->deltatol          = 1.e-6;
#else
  snes->deltatol          = 1.e-12;
#endif
  snes->divtol            = 1.e4;
  snes->rnorm0            = 0;
  snes->nfuncs            = 0;
  snes->numFailures       = 0;
  snes->maxFailures       = 1;
  snes->linear_its        = 0;
  snes->lagjacobian       = 1;
  snes->jac_iter          = 0;
  snes->lagjac_persist    = PETSC_FALSE;
  snes->lagpreconditioner = 1;
  snes->pre_iter          = 0;
  snes->lagpre_persist    = PETSC_FALSE;
  snes->numbermonitors    = 0;
  snes->numberreasonviews = 0;
  snes->data              = NULL;
  snes->setupcalled       = PETSC_FALSE;
  snes->ksp_ewconv        = PETSC_FALSE;
  snes->nwork             = 0;
  snes->work              = NULL;
  snes->nvwork            = 0;
  snes->vwork             = NULL;
  snes->conv_hist_len     = 0;
  snes->conv_hist_max     = 0;
  snes->conv_hist         = NULL;
  snes->conv_hist_its     = NULL;
  snes->conv_hist_reset   = PETSC_TRUE;
  snes->counters_reset    = PETSC_TRUE;
  snes->vec_func_init_set = PETSC_FALSE;
  snes->reason            = SNES_CONVERGED_ITERATING;
  snes->npcside           = PC_RIGHT;
  snes->setfromoptionscalled = 0;

  snes->mf          = PETSC_FALSE;
  snes->mf_operator = PETSC_FALSE;
  snes->mf_version  = 1;

  snes->numLinearSolveFailures = 0;
  snes->maxLinearSolveFailures = 1;

  snes->vizerotolerance = 1.e-8;
  snes->checkjacdomainerror = PetscDefined(USE_DEBUG) ? PETSC_TRUE : PETSC_FALSE;

  /* Set this to true if the implementation of SNESSolve_XXX does compute the residual at the final solution. */
  snes->alwayscomputesfinalresidual = PETSC_FALSE;

  /* Create context to compute Eisenstat-Walker relative tolerance for KSP */
  PetscCall(PetscNewLog(snes,&kctx));

  snes->kspconvctx  = (void*)kctx;
  kctx->version     = 2;
  kctx->rtol_0      = .3; /* Eisenstat and Walker suggest rtol_0=.5, but
                             this was too large for some test cases */
  kctx->rtol_last   = 0.0;
  kctx->rtol_max    = .9;
  kctx->gamma       = 1.0;
  kctx->alpha       = .5*(1.0 + PetscSqrtReal(5.0));
  kctx->alpha2      = kctx->alpha;
  kctx->threshold   = .1;
  kctx->lresid_last = 0.0;
  kctx->norm_last   = 0.0;

  *outsnes = snes;
  PetscFunctionReturn(0);
}

/*MC
    SNESFunction - Functional form used to convey the nonlinear function to be solved by SNES

     Synopsis:
     #include "petscsnes.h"
     PetscErrorCode SNESFunction(SNES snes,Vec x,Vec f,void *ctx);

     Collective on snes

     Input Parameters:
+     snes - the SNES context
.     x    - state at which to evaluate residual
-     ctx     - optional user-defined function context, passed in with SNESSetFunction()

     Output Parameter:
.     f  - vector to put residual (function value)

   Level: intermediate

.seealso:   SNESSetFunction(), SNESGetFunction()
M*/

/*@C
   SNESSetFunction - Sets the function evaluation routine and function
   vector for use by the SNES routines in solving systems of nonlinear
   equations.

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  r - vector to store function values, may be NULL
.  f - function evaluation routine; see SNESFunction for calling sequence details
-  ctx - [optional] user-defined context for private data for the
         function evaluation routine (may be NULL)

   Notes:
   The Newton-like methods typically solve linear systems of the form
$      f'(x) x = -f(x),
   where f'(x) denotes the Jacobian matrix and f(x) is the function.

   Level: beginner

.seealso: SNESGetFunction(), SNESComputeFunction(), SNESSetJacobian(), SNESSetPicard(), SNESFunction
@*/
PetscErrorCode  SNESSetFunction(SNES snes,Vec r,PetscErrorCode (*f)(SNES,Vec,Vec,void*),void *ctx)
{
  DM             dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (r) {
    PetscValidHeaderSpecific(r,VEC_CLASSID,2);
    PetscCheckSameComm(snes,1,r,2);
    PetscCall(PetscObjectReference((PetscObject)r));
    PetscCall(VecDestroy(&snes->vec_func));
    snes->vec_func = r;
  }
  PetscCall(SNESGetDM(snes,&dm));
  PetscCall(DMSNESSetFunction(dm,f,ctx));
  if (f == SNESPicardComputeFunction) {
    PetscCall(DMSNESSetMFFunction(dm,SNESPicardComputeMFFunction,ctx));
  }
  PetscFunctionReturn(0);
}

/*@C
   SNESSetInitialFunction - Sets the function vector to be used as the
   function norm at the initialization of the method.  In some
   instances, the user has precomputed the function before calling
   SNESSolve.  This function allows one to avoid a redundant call
   to SNESComputeFunction in that case.

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
-  f - vector to store function value

   Notes:
   This should not be modified during the solution procedure.

   This is used extensively in the SNESFAS hierarchy and in nonlinear preconditioning.

   Level: developer

.seealso: SNESSetFunction(), SNESComputeFunction(), SNESSetInitialFunctionNorm()
@*/
PetscErrorCode  SNESSetInitialFunction(SNES snes, Vec f)
{
  Vec            vec_func;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidHeaderSpecific(f,VEC_CLASSID,2);
  PetscCheckSameComm(snes,1,f,2);
  if (snes->npcside == PC_LEFT && snes->functype == SNES_FUNCTION_PRECONDITIONED) {
    snes->vec_func_init_set = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
  PetscCall(SNESGetFunction(snes,&vec_func,NULL,NULL));
  PetscCall(VecCopy(f,vec_func));

  snes->vec_func_init_set = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
   SNESSetNormSchedule - Sets the SNESNormSchedule used in convergence and monitoring
   of the SNES method.

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
-  normschedule - the frequency of norm computation

   Options Database Key:
.  -snes_norm_schedule <none, always, initialonly, finalonly, initialfinalonly> - set the schedule

   Notes:
   Only certain SNES methods support certain SNESNormSchedules.  Most require evaluation
   of the nonlinear function and the taking of its norm at every iteration to
   even ensure convergence at all.  However, methods such as custom Gauss-Seidel methods
   (SNESNGS) and the like do not require the norm of the function to be computed, and therefore
   may either be monitored for convergence or not.  As these are often used as nonlinear
   preconditioners, monitoring the norm of their error is not a useful enterprise within
   their solution.

   Level: developer

.seealso: SNESGetNormSchedule(), SNESComputeFunction(), VecNorm(), SNESSetFunction(), SNESSetInitialFunction(), SNESNormSchedule
@*/
PetscErrorCode  SNESSetNormSchedule(SNES snes, SNESNormSchedule normschedule)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  snes->normschedule = normschedule;
  PetscFunctionReturn(0);
}

/*@
   SNESGetNormSchedule - Gets the SNESNormSchedule used in convergence and monitoring
   of the SNES method.

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
-  normschedule - the type of the norm used

   Level: advanced

.seealso: SNESSetNormSchedule(), SNESComputeFunction(), VecNorm(), SNESSetFunction(), SNESSetInitialFunction(), SNESNormSchedule
@*/
PetscErrorCode  SNESGetNormSchedule(SNES snes, SNESNormSchedule *normschedule)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  *normschedule = snes->normschedule;
  PetscFunctionReturn(0);
}

/*@
  SNESSetFunctionNorm - Sets the last computed residual norm.

  Logically Collective on SNES

  Input Parameters:
+ snes - the SNES context

- normschedule - the frequency of norm computation

  Level: developer

.seealso: SNESGetNormSchedule(), SNESComputeFunction(), VecNorm(), SNESSetFunction(), SNESSetInitialFunction(), SNESNormSchedule
@*/
PetscErrorCode SNESSetFunctionNorm(SNES snes, PetscReal norm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  snes->norm = norm;
  PetscFunctionReturn(0);
}

/*@
  SNESGetFunctionNorm - Gets the last computed norm of the residual

  Not Collective

  Input Parameter:
. snes - the SNES context

  Output Parameter:
. norm - the last computed residual norm

  Level: developer

.seealso: SNESSetNormSchedule(), SNESComputeFunction(), VecNorm(), SNESSetFunction(), SNESSetInitialFunction(), SNESNormSchedule
@*/
PetscErrorCode SNESGetFunctionNorm(SNES snes, PetscReal *norm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidRealPointer(norm, 2);
  *norm = snes->norm;
  PetscFunctionReturn(0);
}

/*@
  SNESGetUpdateNorm - Gets the last computed norm of the Newton update

  Not Collective

  Input Parameter:
. snes - the SNES context

  Output Parameter:
. ynorm - the last computed update norm

  Level: developer

.seealso: SNESSetNormSchedule(), SNESComputeFunction(), SNESGetFunctionNorm()
@*/
PetscErrorCode SNESGetUpdateNorm(SNES snes, PetscReal *ynorm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidRealPointer(ynorm, 2);
  *ynorm = snes->ynorm;
  PetscFunctionReturn(0);
}

/*@
  SNESGetSolutionNorm - Gets the last computed norm of the solution

  Not Collective

  Input Parameter:
. snes - the SNES context

  Output Parameter:
. xnorm - the last computed solution norm

  Level: developer

.seealso: SNESSetNormSchedule(), SNESComputeFunction(), SNESGetFunctionNorm(), SNESGetUpdateNorm()
@*/
PetscErrorCode SNESGetSolutionNorm(SNES snes, PetscReal *xnorm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidRealPointer(xnorm, 2);
  *xnorm = snes->xnorm;
  PetscFunctionReturn(0);
}

/*@C
   SNESSetFunctionType - Sets the SNESNormSchedule used in convergence and monitoring
   of the SNES method.

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
-  normschedule - the frequency of norm computation

   Notes:
   Only certain SNES methods support certain SNESNormSchedules.  Most require evaluation
   of the nonlinear function and the taking of its norm at every iteration to
   even ensure convergence at all.  However, methods such as custom Gauss-Seidel methods
   (SNESNGS) and the like do not require the norm of the function to be computed, and therefore
   may either be monitored for convergence or not.  As these are often used as nonlinear
   preconditioners, monitoring the norm of their error is not a useful enterprise within
   their solution.

   Level: developer

.seealso: SNESGetNormSchedule(), SNESComputeFunction(), VecNorm(), SNESSetFunction(), SNESSetInitialFunction(), SNESNormSchedule
@*/
PetscErrorCode  SNESSetFunctionType(SNES snes, SNESFunctionType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  snes->functype = type;
  PetscFunctionReturn(0);
}

/*@C
   SNESGetFunctionType - Gets the SNESNormSchedule used in convergence and monitoring
   of the SNES method.

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
-  normschedule - the type of the norm used

   Level: advanced

.seealso: SNESSetNormSchedule(), SNESComputeFunction(), VecNorm(), SNESSetFunction(), SNESSetInitialFunction(), SNESNormSchedule
@*/
PetscErrorCode  SNESGetFunctionType(SNES snes, SNESFunctionType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  *type = snes->functype;
  PetscFunctionReturn(0);
}

/*MC
    SNESNGSFunction - function used to convey a Gauss-Seidel sweep on the nonlinear function

     Synopsis:
     #include <petscsnes.h>
$    SNESNGSFunction(SNES snes,Vec x,Vec b,void *ctx);

     Collective on snes

     Input Parameters:
+  X   - solution vector
.  B   - RHS vector
-  ctx - optional user-defined Gauss-Seidel context

     Output Parameter:
.  X   - solution vector

   Level: intermediate

.seealso:   SNESSetNGS(), SNESGetNGS()
M*/

/*@C
   SNESSetNGS - Sets the user nonlinear Gauss-Seidel routine for
   use with composed nonlinear solvers.

   Input Parameters:
+  snes   - the SNES context
.  f - function evaluation routine to apply Gauss-Seidel see SNESNGSFunction
-  ctx    - [optional] user-defined context for private data for the
            smoother evaluation routine (may be NULL)

   Notes:
   The NGS routines are used by the composed nonlinear solver to generate
    a problem appropriate update to the solution, particularly FAS.

   Level: intermediate

.seealso: SNESGetFunction(), SNESComputeNGS()
@*/
PetscErrorCode SNESSetNGS(SNES snes,PetscErrorCode (*f)(SNES,Vec,Vec,void*),void *ctx)
{
  DM             dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscCall(SNESGetDM(snes,&dm));
  PetscCall(DMSNESSetNGS(dm,f,ctx));
  PetscFunctionReturn(0);
}

/*
     This is used for -snes_mf_operator; it uses a duplicate of snes->jacobian_pre because snes->jacobian_pre cannot be
   changed during the KSPSolve()
*/
PetscErrorCode SNESPicardComputeMFFunction(SNES snes,Vec x,Vec f,void *ctx)
{
  DM             dm;
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscCall(SNESGetDM(snes,&dm));
  PetscCall(DMGetDMSNES(dm,&sdm));
  PetscCheck(sdm->ops->computepjacobian,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "Must call SNESSetPicard() to provide Picard Jacobian.");
  /*  A(x)*x - b(x) */
  if (sdm->ops->computepfunction) {
    PetscStackPush("SNES Picard user function");
    PetscCall((*sdm->ops->computepfunction)(snes,x,f,sdm->pctx));
    PetscStackPop;
    PetscCall(VecScale(f,-1.0));
    if (!snes->picard) {
      /* Cannot share nonzero pattern because of the possible use of SNESComputeJacobianDefault() */
      PetscCall(MatDuplicate(snes->jacobian_pre,MAT_DO_NOT_COPY_VALUES,&snes->picard));
    }
    PetscStackPush("SNES Picard user Jacobian");
    PetscCall((*sdm->ops->computepjacobian)(snes,x,snes->picard,snes->picard,sdm->pctx));
    PetscStackPop;
    PetscCall(MatMultAdd(snes->picard,x,f,f));
  } else {
    PetscStackPush("SNES Picard user Jacobian");
    PetscCall((*sdm->ops->computepjacobian)(snes,x,snes->picard,snes->picard,sdm->pctx));
    PetscStackPop;
    PetscCall(MatMult(snes->picard,x,f));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SNESPicardComputeFunction(SNES snes,Vec x,Vec f,void *ctx)
{
  DM             dm;
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscCall(SNESGetDM(snes,&dm));
  PetscCall(DMGetDMSNES(dm,&sdm));
  PetscCheck(sdm->ops->computepjacobian,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "Must call SNESSetPicard() to provide Picard Jacobian.");
  /*  A(x)*x - b(x) */
  if (sdm->ops->computepfunction) {
    PetscStackPush("SNES Picard user function");
    PetscCall((*sdm->ops->computepfunction)(snes,x,f,sdm->pctx));
    PetscStackPop;
    PetscCall(VecScale(f,-1.0));
    PetscStackPush("SNES Picard user Jacobian");
    PetscCall((*sdm->ops->computepjacobian)(snes,x,snes->jacobian,snes->jacobian_pre,sdm->pctx));
    PetscStackPop;
    PetscCall(MatMultAdd(snes->jacobian_pre,x,f,f));
  } else {
    PetscStackPush("SNES Picard user Jacobian");
    PetscCall((*sdm->ops->computepjacobian)(snes,x,snes->jacobian,snes->jacobian_pre,sdm->pctx));
    PetscStackPop;
    PetscCall(MatMult(snes->jacobian_pre,x,f));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SNESPicardComputeJacobian(SNES snes,Vec x1,Mat J,Mat B,void *ctx)
{
  PetscFunctionBegin;
  /* the jacobian matrix should be pre-filled in SNESPicardComputeFunction */
  /* must assembly if matrix-free to get the last SNES solution */
  PetscCall(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

/*@C
   SNESSetPicard - Use SNES to solve the system A(x) x = bp(x) + b via a Picard type iteration (Picard linearization)

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  r - vector to store function values, may be NULL
.  bp - function evaluation routine, may be NULL
.  Amat - matrix with which A(x) x - bp(x) - b is to be computed
.  Pmat - matrix from which preconditioner is computed (usually the same as Amat)
.  J  - function to compute matrix values, see SNESJacobianFunction() for details on its calling sequence
-  ctx - [optional] user-defined context for private data for the function evaluation routine (may be NULL)

   Notes:
    It is often better to provide the nonlinear function F() and some approximation to its Jacobian directly and use
    an approximate Newton solver. This interface is provided to allow porting/testing a previous Picard based code in PETSc before converting it to approximate Newton.

    One can call SNESSetPicard() or SNESSetFunction() (and possibly SNESSetJacobian()) but cannot call both

$     Solves the equation A(x) x = bp(x) - b via the defect correction algorithm A(x^{n}) (x^{n+1} - x^{n}) = bp(x^{n}) + b - A(x^{n})x^{n}
$     Note that when an exact solver is used this corresponds to the "classic" Picard A(x^{n}) x^{n+1} = bp(x^{n}) + b iteration.

     Run with -snes_mf_operator to solve the system with Newton's method using A(x^{n}) to construct the preconditioner.

   We implement the defect correction form of the Picard iteration because it converges much more generally when inexact linear solvers are used then
   the direct Picard iteration A(x^n) x^{n+1} = bp(x^n) + b

   There is some controversity over the definition of a Picard iteration for nonlinear systems but almost everyone agrees that it involves a linear solve and some
   believe it is the iteration  A(x^{n}) x^{n+1} = b(x^{n}) hence we use the name Picard. If anyone has an authoritative  reference that defines the Picard iteration
   different please contact us at petsc-dev@mcs.anl.gov and we'll have an entirely new argument :-).

   When used with -snes_mf_operator this will run matrix-free Newton's method where the matrix-vector product is of the true Jacobian of A(x)x - bp(x) -b.

   When used with -snes_fd this will compute the true Jacobian (very slowly one column at at time) and thus represent Newton's method.

   When used with -snes_fd_coloring this will compute the Jacobian via coloring and thus represent a faster implementation of Newton's method. But the
   the nonzero structure of the Jacobian is, in general larger than that of the Picard matrix A so you must provide in A the needed nonzero structure for the correct
   coloring. When using DMDA this may mean creating the matrix A with DMCreateMatrix() using a wider stencil than strictly needed for A or with a DMDA_STENCIL_BOX.
   See the commment in src/snes/tutorials/ex15.c.

   Level: intermediate

.seealso: SNESGetFunction(), SNESSetFunction(), SNESComputeFunction(), SNESSetJacobian(), SNESGetPicard(), SNESLineSearchPreCheckPicard(), SNESJacobianFunction
@*/
PetscErrorCode  SNESSetPicard(SNES snes,Vec r,PetscErrorCode (*bp)(SNES,Vec,Vec,void*),Mat Amat, Mat Pmat, PetscErrorCode (*J)(SNES,Vec,Mat,Mat,void*),void *ctx)
{
  DM             dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(DMSNESSetPicard(dm,bp,J,ctx));
  PetscCall(DMSNESSetMFFunction(dm,SNESPicardComputeMFFunction,ctx));
  PetscCall(SNESSetFunction(snes,r,SNESPicardComputeFunction,ctx));
  PetscCall(SNESSetJacobian(snes,Amat,Pmat,SNESPicardComputeJacobian,ctx));
  PetscFunctionReturn(0);
}

/*@C
   SNESGetPicard - Returns the context for the Picard iteration

   Not Collective, but Vec is parallel if SNES is parallel. Collective if Vec is requested, but has not been created yet.

   Input Parameter:
.  snes - the SNES context

   Output Parameters:
+  r - the function (or NULL)
.  f - the function (or NULL); see SNESFunction for calling sequence details
.  Amat - the matrix used to defined the operation A(x) x - b(x) (or NULL)
.  Pmat  - the matrix from which the preconditioner will be constructed (or NULL)
.  J - the function for matrix evaluation (or NULL); see SNESJacobianFunction for calling sequence details
-  ctx - the function context (or NULL)

   Level: advanced

.seealso: SNESSetPicard(), SNESGetFunction(), SNESGetJacobian(), SNESGetDM(), SNESFunction, SNESJacobianFunction
@*/
PetscErrorCode  SNESGetPicard(SNES snes,Vec *r,PetscErrorCode (**f)(SNES,Vec,Vec,void*),Mat *Amat, Mat *Pmat, PetscErrorCode (**J)(SNES,Vec,Mat,Mat,void*),void **ctx)
{
  DM             dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscCall(SNESGetFunction(snes,r,NULL,NULL));
  PetscCall(SNESGetJacobian(snes,Amat,Pmat,NULL,NULL));
  PetscCall(SNESGetDM(snes,&dm));
  PetscCall(DMSNESGetPicard(dm,f,J,ctx));
  PetscFunctionReturn(0);
}

/*@C
   SNESSetComputeInitialGuess - Sets a routine used to compute an initial guess for the problem

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  func - function evaluation routine
-  ctx - [optional] user-defined context for private data for the
         function evaluation routine (may be NULL)

   Calling sequence of func:
$    func (SNES snes,Vec x,void *ctx);

.  f - function vector
-  ctx - optional user-defined function context

   Level: intermediate

.seealso: SNESGetFunction(), SNESComputeFunction(), SNESSetJacobian()
@*/
PetscErrorCode  SNESSetComputeInitialGuess(SNES snes,PetscErrorCode (*func)(SNES,Vec,void*),void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (func) snes->ops->computeinitialguess = func;
  if (ctx)  snes->initialguessP            = ctx;
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------- */
/*@C
   SNESGetRhs - Gets the vector for solving F(x) = rhs. If rhs is not set
   it assumes a zero right hand side.

   Logically Collective on SNES

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
.  rhs - the right hand side vector or NULL if the right hand side vector is null

   Level: intermediate

.seealso: SNESGetSolution(), SNESGetFunction(), SNESComputeFunction(), SNESSetJacobian(), SNESSetFunction()
@*/
PetscErrorCode  SNESGetRhs(SNES snes,Vec *rhs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidPointer(rhs,2);
  *rhs = snes->vec_rhs;
  PetscFunctionReturn(0);
}

/*@
   SNESComputeFunction - Calls the function that has been set with SNESSetFunction().

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
-  x - input vector

   Output Parameter:
.  y - function vector, as set by SNESSetFunction()

   Notes:
   SNESComputeFunction() is typically used within nonlinear solvers
   implementations, so users would not generally call this routine themselves.

   Level: developer

.seealso: SNESSetFunction(), SNESGetFunction(), SNESComputeMFFunction()
@*/
PetscErrorCode  SNESComputeFunction(SNES snes,Vec x,Vec y)
{
  DM             dm;
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  PetscCheckSameComm(snes,1,x,2);
  PetscCheckSameComm(snes,1,y,3);
  PetscCall(VecValidValues(x,2,PETSC_TRUE));

  PetscCall(SNESGetDM(snes,&dm));
  PetscCall(DMGetDMSNES(dm,&sdm));
  if (sdm->ops->computefunction) {
    if (sdm->ops->computefunction != SNESObjectiveComputeFunctionDefaultFD) {
      PetscCall(PetscLogEventBegin(SNES_FunctionEval,snes,x,y,0));
    }
    PetscCall(VecLockReadPush(x));
    PetscStackPush("SNES user function");
    /* ensure domainerror is false prior to computefunction evaluation (may not have been reset) */
    snes->domainerror = PETSC_FALSE;
    PetscCall((*sdm->ops->computefunction)(snes,x,y,sdm->functionctx));
    PetscStackPop;
    PetscCall(VecLockReadPop(x));
    if (sdm->ops->computefunction != SNESObjectiveComputeFunctionDefaultFD) {
      PetscCall(PetscLogEventEnd(SNES_FunctionEval,snes,x,y,0));
    }
  } else if (snes->vec_rhs) {
    PetscCall(MatMult(snes->jacobian, x, y));
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "Must call SNESSetFunction() or SNESSetDM() before SNESComputeFunction(), likely called from SNESSolve().");
  if (snes->vec_rhs) {
    PetscCall(VecAXPY(y,-1.0,snes->vec_rhs));
  }
  snes->nfuncs++;
  /*
     domainerror might not be set on all processes; so we tag vector locally with Inf and the next inner product or norm will
     propagate the value to all processes
  */
  if (snes->domainerror) {
    PetscCall(VecSetInf(y));
  }
  PetscFunctionReturn(0);
}

/*@
   SNESComputeMFFunction - Calls the function that has been set with SNESSetMFFunction().

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
-  x - input vector

   Output Parameter:
.  y - function vector, as set by SNESSetMFFunction()

   Notes:
       SNESComputeMFFunction() is used within the matrix vector products called by the matrix created with MatCreateSNESMF()
   so users would not generally call this routine themselves.

       Since this function is intended for use with finite differencing it does not subtract the right hand side vector provided with SNESSolve()
    while SNESComputeFunction() does. As such, this routine cannot be used with  MatMFFDSetBase() with a provided F function value even if it applies the
    same function as SNESComputeFunction() if a SNESSolve() right hand side vector is use because the two functions difference would include this right hand side function.

   Level: developer

.seealso: SNESSetFunction(), SNESGetFunction(), SNESComputeFunction(), MatCreateSNESMF
@*/
PetscErrorCode  SNESComputeMFFunction(SNES snes,Vec x,Vec y)
{
  DM             dm;
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  PetscCheckSameComm(snes,1,x,2);
  PetscCheckSameComm(snes,1,y,3);
  PetscCall(VecValidValues(x,2,PETSC_TRUE));

  PetscCall(SNESGetDM(snes,&dm));
  PetscCall(DMGetDMSNES(dm,&sdm));
  PetscCall(PetscLogEventBegin(SNES_FunctionEval,snes,x,y,0));
  PetscCall(VecLockReadPush(x));
  PetscStackPush("SNES user function");
  /* ensure domainerror is false prior to computefunction evaluation (may not have been reset) */
  snes->domainerror = PETSC_FALSE;
  PetscCall((*sdm->ops->computemffunction)(snes,x,y,sdm->mffunctionctx));
  PetscStackPop;
  PetscCall(VecLockReadPop(x));
  PetscCall(PetscLogEventEnd(SNES_FunctionEval,snes,x,y,0));
  snes->nfuncs++;
  /*
     domainerror might not be set on all processes; so we tag vector locally with Inf and the next inner product or norm will
     propagate the value to all processes
  */
  if (snes->domainerror) {
    PetscCall(VecSetInf(y));
  }
  PetscFunctionReturn(0);
}

/*@
   SNESComputeNGS - Calls the Gauss-Seidel function that has been set with  SNESSetNGS().

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  x - input vector
-  b - rhs vector

   Output Parameter:
.  x - new solution vector

   Notes:
   SNESComputeNGS() is typically used within composed nonlinear solver
   implementations, so most users would not generally call this routine
   themselves.

   Level: developer

.seealso: SNESSetNGS(), SNESComputeFunction()
@*/
PetscErrorCode  SNESComputeNGS(SNES snes,Vec b,Vec x)
{
  DM             dm;
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,3);
  if (b) PetscValidHeaderSpecific(b,VEC_CLASSID,2);
  PetscCheckSameComm(snes,1,x,3);
  if (b) PetscCheckSameComm(snes,1,b,2);
  if (b) PetscCall(VecValidValues(b,2,PETSC_TRUE));
  PetscCall(PetscLogEventBegin(SNES_NGSEval,snes,x,b,0));
  PetscCall(SNESGetDM(snes,&dm));
  PetscCall(DMGetDMSNES(dm,&sdm));
  if (sdm->ops->computegs) {
    if (b) PetscCall(VecLockReadPush(b));
    PetscStackPush("SNES user NGS");
    PetscCall((*sdm->ops->computegs)(snes,x,b,sdm->gsctx));
    PetscStackPop;
    if (b) PetscCall(VecLockReadPop(b));
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "Must call SNESSetNGS() before SNESComputeNGS(), likely called from SNESSolve().");
  PetscCall(PetscLogEventEnd(SNES_NGSEval,snes,x,b,0));
  PetscFunctionReturn(0);
}

PetscErrorCode SNESTestJacobian(SNES snes)
{
  Mat               A,B,C,D,jacobian;
  Vec               x = snes->vec_sol,f = snes->vec_func;
  PetscErrorCode    ierr;
  PetscReal         nrm,gnorm;
  PetscReal         threshold = 1.e-5;
  MatType           mattype;
  PetscInt          m,n,M,N;
  void              *functx;
  PetscBool         complete_print = PETSC_FALSE,threshold_print = PETSC_FALSE,test = PETSC_FALSE,flg,istranspose;
  PetscViewer       viewer,mviewer;
  MPI_Comm          comm;
  PetscInt          tabs;
  static PetscBool  directionsprinted = PETSC_FALSE;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscObjectOptionsBegin((PetscObject)snes);PetscCall(ierr);
  PetscCall(PetscOptionsName("-snes_test_jacobian","Compare hand-coded and finite difference Jacobians","None",&test));
  PetscCall(PetscOptionsReal("-snes_test_jacobian", "Threshold for element difference between hand-coded and finite difference being meaningful", "None", threshold, &threshold,NULL));
  PetscCall(PetscOptionsViewer("-snes_test_jacobian_view","View difference between hand-coded and finite difference Jacobians element entries","None",&mviewer,&format,&complete_print));
  if (!complete_print) {
    PetscCall(PetscOptionsDeprecated("-snes_test_jacobian_display","-snes_test_jacobian_view","3.13",NULL));
    PetscCall(PetscOptionsViewer("-snes_test_jacobian_display","Display difference between hand-coded and finite difference Jacobians","None",&mviewer,&format,&complete_print));
  }
  /* for compatibility with PETSc 3.9 and older. */
  PetscCall(PetscOptionsDeprecated("-snes_test_jacobian_display_threshold","-snes_test_jacobian","3.13","-snes_test_jacobian accepts an optional threshold (since v3.10)"));
  PetscCall(PetscOptionsReal("-snes_test_jacobian_display_threshold", "Display difference between hand-coded and finite difference Jacobians which exceed input threshold", "None", threshold, &threshold, &threshold_print));
  ierr = PetscOptionsEnd();PetscCall(ierr);
  if (!test) PetscFunctionReturn(0);

  PetscCall(PetscObjectGetComm((PetscObject)snes,&comm));
  PetscCall(PetscViewerASCIIGetStdout(comm,&viewer));
  PetscCall(PetscViewerASCIIGetTab(viewer, &tabs));
  PetscCall(PetscViewerASCIISetTab(viewer, ((PetscObject)snes)->tablevel));
  PetscCall(PetscViewerASCIIPrintf(viewer,"  ---------- Testing Jacobian -------------\n"));
  if (!complete_print && !directionsprinted) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Run with -snes_test_jacobian_view and optionally -snes_test_jacobian <threshold> to show difference\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer,"    of hand-coded and finite difference Jacobian entries greater than <threshold>.\n"));
  }
  if (!directionsprinted) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Testing hand-coded Jacobian, if (for double precision runs) ||J - Jfd||_F/||J||_F is\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer,"    O(1.e-8), the hand-coded Jacobian is probably correct.\n"));
    directionsprinted = PETSC_TRUE;
  }
  if (complete_print) {
    PetscCall(PetscViewerPushFormat(mviewer,format));
  }

  PetscCall(PetscObjectTypeCompare((PetscObject)snes->jacobian,MATMFFD,&flg));
  if (!flg) jacobian = snes->jacobian;
  else jacobian = snes->jacobian_pre;

  if (!x) {
    PetscCall(MatCreateVecs(jacobian, &x, NULL));
  } else {
    PetscCall(PetscObjectReference((PetscObject) x));
  }
  if (!f) {
    PetscCall(VecDuplicate(x, &f));
  } else {
    PetscCall(PetscObjectReference((PetscObject) f));
  }
  /* evaluate the function at this point because SNESComputeJacobianDefault() assumes that the function has been evaluated and put into snes->vec_func */
  PetscCall(SNESComputeFunction(snes,x,f));
  PetscCall(VecDestroy(&f));
  PetscCall(PetscObjectTypeCompare((PetscObject)snes,SNESKSPTRANSPOSEONLY,&istranspose));
  while (jacobian) {
    Mat JT = NULL, Jsave = NULL;

    if (istranspose) {
      PetscCall(MatCreateTranspose(jacobian,&JT));
      Jsave = jacobian;
      jacobian = JT;
    }
    PetscCall(PetscObjectBaseTypeCompareAny((PetscObject)jacobian,&flg,MATSEQAIJ,MATMPIAIJ,MATSEQDENSE,MATMPIDENSE,MATSEQBAIJ,MATMPIBAIJ,MATSEQSBAIJ,MATMPISBAIJ,""));
    if (flg) {
      A    = jacobian;
      PetscCall(PetscObjectReference((PetscObject)A));
    } else {
      PetscCall(MatComputeOperator(jacobian,MATAIJ,&A));
    }

    PetscCall(MatGetType(A,&mattype));
    PetscCall(MatGetSize(A,&M,&N));
    PetscCall(MatGetLocalSize(A,&m,&n));
    PetscCall(MatCreate(PetscObjectComm((PetscObject)A),&B));
    PetscCall(MatSetType(B,mattype));
    PetscCall(MatSetSizes(B,m,n,M,N));
    PetscCall(MatSetBlockSizesFromMats(B,A,A));
    PetscCall(MatSetUp(B));
    PetscCall(MatSetOption(B,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));

    PetscCall(SNESGetFunction(snes,NULL,NULL,&functx));
    PetscCall(SNESComputeJacobianDefault(snes,x,B,B,functx));

    PetscCall(MatDuplicate(B,MAT_COPY_VALUES,&D));
    PetscCall(MatAYPX(D,-1.0,A,DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatNorm(D,NORM_FROBENIUS,&nrm));
    PetscCall(MatNorm(A,NORM_FROBENIUS,&gnorm));
    PetscCall(MatDestroy(&D));
    if (!gnorm) gnorm = 1; /* just in case */
    PetscCall(PetscViewerASCIIPrintf(viewer,"  ||J - Jfd||_F/||J||_F = %g, ||J - Jfd||_F = %g\n",(double)(nrm/gnorm),(double)nrm));

    if (complete_print) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  Hand-coded Jacobian ----------\n"));
      PetscCall(MatView(A,mviewer));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  Finite difference Jacobian ----------\n"));
      PetscCall(MatView(B,mviewer));
    }

    if (threshold_print || complete_print) {
      PetscInt          Istart, Iend, *ccols, bncols, cncols, j, row;
      PetscScalar       *cvals;
      const PetscInt    *bcols;
      const PetscScalar *bvals;

      PetscCall(MatCreate(PetscObjectComm((PetscObject)A),&C));
      PetscCall(MatSetType(C,mattype));
      PetscCall(MatSetSizes(C,m,n,M,N));
      PetscCall(MatSetBlockSizesFromMats(C,A,A));
      PetscCall(MatSetUp(C));
      PetscCall(MatSetOption(C,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));

      PetscCall(MatAYPX(B,-1.0,A,DIFFERENT_NONZERO_PATTERN));
      PetscCall(MatGetOwnershipRange(B,&Istart,&Iend));

      for (row = Istart; row < Iend; row++) {
        PetscCall(MatGetRow(B,row,&bncols,&bcols,&bvals));
        PetscCall(PetscMalloc2(bncols,&ccols,bncols,&cvals));
        for (j = 0, cncols = 0; j < bncols; j++) {
          if (PetscAbsScalar(bvals[j]) > threshold) {
            ccols[cncols] = bcols[j];
            cvals[cncols] = bvals[j];
            cncols += 1;
          }
        }
        if (cncols) {
          PetscCall(MatSetValues(C,1,&row,cncols,ccols,cvals,INSERT_VALUES));
        }
        PetscCall(MatRestoreRow(B,row,&bncols,&bcols,&bvals));
        PetscCall(PetscFree2(ccols,cvals));
      }
      PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  Hand-coded minus finite-difference Jacobian with tolerance %g ----------\n",(double)threshold));
      PetscCall(MatView(C,complete_print ? mviewer : viewer));
      PetscCall(MatDestroy(&C));
    }
    PetscCall(MatDestroy(&A));
    PetscCall(MatDestroy(&B));
    PetscCall(MatDestroy(&JT));
    if (Jsave) jacobian = Jsave;
    if (jacobian != snes->jacobian_pre) {
      jacobian = snes->jacobian_pre;
      PetscCall(PetscViewerASCIIPrintf(viewer,"  ---------- Testing Jacobian for preconditioner -------------\n"));
    }
    else jacobian = NULL;
  }
  PetscCall(VecDestroy(&x));
  if (complete_print) {
    PetscCall(PetscViewerPopFormat(mviewer));
  }
  if (mviewer) PetscCall(PetscViewerDestroy(&mviewer));
  PetscCall(PetscViewerASCIISetTab(viewer,tabs));
  PetscFunctionReturn(0);
}

/*@
   SNESComputeJacobian - Computes the Jacobian matrix that has been set with SNESSetJacobian().

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
-  x - input vector

   Output Parameters:
+  A - Jacobian matrix
-  B - optional preconditioning matrix

  Options Database Keys:
+    -snes_lag_preconditioner <lag> - how often to rebuild preconditioner
.    -snes_lag_jacobian <lag> - how often to rebuild Jacobian
.    -snes_test_jacobian <optional threshold> - compare the user provided Jacobian with one compute via finite differences to check for errors.  If a threshold is given, display only those entries whose difference is greater than the threshold.
.    -snes_test_jacobian_view - display the user provided Jacobian, the finite difference Jacobian and the difference between them to help users detect the location of errors in the user provided Jacobian
.    -snes_compare_explicit - Compare the computed Jacobian to the finite difference Jacobian and output the differences
.    -snes_compare_explicit_draw  - Compare the computed Jacobian to the finite difference Jacobian and draw the result
.    -snes_compare_explicit_contour  - Compare the computed Jacobian to the finite difference Jacobian and draw a contour plot with the result
.    -snes_compare_operator  - Make the comparison options above use the operator instead of the preconditioning matrix
.    -snes_compare_coloring - Compute the finite difference Jacobian using coloring and display norms of difference
.    -snes_compare_coloring_display - Compute the finite difference Jacobian using coloring and display verbose differences
.    -snes_compare_coloring_threshold - Display only those matrix entries that differ by more than a given threshold
.    -snes_compare_coloring_threshold_atol - Absolute tolerance for difference in matrix entries to be displayed by -snes_compare_coloring_threshold
.    -snes_compare_coloring_threshold_rtol - Relative tolerance for difference in matrix entries to be displayed by -snes_compare_coloring_threshold
.    -snes_compare_coloring_draw - Compute the finite difference Jacobian using coloring and draw differences
-    -snes_compare_coloring_draw_contour - Compute the finite difference Jacobian using coloring and show contours of matrices and differences

   Notes:
   Most users should not need to explicitly call this routine, as it
   is used internally within the nonlinear solvers.

   Developer Notes:
    This has duplicative ways of checking the accuracy of the user provided Jacobian (see the options above). This is for historical reasons, the routine SNESTestJacobian() use to used
      for with the SNESType of test that has been removed.

   Level: developer

.seealso:  SNESSetJacobian(), KSPSetOperators(), MatStructure, SNESSetLagPreconditioner(), SNESSetLagJacobian()
@*/
PetscErrorCode  SNESComputeJacobian(SNES snes,Vec X,Mat A,Mat B)
{
  PetscBool      flag;
  DM             dm;
  DMSNES         sdm;
  KSP            ksp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscCheckSameComm(snes,1,X,2);
  PetscCall(VecValidValues(X,2,PETSC_TRUE));
  PetscCall(SNESGetDM(snes,&dm));
  PetscCall(DMGetDMSNES(dm,&sdm));

  PetscCheck(sdm->ops->computejacobian,PetscObjectComm((PetscObject)snes),PETSC_ERR_USER,"Must call SNESSetJacobian(), DMSNESSetJacobian(), DMDASNESSetJacobianLocal(), etc");

  /* make sure that MatAssemblyBegin/End() is called on A matrix if it is matrix free */

  if (snes->lagjacobian == -2) {
    snes->lagjacobian = -1;

    PetscCall(PetscInfo(snes,"Recomputing Jacobian/preconditioner because lag is -2 (means compute Jacobian, but then never again) \n"));
  } else if (snes->lagjacobian == -1) {
    PetscCall(PetscInfo(snes,"Reusing Jacobian/preconditioner because lag is -1\n"));
    PetscCall(PetscObjectTypeCompare((PetscObject)A,MATMFFD,&flag));
    if (flag) {
      PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
    }
    PetscFunctionReturn(0);
  } else if (snes->lagjacobian > 1 && (snes->iter + snes->jac_iter) % snes->lagjacobian) {
    PetscCall(PetscInfo(snes,"Reusing Jacobian/preconditioner because lag is %D and SNES iteration is %D\n",snes->lagjacobian,snes->iter));
    PetscCall(PetscObjectTypeCompare((PetscObject)A,MATMFFD,&flag));
    if (flag) {
      PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
    }
    PetscFunctionReturn(0);
  }
  if (snes->npc && snes->npcside == PC_LEFT) {
    PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
    PetscFunctionReturn(0);
  }

  PetscCall(PetscLogEventBegin(SNES_JacobianEval,snes,X,A,B));
  PetscCall(VecLockReadPush(X));
  PetscStackPush("SNES user Jacobian function");
  PetscCall((*sdm->ops->computejacobian)(snes,X,A,B,sdm->jacobianctx));
  PetscStackPop;
  PetscCall(VecLockReadPop(X));
  PetscCall(PetscLogEventEnd(SNES_JacobianEval,snes,X,A,B));

  /* attach latest linearization point to the preconditioning matrix */
  PetscCall(PetscObjectCompose((PetscObject)B,"__SNES_latest_X",(PetscObject)X));

  /* the next line ensures that snes->ksp exists */
  PetscCall(SNESGetKSP(snes,&ksp));
  if (snes->lagpreconditioner == -2) {
    PetscCall(PetscInfo(snes,"Rebuilding preconditioner exactly once since lag is -2\n"));
    PetscCall(KSPSetReusePreconditioner(snes->ksp,PETSC_FALSE));
    snes->lagpreconditioner = -1;
  } else if (snes->lagpreconditioner == -1) {
    PetscCall(PetscInfo(snes,"Reusing preconditioner because lag is -1\n"));
    PetscCall(KSPSetReusePreconditioner(snes->ksp,PETSC_TRUE));
  } else if (snes->lagpreconditioner > 1 && (snes->iter + snes->pre_iter) % snes->lagpreconditioner) {
    PetscCall(PetscInfo(snes,"Reusing preconditioner because lag is %D and SNES iteration is %D\n",snes->lagpreconditioner,snes->iter));
    PetscCall(KSPSetReusePreconditioner(snes->ksp,PETSC_TRUE));
  } else {
    PetscCall(PetscInfo(snes,"Rebuilding preconditioner\n"));
    PetscCall(KSPSetReusePreconditioner(snes->ksp,PETSC_FALSE));
  }

  PetscCall(SNESTestJacobian(snes));
  /* make sure user returned a correct Jacobian and preconditioner */
  /* PetscValidHeaderSpecific(A,MAT_CLASSID,3);
    PetscValidHeaderSpecific(B,MAT_CLASSID,4);   */
  {
    PetscBool flag = PETSC_FALSE,flag_draw = PETSC_FALSE,flag_contour = PETSC_FALSE,flag_operator = PETSC_FALSE;
    PetscCall(PetscOptionsGetViewer(PetscObjectComm((PetscObject)snes),((PetscObject) snes)->options,((PetscObject)snes)->prefix,"-snes_compare_explicit",NULL,NULL,&flag));
    PetscCall(PetscOptionsGetViewer(PetscObjectComm((PetscObject)snes),((PetscObject) snes)->options,((PetscObject)snes)->prefix,"-snes_compare_explicit_draw",NULL,NULL,&flag_draw));
    PetscCall(PetscOptionsGetViewer(PetscObjectComm((PetscObject)snes),((PetscObject) snes)->options,((PetscObject)snes)->prefix,"-snes_compare_explicit_draw_contour",NULL,NULL,&flag_contour));
    PetscCall(PetscOptionsGetViewer(PetscObjectComm((PetscObject)snes),((PetscObject) snes)->options,((PetscObject)snes)->prefix,"-snes_compare_operator",NULL,NULL,&flag_operator));
    if (flag || flag_draw || flag_contour) {
      Mat          Bexp_mine = NULL,Bexp,FDexp;
      PetscViewer  vdraw,vstdout;
      PetscBool    flg;
      if (flag_operator) {
        PetscCall(MatComputeOperator(A,MATAIJ,&Bexp_mine));
        Bexp = Bexp_mine;
      } else {
        /* See if the preconditioning matrix can be viewed and added directly */
        PetscCall(PetscObjectBaseTypeCompareAny((PetscObject)B,&flg,MATSEQAIJ,MATMPIAIJ,MATSEQDENSE,MATMPIDENSE,MATSEQBAIJ,MATMPIBAIJ,MATSEQSBAIJ,MATMPIBAIJ,""));
        if (flg) Bexp = B;
        else {
          /* If the "preconditioning" matrix is itself MATSHELL or some other type without direct support */
          PetscCall(MatComputeOperator(B,MATAIJ,&Bexp_mine));
          Bexp = Bexp_mine;
        }
      }
      PetscCall(MatConvert(Bexp,MATSAME,MAT_INITIAL_MATRIX,&FDexp));
      PetscCall(SNESComputeJacobianDefault(snes,X,FDexp,FDexp,NULL));
      PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)snes),&vstdout));
      if (flag_draw || flag_contour) {
        PetscCall(PetscViewerDrawOpen(PetscObjectComm((PetscObject)snes),NULL,"Explicit Jacobians",PETSC_DECIDE,PETSC_DECIDE,300,300,&vdraw));
        if (flag_contour) PetscCall(PetscViewerPushFormat(vdraw,PETSC_VIEWER_DRAW_CONTOUR));
      } else vdraw = NULL;
      PetscCall(PetscViewerASCIIPrintf(vstdout,"Explicit %s\n",flag_operator ? "Jacobian" : "preconditioning Jacobian"));
      if (flag) PetscCall(MatView(Bexp,vstdout));
      if (vdraw) PetscCall(MatView(Bexp,vdraw));
      PetscCall(PetscViewerASCIIPrintf(vstdout,"Finite difference Jacobian\n"));
      if (flag) PetscCall(MatView(FDexp,vstdout));
      if (vdraw) PetscCall(MatView(FDexp,vdraw));
      PetscCall(MatAYPX(FDexp,-1.0,Bexp,SAME_NONZERO_PATTERN));
      PetscCall(PetscViewerASCIIPrintf(vstdout,"User-provided matrix minus finite difference Jacobian\n"));
      if (flag) PetscCall(MatView(FDexp,vstdout));
      if (vdraw) {              /* Always use contour for the difference */
        PetscCall(PetscViewerPushFormat(vdraw,PETSC_VIEWER_DRAW_CONTOUR));
        PetscCall(MatView(FDexp,vdraw));
        PetscCall(PetscViewerPopFormat(vdraw));
      }
      if (flag_contour) PetscCall(PetscViewerPopFormat(vdraw));
      PetscCall(PetscViewerDestroy(&vdraw));
      PetscCall(MatDestroy(&Bexp_mine));
      PetscCall(MatDestroy(&FDexp));
    }
  }
  {
    PetscBool flag = PETSC_FALSE,flag_display = PETSC_FALSE,flag_draw = PETSC_FALSE,flag_contour = PETSC_FALSE,flag_threshold = PETSC_FALSE;
    PetscReal threshold_atol = PETSC_SQRT_MACHINE_EPSILON,threshold_rtol = 10*PETSC_SQRT_MACHINE_EPSILON;
    PetscCall(PetscOptionsGetViewer(PetscObjectComm((PetscObject)snes),((PetscObject)snes)->options,((PetscObject)snes)->prefix,"-snes_compare_coloring",NULL,NULL,&flag));
    PetscCall(PetscOptionsGetViewer(PetscObjectComm((PetscObject)snes),((PetscObject)snes)->options,((PetscObject)snes)->prefix,"-snes_compare_coloring_display",NULL,NULL,&flag_display));
    PetscCall(PetscOptionsGetViewer(PetscObjectComm((PetscObject)snes),((PetscObject)snes)->options,((PetscObject)snes)->prefix,"-snes_compare_coloring_draw",NULL,NULL,&flag_draw));
    PetscCall(PetscOptionsGetViewer(PetscObjectComm((PetscObject)snes),((PetscObject)snes)->options,((PetscObject)snes)->prefix,"-snes_compare_coloring_draw_contour",NULL,NULL,&flag_contour));
    PetscCall(PetscOptionsGetViewer(PetscObjectComm((PetscObject)snes),((PetscObject)snes)->options,((PetscObject)snes)->prefix,"-snes_compare_coloring_threshold",NULL,NULL,&flag_threshold));
    if (flag_threshold) {
      PetscCall(PetscOptionsGetReal(((PetscObject)snes)->options,((PetscObject)snes)->prefix,"-snes_compare_coloring_threshold_rtol",&threshold_rtol,NULL));
      PetscCall(PetscOptionsGetReal(((PetscObject)snes)->options,((PetscObject)snes)->prefix,"-snes_compare_coloring_threshold_atol",&threshold_atol,NULL));
    }
    if (flag || flag_display || flag_draw || flag_contour || flag_threshold) {
      Mat            Bfd;
      PetscViewer    vdraw,vstdout;
      MatColoring    coloring;
      ISColoring     iscoloring;
      MatFDColoring  matfdcoloring;
      PetscErrorCode (*func)(SNES,Vec,Vec,void*);
      void           *funcctx;
      PetscReal      norm1,norm2,normmax;

      PetscCall(MatDuplicate(B,MAT_DO_NOT_COPY_VALUES,&Bfd));
      PetscCall(MatColoringCreate(Bfd,&coloring));
      PetscCall(MatColoringSetType(coloring,MATCOLORINGSL));
      PetscCall(MatColoringSetFromOptions(coloring));
      PetscCall(MatColoringApply(coloring,&iscoloring));
      PetscCall(MatColoringDestroy(&coloring));
      PetscCall(MatFDColoringCreate(Bfd,iscoloring,&matfdcoloring));
      PetscCall(MatFDColoringSetFromOptions(matfdcoloring));
      PetscCall(MatFDColoringSetUp(Bfd,iscoloring,matfdcoloring));
      PetscCall(ISColoringDestroy(&iscoloring));

      /* This method of getting the function is currently unreliable since it doesn't work for DM local functions. */
      PetscCall(SNESGetFunction(snes,NULL,&func,&funcctx));
      PetscCall(MatFDColoringSetFunction(matfdcoloring,(PetscErrorCode (*)(void))func,funcctx));
      PetscCall(PetscObjectSetOptionsPrefix((PetscObject)matfdcoloring,((PetscObject)snes)->prefix));
      PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)matfdcoloring,"coloring_"));
      PetscCall(MatFDColoringSetFromOptions(matfdcoloring));
      PetscCall(MatFDColoringApply(Bfd,matfdcoloring,X,snes));
      PetscCall(MatFDColoringDestroy(&matfdcoloring));

      PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)snes),&vstdout));
      if (flag_draw || flag_contour) {
        PetscCall(PetscViewerDrawOpen(PetscObjectComm((PetscObject)snes),NULL,"Colored Jacobians",PETSC_DECIDE,PETSC_DECIDE,300,300,&vdraw));
        if (flag_contour) PetscCall(PetscViewerPushFormat(vdraw,PETSC_VIEWER_DRAW_CONTOUR));
      } else vdraw = NULL;
      PetscCall(PetscViewerASCIIPrintf(vstdout,"Explicit preconditioning Jacobian\n"));
      if (flag_display) PetscCall(MatView(B,vstdout));
      if (vdraw) PetscCall(MatView(B,vdraw));
      PetscCall(PetscViewerASCIIPrintf(vstdout,"Colored Finite difference Jacobian\n"));
      if (flag_display) PetscCall(MatView(Bfd,vstdout));
      if (vdraw) PetscCall(MatView(Bfd,vdraw));
      PetscCall(MatAYPX(Bfd,-1.0,B,SAME_NONZERO_PATTERN));
      PetscCall(MatNorm(Bfd,NORM_1,&norm1));
      PetscCall(MatNorm(Bfd,NORM_FROBENIUS,&norm2));
      PetscCall(MatNorm(Bfd,NORM_MAX,&normmax));
      PetscCall(PetscViewerASCIIPrintf(vstdout,"User-provided matrix minus finite difference Jacobian, norm1=%g normFrob=%g normmax=%g\n",(double)norm1,(double)norm2,(double)normmax));
      if (flag_display) PetscCall(MatView(Bfd,vstdout));
      if (vdraw) {              /* Always use contour for the difference */
        PetscCall(PetscViewerPushFormat(vdraw,PETSC_VIEWER_DRAW_CONTOUR));
        PetscCall(MatView(Bfd,vdraw));
        PetscCall(PetscViewerPopFormat(vdraw));
      }
      if (flag_contour) PetscCall(PetscViewerPopFormat(vdraw));

      if (flag_threshold) {
        PetscInt bs,rstart,rend,i;
        PetscCall(MatGetBlockSize(B,&bs));
        PetscCall(MatGetOwnershipRange(B,&rstart,&rend));
        for (i=rstart; i<rend; i++) {
          const PetscScalar *ba,*ca;
          const PetscInt    *bj,*cj;
          PetscInt          bn,cn,j,maxentrycol = -1,maxdiffcol = -1,maxrdiffcol = -1;
          PetscReal         maxentry = 0,maxdiff = 0,maxrdiff = 0;
          PetscCall(MatGetRow(B,i,&bn,&bj,&ba));
          PetscCall(MatGetRow(Bfd,i,&cn,&cj,&ca));
          PetscCheck(bn == cn,((PetscObject)A)->comm,PETSC_ERR_PLIB,"Unexpected different nonzero pattern in -snes_compare_coloring_threshold");
          for (j=0; j<bn; j++) {
            PetscReal rdiff = PetscAbsScalar(ca[j]) / (threshold_atol + threshold_rtol*PetscAbsScalar(ba[j]));
            if (PetscAbsScalar(ba[j]) > PetscAbs(maxentry)) {
              maxentrycol = bj[j];
              maxentry    = PetscRealPart(ba[j]);
            }
            if (PetscAbsScalar(ca[j]) > PetscAbs(maxdiff)) {
              maxdiffcol = bj[j];
              maxdiff    = PetscRealPart(ca[j]);
            }
            if (rdiff > maxrdiff) {
              maxrdiffcol = bj[j];
              maxrdiff    = rdiff;
            }
          }
          if (maxrdiff > 1) {
            PetscCall(PetscViewerASCIIPrintf(vstdout,"row %D (maxentry=%g at %D, maxdiff=%g at %D, maxrdiff=%g at %D):",i,(double)maxentry,maxentrycol,(double)maxdiff,maxdiffcol,(double)maxrdiff,maxrdiffcol));
            for (j=0; j<bn; j++) {
              PetscReal rdiff;
              rdiff = PetscAbsScalar(ca[j]) / (threshold_atol + threshold_rtol*PetscAbsScalar(ba[j]));
              if (rdiff > 1) {
                PetscCall(PetscViewerASCIIPrintf(vstdout," (%D,%g:%g)",bj[j],(double)PetscRealPart(ba[j]),(double)PetscRealPart(ca[j])));
              }
            }
            PetscCall(PetscViewerASCIIPrintf(vstdout,"\n",i,maxentry,maxdiff,maxrdiff));
          }
          PetscCall(MatRestoreRow(B,i,&bn,&bj,&ba));
          PetscCall(MatRestoreRow(Bfd,i,&cn,&cj,&ca));
        }
      }
      PetscCall(PetscViewerDestroy(&vdraw));
      PetscCall(MatDestroy(&Bfd));
    }
  }
  PetscFunctionReturn(0);
}

/*MC
    SNESJacobianFunction - Function used to convey the nonlinear Jacobian of the function to be solved by SNES

     Synopsis:
     #include "petscsnes.h"
     PetscErrorCode SNESJacobianFunction(SNES snes,Vec x,Mat Amat,Mat Pmat,void *ctx);

     Collective on snes

    Input Parameters:
+  x - input vector, the Jacobian is to be computed at this value
-  ctx - [optional] user-defined Jacobian context

    Output Parameters:
+  Amat - the matrix that defines the (approximate) Jacobian
-  Pmat - the matrix to be used in constructing the preconditioner, usually the same as Amat.

   Level: intermediate

.seealso:   SNESSetFunction(), SNESGetFunction(), SNESSetJacobian(), SNESGetJacobian()
M*/

/*@C
   SNESSetJacobian - Sets the function to compute Jacobian as well as the
   location to store the matrix.

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  Amat - the matrix that defines the (approximate) Jacobian
.  Pmat - the matrix to be used in constructing the preconditioner, usually the same as Amat.
.  J - Jacobian evaluation routine (if NULL then SNES retains any previously set value), see SNESJacobianFunction for details
-  ctx - [optional] user-defined context for private data for the
         Jacobian evaluation routine (may be NULL) (if NULL then SNES retains any previously set value)

   Notes:
   If the Amat matrix and Pmat matrix are different you must call MatAssemblyBegin/End() on
   each matrix.

   If you know the operator Amat has a null space you can use MatSetNullSpace() and MatSetTransposeNullSpace() to supply the null
   space to Amat and the KSP solvers will automatically use that null space as needed during the solution process.

   If using SNESComputeJacobianDefaultColor() to assemble a Jacobian, the ctx argument
   must be a MatFDColoring.

   Other defect-correction schemes can be used by computing a different matrix in place of the Jacobian.  One common
   example is to use the "Picard linearization" which only differentiates through the highest order parts of each term.

   Level: beginner

.seealso: KSPSetOperators(), SNESSetFunction(), MatMFFDComputeJacobian(), SNESComputeJacobianDefaultColor(), MatStructure, J,
          SNESSetPicard(), SNESJacobianFunction
@*/
PetscErrorCode  SNESSetJacobian(SNES snes,Mat Amat,Mat Pmat,PetscErrorCode (*J)(SNES,Vec,Mat,Mat,void*),void *ctx)
{
  DM             dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (Amat) PetscValidHeaderSpecific(Amat,MAT_CLASSID,2);
  if (Pmat) PetscValidHeaderSpecific(Pmat,MAT_CLASSID,3);
  if (Amat) PetscCheckSameComm(snes,1,Amat,2);
  if (Pmat) PetscCheckSameComm(snes,1,Pmat,3);
  PetscCall(SNESGetDM(snes,&dm));
  PetscCall(DMSNESSetJacobian(dm,J,ctx));
  if (Amat) {
    PetscCall(PetscObjectReference((PetscObject)Amat));
    PetscCall(MatDestroy(&snes->jacobian));

    snes->jacobian = Amat;
  }
  if (Pmat) {
    PetscCall(PetscObjectReference((PetscObject)Pmat));
    PetscCall(MatDestroy(&snes->jacobian_pre));

    snes->jacobian_pre = Pmat;
  }
  PetscFunctionReturn(0);
}

/*@C
   SNESGetJacobian - Returns the Jacobian matrix and optionally the user
   provided context for evaluating the Jacobian.

   Not Collective, but Mat object will be parallel if SNES object is

   Input Parameter:
.  snes - the nonlinear solver context

   Output Parameters:
+  Amat - location to stash (approximate) Jacobian matrix (or NULL)
.  Pmat - location to stash matrix used to compute the preconditioner (or NULL)
.  J - location to put Jacobian function (or NULL), see SNESJacobianFunction for details on its calling sequence
-  ctx - location to stash Jacobian ctx (or NULL)

   Level: advanced

.seealso: SNESSetJacobian(), SNESComputeJacobian(), SNESJacobianFunction, SNESGetFunction()
@*/
PetscErrorCode SNESGetJacobian(SNES snes,Mat *Amat,Mat *Pmat,PetscErrorCode (**J)(SNES,Vec,Mat,Mat,void*),void **ctx)
{
  DM             dm;
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (Amat) *Amat = snes->jacobian;
  if (Pmat) *Pmat = snes->jacobian_pre;
  PetscCall(SNESGetDM(snes,&dm));
  PetscCall(DMGetDMSNES(dm,&sdm));
  if (J) *J = sdm->ops->computejacobian;
  if (ctx) *ctx = sdm->jacobianctx;
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESSetDefaultComputeJacobian(SNES snes)
{
  DM             dm;
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscCall(SNESGetDM(snes,&dm));
  PetscCall(DMGetDMSNES(dm,&sdm));
  if (!sdm->ops->computejacobian && snes->jacobian_pre) {
    DM        dm;
    PetscBool isdense,ismf;

    PetscCall(SNESGetDM(snes,&dm));
    PetscCall(PetscObjectTypeCompareAny((PetscObject)snes->jacobian_pre,&isdense,MATSEQDENSE,MATMPIDENSE,MATDENSE,NULL));
    PetscCall(PetscObjectTypeCompareAny((PetscObject)snes->jacobian_pre,&ismf,MATMFFD,MATSHELL,NULL));
    if (isdense) {
      PetscCall(DMSNESSetJacobian(dm,SNESComputeJacobianDefault,NULL));
    } else if (!ismf) {
      PetscCall(DMSNESSetJacobian(dm,SNESComputeJacobianDefaultColor,NULL));
    }
  }
  PetscFunctionReturn(0);
}

/*@
   SNESSetUp - Sets up the internal data structures for the later use
   of a nonlinear solver.

   Collective on SNES

   Input Parameters:
.  snes - the SNES context

   Notes:
   For basic use of the SNES solvers the user need not explicitly call
   SNESSetUp(), since these actions will automatically occur during
   the call to SNESSolve().  However, if one wishes to control this
   phase separately, SNESSetUp() should be called after SNESCreate()
   and optional routines of the form SNESSetXXX(), but before SNESSolve().

   Level: advanced

.seealso: SNESCreate(), SNESSolve(), SNESDestroy()
@*/
PetscErrorCode  SNESSetUp(SNES snes)
{
  DM             dm;
  DMSNES         sdm;
  SNESLineSearch linesearch, pclinesearch;
  void           *lsprectx,*lspostctx;
  PetscErrorCode (*precheck)(SNESLineSearch,Vec,Vec,PetscBool*,void*);
  PetscErrorCode (*postcheck)(SNESLineSearch,Vec,Vec,Vec,PetscBool*,PetscBool*,void*);
  PetscErrorCode (*func)(SNES,Vec,Vec,void*);
  Vec            f,fpc;
  void           *funcctx;
  PetscErrorCode (*jac)(SNES,Vec,Mat,Mat,void*);
  void           *jacctx,*appctx;
  Mat            j,jpre;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (snes->setupcalled) PetscFunctionReturn(0);
  PetscCall(PetscLogEventBegin(SNES_Setup,snes,0,0,0));

  if (!((PetscObject)snes)->type_name) {
    PetscCall(SNESSetType(snes,SNESNEWTONLS));
  }

  PetscCall(SNESGetFunction(snes,&snes->vec_func,NULL,NULL));

  PetscCall(SNESGetDM(snes,&dm));
  PetscCall(DMGetDMSNES(dm,&sdm));
  PetscCheck(sdm->ops->computefunction,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Function never provided to SNES object");
  PetscCall(SNESSetDefaultComputeJacobian(snes));

  if (!snes->vec_func) {
    PetscCall(DMCreateGlobalVector(dm,&snes->vec_func));
  }

  if (!snes->ksp) {
    PetscCall(SNESGetKSP(snes, &snes->ksp));
  }

  if (snes->linesearch) {
    PetscCall(SNESGetLineSearch(snes, &snes->linesearch));
    PetscCall(SNESLineSearchSetFunction(snes->linesearch,SNESComputeFunction));
  }

  if (snes->npc && snes->npcside == PC_LEFT) {
    snes->mf          = PETSC_TRUE;
    snes->mf_operator = PETSC_FALSE;
  }

  if (snes->npc) {
    /* copy the DM over */
    PetscCall(SNESGetDM(snes,&dm));
    PetscCall(SNESSetDM(snes->npc,dm));

    PetscCall(SNESGetFunction(snes,&f,&func,&funcctx));
    PetscCall(VecDuplicate(f,&fpc));
    PetscCall(SNESSetFunction(snes->npc,fpc,func,funcctx));
    PetscCall(SNESGetJacobian(snes,&j,&jpre,&jac,&jacctx));
    PetscCall(SNESSetJacobian(snes->npc,j,jpre,jac,jacctx));
    PetscCall(SNESGetApplicationContext(snes,&appctx));
    PetscCall(SNESSetApplicationContext(snes->npc,appctx));
    PetscCall(VecDestroy(&fpc));

    /* copy the function pointers over */
    PetscCall(PetscObjectCopyFortranFunctionPointers((PetscObject)snes,(PetscObject)snes->npc));

    /* default to 1 iteration */
    PetscCall(SNESSetTolerances(snes->npc,0.0,0.0,0.0,1,snes->npc->max_funcs));
    if (snes->npcside == PC_RIGHT) {
      PetscCall(SNESSetNormSchedule(snes->npc,SNES_NORM_FINAL_ONLY));
    } else {
      PetscCall(SNESSetNormSchedule(snes->npc,SNES_NORM_NONE));
    }
    PetscCall(SNESSetFromOptions(snes->npc));

    /* copy the line search context over */
    if (snes->linesearch && snes->npc->linesearch) {
      PetscCall(SNESGetLineSearch(snes,&linesearch));
      PetscCall(SNESGetLineSearch(snes->npc,&pclinesearch));
      PetscCall(SNESLineSearchGetPreCheck(linesearch,&precheck,&lsprectx));
      PetscCall(SNESLineSearchGetPostCheck(linesearch,&postcheck,&lspostctx));
      PetscCall(SNESLineSearchSetPreCheck(pclinesearch,precheck,lsprectx));
      PetscCall(SNESLineSearchSetPostCheck(pclinesearch,postcheck,lspostctx));
      PetscCall(PetscObjectCopyFortranFunctionPointers((PetscObject)linesearch, (PetscObject)pclinesearch));
    }
  }
  if (snes->mf) {
    PetscCall(SNESSetUpMatrixFree_Private(snes, snes->mf_operator, snes->mf_version));
  }
  if (snes->ops->usercompute && !snes->user) {
    PetscCall((*snes->ops->usercompute)(snes,(void**)&snes->user));
  }

  snes->jac_iter = 0;
  snes->pre_iter = 0;

  if (snes->ops->setup) {
    PetscCall((*snes->ops->setup)(snes));
  }

  PetscCall(SNESSetDefaultComputeJacobian(snes));

  if (snes->npc && snes->npcside == PC_LEFT) {
    if (snes->functype == SNES_FUNCTION_PRECONDITIONED) {
      if (snes->linesearch) {
        PetscCall(SNESGetLineSearch(snes,&linesearch));
        PetscCall(SNESLineSearchSetFunction(linesearch,SNESComputeFunctionDefaultNPC));
      }
    }
  }
  PetscCall(PetscLogEventEnd(SNES_Setup,snes,0,0,0));
  snes->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
   SNESReset - Resets a SNES context to the snessetupcalled = 0 state and removes any allocated Vecs and Mats

   Collective on SNES

   Input Parameter:
.  snes - iterative context obtained from SNESCreate()

   Level: intermediate

   Notes:
    Also calls the application context destroy routine set with SNESSetComputeApplicationContext()

.seealso: SNESCreate(), SNESSetUp(), SNESSolve()
@*/
PetscErrorCode  SNESReset(SNES snes)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (snes->ops->userdestroy && snes->user) {
    PetscCall((*snes->ops->userdestroy)((void**)&snes->user));
    snes->user = NULL;
  }
  if (snes->npc) {
    PetscCall(SNESReset(snes->npc));
  }

  if (snes->ops->reset) {
    PetscCall((*snes->ops->reset)(snes));
  }
  if (snes->ksp) {
    PetscCall(KSPReset(snes->ksp));
  }

  if (snes->linesearch) {
    PetscCall(SNESLineSearchReset(snes->linesearch));
  }

  PetscCall(VecDestroy(&snes->vec_rhs));
  PetscCall(VecDestroy(&snes->vec_sol));
  PetscCall(VecDestroy(&snes->vec_sol_update));
  PetscCall(VecDestroy(&snes->vec_func));
  PetscCall(MatDestroy(&snes->jacobian));
  PetscCall(MatDestroy(&snes->jacobian_pre));
  PetscCall(MatDestroy(&snes->picard));
  PetscCall(VecDestroyVecs(snes->nwork,&snes->work));
  PetscCall(VecDestroyVecs(snes->nvwork,&snes->vwork));

  snes->alwayscomputesfinalresidual = PETSC_FALSE;

  snes->nwork       = snes->nvwork = 0;
  snes->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@
   SNESConvergedReasonViewCancel - Clears all the reasonview functions for a SNES object.

   Collective on SNES

   Input Parameter:
.  snes - iterative context obtained from SNESCreate()

   Level: intermediate

.seealso: SNESCreate(), SNESDestroy(), SNESReset()
@*/
PetscErrorCode  SNESConvergedReasonViewCancel(SNES snes)
{
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  for (i=0; i<snes->numberreasonviews; i++) {
    if (snes->reasonviewdestroy[i]) {
      PetscCall((*snes->reasonviewdestroy[i])(&snes->reasonviewcontext[i]));
    }
  }
  snes->numberreasonviews = 0;
  PetscFunctionReturn(0);
}

/*@C
   SNESDestroy - Destroys the nonlinear solver context that was created
   with SNESCreate().

   Collective on SNES

   Input Parameter:
.  snes - the SNES context

   Level: beginner

.seealso: SNESCreate(), SNESSolve()
@*/
PetscErrorCode  SNESDestroy(SNES *snes)
{
  PetscFunctionBegin;
  if (!*snes) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*snes),SNES_CLASSID,1);
  if (--((PetscObject)(*snes))->refct > 0) {*snes = NULL; PetscFunctionReturn(0);}

  PetscCall(SNESReset((*snes)));
  PetscCall(SNESDestroy(&(*snes)->npc));

  /* if memory was published with SAWs then destroy it */
  PetscCall(PetscObjectSAWsViewOff((PetscObject)*snes));
  if ((*snes)->ops->destroy) PetscCall((*((*snes))->ops->destroy)((*snes)));

  if ((*snes)->dm) PetscCall(DMCoarsenHookRemove((*snes)->dm,DMCoarsenHook_SNESVecSol,DMRestrictHook_SNESVecSol,*snes));
  PetscCall(DMDestroy(&(*snes)->dm));
  PetscCall(KSPDestroy(&(*snes)->ksp));
  PetscCall(SNESLineSearchDestroy(&(*snes)->linesearch));

  PetscCall(PetscFree((*snes)->kspconvctx));
  if ((*snes)->ops->convergeddestroy) {
    PetscCall((*(*snes)->ops->convergeddestroy)((*snes)->cnvP));
  }
  if ((*snes)->conv_hist_alloc) {
    PetscCall(PetscFree2((*snes)->conv_hist,(*snes)->conv_hist_its));
  }
  PetscCall(SNESMonitorCancel((*snes)));
  PetscCall(SNESConvergedReasonViewCancel((*snes)));
  PetscCall(PetscHeaderDestroy(snes));
  PetscFunctionReturn(0);
}

/* ----------- Routines to set solver parameters ---------- */

/*@
   SNESSetLagPreconditioner - Determines when the preconditioner is rebuilt in the nonlinear solve.

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
-  lag - 1 means rebuild every time the Jacobian is computed within a single nonlinear solve, 2 means every second time
         the Jacobian is built etc. -2 indicates rebuild preconditioner at next chance but then never rebuild after that

   Options Database Keys:
+    -snes_lag_jacobian_persists <true,false> - sets the persistence
.    -snes_lag_jacobian <-2,1,2,...> - sets the lag
.    -snes_lag_preconditioner_persists <true,false> - sets the persistence
-    -snes_lag_preconditioner <-2,1,2,...> - sets the lag

   Notes:
   The default is 1
   The preconditioner is ALWAYS built in the first iteration of a nonlinear solve unless lag is -1 or SNESSetLagPreconditionerPersists() was called

   SNESSetLagPreconditionerPersists() allows using the same uniform lagging (for example every second solve) across multiple solves.

   Level: intermediate

.seealso: SNESSetTrustRegionTolerance(), SNESGetLagPreconditioner(), SNESSetLagJacobian(), SNESGetLagJacobian(), SNESSetLagPreconditionerPersists(),
          SNESSetLagJacobianPersists()

@*/
PetscErrorCode  SNESSetLagPreconditioner(SNES snes,PetscInt lag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscCheck(lag >= -2,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Lag must be -2, -1, 1 or greater");
  PetscCheck(lag,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Lag cannot be 0");
  PetscValidLogicalCollectiveInt(snes,lag,2);
  snes->lagpreconditioner = lag;
  PetscFunctionReturn(0);
}

/*@
   SNESSetGridSequence - sets the number of steps of grid sequencing that SNES does

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
-  steps - the number of refinements to do, defaults to 0

   Options Database Keys:
.    -snes_grid_sequence <steps> - Use grid sequencing to generate initial guess

   Level: intermediate

   Notes:
   Use SNESGetSolution() to extract the fine grid solution after grid sequencing.

.seealso: SNESSetTrustRegionTolerance(), SNESGetLagPreconditioner(), SNESSetLagJacobian(), SNESGetLagJacobian(), SNESGetGridSequence()

@*/
PetscErrorCode  SNESSetGridSequence(SNES snes,PetscInt steps)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidLogicalCollectiveInt(snes,steps,2);
  snes->gridsequence = steps;
  PetscFunctionReturn(0);
}

/*@
   SNESGetGridSequence - gets the number of steps of grid sequencing that SNES does

   Logically Collective on SNES

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
.  steps - the number of refinements to do, defaults to 0

   Options Database Keys:
.    -snes_grid_sequence <steps> - set number of refinements

   Level: intermediate

   Notes:
   Use SNESGetSolution() to extract the fine grid solution after grid sequencing.

.seealso: SNESSetTrustRegionTolerance(), SNESGetLagPreconditioner(), SNESSetLagJacobian(), SNESGetLagJacobian(), SNESSetGridSequence()

@*/
PetscErrorCode  SNESGetGridSequence(SNES snes,PetscInt *steps)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  *steps = snes->gridsequence;
  PetscFunctionReturn(0);
}

/*@
   SNESGetLagPreconditioner - Indicates how often the preconditioner is rebuilt

   Not Collective

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
.   lag - -1 indicates NEVER rebuild, 1 means rebuild every time the Jacobian is computed within a single nonlinear solve, 2 means every second time
         the Jacobian is built etc. -2 indicates rebuild preconditioner at next chance but then never rebuild after that

   Options Database Keys:
+    -snes_lag_jacobian_persists <true,false> - sets the persistence
.    -snes_lag_jacobian <-2,1,2,...> - sets the lag
.    -snes_lag_preconditioner_persists <true,false> - sets the persistence
-    -snes_lag_preconditioner <-2,1,2,...> - sets the lag

   Notes:
   The default is 1
   The preconditioner is ALWAYS built in the first iteration of a nonlinear solve unless lag is -1

   Level: intermediate

.seealso: SNESSetTrustRegionTolerance(), SNESSetLagPreconditioner(), SNESSetLagJacobianPersists(), SNESSetLagPreconditionerPersists()

@*/
PetscErrorCode  SNESGetLagPreconditioner(SNES snes,PetscInt *lag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  *lag = snes->lagpreconditioner;
  PetscFunctionReturn(0);
}

/*@
   SNESSetLagJacobian - Determines when the Jacobian is rebuilt in the nonlinear solve. See SNESSetLagPreconditioner() for determining how
     often the preconditioner is rebuilt.

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
-  lag - -1 indicates NEVER rebuild, 1 means rebuild every time the Jacobian is computed within a single nonlinear solve, 2 means every second time
         the Jacobian is built etc. -2 means rebuild at next chance but then never again

   Options Database Keys:
+    -snes_lag_jacobian_persists <true,false> - sets the persistence
.    -snes_lag_jacobian <-2,1,2,...> - sets the lag
.    -snes_lag_preconditioner_persists <true,false> - sets the persistence
-    -snes_lag_preconditioner <-2,1,2,...> - sets the lag.

   Notes:
   The default is 1
   The Jacobian is ALWAYS built in the first iteration of a nonlinear solve unless lag is -1
   If  -1 is used before the very first nonlinear solve the CODE WILL FAIL! because no Jacobian is used, use -2 to indicate you want it recomputed
   at the next Newton step but never again (unless it is reset to another value)

   Level: intermediate

.seealso: SNESSetTrustRegionTolerance(), SNESGetLagPreconditioner(), SNESSetLagPreconditioner(), SNESGetLagJacobianPersists(), SNESSetLagPreconditionerPersists()

@*/
PetscErrorCode  SNESSetLagJacobian(SNES snes,PetscInt lag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscCheck(lag >= -2,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Lag must be -2, -1, 1 or greater");
  PetscCheck(lag,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Lag cannot be 0");
  PetscValidLogicalCollectiveInt(snes,lag,2);
  snes->lagjacobian = lag;
  PetscFunctionReturn(0);
}

/*@
   SNESGetLagJacobian - Indicates how often the Jacobian is rebuilt. See SNESGetLagPreconditioner() to determine when the preconditioner is rebuilt

   Not Collective

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
.   lag - -1 indicates NEVER rebuild, 1 means rebuild every time the Jacobian is computed within a single nonlinear solve, 2 means every second time
         the Jacobian is built etc.

   Notes:
   The default is 1
   The jacobian is ALWAYS built in the first iteration of a nonlinear solve unless lag is -1 or SNESSetLagJacobianPersists() was called.

   Level: intermediate

.seealso: SNESSetTrustRegionTolerance(), SNESSetLagJacobian(), SNESSetLagPreconditioner(), SNESGetLagPreconditioner(), SNESSetLagJacobianPersists(), SNESSetLagPreconditionerPersists()

@*/
PetscErrorCode  SNESGetLagJacobian(SNES snes,PetscInt *lag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  *lag = snes->lagjacobian;
  PetscFunctionReturn(0);
}

/*@
   SNESSetLagJacobianPersists - Set whether or not the Jacobian lagging persists through multiple solves

   Logically collective on SNES

   Input Parameters:
+  snes - the SNES context
-   flg - jacobian lagging persists if true

   Options Database Keys:
+    -snes_lag_jacobian_persists <true,false> - sets the persistence
.    -snes_lag_jacobian <-2,1,2,...> - sets the lag
.    -snes_lag_preconditioner_persists <true,false> - sets the persistence
-    -snes_lag_preconditioner <-2,1,2,...> - sets the lag

   Notes:
    This is useful both for nonlinear preconditioning, where it's appropriate to have the Jacobian be stale by
   several solves, and for implicit time-stepping, where Jacobian lagging in the inner nonlinear solve over several
   timesteps may present huge efficiency gains.

   Level: developer

.seealso: SNESSetLagPreconditionerPersists(), SNESSetLagJacobian(), SNESGetLagJacobian(), SNESGetNPC(), SNESSetLagJacobianPersists()

@*/
PetscErrorCode  SNESSetLagJacobianPersists(SNES snes,PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidLogicalCollectiveBool(snes,flg,2);
  snes->lagjac_persist = flg;
  PetscFunctionReturn(0);
}

/*@
   SNESSetLagPreconditionerPersists - Set whether or not the preconditioner lagging persists through multiple nonlinear solves

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
-   flg - preconditioner lagging persists if true

   Options Database Keys:
+    -snes_lag_jacobian_persists <true,false> - sets the persistence
.    -snes_lag_jacobian <-2,1,2,...> - sets the lag
.    -snes_lag_preconditioner_persists <true,false> - sets the persistence
-    -snes_lag_preconditioner <-2,1,2,...> - sets the lag

   Notes:
    This is useful both for nonlinear preconditioning, where it's appropriate to have the preconditioner be stale
   by several solves, and for implicit time-stepping, where preconditioner lagging in the inner nonlinear solve over
   several timesteps may present huge efficiency gains.

   Level: developer

.seealso: SNESSetLagJacobianPersists(), SNESSetLagJacobian(), SNESGetLagJacobian(), SNESGetNPC(), SNESSetLagPreconditioner()

@*/
PetscErrorCode  SNESSetLagPreconditionerPersists(SNES snes,PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidLogicalCollectiveBool(snes,flg,2);
  snes->lagpre_persist = flg;
  PetscFunctionReturn(0);
}

/*@
   SNESSetForceIteration - force SNESSolve() to take at least one iteration regardless of the initial residual norm

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
-  force - PETSC_TRUE require at least one iteration

   Options Database Keys:
.    -snes_force_iteration <force> - Sets forcing an iteration

   Notes:
   This is used sometimes with TS to prevent TS from detecting a false steady state solution

   Level: intermediate

.seealso: SNESSetTrustRegionTolerance(), SNESSetDivergenceTolerance()
@*/
PetscErrorCode  SNESSetForceIteration(SNES snes,PetscBool force)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  snes->forceiteration = force;
  PetscFunctionReturn(0);
}

/*@
   SNESGetForceIteration - Whether or not to force SNESSolve() take at least one iteration regardless of the initial residual norm

   Logically Collective on SNES

   Input Parameters:
.  snes - the SNES context

   Output Parameter:
.  force - PETSC_TRUE requires at least one iteration.

   Level: intermediate

.seealso: SNESSetForceIteration(), SNESSetTrustRegionTolerance(), SNESSetDivergenceTolerance()
@*/
PetscErrorCode  SNESGetForceIteration(SNES snes,PetscBool *force)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  *force = snes->forceiteration;
  PetscFunctionReturn(0);
}

/*@
   SNESSetTolerances - Sets various parameters used in convergence tests.

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  abstol - absolute convergence tolerance
.  rtol - relative convergence tolerance
.  stol -  convergence tolerance in terms of the norm of the change in the solution between steps,  || delta x || < stol*|| x ||
.  maxit - maximum number of iterations
-  maxf - maximum number of function evaluations (-1 indicates no limit)

   Options Database Keys:
+    -snes_atol <abstol> - Sets abstol
.    -snes_rtol <rtol> - Sets rtol
.    -snes_stol <stol> - Sets stol
.    -snes_max_it <maxit> - Sets maxit
-    -snes_max_funcs <maxf> - Sets maxf

   Notes:
   The default maximum number of iterations is 50.
   The default maximum number of function evaluations is 1000.

   Level: intermediate

.seealso: SNESSetTrustRegionTolerance(), SNESSetDivergenceTolerance(), SNESSetForceIteration()
@*/
PetscErrorCode  SNESSetTolerances(SNES snes,PetscReal abstol,PetscReal rtol,PetscReal stol,PetscInt maxit,PetscInt maxf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidLogicalCollectiveReal(snes,abstol,2);
  PetscValidLogicalCollectiveReal(snes,rtol,3);
  PetscValidLogicalCollectiveReal(snes,stol,4);
  PetscValidLogicalCollectiveInt(snes,maxit,5);
  PetscValidLogicalCollectiveInt(snes,maxf,6);

  if (abstol != PETSC_DEFAULT) {
    PetscCheck(abstol >= 0.0,PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_OUTOFRANGE,"Absolute tolerance %g must be non-negative",(double)abstol);
    snes->abstol = abstol;
  }
  if (rtol != PETSC_DEFAULT) {
    PetscCheck(rtol >= 0.0 && 1.0 > rtol,PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_OUTOFRANGE,"Relative tolerance %g must be non-negative and less than 1.0",(double)rtol);
    snes->rtol = rtol;
  }
  if (stol != PETSC_DEFAULT) {
    PetscCheck(stol >= 0.0,PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_OUTOFRANGE,"Step tolerance %g must be non-negative",(double)stol);
    snes->stol = stol;
  }
  if (maxit != PETSC_DEFAULT) {
    PetscCheck(maxit >= 0,PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_OUTOFRANGE,"Maximum number of iterations %D must be non-negative",maxit);
    snes->max_its = maxit;
  }
  if (maxf != PETSC_DEFAULT) {
    PetscCheck(maxf >= -1,PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_OUTOFRANGE,"Maximum number of function evaluations %D must be -1 or nonnegative",maxf);
    snes->max_funcs = maxf;
  }
  snes->tolerancesset = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
   SNESSetDivergenceTolerance - Sets the divergence tolerance used for the SNES divergence test.

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
-  divtol - the divergence tolerance. Use -1 to deactivate the test.

   Options Database Keys:
.    -snes_divergence_tolerance <divtol> - Sets divtol

   Notes:
   The default divergence tolerance is 1e4.

   Level: intermediate

.seealso: SNESSetTolerances(), SNESGetDivergenceTolerance
@*/
PetscErrorCode  SNESSetDivergenceTolerance(SNES snes,PetscReal divtol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidLogicalCollectiveReal(snes,divtol,2);

  if (divtol != PETSC_DEFAULT) {
    snes->divtol = divtol;
  }
  else {
    snes->divtol = 1.0e4;
  }
  PetscFunctionReturn(0);
}

/*@
   SNESGetTolerances - Gets various parameters used in convergence tests.

   Not Collective

   Input Parameters:
+  snes - the SNES context
.  atol - absolute convergence tolerance
.  rtol - relative convergence tolerance
.  stol -  convergence tolerance in terms of the norm
           of the change in the solution between steps
.  maxit - maximum number of iterations
-  maxf - maximum number of function evaluations

   Notes:
   The user can specify NULL for any parameter that is not needed.

   Level: intermediate

.seealso: SNESSetTolerances()
@*/
PetscErrorCode  SNESGetTolerances(SNES snes,PetscReal *atol,PetscReal *rtol,PetscReal *stol,PetscInt *maxit,PetscInt *maxf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (atol)  *atol  = snes->abstol;
  if (rtol)  *rtol  = snes->rtol;
  if (stol)  *stol  = snes->stol;
  if (maxit) *maxit = snes->max_its;
  if (maxf)  *maxf  = snes->max_funcs;
  PetscFunctionReturn(0);
}

/*@
   SNESGetDivergenceTolerance - Gets divergence tolerance used in divergence test.

   Not Collective

   Input Parameters:
+  snes - the SNES context
-  divtol - divergence tolerance

   Level: intermediate

.seealso: SNESSetDivergenceTolerance()
@*/
PetscErrorCode  SNESGetDivergenceTolerance(SNES snes,PetscReal *divtol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (divtol) *divtol = snes->divtol;
  PetscFunctionReturn(0);
}

/*@
   SNESSetTrustRegionTolerance - Sets the trust region parameter tolerance.

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
-  tol - tolerance

   Options Database Key:
.  -snes_trtol <tol> - Sets tol

   Level: intermediate

.seealso: SNESSetTolerances()
@*/
PetscErrorCode  SNESSetTrustRegionTolerance(SNES snes,PetscReal tol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidLogicalCollectiveReal(snes,tol,2);
  snes->deltatol = tol;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode  SNESMonitorRange_Private(SNES,PetscInt,PetscReal*);

PetscErrorCode  SNESMonitorLGRange(SNES snes,PetscInt n,PetscReal rnorm,void *monctx)
{
  PetscDrawLG      lg;
  PetscReal        x,y,per;
  PetscViewer      v = (PetscViewer)monctx;
  static PetscReal prev; /* should be in the context */
  PetscDraw        draw;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,PETSC_VIEWER_CLASSID,4);
  PetscCall(PetscViewerDrawGetDrawLG(v,0,&lg));
  if (!n) PetscCall(PetscDrawLGReset(lg));
  PetscCall(PetscDrawLGGetDraw(lg,&draw));
  PetscCall(PetscDrawSetTitle(draw,"Residual norm"));
  x    = (PetscReal)n;
  if (rnorm > 0.0) y = PetscLog10Real(rnorm);
  else y = -15.0;
  PetscCall(PetscDrawLGAddPoint(lg,&x,&y));
  if (n < 20 || !(n % 5) || snes->reason) {
    PetscCall(PetscDrawLGDraw(lg));
    PetscCall(PetscDrawLGSave(lg));
  }

  PetscCall(PetscViewerDrawGetDrawLG(v,1,&lg));
  if (!n) PetscCall(PetscDrawLGReset(lg));
  PetscCall(PetscDrawLGGetDraw(lg,&draw));
  PetscCall(PetscDrawSetTitle(draw,"% elemts > .2*max elemt"));
  PetscCall(SNESMonitorRange_Private(snes,n,&per));
  x    = (PetscReal)n;
  y    = 100.0*per;
  PetscCall(PetscDrawLGAddPoint(lg,&x,&y));
  if (n < 20 || !(n % 5) || snes->reason) {
    PetscCall(PetscDrawLGDraw(lg));
    PetscCall(PetscDrawLGSave(lg));
  }

  PetscCall(PetscViewerDrawGetDrawLG(v,2,&lg));
  if (!n) {prev = rnorm;PetscCall(PetscDrawLGReset(lg));}
  PetscCall(PetscDrawLGGetDraw(lg,&draw));
  PetscCall(PetscDrawSetTitle(draw,"(norm -oldnorm)/oldnorm"));
  x    = (PetscReal)n;
  y    = (prev - rnorm)/prev;
  PetscCall(PetscDrawLGAddPoint(lg,&x,&y));
  if (n < 20 || !(n % 5) || snes->reason) {
    PetscCall(PetscDrawLGDraw(lg));
    PetscCall(PetscDrawLGSave(lg));
  }

  PetscCall(PetscViewerDrawGetDrawLG(v,3,&lg));
  if (!n) PetscCall(PetscDrawLGReset(lg));
  PetscCall(PetscDrawLGGetDraw(lg,&draw));
  PetscCall(PetscDrawSetTitle(draw,"(norm -oldnorm)/oldnorm*(% > .2 max)"));
  x    = (PetscReal)n;
  y    = (prev - rnorm)/(prev*per);
  if (n > 2) { /*skip initial crazy value */
    PetscCall(PetscDrawLGAddPoint(lg,&x,&y));
  }
  if (n < 20 || !(n % 5) || snes->reason) {
    PetscCall(PetscDrawLGDraw(lg));
    PetscCall(PetscDrawLGSave(lg));
  }
  prev = rnorm;
  PetscFunctionReturn(0);
}

/*@
   SNESMonitor - runs the user provided monitor routines, if they exist

   Collective on SNES

   Input Parameters:
+  snes - nonlinear solver context obtained from SNESCreate()
.  iter - iteration number
-  rnorm - relative norm of the residual

   Notes:
   This routine is called by the SNES implementations.
   It does not typically need to be called by the user.

   Level: developer

.seealso: SNESMonitorSet()
@*/
PetscErrorCode  SNESMonitor(SNES snes,PetscInt iter,PetscReal rnorm)
{
  PetscInt       i,n = snes->numbermonitors;

  PetscFunctionBegin;
  PetscCall(VecLockReadPush(snes->vec_sol));
  for (i=0; i<n; i++) {
    PetscCall((*snes->monitor[i])(snes,iter,rnorm,snes->monitorcontext[i]));
  }
  PetscCall(VecLockReadPop(snes->vec_sol));
  PetscFunctionReturn(0);
}

/* ------------ Routines to set performance monitoring options ----------- */

/*MC
    SNESMonitorFunction - functional form passed to SNESMonitorSet() to monitor convergence of nonlinear solver

     Synopsis:
     #include <petscsnes.h>
$    PetscErrorCode SNESMonitorFunction(SNES snes,PetscInt its, PetscReal norm,void *mctx)

     Collective on snes

    Input Parameters:
+    snes - the SNES context
.    its - iteration number
.    norm - 2-norm function value (may be estimated)
-    mctx - [optional] monitoring context

   Level: advanced

.seealso:   SNESMonitorSet(), SNESMonitorGet()
M*/

/*@C
   SNESMonitorSet - Sets an ADDITIONAL function that is to be used at every
   iteration of the nonlinear solver to display the iteration's
   progress.

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  f - the monitor function, see SNESMonitorFunction for the calling sequence
.  mctx - [optional] user-defined context for private data for the
          monitor routine (use NULL if no context is desired)
-  monitordestroy - [optional] routine that frees monitor context
          (may be NULL)

   Options Database Keys:
+    -snes_monitor        - sets SNESMonitorDefault()
.    -snes_monitor draw::draw_lg - sets line graph monitor,
-    -snes_monitor_cancel - cancels all monitors that have
                            been hardwired into a code by
                            calls to SNESMonitorSet(), but
                            does not cancel those set via
                            the options database.

   Notes:
   Several different monitoring routines may be set by calling
   SNESMonitorSet() multiple times; all will be called in the
   order in which they were set.

   Fortran Notes:
    Only a single monitor function can be set for each SNES object

   Level: intermediate

.seealso: SNESMonitorDefault(), SNESMonitorCancel(), SNESMonitorFunction
@*/
PetscErrorCode  SNESMonitorSet(SNES snes,PetscErrorCode (*f)(SNES,PetscInt,PetscReal,void*),void *mctx,PetscErrorCode (*monitordestroy)(void**))
{
  PetscInt       i;
  PetscBool      identical;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  for (i=0; i<snes->numbermonitors;i++) {
    PetscCall(PetscMonitorCompare((PetscErrorCode (*)(void))f,mctx,monitordestroy,(PetscErrorCode (*)(void))snes->monitor[i],snes->monitorcontext[i],snes->monitordestroy[i],&identical));
    if (identical) PetscFunctionReturn(0);
  }
  PetscCheck(snes->numbermonitors < MAXSNESMONITORS,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many monitors set");
  snes->monitor[snes->numbermonitors]          = f;
  snes->monitordestroy[snes->numbermonitors]   = monitordestroy;
  snes->monitorcontext[snes->numbermonitors++] = (void*)mctx;
  PetscFunctionReturn(0);
}

/*@
   SNESMonitorCancel - Clears all the monitor functions for a SNES object.

   Logically Collective on SNES

   Input Parameters:
.  snes - the SNES context

   Options Database Key:
.  -snes_monitor_cancel - cancels all monitors that have been hardwired
    into a code by calls to SNESMonitorSet(), but does not cancel those
    set via the options database

   Notes:
   There is no way to clear one specific monitor from a SNES object.

   Level: intermediate

.seealso: SNESMonitorDefault(), SNESMonitorSet()
@*/
PetscErrorCode  SNESMonitorCancel(SNES snes)
{
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  for (i=0; i<snes->numbermonitors; i++) {
    if (snes->monitordestroy[i]) {
      PetscCall((*snes->monitordestroy[i])(&snes->monitorcontext[i]));
    }
  }
  snes->numbermonitors = 0;
  PetscFunctionReturn(0);
}

/*MC
    SNESConvergenceTestFunction - functional form used for testing of convergence of nonlinear solver

     Synopsis:
     #include <petscsnes.h>
$     PetscErrorCode SNESConvergenceTest(SNES snes,PetscInt it,PetscReal xnorm,PetscReal gnorm,PetscReal f,SNESConvergedReason *reason,void *cctx)

     Collective on snes

    Input Parameters:
+    snes - the SNES context
.    it - current iteration (0 is the first and is before any Newton step)
.    xnorm - 2-norm of current iterate
.    gnorm - 2-norm of current step
.    f - 2-norm of function
-    cctx - [optional] convergence context

    Output Parameter:
.    reason - reason for convergence/divergence, only needs to be set when convergence or divergence is detected

   Level: intermediate

.seealso:   SNESSetConvergenceTest(), SNESGetConvergenceTest()
M*/

/*@C
   SNESSetConvergenceTest - Sets the function that is to be used
   to test for convergence of the nonlinear iterative solution.

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  SNESConvergenceTestFunction - routine to test for convergence
.  cctx - [optional] context for private data for the convergence routine  (may be NULL)
-  destroy - [optional] destructor for the context (may be NULL; PETSC_NULL_FUNCTION in Fortran)

   Level: advanced

.seealso: SNESConvergedDefault(), SNESConvergedSkip(), SNESConvergenceTestFunction
@*/
PetscErrorCode  SNESSetConvergenceTest(SNES snes,PetscErrorCode (*SNESConvergenceTestFunction)(SNES,PetscInt,PetscReal,PetscReal,PetscReal,SNESConvergedReason*,void*),void *cctx,PetscErrorCode (*destroy)(void*))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (!SNESConvergenceTestFunction) SNESConvergenceTestFunction = SNESConvergedSkip;
  if (snes->ops->convergeddestroy) {
    PetscCall((*snes->ops->convergeddestroy)(snes->cnvP));
  }
  snes->ops->converged        = SNESConvergenceTestFunction;
  snes->ops->convergeddestroy = destroy;
  snes->cnvP                  = cctx;
  PetscFunctionReturn(0);
}

/*@
   SNESGetConvergedReason - Gets the reason the SNES iteration was stopped.

   Not Collective

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
.  reason - negative value indicates diverged, positive value converged, see SNESConvergedReason or the
            manual pages for the individual convergence tests for complete lists

   Options Database:
.   -snes_converged_reason - prints the reason to standard out

   Level: intermediate

   Notes:
    Should only be called after the call the SNESSolve() is complete, if it is called earlier it returns the value SNES__CONVERGED_ITERATING.

.seealso: SNESSetConvergenceTest(), SNESSetConvergedReason(), SNESConvergedReason
@*/
PetscErrorCode SNESGetConvergedReason(SNES snes,SNESConvergedReason *reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidPointer(reason,2);
  *reason = snes->reason;
  PetscFunctionReturn(0);
}

/*@C
   SNESGetConvergedReasonString - Return a human readable string for snes converged reason

   Not Collective

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
.  strreason - a human readable string that describes SNES converged reason

   Level: beginner

.seealso: SNESGetConvergedReason()
@*/
PetscErrorCode SNESGetConvergedReasonString(SNES snes, const char** strreason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidPointer(strreason,2);
  *strreason = SNESConvergedReasons[snes->reason];
  PetscFunctionReturn(0);
}

/*@
   SNESSetConvergedReason - Sets the reason the SNES iteration was stopped.

   Not Collective

   Input Parameters:
+  snes - the SNES context
-  reason - negative value indicates diverged, positive value converged, see SNESConvergedReason or the
            manual pages for the individual convergence tests for complete lists

   Level: intermediate

.seealso: SNESGetConvergedReason(), SNESSetConvergenceTest(), SNESConvergedReason
@*/
PetscErrorCode SNESSetConvergedReason(SNES snes,SNESConvergedReason reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  snes->reason = reason;
  PetscFunctionReturn(0);
}

/*@
   SNESSetConvergenceHistory - Sets the array used to hold the convergence history.

   Logically Collective on SNES

   Input Parameters:
+  snes - iterative context obtained from SNESCreate()
.  a   - array to hold history, this array will contain the function norms computed at each step
.  its - integer array holds the number of linear iterations for each solve.
.  na  - size of a and its
-  reset - PETSC_TRUE indicates each new nonlinear solve resets the history counter to zero,
           else it continues storing new values for new nonlinear solves after the old ones

   Notes:
   If 'a' and 'its' are NULL then space is allocated for the history. If 'na' PETSC_DECIDE or PETSC_DEFAULT then a
   default array of length 10000 is allocated.

   This routine is useful, e.g., when running a code for purposes
   of accurate performance monitoring, when no I/O should be done
   during the section of code that is being timed.

   Level: intermediate

.seealso: SNESGetConvergenceHistory()

@*/
PetscErrorCode  SNESSetConvergenceHistory(SNES snes,PetscReal a[],PetscInt its[],PetscInt na,PetscBool reset)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (a) PetscValidRealPointer(a,2);
  if (its) PetscValidIntPointer(its,3);
  if (!a) {
    if (na == PETSC_DECIDE || na == PETSC_DEFAULT) na = 1000;
    PetscCall(PetscCalloc2(na,&a,na,&its));
    snes->conv_hist_alloc = PETSC_TRUE;
  }
  snes->conv_hist       = a;
  snes->conv_hist_its   = its;
  snes->conv_hist_max   = (size_t)na;
  snes->conv_hist_len   = 0;
  snes->conv_hist_reset = reset;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MATLAB_ENGINE)
#include <engine.h>   /* MATLAB include file */
#include <mex.h>      /* MATLAB include file */

PETSC_EXTERN mxArray *SNESGetConvergenceHistoryMatlab(SNES snes)
{
  mxArray   *mat;
  PetscInt  i;
  PetscReal *ar;

  PetscFunctionBegin;
  mat = mxCreateDoubleMatrix(snes->conv_hist_len,1,mxREAL);
  ar  = (PetscReal*) mxGetData(mat);
  for (i=0; i<snes->conv_hist_len; i++) ar[i] = snes->conv_hist[i];
  PetscFunctionReturn(mat);
}
#endif

/*@C
   SNESGetConvergenceHistory - Gets the array used to hold the convergence history.

   Not Collective

   Input Parameter:
.  snes - iterative context obtained from SNESCreate()

   Output Parameters:
+  a   - array to hold history
.  its - integer array holds the number of linear iterations (or
         negative if not converged) for each solve.
-  na  - size of a and its

   Notes:
    The calling sequence for this routine in Fortran is
$   call SNESGetConvergenceHistory(SNES snes, integer na, integer ierr)

   This routine is useful, e.g., when running a code for purposes
   of accurate performance monitoring, when no I/O should be done
   during the section of code that is being timed.

   Level: intermediate

.seealso: SNESSetConvergenceHistory()

@*/
PetscErrorCode  SNESGetConvergenceHistory(SNES snes,PetscReal *a[],PetscInt *its[],PetscInt *na)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (a)   *a   = snes->conv_hist;
  if (its) *its = snes->conv_hist_its;
  if (na)  *na  = (PetscInt) snes->conv_hist_len;
  PetscFunctionReturn(0);
}

/*@C
  SNESSetUpdate - Sets the general-purpose update function called
  at the beginning of every iteration of the nonlinear solve. Specifically
  it is called just before the Jacobian is "evaluated".

  Logically Collective on SNES

  Input Parameters:
+ snes - The nonlinear solver context
- func - The function

  Calling sequence of func:
$ func (SNES snes, PetscInt step);

. step - The current step of the iteration

  Level: advanced

  Note:
     This is NOT what one uses to update the ghost points before a function evaluation, that should be done at the beginning of your FormFunction()
     This is not used by most users.

     There are a varity of function hooks one many set that are called at different stages of the nonlinear solution process, see the functions listed below.

.seealso SNESSetJacobian(), SNESSolve(), SNESLineSearchSetPreCheck(), SNESLineSearchSetPostCheck(), SNESNewtonTRSetPreCheck(), SNESNewtonTRSetPostCheck(),
         SNESMonitorSet(), SNESSetDivergenceTest()
@*/
PetscErrorCode  SNESSetUpdate(SNES snes, PetscErrorCode (*func)(SNES, PetscInt))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID,1);
  snes->ops->update = func;
  PetscFunctionReturn(0);
}

/*
   SNESScaleStep_Private - Scales a step so that its length is less than the
   positive parameter delta.

    Input Parameters:
+   snes - the SNES context
.   y - approximate solution of linear system
.   fnorm - 2-norm of current function
-   delta - trust region size

    Output Parameters:
+   gpnorm - predicted function norm at the new point, assuming local
    linearization.  The value is zero if the step lies within the trust
    region, and exceeds zero otherwise.
-   ynorm - 2-norm of the step

    Note:
    For non-trust region methods such as SNESNEWTONLS, the parameter delta
    is set to be the maximum allowable step size.

*/
PetscErrorCode SNESScaleStep_Private(SNES snes,Vec y,PetscReal *fnorm,PetscReal *delta,PetscReal *gpnorm,PetscReal *ynorm)
{
  PetscReal      nrm;
  PetscScalar    cnorm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidHeaderSpecific(y,VEC_CLASSID,2);
  PetscCheckSameComm(snes,1,y,2);

  PetscCall(VecNorm(y,NORM_2,&nrm));
  if (nrm > *delta) {
    nrm     = *delta/nrm;
    *gpnorm = (1.0 - nrm)*(*fnorm);
    cnorm   = nrm;
    PetscCall(VecScale(y,cnorm));
    *ynorm  = *delta;
  } else {
    *gpnorm = 0.0;
    *ynorm  = nrm;
  }
  PetscFunctionReturn(0);
}

/*@C
   SNESConvergedReasonView - Displays the reason a SNES solve converged or diverged to a viewer

   Collective on SNES

   Parameter:
+  snes - iterative context obtained from SNESCreate()
-  viewer - the viewer to display the reason

   Options Database Keys:
+  -snes_converged_reason - print reason for converged or diverged, also prints number of iterations
-  -snes_converged_reason ::failed - only print reason and number of iterations when diverged

  Notes:
     To change the format of the output call PetscViewerPushFormat(viewer,format) before this call. Use PETSC_VIEWER_DEFAULT for the default,
     use PETSC_VIEWER_FAILED to only display a reason if it fails.

   Level: beginner

.seealso: SNESCreate(), SNESSetUp(), SNESDestroy(), SNESSetTolerances(), SNESConvergedDefault(), SNESGetConvergedReason(), SNESConvergedReasonViewFromOptions(),
          PetscViewerPushFormat(), PetscViewerPopFormat()

@*/
PetscErrorCode  SNESConvergedReasonView(SNES snes,PetscViewer viewer)
{
  PetscViewerFormat format;
  PetscBool         isAscii;

  PetscFunctionBegin;
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)snes));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isAscii));
  if (isAscii) {
    PetscCall(PetscViewerGetFormat(viewer, &format));
    PetscCall(PetscViewerASCIIAddTab(viewer,((PetscObject)snes)->tablevel));
    if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      DM              dm;
      Vec             u;
      PetscDS         prob;
      PetscInt        Nf, f;
      PetscErrorCode (**exactSol)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void *);
      void            **exactCtx;
      PetscReal       error;

      PetscCall(SNESGetDM(snes, &dm));
      PetscCall(SNESGetSolution(snes, &u));
      PetscCall(DMGetDS(dm, &prob));
      PetscCall(PetscDSGetNumFields(prob, &Nf));
      PetscCall(PetscMalloc2(Nf, &exactSol, Nf, &exactCtx));
      for (f = 0; f < Nf; ++f) PetscCall(PetscDSGetExactSolution(prob, f, &exactSol[f], &exactCtx[f]));
      PetscCall(DMComputeL2Diff(dm, 0.0, exactSol, exactCtx, u, &error));
      PetscCall(PetscFree2(exactSol, exactCtx));
      if (error < 1.0e-11) PetscCall(PetscViewerASCIIPrintf(viewer, "L_2 Error: < 1.0e-11\n"));
      else                 PetscCall(PetscViewerASCIIPrintf(viewer, "L_2 Error: %g\n", error));
    }
    if (snes->reason > 0 && format != PETSC_VIEWER_FAILED) {
      if (((PetscObject) snes)->prefix) {
        PetscCall(PetscViewerASCIIPrintf(viewer,"Nonlinear %s solve converged due to %s iterations %D\n",((PetscObject) snes)->prefix,SNESConvergedReasons[snes->reason],snes->iter));
      } else {
        PetscCall(PetscViewerASCIIPrintf(viewer,"Nonlinear solve converged due to %s iterations %D\n",SNESConvergedReasons[snes->reason],snes->iter));
      }
    } else if (snes->reason <= 0) {
      if (((PetscObject) snes)->prefix) {
        PetscCall(PetscViewerASCIIPrintf(viewer,"Nonlinear %s solve did not converge due to %s iterations %D\n",((PetscObject) snes)->prefix,SNESConvergedReasons[snes->reason],snes->iter));
      } else {
        PetscCall(PetscViewerASCIIPrintf(viewer,"Nonlinear solve did not converge due to %s iterations %D\n",SNESConvergedReasons[snes->reason],snes->iter));
      }
    }
    PetscCall(PetscViewerASCIISubtractTab(viewer,((PetscObject)snes)->tablevel));
  }
  PetscFunctionReturn(0);
}

/*@C
   SNESConvergedReasonViewSet - Sets an ADDITIONAL function that is to be used at the
    end of the nonlinear solver to display the conver reason of the nonlinear solver.

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  f - the snes converged reason view function
.  vctx - [optional] user-defined context for private data for the
          snes converged reason view routine (use NULL if no context is desired)
-  reasonviewdestroy - [optional] routine that frees reasonview context
          (may be NULL)

   Options Database Keys:
+    -snes_converged_reason        - sets a default SNESConvergedReasonView()
-    -snes_converged_reason_view_cancel - cancels all converged reason viewers that have
                            been hardwired into a code by
                            calls to SNESConvergedReasonViewSet(), but
                            does not cancel those set via
                            the options database.

   Notes:
   Several different converged reason view routines may be set by calling
   SNESConvergedReasonViewSet() multiple times; all will be called in the
   order in which they were set.

   Level: intermediate

.seealso: SNESConvergedReasonView(), SNESConvergedReasonViewCancel()
@*/
PetscErrorCode  SNESConvergedReasonViewSet(SNES snes,PetscErrorCode (*f)(SNES,void*),void *vctx,PetscErrorCode (*reasonviewdestroy)(void**))
{
  PetscInt       i;
  PetscBool      identical;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  for (i=0; i<snes->numberreasonviews;i++) {
    PetscCall(PetscMonitorCompare((PetscErrorCode (*)(void))f,vctx,reasonviewdestroy,(PetscErrorCode (*)(void))snes->reasonview[i],snes->reasonviewcontext[i],snes->reasonviewdestroy[i],&identical));
    if (identical) PetscFunctionReturn(0);
  }
  PetscCheck(snes->numberreasonviews < MAXSNESREASONVIEWS,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many SNES reasonview set");
  snes->reasonview[snes->numberreasonviews]          = f;
  snes->reasonviewdestroy[snes->numberreasonviews]   = reasonviewdestroy;
  snes->reasonviewcontext[snes->numberreasonviews++] = (void*)vctx;
  PetscFunctionReturn(0);
}

/*@
  SNESConvergedReasonViewFromOptions - Processes command line options to determine if/how a SNESReason is to be viewed.
                                       All the user-provided convergedReasonView routines will be involved as well, if they exist.

  Collective on SNES

  Input Parameters:
. snes   - the SNES object

  Level: intermediate

.seealso: SNESCreate(), SNESSetUp(), SNESDestroy(), SNESSetTolerances(), SNESConvergedDefault(), SNESGetConvergedReason(), SNESConvergedReasonView()

@*/
PetscErrorCode SNESConvergedReasonViewFromOptions(SNES snes)
{
  PetscViewer       viewer;
  PetscBool         flg;
  static PetscBool  incall = PETSC_FALSE;
  PetscViewerFormat format;
  PetscInt          i;

  PetscFunctionBegin;
  if (incall) PetscFunctionReturn(0);
  incall = PETSC_TRUE;

  /* All user-provided viewers are called first, if they exist. */
  for (i=0; i<snes->numberreasonviews; i++) {
    PetscCall((*snes->reasonview[i])(snes,snes->reasonviewcontext[i]));
  }

  /* Call PETSc default routine if users ask for it */
  PetscCall(PetscOptionsGetViewer(PetscObjectComm((PetscObject)snes),((PetscObject)snes)->options,((PetscObject)snes)->prefix,"-snes_converged_reason",&viewer,&format,&flg));
  if (flg) {
    PetscCall(PetscViewerPushFormat(viewer,format));
    PetscCall(SNESConvergedReasonView(snes,viewer));
    PetscCall(PetscViewerPopFormat(viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }
  incall = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@
   SNESSolve - Solves a nonlinear system F(x) = b.
   Call SNESSolve() after calling SNESCreate() and optional routines of the form SNESSetXXX().

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  b - the constant part of the equation F(x) = b, or NULL to use zero.
-  x - the solution vector.

   Notes:
   The user should initialize the vector,x, with the initial guess
   for the nonlinear solve prior to calling SNESSolve().  In particular,
   to employ an initial guess of zero, the user should explicitly set
   this vector to zero by calling VecSet().

   Level: beginner

.seealso: SNESCreate(), SNESDestroy(), SNESSetFunction(), SNESSetJacobian(), SNESSetGridSequence(), SNESGetSolution(),
          SNESNewtonTRSetPreCheck(), SNESNewtonTRGetPreCheck(), SNESNewtonTRSetPostCheck(), SNESNewtonTRGetPostCheck(),
          SNESLineSearchSetPostCheck(), SNESLineSearchGetPostCheck(), SNESLineSearchSetPreCheck(), SNESLineSearchGetPreCheck()
@*/
PetscErrorCode  SNESSolve(SNES snes,Vec b,Vec x)
{
  PetscBool         flg;
  PetscInt          grid;
  Vec               xcreated = NULL;
  DM                dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (x) PetscValidHeaderSpecific(x,VEC_CLASSID,3);
  if (x) PetscCheckSameComm(snes,1,x,3);
  if (b) PetscValidHeaderSpecific(b,VEC_CLASSID,2);
  if (b) PetscCheckSameComm(snes,1,b,2);

  /* High level operations using the nonlinear solver */
  {
    PetscViewer       viewer;
    PetscViewerFormat format;
    PetscInt          num;
    PetscBool         flg;
    static PetscBool  incall = PETSC_FALSE;

    if (!incall) {
      /* Estimate the convergence rate of the discretization */
      PetscCall(PetscOptionsGetViewer(PetscObjectComm((PetscObject) snes),((PetscObject)snes)->options, ((PetscObject) snes)->prefix, "-snes_convergence_estimate", &viewer, &format, &flg));
      if (flg) {
        PetscConvEst conv;
        DM           dm;
        PetscReal   *alpha; /* Convergence rate of the solution error for each field in the L_2 norm */
        PetscInt     Nf;

        incall = PETSC_TRUE;
        PetscCall(SNESGetDM(snes, &dm));
        PetscCall(DMGetNumFields(dm, &Nf));
        PetscCall(PetscCalloc1(Nf, &alpha));
        PetscCall(PetscConvEstCreate(PetscObjectComm((PetscObject) snes), &conv));
        PetscCall(PetscConvEstSetSolver(conv, (PetscObject) snes));
        PetscCall(PetscConvEstSetFromOptions(conv));
        PetscCall(PetscConvEstSetUp(conv));
        PetscCall(PetscConvEstGetConvRate(conv, alpha));
        PetscCall(PetscViewerPushFormat(viewer, format));
        PetscCall(PetscConvEstRateView(conv, alpha, viewer));
        PetscCall(PetscViewerPopFormat(viewer));
        PetscCall(PetscViewerDestroy(&viewer));
        PetscCall(PetscConvEstDestroy(&conv));
        PetscCall(PetscFree(alpha));
        incall = PETSC_FALSE;
      }
      /* Adaptively refine the initial grid */
      num  = 1;
      PetscCall(PetscOptionsGetInt(NULL, ((PetscObject) snes)->prefix, "-snes_adapt_initial", &num, &flg));
      if (flg) {
        DMAdaptor adaptor;

        incall = PETSC_TRUE;
        PetscCall(DMAdaptorCreate(PetscObjectComm((PetscObject)snes), &adaptor));
        PetscCall(DMAdaptorSetSolver(adaptor, snes));
        PetscCall(DMAdaptorSetSequenceLength(adaptor, num));
        PetscCall(DMAdaptorSetFromOptions(adaptor));
        PetscCall(DMAdaptorSetUp(adaptor));
        PetscCall(DMAdaptorAdapt(adaptor, x, DM_ADAPTATION_INITIAL, &dm, &x));
        PetscCall(DMAdaptorDestroy(&adaptor));
        incall = PETSC_FALSE;
      }
      /* Use grid sequencing to adapt */
      num  = 0;
      PetscCall(PetscOptionsGetInt(NULL, ((PetscObject) snes)->prefix, "-snes_adapt_sequence", &num, NULL));
      if (num) {
        DMAdaptor adaptor;

        incall = PETSC_TRUE;
        PetscCall(DMAdaptorCreate(PetscObjectComm((PetscObject)snes), &adaptor));
        PetscCall(DMAdaptorSetSolver(adaptor, snes));
        PetscCall(DMAdaptorSetSequenceLength(adaptor, num));
        PetscCall(DMAdaptorSetFromOptions(adaptor));
        PetscCall(DMAdaptorSetUp(adaptor));
        PetscCall(DMAdaptorAdapt(adaptor, x, DM_ADAPTATION_SEQUENTIAL, &dm, &x));
        PetscCall(DMAdaptorDestroy(&adaptor));
        incall = PETSC_FALSE;
      }
    }
  }
  if (!x) { x = snes->vec_sol; }
  if (!x) {
    PetscCall(SNESGetDM(snes,&dm));
    PetscCall(DMCreateGlobalVector(dm,&xcreated));
    x    = xcreated;
  }
  PetscCall(SNESViewFromOptions(snes,NULL,"-snes_view_pre"));

  for (grid=0; grid<snes->gridsequence; grid++) PetscCall(PetscViewerASCIIPushTab(PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)snes))));
  for (grid=0; grid<snes->gridsequence+1; grid++) {

    /* set solution vector */
    if (!grid) PetscCall(PetscObjectReference((PetscObject)x));
    PetscCall(VecDestroy(&snes->vec_sol));
    snes->vec_sol = x;
    PetscCall(SNESGetDM(snes,&dm));

    /* set affine vector if provided */
    if (b) PetscCall(PetscObjectReference((PetscObject)b));
    PetscCall(VecDestroy(&snes->vec_rhs));
    snes->vec_rhs = b;

    if (snes->vec_rhs) PetscCheck(snes->vec_func != snes->vec_rhs,PETSC_COMM_SELF,PETSC_ERR_ARG_IDN,"Right hand side vector cannot be function vector");
    PetscCheck(snes->vec_func != snes->vec_sol,PETSC_COMM_SELF,PETSC_ERR_ARG_IDN,"Solution vector cannot be function vector");
    PetscCheck(snes->vec_rhs  != snes->vec_sol,PETSC_COMM_SELF,PETSC_ERR_ARG_IDN,"Solution vector cannot be right hand side vector");
    if (!snes->vec_sol_update /* && snes->vec_sol */) {
      PetscCall(VecDuplicate(snes->vec_sol,&snes->vec_sol_update));
      PetscCall(PetscLogObjectParent((PetscObject)snes,(PetscObject)snes->vec_sol_update));
    }
    PetscCall(DMShellSetGlobalVector(dm,snes->vec_sol));
    PetscCall(SNESSetUp(snes));

    if (!grid) {
      if (snes->ops->computeinitialguess) {
        PetscCall((*snes->ops->computeinitialguess)(snes,snes->vec_sol,snes->initialguessP));
      }
    }

    if (snes->conv_hist_reset) snes->conv_hist_len = 0;
    if (snes->counters_reset) {snes->nfuncs = 0; snes->linear_its = 0; snes->numFailures = 0;}

    PetscCall(PetscLogEventBegin(SNES_Solve,snes,0,0,0));
    PetscCall((*snes->ops->solve)(snes));
    PetscCall(PetscLogEventEnd(SNES_Solve,snes,0,0,0));
    PetscCheck(snes->reason,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Internal error, solver returned without setting converged reason");
    snes->domainerror = PETSC_FALSE; /* clear the flag if it has been set */

    if (snes->lagjac_persist) snes->jac_iter += snes->iter;
    if (snes->lagpre_persist) snes->pre_iter += snes->iter;

    PetscCall(PetscOptionsGetViewer(PetscObjectComm((PetscObject)snes),((PetscObject)snes)->options,((PetscObject)snes)->prefix,"-snes_test_local_min",NULL,NULL,&flg));
    if (flg && !PetscPreLoadingOn) PetscCall(SNESTestLocalMin(snes));
    /* Call converged reason views. This may involve user-provided viewers as well */
    PetscCall(SNESConvergedReasonViewFromOptions(snes));

    if (snes->errorifnotconverged) PetscCheck(snes->reason >= 0,PetscObjectComm((PetscObject)snes),PETSC_ERR_NOT_CONVERGED,"SNESSolve has not converged");
    if (snes->reason < 0) break;
    if (grid <  snes->gridsequence) {
      DM  fine;
      Vec xnew;
      Mat interp;

      PetscCall(DMRefine(snes->dm,PetscObjectComm((PetscObject)snes),&fine));
      PetscCheck(fine,PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_INCOMP,"DMRefine() did not perform any refinement, cannot continue grid sequencing");
      PetscCall(DMCreateInterpolation(snes->dm,fine,&interp,NULL));
      PetscCall(DMCreateGlobalVector(fine,&xnew));
      PetscCall(MatInterpolate(interp,x,xnew));
      PetscCall(DMInterpolate(snes->dm,interp,fine));
      PetscCall(MatDestroy(&interp));
      x    = xnew;

      PetscCall(SNESReset(snes));
      PetscCall(SNESSetDM(snes,fine));
      PetscCall(SNESResetFromOptions(snes));
      PetscCall(DMDestroy(&fine));
      PetscCall(PetscViewerASCIIPopTab(PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)snes))));
    }
  }
  PetscCall(SNESViewFromOptions(snes,NULL,"-snes_view"));
  PetscCall(VecViewFromOptions(snes->vec_sol,(PetscObject)snes,"-snes_view_solution"));
  PetscCall(DMMonitor(snes->dm));
  PetscCall(SNESMonitorPauseFinal_Internal(snes));

  PetscCall(VecDestroy(&xcreated));
  PetscCall(PetscObjectSAWsBlock((PetscObject)snes));
  PetscFunctionReturn(0);
}

/* --------- Internal routines for SNES Package --------- */

/*@C
   SNESSetType - Sets the method for the nonlinear solver.

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
-  type - a known method

   Options Database Key:
.  -snes_type <type> - Sets the method; use -help for a list
   of available methods (for instance, newtonls or newtontr)

   Notes:
   See "petsc/include/petscsnes.h" for available methods (for instance)
+    SNESNEWTONLS - Newton's method with line search
     (systems of nonlinear equations)
-    SNESNEWTONTR - Newton's method with trust region
     (systems of nonlinear equations)

  Normally, it is best to use the SNESSetFromOptions() command and then
  set the SNES solver type from the options database rather than by using
  this routine.  Using the options database provides the user with
  maximum flexibility in evaluating the many nonlinear solvers.
  The SNESSetType() routine is provided for those situations where it
  is necessary to set the nonlinear solver independently of the command
  line or options database.  This might be the case, for example, when
  the choice of solver changes during the execution of the program,
  and the user's application is taking responsibility for choosing the
  appropriate method.

    Developer Notes:
    SNESRegister() adds a constructor for a new SNESType to SNESList, SNESSetType() locates
    the constructor in that list and calls it to create the spexific object.

  Level: intermediate

.seealso: SNESType, SNESCreate(), SNESDestroy(), SNESGetType(), SNESSetFromOptions()

@*/
PetscErrorCode  SNESSetType(SNES snes,SNESType type)
{
  PetscBool      match;
  PetscErrorCode (*r)(SNES);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidCharPointer(type,2);

  PetscCall(PetscObjectTypeCompare((PetscObject)snes,type,&match));
  if (match) PetscFunctionReturn(0);

  PetscCall(PetscFunctionListFind(SNESList,type,&r));
  PetscCheck(r,PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested SNES type %s",type);
  /* Destroy the previous private SNES context */
  if (snes->ops->destroy) {
    PetscCall((*(snes)->ops->destroy)(snes));
    snes->ops->destroy = NULL;
  }
  /* Reinitialize function pointers in SNESOps structure */
  snes->ops->setup          = NULL;
  snes->ops->solve          = NULL;
  snes->ops->view           = NULL;
  snes->ops->setfromoptions = NULL;
  snes->ops->destroy        = NULL;

  /* It may happen the user has customized the line search before calling SNESSetType */
  if (((PetscObject)snes)->type_name) PetscCall(SNESLineSearchDestroy(&snes->linesearch));

  /* Call the SNESCreate_XXX routine for this particular Nonlinear solver */
  snes->setupcalled = PETSC_FALSE;

  PetscCall(PetscObjectChangeTypeName((PetscObject)snes,type));
  PetscCall((*r)(snes));
  PetscFunctionReturn(0);
}

/*@C
   SNESGetType - Gets the SNES method type and name (as a string).

   Not Collective

   Input Parameter:
.  snes - nonlinear solver context

   Output Parameter:
.  type - SNES method (a character string)

   Level: intermediate

@*/
PetscErrorCode  SNESGetType(SNES snes,SNESType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)snes)->type_name;
  PetscFunctionReturn(0);
}

/*@
  SNESSetSolution - Sets the solution vector for use by the SNES routines.

  Logically Collective on SNES

  Input Parameters:
+ snes - the SNES context obtained from SNESCreate()
- u    - the solution vector

  Level: beginner

@*/
PetscErrorCode SNESSetSolution(SNES snes, Vec u)
{
  DM             dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidHeaderSpecific(u, VEC_CLASSID, 2);
  PetscCall(PetscObjectReference((PetscObject) u));
  PetscCall(VecDestroy(&snes->vec_sol));

  snes->vec_sol = u;

  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(DMShellSetGlobalVector(dm, u));
  PetscFunctionReturn(0);
}

/*@
   SNESGetSolution - Returns the vector where the approximate solution is
   stored. This is the fine grid solution when using SNESSetGridSequence().

   Not Collective, but Vec is parallel if SNES is parallel

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
.  x - the solution

   Level: intermediate

.seealso:  SNESGetSolutionUpdate(), SNESGetFunction()
@*/
PetscErrorCode  SNESGetSolution(SNES snes,Vec *x)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidPointer(x,2);
  *x = snes->vec_sol;
  PetscFunctionReturn(0);
}

/*@
   SNESGetSolutionUpdate - Returns the vector where the solution update is
   stored.

   Not Collective, but Vec is parallel if SNES is parallel

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
.  x - the solution update

   Level: advanced

.seealso: SNESGetSolution(), SNESGetFunction()
@*/
PetscErrorCode  SNESGetSolutionUpdate(SNES snes,Vec *x)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidPointer(x,2);
  *x = snes->vec_sol_update;
  PetscFunctionReturn(0);
}

/*@C
   SNESGetFunction - Returns the vector where the function is stored.

   Not Collective, but Vec is parallel if SNES is parallel. Collective if Vec is requested, but has not been created yet.

   Input Parameter:
.  snes - the SNES context

   Output Parameters:
+  r - the vector that is used to store residuals (or NULL if you don't want it)
.  f - the function (or NULL if you don't want it); see SNESFunction for calling sequence details
-  ctx - the function context (or NULL if you don't want it)

   Level: advanced

    Notes: The vector r DOES NOT, in general contain the current value of the SNES nonlinear function

.seealso: SNESSetFunction(), SNESGetSolution(), SNESFunction
@*/
PetscErrorCode  SNESGetFunction(SNES snes,Vec *r,PetscErrorCode (**f)(SNES,Vec,Vec,void*),void **ctx)
{
  DM             dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (r) {
    if (!snes->vec_func) {
      if (snes->vec_rhs) {
        PetscCall(VecDuplicate(snes->vec_rhs,&snes->vec_func));
      } else if (snes->vec_sol) {
        PetscCall(VecDuplicate(snes->vec_sol,&snes->vec_func));
      } else if (snes->dm) {
        PetscCall(DMCreateGlobalVector(snes->dm,&snes->vec_func));
      }
    }
    *r = snes->vec_func;
  }
  PetscCall(SNESGetDM(snes,&dm));
  PetscCall(DMSNESGetFunction(dm,f,ctx));
  PetscFunctionReturn(0);
}

/*@C
   SNESGetNGS - Returns the NGS function and context.

   Input Parameter:
.  snes - the SNES context

   Output Parameters:
+  f - the function (or NULL) see SNESNGSFunction for details
-  ctx    - the function context (or NULL)

   Level: advanced

.seealso: SNESSetNGS(), SNESGetFunction()
@*/

PetscErrorCode SNESGetNGS (SNES snes, PetscErrorCode (**f)(SNES, Vec, Vec, void*), void ** ctx)
{
  DM             dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscCall(SNESGetDM(snes,&dm));
  PetscCall(DMSNESGetNGS(dm,f,ctx));
  PetscFunctionReturn(0);
}

/*@C
   SNESSetOptionsPrefix - Sets the prefix used for searching for all
   SNES options in the database.

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
-  prefix - the prefix to prepend to all option names

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.seealso: SNESSetFromOptions()
@*/
PetscErrorCode  SNESSetOptionsPrefix(SNES snes,const char prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)snes,prefix));
  if (!snes->ksp) PetscCall(SNESGetKSP(snes,&snes->ksp));
  if (snes->linesearch) {
    PetscCall(SNESGetLineSearch(snes,&snes->linesearch));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)snes->linesearch,prefix));
  }
  PetscCall(KSPSetOptionsPrefix(snes->ksp,prefix));
  PetscFunctionReturn(0);
}

/*@C
   SNESAppendOptionsPrefix - Appends to the prefix used for searching for all
   SNES options in the database.

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
-  prefix - the prefix to prepend to all option names

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.seealso: SNESGetOptionsPrefix()
@*/
PetscErrorCode  SNESAppendOptionsPrefix(SNES snes,const char prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)snes,prefix));
  if (!snes->ksp) PetscCall(SNESGetKSP(snes,&snes->ksp));
  if (snes->linesearch) {
    PetscCall(SNESGetLineSearch(snes,&snes->linesearch));
    PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)snes->linesearch,prefix));
  }
  PetscCall(KSPAppendOptionsPrefix(snes->ksp,prefix));
  PetscFunctionReturn(0);
}

/*@C
   SNESGetOptionsPrefix - Sets the prefix used for searching for all
   SNES options in the database.

   Not Collective

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
.  prefix - pointer to the prefix string used

   Notes:
    On the fortran side, the user should pass in a string 'prefix' of
   sufficient length to hold the prefix.

   Level: advanced

.seealso: SNESAppendOptionsPrefix()
@*/
PetscErrorCode  SNESGetOptionsPrefix(SNES snes,const char *prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)snes,prefix));
  PetscFunctionReturn(0);
}

/*@C
  SNESRegister - Adds a method to the nonlinear solver package.

   Not collective

   Input Parameters:
+  name_solver - name of a new user-defined solver
-  routine_create - routine to create method context

   Notes:
   SNESRegister() may be called multiple times to add several user-defined solvers.

   Sample usage:
.vb
   SNESRegister("my_solver",MySolverCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     SNESSetType(snes,"my_solver")
   or at runtime via the option
$     -snes_type my_solver

   Level: advanced

    Note: If your function is not being put into a shared library then use SNESRegister() instead

.seealso: SNESRegisterAll(), SNESRegisterDestroy()

  Level: advanced
@*/
PetscErrorCode  SNESRegister(const char sname[],PetscErrorCode (*function)(SNES))
{
  PetscFunctionBegin;
  PetscCall(SNESInitializePackage());
  PetscCall(PetscFunctionListAdd(&SNESList,sname,function));
  PetscFunctionReturn(0);
}

PetscErrorCode  SNESTestLocalMin(SNES snes)
{
  PetscInt       N,i,j;
  Vec            u,uh,fh;
  PetscScalar    value;
  PetscReal      norm;

  PetscFunctionBegin;
  PetscCall(SNESGetSolution(snes,&u));
  PetscCall(VecDuplicate(u,&uh));
  PetscCall(VecDuplicate(u,&fh));

  /* currently only works for sequential */
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)snes),"Testing FormFunction() for local min\n"));
  PetscCall(VecGetSize(u,&N));
  for (i=0; i<N; i++) {
    PetscCall(VecCopy(u,uh));
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)snes),"i = %D\n",i));
    for (j=-10; j<11; j++) {
      value = PetscSign(j)*PetscExpReal(PetscAbs(j)-10.0);
      PetscCall(VecSetValue(uh,i,value,ADD_VALUES));
      PetscCall(SNESComputeFunction(snes,uh,fh));
      PetscCall(VecNorm(fh,NORM_2,&norm));
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)snes),"       j norm %D %18.16e\n",j,norm));
      value = -value;
      PetscCall(VecSetValue(uh,i,value,ADD_VALUES));
    }
  }
  PetscCall(VecDestroy(&uh));
  PetscCall(VecDestroy(&fh));
  PetscFunctionReturn(0);
}

/*@
   SNESKSPSetUseEW - Sets SNES use Eisenstat-Walker method for
   computing relative tolerance for linear solvers within an inexact
   Newton method.

   Logically Collective on SNES

   Input Parameters:
+  snes - SNES context
-  flag - PETSC_TRUE or PETSC_FALSE

    Options Database:
+  -snes_ksp_ew - use Eisenstat-Walker method for determining linear system convergence
.  -snes_ksp_ew_version ver - version of  Eisenstat-Walker method
.  -snes_ksp_ew_rtol0 <rtol0> - Sets rtol0
.  -snes_ksp_ew_rtolmax <rtolmax> - Sets rtolmax
.  -snes_ksp_ew_gamma <gamma> - Sets gamma
.  -snes_ksp_ew_alpha <alpha> - Sets alpha
.  -snes_ksp_ew_alpha2 <alpha2> - Sets alpha2
-  -snes_ksp_ew_threshold <threshold> - Sets threshold

   Notes:
   Currently, the default is to use a constant relative tolerance for
   the inner linear solvers.  Alternatively, one can use the
   Eisenstat-Walker method, where the relative convergence tolerance
   is reset at each Newton iteration according progress of the nonlinear
   solver.

   Level: advanced

   Reference:
   S. C. Eisenstat and H. F. Walker, "Choosing the forcing terms in an
   inexact Newton method", SISC 17 (1), pp.16-32, 1996.

.seealso: SNESKSPGetUseEW(), SNESKSPGetParametersEW(), SNESKSPSetParametersEW()
@*/
PetscErrorCode  SNESKSPSetUseEW(SNES snes,PetscBool flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidLogicalCollectiveBool(snes,flag,2);
  snes->ksp_ewconv = flag;
  PetscFunctionReturn(0);
}

/*@
   SNESKSPGetUseEW - Gets if SNES is using Eisenstat-Walker method
   for computing relative tolerance for linear solvers within an
   inexact Newton method.

   Not Collective

   Input Parameter:
.  snes - SNES context

   Output Parameter:
.  flag - PETSC_TRUE or PETSC_FALSE

   Notes:
   Currently, the default is to use a constant relative tolerance for
   the inner linear solvers.  Alternatively, one can use the
   Eisenstat-Walker method, where the relative convergence tolerance
   is reset at each Newton iteration according progress of the nonlinear
   solver.

   Level: advanced

   Reference:
   S. C. Eisenstat and H. F. Walker, "Choosing the forcing terms in an
   inexact Newton method", SISC 17 (1), pp.16-32, 1996.

.seealso: SNESKSPSetUseEW(), SNESKSPGetParametersEW(), SNESKSPSetParametersEW()
@*/
PetscErrorCode  SNESKSPGetUseEW(SNES snes, PetscBool  *flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidBoolPointer(flag,2);
  *flag = snes->ksp_ewconv;
  PetscFunctionReturn(0);
}

/*@
   SNESKSPSetParametersEW - Sets parameters for Eisenstat-Walker
   convergence criteria for the linear solvers within an inexact
   Newton method.

   Logically Collective on SNES

   Input Parameters:
+    snes - SNES context
.    version - version 1, 2 (default is 2) or 3
.    rtol_0 - initial relative tolerance (0 <= rtol_0 < 1)
.    rtol_max - maximum relative tolerance (0 <= rtol_max < 1)
.    gamma - multiplicative factor for version 2 rtol computation
             (0 <= gamma2 <= 1)
.    alpha - power for version 2 rtol computation (1 < alpha <= 2)
.    alpha2 - power for safeguard
-    threshold - threshold for imposing safeguard (0 < threshold < 1)

   Note:
   Version 3 was contributed by Luis Chacon, June 2006.

   Use PETSC_DEFAULT to retain the default for any of the parameters.

   Level: advanced

   Reference:
   S. C. Eisenstat and H. F. Walker, "Choosing the forcing terms in an
   inexact Newton method", Utah State University Math. Stat. Dept. Res.
   Report 6/94/75, June, 1994, to appear in SIAM J. Sci. Comput.

.seealso: SNESKSPSetUseEW(), SNESKSPGetUseEW(), SNESKSPGetParametersEW()
@*/
PetscErrorCode  SNESKSPSetParametersEW(SNES snes,PetscInt version,PetscReal rtol_0,PetscReal rtol_max,PetscReal gamma,PetscReal alpha,PetscReal alpha2,PetscReal threshold)
{
  SNESKSPEW *kctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  kctx = (SNESKSPEW*)snes->kspconvctx;
  PetscCheck(kctx,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"No Eisenstat-Walker context existing");
  PetscValidLogicalCollectiveInt(snes,version,2);
  PetscValidLogicalCollectiveReal(snes,rtol_0,3);
  PetscValidLogicalCollectiveReal(snes,rtol_max,4);
  PetscValidLogicalCollectiveReal(snes,gamma,5);
  PetscValidLogicalCollectiveReal(snes,alpha,6);
  PetscValidLogicalCollectiveReal(snes,alpha2,7);
  PetscValidLogicalCollectiveReal(snes,threshold,8);

  if (version != PETSC_DEFAULT)   kctx->version   = version;
  if (rtol_0 != PETSC_DEFAULT)    kctx->rtol_0    = rtol_0;
  if (rtol_max != PETSC_DEFAULT)  kctx->rtol_max  = rtol_max;
  if (gamma != PETSC_DEFAULT)     kctx->gamma     = gamma;
  if (alpha != PETSC_DEFAULT)     kctx->alpha     = alpha;
  if (alpha2 != PETSC_DEFAULT)    kctx->alpha2    = alpha2;
  if (threshold != PETSC_DEFAULT) kctx->threshold = threshold;

  PetscCheck(kctx->version >= 1 && kctx->version <= 3,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Only versions 1, 2 and 3 are supported: %D",kctx->version);
  PetscCheckFalse(kctx->rtol_0 < 0.0 || kctx->rtol_0 >= 1.0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"0.0 <= rtol_0 < 1.0: %g",(double)kctx->rtol_0);
  PetscCheckFalse(kctx->rtol_max < 0.0 || kctx->rtol_max >= 1.0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"0.0 <= rtol_max (%g) < 1.0",(double)kctx->rtol_max);
  PetscCheckFalse(kctx->gamma < 0.0 || kctx->gamma > 1.0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"0.0 <= gamma (%g) <= 1.0",(double)kctx->gamma);
  PetscCheckFalse(kctx->alpha <= 1.0 || kctx->alpha > 2.0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"1.0 < alpha (%g) <= 2.0",(double)kctx->alpha);
  PetscCheckFalse(kctx->threshold <= 0.0 || kctx->threshold >= 1.0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"0.0 < threshold (%g) < 1.0",(double)kctx->threshold);
  PetscFunctionReturn(0);
}

/*@
   SNESKSPGetParametersEW - Gets parameters for Eisenstat-Walker
   convergence criteria for the linear solvers within an inexact
   Newton method.

   Not Collective

   Input Parameter:
.    snes - SNES context

   Output Parameters:
+    version - version 1, 2 (default is 2) or 3
.    rtol_0 - initial relative tolerance (0 <= rtol_0 < 1)
.    rtol_max - maximum relative tolerance (0 <= rtol_max < 1)
.    gamma - multiplicative factor for version 2 rtol computation (0 <= gamma2 <= 1)
.    alpha - power for version 2 rtol computation (1 < alpha <= 2)
.    alpha2 - power for safeguard
-    threshold - threshold for imposing safeguard (0 < threshold < 1)

   Level: advanced

.seealso: SNESKSPSetUseEW(), SNESKSPGetUseEW(), SNESKSPSetParametersEW()
@*/
PetscErrorCode  SNESKSPGetParametersEW(SNES snes,PetscInt *version,PetscReal *rtol_0,PetscReal *rtol_max,PetscReal *gamma,PetscReal *alpha,PetscReal *alpha2,PetscReal *threshold)
{
  SNESKSPEW *kctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  kctx = (SNESKSPEW*)snes->kspconvctx;
  PetscCheck(kctx,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"No Eisenstat-Walker context existing");
  if (version)   *version   = kctx->version;
  if (rtol_0)    *rtol_0    = kctx->rtol_0;
  if (rtol_max)  *rtol_max  = kctx->rtol_max;
  if (gamma)     *gamma     = kctx->gamma;
  if (alpha)     *alpha     = kctx->alpha;
  if (alpha2)    *alpha2    = kctx->alpha2;
  if (threshold) *threshold = kctx->threshold;
  PetscFunctionReturn(0);
}

 PetscErrorCode KSPPreSolve_SNESEW(KSP ksp, Vec b, Vec x, SNES snes)
{
  SNESKSPEW      *kctx = (SNESKSPEW*)snes->kspconvctx;
  PetscReal      rtol  = PETSC_DEFAULT,stol;

  PetscFunctionBegin;
  if (!snes->ksp_ewconv) PetscFunctionReturn(0);
  if (!snes->iter) {
    rtol = kctx->rtol_0; /* first time in, so use the original user rtol */
    PetscCall(VecNorm(snes->vec_func,NORM_2,&kctx->norm_first));
  }
  else {
    if (kctx->version == 1) {
      rtol = (snes->norm - kctx->lresid_last)/kctx->norm_last;
      if (rtol < 0.0) rtol = -rtol;
      stol = PetscPowReal(kctx->rtol_last,kctx->alpha2);
      if (stol > kctx->threshold) rtol = PetscMax(rtol,stol);
    } else if (kctx->version == 2) {
      rtol = kctx->gamma * PetscPowReal(snes->norm/kctx->norm_last,kctx->alpha);
      stol = kctx->gamma * PetscPowReal(kctx->rtol_last,kctx->alpha);
      if (stol > kctx->threshold) rtol = PetscMax(rtol,stol);
    } else if (kctx->version == 3) { /* contributed by Luis Chacon, June 2006. */
      rtol = kctx->gamma * PetscPowReal(snes->norm/kctx->norm_last,kctx->alpha);
      /* safeguard: avoid sharp decrease of rtol */
      stol = kctx->gamma*PetscPowReal(kctx->rtol_last,kctx->alpha);
      stol = PetscMax(rtol,stol);
      rtol = PetscMin(kctx->rtol_0,stol);
      /* safeguard: avoid oversolving */
      stol = kctx->gamma*(kctx->norm_first*snes->rtol)/snes->norm;
      stol = PetscMax(rtol,stol);
      rtol = PetscMin(kctx->rtol_0,stol);
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Only versions 1, 2 or 3 are supported: %D",kctx->version);
  }
  /* safeguard: avoid rtol greater than one */
  rtol = PetscMin(rtol,kctx->rtol_max);
  PetscCall(KSPSetTolerances(ksp,rtol,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT));
  PetscCall(PetscInfo(snes,"iter %D, Eisenstat-Walker (version %D) KSP rtol=%g\n",snes->iter,kctx->version,(double)rtol));
  PetscFunctionReturn(0);
}

PetscErrorCode KSPPostSolve_SNESEW(KSP ksp, Vec b, Vec x, SNES snes)
{
  SNESKSPEW      *kctx = (SNESKSPEW*)snes->kspconvctx;
  PCSide         pcside;
  Vec            lres;

  PetscFunctionBegin;
  if (!snes->ksp_ewconv) PetscFunctionReturn(0);
  PetscCall(KSPGetTolerances(ksp,&kctx->rtol_last,NULL,NULL,NULL));
  kctx->norm_last = snes->norm;
  if (kctx->version == 1) {
    PC        pc;
    PetscBool isNone;

    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PetscObjectTypeCompare((PetscObject) pc, PCNONE, &isNone));
    PetscCall(KSPGetPCSide(ksp,&pcside));
     if (pcside == PC_RIGHT || isNone) { /* XXX Should we also test KSP_UNPRECONDITIONED_NORM ? */
      /* KSP residual is true linear residual */
      PetscCall(KSPGetResidualNorm(ksp,&kctx->lresid_last));
    } else {
      /* KSP residual is preconditioned residual */
      /* compute true linear residual norm */
      PetscCall(VecDuplicate(b,&lres));
      PetscCall(MatMult(snes->jacobian,x,lres));
      PetscCall(VecAYPX(lres,-1.0,b));
      PetscCall(VecNorm(lres,NORM_2,&kctx->lresid_last));
      PetscCall(VecDestroy(&lres));
    }
  }
  PetscFunctionReturn(0);
}

/*@
   SNESGetKSP - Returns the KSP context for a SNES solver.

   Not Collective, but if SNES object is parallel, then KSP object is parallel

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
.  ksp - the KSP context

   Notes:
   The user can then directly manipulate the KSP context to set various
   options, etc.  Likewise, the user can then extract and manipulate the
   PC contexts as well.

   Level: beginner

.seealso: KSPGetPC(), SNESCreate(), KSPCreate(), SNESSetKSP()
@*/
PetscErrorCode  SNESGetKSP(SNES snes,KSP *ksp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidPointer(ksp,2);

  if (!snes->ksp) {
    PetscCall(KSPCreate(PetscObjectComm((PetscObject)snes),&snes->ksp));
    PetscCall(PetscObjectIncrementTabLevel((PetscObject)snes->ksp,(PetscObject)snes,1));
    PetscCall(PetscLogObjectParent((PetscObject)snes,(PetscObject)snes->ksp));

    PetscCall(KSPSetPreSolve(snes->ksp,(PetscErrorCode (*)(KSP,Vec,Vec,void*))KSPPreSolve_SNESEW,snes));
    PetscCall(KSPSetPostSolve(snes->ksp,(PetscErrorCode (*)(KSP,Vec,Vec,void*))KSPPostSolve_SNESEW,snes));

    PetscCall(KSPMonitorSetFromOptions(snes->ksp, "-snes_monitor_ksp", "snes_preconditioned_residual", snes));
    PetscCall(PetscObjectSetOptions((PetscObject)snes->ksp,((PetscObject)snes)->options));
  }
  *ksp = snes->ksp;
  PetscFunctionReturn(0);
}

#include <petsc/private/dmimpl.h>
/*@
   SNESSetDM - Sets the DM that may be used by some nonlinear solvers or their underlying preconditioners

   Logically Collective on SNES

   Input Parameters:
+  snes - the nonlinear solver context
-  dm - the dm, cannot be NULL

   Notes:
   A DM can only be used for solving one problem at a time because information about the problem is stored on the DM,
   even when not using interfaces like DMSNESSetFunction().  Use DMClone() to get a distinct DM when solving different
   problems using the same function space.

   Level: intermediate

.seealso: SNESGetDM(), KSPSetDM(), KSPGetDM()
@*/
PetscErrorCode  SNESSetDM(SNES snes,DM dm)
{
  KSP            ksp;
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidHeaderSpecific(dm,DM_CLASSID,2);
  PetscCall(PetscObjectReference((PetscObject)dm));
  if (snes->dm) {               /* Move the DMSNES context over to the new DM unless the new DM already has one */
    if (snes->dm->dmsnes && !dm->dmsnes) {
      PetscCall(DMCopyDMSNES(snes->dm,dm));
      PetscCall(DMGetDMSNES(snes->dm,&sdm));
      if (sdm->originaldm == snes->dm) sdm->originaldm = dm; /* Grant write privileges to the replacement DM */
    }
    PetscCall(DMCoarsenHookRemove(snes->dm,DMCoarsenHook_SNESVecSol,DMRestrictHook_SNESVecSol,snes));
    PetscCall(DMDestroy(&snes->dm));
  }
  snes->dm     = dm;
  snes->dmAuto = PETSC_FALSE;

  PetscCall(SNESGetKSP(snes,&ksp));
  PetscCall(KSPSetDM(ksp,dm));
  PetscCall(KSPSetDMActive(ksp,PETSC_FALSE));
  if (snes->npc) {
    PetscCall(SNESSetDM(snes->npc,snes->dm));
    PetscCall(SNESSetNPCSide(snes,snes->npcside));
  }
  PetscFunctionReturn(0);
}

/*@
   SNESGetDM - Gets the DM that may be used by some preconditioners

   Not Collective but DM obtained is parallel on SNES

   Input Parameter:
. snes - the preconditioner context

   Output Parameter:
.  dm - the dm

   Level: intermediate

.seealso: SNESSetDM(), KSPSetDM(), KSPGetDM()
@*/
PetscErrorCode  SNESGetDM(SNES snes,DM *dm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (!snes->dm) {
    PetscCall(DMShellCreate(PetscObjectComm((PetscObject)snes),&snes->dm));
    snes->dmAuto = PETSC_TRUE;
  }
  *dm = snes->dm;
  PetscFunctionReturn(0);
}

/*@
  SNESSetNPC - Sets the nonlinear preconditioner to be used.

  Collective on SNES

  Input Parameters:
+ snes - iterative context obtained from SNESCreate()
- pc   - the preconditioner object

  Notes:
  Use SNESGetNPC() to retrieve the preconditioner context (for example,
  to configure it using the API).

  Level: developer

.seealso: SNESGetNPC(), SNESHasNPC()
@*/
PetscErrorCode SNESSetNPC(SNES snes, SNES pc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidHeaderSpecific(pc, SNES_CLASSID, 2);
  PetscCheckSameComm(snes, 1, pc, 2);
  PetscCall(PetscObjectReference((PetscObject) pc));
  PetscCall(SNESDestroy(&snes->npc));
  snes->npc = pc;
  PetscCall(PetscLogObjectParent((PetscObject)snes, (PetscObject)snes->npc));
  PetscFunctionReturn(0);
}

/*@
  SNESGetNPC - Creates a nonlinear preconditioning solver (SNES) to be used to precondition the nonlinear solver.

  Not Collective; but any changes to the obtained SNES object must be applied collectively

  Input Parameter:
. snes - iterative context obtained from SNESCreate()

  Output Parameter:
. pc - preconditioner context

  Options Database:
. -npc_snes_type <type> - set the type of the SNES to use as the nonlinear preconditioner

  Notes:
    If a SNES was previously set with SNESSetNPC() then that SNES is returned, otherwise a new SNES object is created.

    The (preconditioner) SNES returned automatically inherits the same nonlinear function and Jacobian supplied to the original
    SNES during SNESSetUp()

  Level: developer

.seealso: SNESSetNPC(), SNESHasNPC(), SNES, SNESCreate()
@*/
PetscErrorCode SNESGetNPC(SNES snes, SNES *pc)
{
  const char     *optionsprefix;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidPointer(pc, 2);
  if (!snes->npc) {
    PetscCall(SNESCreate(PetscObjectComm((PetscObject)snes),&snes->npc));
    PetscCall(PetscObjectIncrementTabLevel((PetscObject)snes->npc,(PetscObject)snes,1));
    PetscCall(PetscLogObjectParent((PetscObject)snes,(PetscObject)snes->npc));
    PetscCall(SNESGetOptionsPrefix(snes,&optionsprefix));
    PetscCall(SNESSetOptionsPrefix(snes->npc,optionsprefix));
    PetscCall(SNESAppendOptionsPrefix(snes->npc,"npc_"));
    PetscCall(SNESSetCountersReset(snes->npc,PETSC_FALSE));
  }
  *pc = snes->npc;
  PetscFunctionReturn(0);
}

/*@
  SNESHasNPC - Returns whether a nonlinear preconditioner exists

  Not Collective

  Input Parameter:
. snes - iterative context obtained from SNESCreate()

  Output Parameter:
. has_npc - whether the SNES has an NPC or not

  Level: developer

.seealso: SNESSetNPC(), SNESGetNPC()
@*/
PetscErrorCode SNESHasNPC(SNES snes, PetscBool *has_npc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  *has_npc = (PetscBool) (snes->npc ? PETSC_TRUE : PETSC_FALSE);
  PetscFunctionReturn(0);
}

/*@
    SNESSetNPCSide - Sets the preconditioning side.

    Logically Collective on SNES

    Input Parameter:
.   snes - iterative context obtained from SNESCreate()

    Output Parameter:
.   side - the preconditioning side, where side is one of
.vb
      PC_LEFT - left preconditioning
      PC_RIGHT - right preconditioning (default for most nonlinear solvers)
.ve

    Options Database Keys:
.   -snes_npc_side <right,left> - nonlinear preconditioner side

    Notes:
    SNESNRICHARDSON and SNESNCG only support left preconditioning.

    Level: intermediate

.seealso: SNESGetNPCSide(), KSPSetPCSide()
@*/
PetscErrorCode  SNESSetNPCSide(SNES snes,PCSide side)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidLogicalCollectiveEnum(snes,side,2);
  if (side == PC_SIDE_DEFAULT) side = PC_RIGHT;
  PetscCheck((side == PC_LEFT) || (side == PC_RIGHT),PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONG,"Only PC_LEFT and PC_RIGHT are supported");
  snes->npcside = side;
  PetscFunctionReturn(0);
}

/*@
    SNESGetNPCSide - Gets the preconditioning side.

    Not Collective

    Input Parameter:
.   snes - iterative context obtained from SNESCreate()

    Output Parameter:
.   side - the preconditioning side, where side is one of
.vb
      PC_LEFT - left preconditioning
      PC_RIGHT - right preconditioning (default for most nonlinear solvers)
.ve

    Level: intermediate

.seealso: SNESSetNPCSide(), KSPGetPCSide()
@*/
PetscErrorCode  SNESGetNPCSide(SNES snes,PCSide *side)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidPointer(side,2);
  *side = snes->npcside;
  PetscFunctionReturn(0);
}

/*@
  SNESSetLineSearch - Sets the linesearch on the SNES instance.

  Collective on SNES

  Input Parameters:
+ snes - iterative context obtained from SNESCreate()
- linesearch   - the linesearch object

  Notes:
  Use SNESGetLineSearch() to retrieve the preconditioner context (for example,
  to configure it using the API).

  Level: developer

.seealso: SNESGetLineSearch()
@*/
PetscErrorCode SNESSetLineSearch(SNES snes, SNESLineSearch linesearch)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidHeaderSpecific(linesearch, SNESLINESEARCH_CLASSID, 2);
  PetscCheckSameComm(snes, 1, linesearch, 2);
  PetscCall(PetscObjectReference((PetscObject) linesearch));
  PetscCall(SNESLineSearchDestroy(&snes->linesearch));

  snes->linesearch = linesearch;

  PetscCall(PetscLogObjectParent((PetscObject)snes, (PetscObject)snes->linesearch));
  PetscFunctionReturn(0);
}

/*@
  SNESGetLineSearch - Returns a pointer to the line search context set with SNESSetLineSearch()
  or creates a default line search instance associated with the SNES and returns it.

  Not Collective

  Input Parameter:
. snes - iterative context obtained from SNESCreate()

  Output Parameter:
. linesearch - linesearch context

  Level: beginner

.seealso: SNESSetLineSearch(), SNESLineSearchCreate()
@*/
PetscErrorCode SNESGetLineSearch(SNES snes, SNESLineSearch *linesearch)
{
  const char     *optionsprefix;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidPointer(linesearch, 2);
  if (!snes->linesearch) {
    PetscCall(SNESGetOptionsPrefix(snes, &optionsprefix));
    PetscCall(SNESLineSearchCreate(PetscObjectComm((PetscObject)snes), &snes->linesearch));
    PetscCall(SNESLineSearchSetSNES(snes->linesearch, snes));
    PetscCall(SNESLineSearchAppendOptionsPrefix(snes->linesearch, optionsprefix));
    PetscCall(PetscObjectIncrementTabLevel((PetscObject) snes->linesearch, (PetscObject) snes, 1));
    PetscCall(PetscLogObjectParent((PetscObject)snes, (PetscObject)snes->linesearch));
  }
  *linesearch = snes->linesearch;
  PetscFunctionReturn(0);
}
