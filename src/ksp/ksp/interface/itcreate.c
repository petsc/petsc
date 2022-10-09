/*
     The basic KSP routines, Create, View etc. are here.
*/
#include <petsc/private/kspimpl.h> /*I "petscksp.h" I*/

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
  KSPLoad - Loads a `KSP` that has been stored in a `PETSCVIEWERBINARY`  with `KSPView()`.

  Collective on viewer

  Input Parameters:
+ newdm - the newly loaded `KSP`, this needs to have been created with `KSPCreate()` or
           some related function before a call to `KSPLoad()`.
- viewer - binary file viewer, obtained from `PetscViewerBinaryOpen()`

   Level: intermediate

  Note:
   The type is determined by the data in the file, any type set into the `KSP` before this call is ignored.

.seealso: `KSP`, `PetscViewerBinaryOpen()`, `KSPView()`, `MatLoad()`, `VecLoad()`
@*/
PetscErrorCode KSPLoad(KSP newdm, PetscViewer viewer)
{
  PetscBool isbinary;
  PetscInt  classid;
  char      type[256];
  PC        pc;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(newdm, KSP_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERBINARY, &isbinary));
  PetscCheck(isbinary, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid viewer; open viewer with PetscViewerBinaryOpen()");

  PetscCall(PetscViewerBinaryRead(viewer, &classid, 1, NULL, PETSC_INT));
  PetscCheck(classid == KSP_FILE_CLASSID, PetscObjectComm((PetscObject)newdm), PETSC_ERR_ARG_WRONG, "Not KSP next in file");
  PetscCall(PetscViewerBinaryRead(viewer, type, 256, NULL, PETSC_CHAR));
  PetscCall(KSPSetType(newdm, type));
  PetscTryTypeMethod(newdm, load, viewer);
  PetscCall(KSPGetPC(newdm, &pc));
  PetscCall(PCLoad(pc, viewer));
  PetscFunctionReturn(0);
}

#include <petscdraw.h>
#if defined(PETSC_HAVE_SAWS)
  #include <petscviewersaws.h>
#endif
/*@C
   KSPView - Prints the `KSP` data structure.

   Collective on ksp

   Input Parameters:
+  ksp - the Krylov space context
-  viewer - visualization context

   Options Database Keys:
.  -ksp_view - print the `KSP` data structure at the end of each `KSPSolve()` call

   Notes:
   The available visualization contexts include
+     `PETSC_VIEWER_STDOUT_SELF` - standard output (default)
-     `PETSC_VIEWER_STDOUT_WORLD` - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their
         data to the first processor to print.

   The available formats include
+     `PETSC_VIEWER_DEFAULT` - standard output (default)
-     `PETSC_VIEWER_ASCII_INFO_DETAIL` - more verbose output for PCBJACOBI and PCASM

   The user can open an alternative visualization context with
   `PetscViewerASCIIOpen()` - output to a specified file.

  In the debugger you can do call `KSPView(ksp,0)` to display the `KSP`. (The same holds for any PETSc object viewer).

   Level: beginner

.seealso: `KSP`, `PetscViewer`, `PCView()`, `PetscViewerASCIIOpen()`
@*/
PetscErrorCode KSPView(KSP ksp, PetscViewer viewer)
{
  PetscBool iascii, isbinary, isdraw, isstring;
#if defined(PETSC_HAVE_SAWS)
  PetscBool issaws;
#endif

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)ksp), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(ksp, 1, viewer, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERBINARY, &isbinary));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERDRAW, &isdraw));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERSTRING, &isstring));
#if defined(PETSC_HAVE_SAWS)
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERSAWS, &issaws));
#endif
  if (iascii) {
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)ksp, viewer));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscTryTypeMethod(ksp, view, viewer);
    PetscCall(PetscViewerASCIIPopTab(viewer));
    if (ksp->guess_zero) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  maximum iterations=%" PetscInt_FMT ", initial guess is zero\n", ksp->max_it));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  maximum iterations=%" PetscInt_FMT ", nonzero initial guess\n", ksp->max_it));
    }
    if (ksp->guess_knoll) PetscCall(PetscViewerASCIIPrintf(viewer, "  using preconditioner applied to right hand side for initial guess\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  tolerances:  relative=%g, absolute=%g, divergence=%g\n", (double)ksp->rtol, (double)ksp->abstol, (double)ksp->divtol));
    if (ksp->pc_side == PC_RIGHT) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  right preconditioning\n"));
    } else if (ksp->pc_side == PC_SYMMETRIC) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  symmetric preconditioning\n"));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  left preconditioning\n"));
    }
    if (ksp->guess) {
      PetscCall(PetscViewerASCIIPushTab(viewer));
      PetscCall(KSPGuessView(ksp->guess, viewer));
      PetscCall(PetscViewerASCIIPopTab(viewer));
    }
    if (ksp->dscale) PetscCall(PetscViewerASCIIPrintf(viewer, "  diagonally scaled system\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  using %s norm type for convergence test\n", KSPNormTypes[ksp->normtype]));
  } else if (isbinary) {
    PetscInt    classid = KSP_FILE_CLASSID;
    MPI_Comm    comm;
    PetscMPIInt rank;
    char        type[256];

    PetscCall(PetscObjectGetComm((PetscObject)ksp, &comm));
    PetscCallMPI(MPI_Comm_rank(comm, &rank));
    if (rank == 0) {
      PetscCall(PetscViewerBinaryWrite(viewer, &classid, 1, PETSC_INT));
      PetscCall(PetscStrncpy(type, ((PetscObject)ksp)->type_name, 256));
      PetscCall(PetscViewerBinaryWrite(viewer, type, 256, PETSC_CHAR));
    }
    PetscTryTypeMethod(ksp, view, viewer);
  } else if (isstring) {
    const char *type;
    PetscCall(KSPGetType(ksp, &type));
    PetscCall(PetscViewerStringSPrintf(viewer, " KSPType: %-7.7s", type));
    PetscTryTypeMethod(ksp, view, viewer);
  } else if (isdraw) {
    PetscDraw draw;
    char      str[36];
    PetscReal x, y, bottom, h;
    PetscBool flg;

    PetscCall(PetscViewerDrawGetDraw(viewer, 0, &draw));
    PetscCall(PetscDrawGetCurrentPoint(draw, &x, &y));
    PetscCall(PetscObjectTypeCompare((PetscObject)ksp, KSPPREONLY, &flg));
    if (!flg) {
      PetscCall(PetscStrncpy(str, "KSP: ", sizeof(str)));
      PetscCall(PetscStrlcat(str, ((PetscObject)ksp)->type_name, sizeof(str)));
      PetscCall(PetscDrawStringBoxed(draw, x, y, PETSC_DRAW_RED, PETSC_DRAW_BLACK, str, NULL, &h));
      bottom = y - h;
    } else {
      bottom = y;
    }
    PetscCall(PetscDrawPushCurrentPoint(draw, x, bottom));
#if defined(PETSC_HAVE_SAWS)
  } else if (issaws) {
    PetscMPIInt rank;
    const char *name;

    PetscCall(PetscObjectGetName((PetscObject)ksp, &name));
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
    if (!((PetscObject)ksp)->amsmem && rank == 0) {
      char dir[1024];

      PetscCall(PetscObjectViewSAWs((PetscObject)ksp, viewer));
      PetscCall(PetscSNPrintf(dir, 1024, "/PETSc/Objects/%s/its", name));
      PetscCallSAWs(SAWs_Register, (dir, &ksp->its, 1, SAWs_READ, SAWs_INT));
      if (!ksp->res_hist) PetscCall(KSPSetResidualHistory(ksp, NULL, PETSC_DECIDE, PETSC_TRUE));
      PetscCall(PetscSNPrintf(dir, 1024, "/PETSc/Objects/%s/res_hist", name));
      PetscCallSAWs(SAWs_Register, (dir, ksp->res_hist, 10, SAWs_READ, SAWs_DOUBLE));
    }
#endif
  } else PetscTryTypeMethod(ksp, view, viewer);
  if (ksp->pc) PetscCall(PCView(ksp->pc, viewer));
  if (isdraw) {
    PetscDraw draw;
    PetscCall(PetscViewerDrawGetDraw(viewer, 0, &draw));
    PetscCall(PetscDrawPopCurrentPoint(draw));
  }
  PetscFunctionReturn(0);
}

/*@C
   KSPViewFromOptions - View a `KSP` object based on values in the options database

   Collective on A

   Input Parameters:
+  A - Krylov solver context
.  obj - Optional object
-  name - command line option

   Level: intermediate

.seealso: `KSP`, `KSPView`, `PetscObjectViewFromOptions()`, `KSPCreate()`
@*/
PetscErrorCode KSPViewFromOptions(KSP A, PetscObject obj, const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, KSP_CLASSID, 1);
  PetscCall(PetscObjectViewFromOptions((PetscObject)A, obj, name));
  PetscFunctionReturn(0);
}

/*@
   KSPSetNormType - Sets the norm that is used for convergence testing.

   Logically Collective on ksp

   Input Parameters:
+  ksp - Krylov solver context
-  normtype - one of
.vb
   KSP_NORM_NONE - skips computing the norm, this should generally only be used if you are using
                 the Krylov method as a smoother with a fixed small number of iterations.
                 Implicitly sets KSPConvergedSkip() as KSP convergence test.
                 Note that certain algorithms such as KSPGMRES ALWAYS require the norm calculation,
                 for these methods the norms are still computed, they are just not used in
                 the convergence test.
   KSP_NORM_PRECONDITIONED - the default for left preconditioned solves, uses the l2 norm
                 of the preconditioned residual P^{-1}(b - A x)
   KSP_NORM_UNPRECONDITIONED - uses the l2 norm of the true b - Ax residual.
   KSP_NORM_NATURAL - supported  by KSPCG, KSPCR, KSPCGNE, KSPCGS
.ve

   Options Database Key:
.   -ksp_norm_type <none,preconditioned,unpreconditioned,natural> - set `KSP` norm type

   Level: advanced

   Note:
   Not all combinations of preconditioner side (see `KSPSetPCSide()`) and norm type are supported by all Krylov methods.
   If only one is set, PETSc tries to automatically change the other to find a compatible pair.  If no such combination
   is supported, PETSc will generate an error.

   Developer Note:
   Supported combinations of norm and preconditioner side are set using `KSPSetSupportedNorm()`.

.seealso: `KSPSetUp()`, `KSPSolve()`, `KSPDestroy()`, `KSPConvergedSkip()`, `KSPSetCheckNormIteration()`, `KSPSetPCSide()`, `KSPGetPCSide()`, `KSPNormType`
@*/
PetscErrorCode KSPSetNormType(KSP ksp, KSPNormType normtype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(ksp, normtype, 2);
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
   Currently only works with `KSPCG`, `KSPBCGS` and `KSPIBCGS`

   Use `KSPSetNormType`(ksp,`KSP_NORM_NONE`) to never check the norm

   On steps where the norm is not computed, the previous norm is still in the variable, so if you run with, for example,
    -ksp_monitor the residual norm will appear to be unchanged for several iterations (though it is not really unchanged).
   Level: advanced

.seealso: `KSP`, `KSPSetUp()`, `KSPSolve()`, `KSPDestroy()`, `KSPConvergedSkip()`, `KSPSetNormType()`
@*/
PetscErrorCode KSPSetCheckNormIteration(KSP ksp, PetscInt it)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidLogicalCollectiveInt(ksp, it, 2);
  ksp->chknorm = it;
  PetscFunctionReturn(0);
}

/*@
   KSPSetLagNorm - Lags the residual norm calculation so that it is computed as part of the `MPI_Allreduce()` for
   computing the inner products for the next iteration.  This can reduce communication costs at the expense of doing
   one additional iteration.

   Logically Collective on ksp

   Input Parameters:
+  ksp - Krylov solver context
-  flg - `PETSC_TRUE` or `PETSC_FALSE`

   Options Database Keys:
.  -ksp_lag_norm - lag the calculated residual norm

   Level: advanced

   Notes:
   Currently only works with `KSPIBCGS`.

   Use `KSPSetNormType`(ksp,`KSP_NORM_NONE`) to never check the norm

   If you lag the norm and run with, for example, -ksp_monitor, the residual norm reported will be the lagged one.

.seealso: `KSPSetUp()`, `KSPSolve()`, `KSPDestroy()`, `KSPConvergedSkip()`, `KSPSetNormType()`, `KSPSetCheckNormIteration()`
@*/
PetscErrorCode KSPSetLagNorm(KSP ksp, PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidLogicalCollectiveBool(ksp, flg, 2);
  ksp->lagnorm = flg;
  PetscFunctionReturn(0);
}

/*@
   KSPSetSupportedNorm - Sets a norm and preconditioner side supported by a `KSP`

   Logically Collective

   Input Parameters:
+  ksp - Krylov method
.  normtype - supported norm type
.  pcside - preconditioner side that can be used with this norm
-  priority - positive integer preference for this combination; larger values have higher priority

   Level: developer

   Note:
   This function should be called from the implementation files `KSPCreate_XXX()` to declare
   which norms and preconditioner sides are supported. Users should not need to call this
   function.

.seealso: `KSP`, `KSPNormType`, `PCSide`, `KSPSetNormType()`, `KSPSetPCSide()`
@*/
PetscErrorCode KSPSetSupportedNorm(KSP ksp, KSPNormType normtype, PCSide pcside, PetscInt priority)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  ksp->normsupporttable[normtype][pcside] = priority;
  PetscFunctionReturn(0);
}

PetscErrorCode KSPNormSupportTableReset_Private(KSP ksp)
{
  PetscFunctionBegin;
  PetscCall(PetscMemzero(ksp->normsupporttable, sizeof(ksp->normsupporttable)));
  ksp->pc_side  = ksp->pc_side_set;
  ksp->normtype = ksp->normtype_set;
  PetscFunctionReturn(0);
}

PetscErrorCode KSPSetUpNorms_Private(KSP ksp, PetscBool errorifnotsupported, KSPNormType *normtype, PCSide *pcside)
{
  PetscInt i, j, best, ibest = 0, jbest = 0;

  PetscFunctionBegin;
  best = 0;
  for (i = 0; i < KSP_NORM_MAX; i++) {
    for (j = 0; j < PC_SIDE_MAX; j++) {
      if ((ksp->normtype == KSP_NORM_DEFAULT || ksp->normtype == i) && (ksp->pc_side == PC_SIDE_DEFAULT || ksp->pc_side == j) && (ksp->normsupporttable[i][j] > best)) {
        best  = ksp->normsupporttable[i][j];
        ibest = i;
        jbest = j;
      }
    }
  }
  if (best < 1 && errorifnotsupported) {
    PetscCheck(ksp->normtype != KSP_NORM_DEFAULT || ksp->pc_side != PC_SIDE_DEFAULT, PetscObjectComm((PetscObject)ksp), PETSC_ERR_PLIB, "The %s KSP implementation did not call KSPSetSupportedNorm()", ((PetscObject)ksp)->type_name);
    PetscCheck(ksp->normtype != KSP_NORM_DEFAULT, PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "KSP %s does not support preconditioner side %s", ((PetscObject)ksp)->type_name, PCSides[ksp->pc_side]);
    PetscCheck(ksp->pc_side != PC_SIDE_DEFAULT, PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "KSP %s does not support norm type %s", ((PetscObject)ksp)->type_name, KSPNormTypes[ksp->normtype]);
    SETERRQ(PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "KSP %s does not support norm type %s with preconditioner side %s", ((PetscObject)ksp)->type_name, KSPNormTypes[ksp->normtype], PCSides[ksp->pc_side]);
  }
  if (normtype) *normtype = (KSPNormType)ibest;
  if (pcside) *pcside = (PCSide)jbest;
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

.seealso: `KSPNormType`, `KSPSetNormType()`, `KSPConvergedSkip()`
@*/
PetscErrorCode KSPGetNormType(KSP ksp, KSPNormType *normtype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidPointer(normtype, 2);
  PetscCall(KSPSetUpNorms_Private(ksp, PETSC_TRUE, &ksp->normtype, &ksp->pc_side));
  *normtype = ksp->normtype;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_SAWS)
  #include <petscviewersaws.h>
#endif

/*@
   KSPSetOperators - Sets the matrix associated with the linear system
   and a (possibly) different one from which the preconditioner will be built

   Collective on ksp

   Input Parameters:
+  ksp - the `KSP` context
.  Amat - the matrix that defines the linear system
-  Pmat - the matrix to be used in constructing the preconditioner, usually the same as Amat.

    Level: beginner

   Notes:
    If you know the operator Amat has a null space you can use `MatSetNullSpace()` and `MatSetTransposeNullSpace()` to supply the null
    space to Amat and the `KSP` solvers will automatically use that null space as needed during the solution process.

    All future calls to `KSPSetOperators()` must use the same size matrices!

    Passing a NULL for Amat or Pmat removes the matrix that is currently used.

    If you wish to replace either Amat or Pmat but leave the other one untouched then
    first call KSPGetOperators() to get the one you wish to keep, call `PetscObjectReference()`
    on it and then pass it back in in your call to `KSPSetOperators()`.

   Developer Notes:
   If the operators have NOT been set with `KSPSetOperators()` then the operators
      are created in the `PC` and returned to the user. In this case, if both operators
      mat and pmat are requested, two DIFFERENT operators will be returned. If
      only one is requested both operators in the PC will be the same (i.e. as
      if one had called `KSPSetOperators()` with the same argument for both `Mat`s).
      The user must set the sizes of the returned matrices and their type etc just
      as if the user created them with `MatCreate()`. For example,

.vb
         KSPGetOperators(ksp/pc,&mat,NULL); is equivalent to
           set size, type, etc of mat

         MatCreate(comm,&mat);
         KSP/PCSetOperators(ksp/pc,mat,mat);
         PetscObjectDereference((PetscObject)mat);
           set size, type, etc of mat

     and

         KSP/PCGetOperators(ksp/pc,&mat,&pmat); is equivalent to
           set size, type, etc of mat and pmat

         MatCreate(comm,&mat);
         MatCreate(comm,&pmat);
         KSP/PCSetOperators(ksp/pc,mat,pmat);
         PetscObjectDereference((PetscObject)mat);
         PetscObjectDereference((PetscObject)pmat);
           set size, type, etc of mat and pmat
.ve

    The rationale for this support is so that when creating a `TS`, `SNES`, or `KSP` the hierarchy
    of underlying objects (i.e. `SNES`, `KSP`, `PC`, `Mat`) and their livespans can be completely
    managed by the top most level object (i.e. the `TS`, `SNES`, or `KSP`). Another way to look
    at this is when you create a `SNES` you do not NEED to create a `KSP` and attach it to
    the `SNES` object (the `SNES` object manages it for you). Similarly when you create a `KSP`
    you do not need to attach a `PC` to it (the `KSP` object manages the `PC` object for you).
    Thus, why should YOU have to create the `Mat` and attach it to the `SNES`/`KSP`/`PC`, when
    it can be created for you?

.seealso: `KSP`, `Mat`, `KSPSolve()`, `KSPGetPC()`, `PCGetOperators()`, `PCSetOperators()`, `KSPGetOperators()`, `KSPSetComputeOperators()`, `KSPSetComputeInitialGuess()`, `KSPSetComputeRHS()`
@*/
PetscErrorCode KSPSetOperators(KSP ksp, Mat Amat, Mat Pmat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  if (Amat) PetscValidHeaderSpecific(Amat, MAT_CLASSID, 2);
  if (Pmat) PetscValidHeaderSpecific(Pmat, MAT_CLASSID, 3);
  if (Amat) PetscCheckSameComm(ksp, 1, Amat, 2);
  if (Pmat) PetscCheckSameComm(ksp, 1, Pmat, 3);
  if (!ksp->pc) PetscCall(KSPGetPC(ksp, &ksp->pc));
  PetscCall(PCSetOperators(ksp->pc, Amat, Pmat));
  if (ksp->setupstage == KSP_SETUP_NEWRHS) ksp->setupstage = KSP_SETUP_NEWMATRIX; /* so that next solve call will call PCSetUp() on new matrix */
  PetscFunctionReturn(0);
}

/*@
   KSPGetOperators - Gets the matrix associated with the linear system
   and a (possibly) different one used to construct the preconditioner.

   Collective on ksp

   Input Parameter:
.  ksp - the `KSP` context

   Output Parameters:
+  Amat - the matrix that defines the linear system
-  Pmat - the matrix to be used in constructing the preconditioner, usually the same as Amat.

    Level: intermediate

   Note:
    DOES NOT increase the reference counts of the matrix, so you should NOT destroy them.

.seealso: `KSP`, `KSPSolve()`, `KSPGetPC()`, `PCGetOperators()`, `PCSetOperators()`, `KSPSetOperators()`, `KSPGetOperatorsSet()`
@*/
PetscErrorCode KSPGetOperators(KSP ksp, Mat *Amat, Mat *Pmat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  if (!ksp->pc) PetscCall(KSPGetPC(ksp, &ksp->pc));
  PetscCall(PCGetOperators(ksp->pc, Amat, Pmat));
  PetscFunctionReturn(0);
}

/*@C
   KSPGetOperatorsSet - Determines if the matrix associated with the linear system and
   possibly a different one associated with the preconditioner have been set in the `KSP`.

   Not collective, though the results on all processes should be the same

   Input Parameter:
.  pc - the `KSP` context

   Output Parameters:
+  mat - the matrix associated with the linear system was set
-  pmat - matrix associated with the preconditioner was set, usually the same

   Level: intermediate

   Note:
   This routine exists because if you call `KSPGetOperators()` on a `KSP` that does not yet have operators they are
   automatically created in the call.

.seealso: `KSP`, `PCSetOperators()`, `KSPGetOperators()`, `KSPSetOperators()`, `PCGetOperators()`, `PCGetOperatorsSet()`
@*/
PetscErrorCode KSPGetOperatorsSet(KSP ksp, PetscBool *mat, PetscBool *pmat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  if (!ksp->pc) PetscCall(KSPGetPC(ksp, &ksp->pc));
  PetscCall(PCGetOperatorsSet(ksp->pc, mat, pmat));
  PetscFunctionReturn(0);
}

/*@C
   KSPSetPreSolve - Sets a function that is called at the beginning of each `KSPSolve()`

   Logically Collective on ksp

   Input Parameters:
+   ksp - the solver object
.   presolve - the function to call before the solve
-   prectx - any context needed by the function

   Calling sequence of presolve:
$  func(KSP ksp,Vec rhs,Vec x,void *ctx)

+  ksp - the `KSP` context
.  rhs - the right-hand side vector
.  x - the solution vector
-  ctx - optional user-provided context

   Level: developer

.seealso: `KSPSetUp()`, `KSPSolve()`, `KSPDestroy()`, `KSP`, `KSPSetPostSolve()`, `PCEISENSTAT`
@*/
PetscErrorCode KSPSetPreSolve(KSP ksp, PetscErrorCode (*presolve)(KSP, Vec, Vec, void *), void *prectx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  ksp->presolve = presolve;
  ksp->prectx   = prectx;
  PetscFunctionReturn(0);
}

/*@C
   KSPSetPostSolve - Sets a function that is called at the end of each `KSPSolve()` (whether it converges or not)

   Logically Collective on ksp

   Input Parameters:
+   ksp - the solver object
.   postsolve - the function to call after the solve
-   postctx - any context needed by the function

   Calling sequence of postsolve:
$  func(KSP ksp,Vec rhs,Vec x,void *ctx)

+  ksp - the `KSP` context
.  rhs - the right-hand side vector
.  x - the solution vector
-  ctx - optional user-provided context

   Level: developer

.seealso: `KSPSetUp()`, `KSPSolve()`, `KSPDestroy()`, `KSP`, `KSPSetPreSolve()`, `PCEISENSTAT`
@*/
PetscErrorCode KSPSetPostSolve(KSP ksp, PetscErrorCode (*postsolve)(KSP, Vec, Vec, void *), void *postctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  ksp->postsolve = postsolve;
  ksp->postctx   = postctx;
  PetscFunctionReturn(0);
}

/*@
   KSPCreate - Creates the `KSP` context.

   Collective

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  ksp - location to put the `KSP` context

   Note:
   The default `KSPType` is `KSPGMRES` with a restart of 30, using modified Gram-Schmidt orthogonalization.

   Level: beginner

.seealso: [](chapter_ksp), `KSPSetUp()`, `KSPSolve()`, `KSPDestroy()`, `KSP`, `KSPGMRES`, `KSPType`
@*/
PetscErrorCode KSPCreate(MPI_Comm comm, KSP *inksp)
{
  KSP   ksp;
  void *ctx;

  PetscFunctionBegin;
  PetscValidPointer(inksp, 2);
  *inksp = NULL;
  PetscCall(KSPInitializePackage());

  PetscCall(PetscHeaderCreate(ksp, KSP_CLASSID, "KSP", "Krylov Method", "KSP", comm, KSPDestroy, KSPView));

  ksp->max_it  = 10000;
  ksp->pc_side = ksp->pc_side_set = PC_SIDE_DEFAULT;
  ksp->rtol                       = 1.e-5;
#if defined(PETSC_USE_REAL_SINGLE)
  ksp->abstol = 1.e-25;
#else
  ksp->abstol = 1.e-50;
#endif
  ksp->divtol = 1.e4;

  ksp->chknorm  = -1;
  ksp->normtype = ksp->normtype_set = KSP_NORM_DEFAULT;
  ksp->rnorm                        = 0.0;
  ksp->its                          = 0;
  ksp->guess_zero                   = PETSC_TRUE;
  ksp->calc_sings                   = PETSC_FALSE;
  ksp->res_hist                     = NULL;
  ksp->res_hist_alloc               = NULL;
  ksp->res_hist_len                 = 0;
  ksp->res_hist_max                 = 0;
  ksp->res_hist_reset               = PETSC_TRUE;
  ksp->err_hist                     = NULL;
  ksp->err_hist_alloc               = NULL;
  ksp->err_hist_len                 = 0;
  ksp->err_hist_max                 = 0;
  ksp->err_hist_reset               = PETSC_TRUE;
  ksp->numbermonitors               = 0;
  ksp->numberreasonviews            = 0;
  ksp->setfromoptionscalled         = 0;
  ksp->nmax                         = PETSC_DECIDE;

  PetscCall(KSPConvergedDefaultCreate(&ctx));
  PetscCall(KSPSetConvergenceTest(ksp, KSPConvergedDefault, ctx, KSPConvergedDefaultDestroy));
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

  PetscCall(KSPNormSupportTableReset_Private(ksp));

  *inksp = ksp;
  PetscFunctionReturn(0);
}

/*@C
   KSPSetType - Builds the `KSP` datastructure for a particular `KSPType`

   Logically Collective on ksp

   Input Parameters:
+  ksp  - the Krylov space context
-  type - a known method

   Options Database Key:
.  -ksp_type  <method> - Sets the method; use -help for a list  of available methods (for instance, cg or gmres)

   Notes:
   See "petsc/include/petscksp.h" for available methods (for instance, `KSPCG` or `KSPGMRES`).

  Normally, it is best to use the `KSPSetFromOptions()` command and
  then set the `KSP` type from the options database rather than by using
  this routine.  Using the options database provides the user with
  maximum flexibility in evaluating the many different Krylov methods.
  The `KSPSetType()` routine is provided for those situations where it
  is necessary to set the iterative solver independently of the command
  line or options database.  This might be the case, for example, when
  the choice of iterative solver changes during the execution of the
  program, and the user's application is taking responsibility for
  choosing the appropriate method.  In other words, this routine is
  not for beginners.

  Level: intermediate

  Developer Note:
  `KSPRegister()` is used to add Krylov types to `KSPList` from which they are accessed by `KSPSetType()`.

.seealso: [](chapter_ksp), `PCSetType()`, `KSPType`, `KSPRegister()`, `KSPCreate()`, `KSP`
@*/
PetscErrorCode KSPSetType(KSP ksp, KSPType type)
{
  PetscBool match;
  PetscErrorCode (*r)(KSP);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidCharPointer(type, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)ksp, type, &match));
  if (match) PetscFunctionReturn(0);

  PetscCall(PetscFunctionListFind(KSPList, type, &r));
  PetscCheck(r, PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unable to find requested KSP type %s", type);
  /* Destroy the previous private KSP context */
  PetscTryTypeMethod(ksp, destroy);
  ksp->ops->destroy = NULL;

  /* Reinitialize function pointers in KSPOps structure */
  PetscCall(PetscMemzero(ksp->ops, sizeof(struct _KSPOps)));
  ksp->ops->buildsolution = KSPBuildSolutionDefault;
  ksp->ops->buildresidual = KSPBuildResidualDefault;
  PetscCall(KSPNormSupportTableReset_Private(ksp));
  ksp->setupnewmatrix = PETSC_FALSE; // restore default (setup not called in case of new matrix)
  /* Call the KSPCreate_XXX routine for this particular Krylov solver */
  ksp->setupstage = KSP_SETUP_NEW;
  PetscCall((*r)(ksp));
  PetscCall(PetscObjectChangeTypeName((PetscObject)ksp, type));
  PetscFunctionReturn(0);
}

/*@C
   KSPGetType - Gets the `KSP` type as a string from the KSP object.

   Not Collective

   Input Parameter:
.  ksp - Krylov context

   Output Parameter:
.  name - name of the `KSP` method

   Level: intermediate

.seealso: [](chapter_ksp), `KSPType`, `KSP`, `KSPSetType()`
@*/
PetscErrorCode KSPGetType(KSP ksp, KSPType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidPointer(type, 2);
  *type = ((PetscObject)ksp)->type_name;
  PetscFunctionReturn(0);
}

/*@C
  KSPRegister -  Adds a method, `KSPType`, to the Krylov subspace solver package.

   Not Collective

   Input Parameters:
+  name_solver - name of a new user-defined solver
-  routine_create - routine to create method context

   Level: advanced

   Note:
   `KSPRegister()` may be called multiple times to add several user-defined solvers.

   Sample usage:
.vb
   KSPRegister("my_solver",MySolverCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$    ` KSPSetType`(ksp,"my_solver")
   or at runtime via the option
$     -ksp_type my_solver

.seealso: [](chapter_ksp), `KSP`, `KSPType`, `KSPSetType`, `KSPRegisterAll()`
@*/
PetscErrorCode KSPRegister(const char sname[], PetscErrorCode (*function)(KSP))
{
  PetscFunctionBegin;
  PetscCall(KSPInitializePackage());
  PetscCall(PetscFunctionListAdd(&KSPList, sname, function));
  PetscFunctionReturn(0);
}

PetscErrorCode KSPMonitorMakeKey_Internal(const char name[], PetscViewerType vtype, PetscViewerFormat format, char key[])
{
  PetscFunctionBegin;
  PetscCall(PetscStrncpy(key, name, PETSC_MAX_PATH_LEN));
  PetscCall(PetscStrlcat(key, ":", PETSC_MAX_PATH_LEN));
  PetscCall(PetscStrlcat(key, vtype, PETSC_MAX_PATH_LEN));
  PetscCall(PetscStrlcat(key, ":", PETSC_MAX_PATH_LEN));
  PetscCall(PetscStrlcat(key, PetscViewerFormats[format], PETSC_MAX_PATH_LEN));
  PetscFunctionReturn(0);
}

/*@C
  KSPMonitorRegister -  Registers a Krylov subspace solver monitor routine that may be accessed with `KSPMonitorSetFromOptions()`

  Not Collective

  Input Parameters:
+ name    - name of a new monitor routine
. vtype   - A `PetscViewerType` for the output
. format  - A `PetscViewerFormat` for the output
. monitor - Monitor routine
. create  - Creation routine, or NULL
- destroy - Destruction routine, or NULL

  Level: advanced

  Note:
  `KSPMonitorRegister()` may be called multiple times to add several user-defined monitors.

  Sample usage:
.vb
  KSPMonitorRegister("my_monitor",PETSCVIEWERASCII,PETSC_VIEWER_ASCII_INFO_DETAIL,MyMonitor,NULL,NULL);
.ve

  Then, your monitor can be chosen with the procedural interface via
$     KSPMonitorSetFromOptions(ksp,"-ksp_monitor_my_monitor","my_monitor",NULL)
  or at runtime via the option
$     -ksp_monitor_my_monitor

.seealso: [](chapter_ksp), `KSP`, `KSPMonitorSet()`, `KSPMonitorRegisterAll()`, `KSPMonitorSetFromOptions()`
@*/
PetscErrorCode KSPMonitorRegister(const char name[], PetscViewerType vtype, PetscViewerFormat format, PetscErrorCode (*monitor)(KSP, PetscInt, PetscReal, PetscViewerAndFormat *), PetscErrorCode (*create)(PetscViewer, PetscViewerFormat, void *, PetscViewerAndFormat **), PetscErrorCode (*destroy)(PetscViewerAndFormat **))
{
  char key[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  PetscCall(KSPInitializePackage());
  PetscCall(KSPMonitorMakeKey_Internal(name, vtype, format, key));
  PetscCall(PetscFunctionListAdd(&KSPMonitorList, key, monitor));
  if (create) PetscCall(PetscFunctionListAdd(&KSPMonitorCreateList, key, create));
  if (destroy) PetscCall(PetscFunctionListAdd(&KSPMonitorDestroyList, key, destroy));
  PetscFunctionReturn(0);
}
