/*
     The basic KSP routines, Create, View etc. are here.
*/
#include <petsc/private/kspimpl.h> /*I "petscksp.h" I*/

/* Logging support */
PetscClassId  KSP_CLASSID;
PetscClassId  DMKSP_CLASSID;
PetscClassId  KSPGUESS_CLASSID;
PetscLogEvent KSP_GMRESOrthogonalization, KSP_SetUp, KSP_Solve, KSP_SolveTranspose, KSP_MatSolve, KSP_MatSolveTranspose;

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

/*@
  KSPLoad - Loads a `KSP` that has been stored in a `PETSCVIEWERBINARY`  with `KSPView()`.

  Collective

  Input Parameters:
+ newdm  - the newly loaded `KSP`, this needs to have been created with `KSPCreate()` or
           some related function before a call to `KSPLoad()`.
- viewer - binary file viewer, obtained from `PetscViewerBinaryOpen()`

  Level: intermediate

  Note:
  The type is determined by the data in the file, any type set into the `KSP` before this call is ignored.

.seealso: [](ch_ksp), `KSP`, `PetscViewerBinaryOpen()`, `KSPView()`, `MatLoad()`, `VecLoad()`
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

#include <petscdraw.h>
#if defined(PETSC_HAVE_SAWS)
  #include <petscviewersaws.h>
#endif
/*@
  KSPView - Prints the various parameters currently set in the `KSP` object. For example, the convergence tolerances and `KSPType`.
  Also views the `PC` and `Mat` contained by the `KSP` with `PCView()` and `MatView()`.

  Collective

  Input Parameters:
+ ksp    - the Krylov space context
- viewer - visualization context

  Options Database Key:
. -ksp_view - print the `KSP` data structure at the end of each `KSPSolve()` call

  Level: beginner

  Notes:
  The available visualization contexts include
+     `PETSC_VIEWER_STDOUT_SELF` - standard output (default)
-     `PETSC_VIEWER_STDOUT_WORLD` - synchronized standard
  output where only the first processor opens
  the file.  All other processors send their
  data to the first processor to print.

  The available formats include
+     `PETSC_VIEWER_DEFAULT` - standard output (default)
-     `PETSC_VIEWER_ASCII_INFO_DETAIL` - more verbose output for `PCBJACOBI` and `PCASM`

  The user can open an alternative visualization context with
  `PetscViewerASCIIOpen()` - output to a specified file.

  Use `KSPViewFromOptions()` to allow the user to select many different `PetscViewerType` and formats from the options database.

  In the debugger you can do call `KSPView(ksp,0)` to display the `KSP`. (The same holds for any PETSc object viewer).

.seealso: [](ch_ksp), `KSP`, `PetscViewer`, `PCView()`, `PetscViewerASCIIOpen()`, `KSPViewFromOptions()`
@*/
PetscErrorCode KSPView(KSP ksp, PetscViewer viewer)
{
  PetscBool isascii, isbinary, isdraw, isstring;
#if defined(PETSC_HAVE_SAWS)
  PetscBool issaws;
#endif

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)ksp), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(ksp, 1, viewer, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERBINARY, &isbinary));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERDRAW, &isdraw));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERSTRING, &isstring));
#if defined(PETSC_HAVE_SAWS)
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERSAWS, &issaws));
#endif
  if (isascii) {
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)ksp, viewer));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscTryTypeMethod(ksp, view, viewer);
    PetscCall(PetscViewerASCIIPopTab(viewer));
    if (ksp->guess_zero) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  maximum iterations=%" PetscInt_FMT ", initial guess is zero\n", ksp->max_it));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  maximum iterations=%" PetscInt_FMT ", nonzero initial guess\n", ksp->max_it));
    }
    if (ksp->min_it) PetscCall(PetscViewerASCIIPrintf(viewer, "  minimum iterations=%" PetscInt_FMT "\n", ksp->min_it));
    if (ksp->guess_knoll) PetscCall(PetscViewerASCIIPrintf(viewer, "  using preconditioner applied to right-hand side for initial guess\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  tolerances: relative=%g, absolute=%g, divergence=%g\n", (double)ksp->rtol, (double)ksp->abstol, (double)ksp->divtol));
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPViewFromOptions - View (print) a `KSP` object based on values in the options database. Also views the `PC` and `Mat` contained by the `KSP`
  with `PCView()` and `MatView()`.

  Collective

  Input Parameters:
+ A    - Krylov solver context
. obj  - Optional object that provides the options prefix used to query the options database
- name - command line option

  Level: intermediate

.seealso: [](ch_ksp), `KSP`, `KSPView()`, `PetscObjectViewFromOptions()`, `KSPCreate()`
@*/
PetscErrorCode KSPViewFromOptions(KSP A, PetscObject obj, const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, KSP_CLASSID, 1);
  PetscCall(PetscObjectViewFromOptions((PetscObject)A, obj, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPSetNormType - Sets the type of residual norm that is used for convergence testing in `KSPSolve()` for the given `KSP` context

  Logically Collective

  Input Parameters:
+ ksp      - Krylov solver context
- normtype - one of
.vb
   KSP_NORM_NONE             - skips computing the norm, this should generally only be used if you are using
                               the Krylov method as a smoother with a fixed small number of iterations.
                               Implicitly sets `KSPConvergedSkip()` as the `KSP` convergence test.
                               Note that certain algorithms such as `KSPGMRES` ALWAYS require the norm calculation,
                               for these methods the norms are still computed, they are just not used in
                               the convergence test.
   KSP_NORM_PRECONDITIONED   - the default for left-preconditioned solves, uses the 2-norm
                               of the preconditioned residual  $B^{-1}(b - A x)$.
   KSP_NORM_UNPRECONDITIONED - uses the 2-norm of the true $b - Ax$ residual.
   KSP_NORM_NATURAL          - uses the $A$ norm of the true $b - Ax$ residual; supported by `KSPCG`, `KSPCR`, `KSPCGNE`, `KSPCGS`
.ve

  Options Database Key:
. -ksp_norm_type <none,preconditioned,unpreconditioned,natural> - set `KSP` norm type

  Level: advanced

  Notes:
  The norm is always of the equations residual $\| b - A x^n \|$  (or an approximation to that norm), they are never a norm of the error in the equation.

  Not all combinations of preconditioner side (see `KSPSetPCSide()`) and norm types are supported by all Krylov methods.
  If only one is set, PETSc tries to automatically change the other to find a compatible pair.  If no such combination
  is supported, PETSc will generate an error.

  Developer Note:
  Supported combinations of norm and preconditioner side are set using `KSPSetSupportedNorm()` for each `KSPType`.

.seealso: [](ch_ksp), `KSPSetUp()`, `KSPSolve()`, `KSPDestroy()`, `KSPConvergedSkip()`, `KSPSetCheckNormIteration()`, `KSPSetPCSide()`, `KSPGetPCSide()`, `KSPNormType`
@*/
PetscErrorCode KSPSetNormType(KSP ksp, KSPNormType normtype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(ksp, normtype, 2);
  ksp->normtype = ksp->normtype_set = normtype;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPSetCheckNormIteration - Sets the first iteration at which the norm of the residual will be
  computed and used in the convergence test of `KSPSolve()` for the given `KSP` context

  Logically Collective

  Input Parameters:
+ ksp - Krylov solver context
- it  - use -1 to check at all iterations

  Level: advanced

  Notes:
  Currently only works with `KSPCG`, `KSPBCGS` and `KSPIBCGS`

  Use `KSPSetNormType`(ksp,`KSP_NORM_NONE`) to never check the norm

  On steps where the norm is not computed, the previous norm is still in the variable, so if you run with, for example,
  `-ksp_monitor` the residual norm will appear to be unchanged for several iterations (though it is not really unchanged).

  Certain methods such as `KSPGMRES` always compute the residual norm, this routine will not change that computation, but it will
  prevent the computed norm from being checked.

.seealso: [](ch_ksp), `KSP`, `KSPSetUp()`, `KSPSolve()`, `KSPDestroy()`, `KSPConvergedSkip()`, `KSPSetNormType()`, `KSPSetLagNorm()`
@*/
PetscErrorCode KSPSetCheckNormIteration(KSP ksp, PetscInt it)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidLogicalCollectiveInt(ksp, it, 2);
  ksp->chknorm = it;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPSetLagNorm - Lags the residual norm calculation so that it is computed as part of the `MPI_Allreduce()` used for
  computing the inner products needed for the next iteration.

  Logically Collective

  Input Parameters:
+ ksp - Krylov solver context
- flg - `PETSC_TRUE` or `PETSC_FALSE`

  Options Database Key:
. -ksp_lag_norm - lag the calculated residual norm

  Level: advanced

  Notes:
  Currently only works with `KSPIBCGS`.

  This can reduce communication costs at the expense of doing
  one additional iteration because the norm used in the convergence test of `KSPSolve()` is one iteration behind the actual
  current residual norm (which has not yet been computed due to the lag).

  Use `KSPSetNormType`(ksp,`KSP_NORM_NONE`) to never check the norm

  If you lag the norm and run with, for example, `-ksp_monitor`, the residual norm reported will be the lagged one.

  `KSPSetCheckNormIteration()` is an alternative way of avoiding the expense of computing the residual norm at each iteration.

.seealso: [](ch_ksp), `KSPSetUp()`, `KSPSolve()`, `KSPDestroy()`, `KSPConvergedSkip()`, `KSPSetNormType()`, `KSPSetCheckNormIteration()`
@*/
PetscErrorCode KSPSetLagNorm(KSP ksp, PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidLogicalCollectiveBool(ksp, flg, 2);
  ksp->lagnorm = flg;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPSetSupportedNorm - Sets a norm and preconditioner side supported by a `KSPType`

  Logically Collective

  Input Parameters:
+ ksp      - Krylov method
. normtype - supported norm type of the type `KSPNormType`
. pcside   - preconditioner side, of the type `PCSide` that can be used with this `KSPNormType`
- priority - positive integer preference for this combination; larger values have higher priority

  Level: developer

  Notes:
  This function should be called from the implementation files `KSPCreate_XXX()` to declare
  which norms and preconditioner sides are supported. Users should not call this
  function.

  This function can be called multiple times for each combination of `KSPNormType` and `PCSide`
  the `KSPType` supports

.seealso: [](ch_ksp), `KSP`, `KSPNormType`, `PCSide`, `KSPSetNormType()`, `KSPSetPCSide()`
@*/
PetscErrorCode KSPSetSupportedNorm(KSP ksp, KSPNormType normtype, PCSide pcside, PetscInt priority)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  ksp->normsupporttable[normtype][pcside] = priority;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPNormSupportTableReset_Private(KSP ksp)
{
  PetscFunctionBegin;
  PetscCall(PetscMemzero(ksp->normsupporttable, sizeof(ksp->normsupporttable)));
  ksp->pc_side  = ksp->pc_side_set;
  ksp->normtype = ksp->normtype_set;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode KSPSetUpNorms_Private(KSP ksp, PetscBool errorifnotsupported, KSPNormType *normtype, PCSide *pcside)
{
  PetscInt i, j, best, ibest = 0, jbest = 0;

  PetscFunctionBegin;
  best = 0;
  for (i = 0; i < KSP_NORM_MAX; i++) {
    for (j = 0; j < PC_SIDE_MAX; j++) {
      if ((ksp->normtype == KSP_NORM_DEFAULT || ksp->normtype == i) && (ksp->pc_side == PC_SIDE_DEFAULT || ksp->pc_side == j) && ksp->normsupporttable[i][j] > best) {
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPGetNormType - Gets the `KSPNormType` that is used for convergence testing during `KSPSolve()` for this `KSP` context

  Not Collective

  Input Parameter:
. ksp - Krylov solver context

  Output Parameter:
. normtype - the `KSPNormType` that is used for convergence testing

  Level: advanced

.seealso: [](ch_ksp), `KSPNormType`, `KSPSetNormType()`, `KSPConvergedSkip()`
@*/
PetscErrorCode KSPGetNormType(KSP ksp, KSPNormType *normtype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscAssertPointer(normtype, 2);
  PetscCall(KSPSetUpNorms_Private(ksp, PETSC_TRUE, &ksp->normtype, &ksp->pc_side));
  *normtype = ksp->normtype;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_HAVE_SAWS)
  #include <petscviewersaws.h>
#endif

/*@
  KSPSetOperators - Sets the matrix associated with the linear system
  and a (possibly) different one from which the preconditioner will be built into the `KSP` context. The matrix will then be used during `KSPSolve()`

  Collective

  Input Parameters:
+ ksp  - the `KSP` context
. Amat - the matrix that defines the linear system
- Pmat - the matrix to be used in constructing the preconditioner, usually the same as `Amat`.

  Level: beginner

  Notes:
.vb
  KSPSetOperators(ksp, Amat, Pmat);
.ve
  is the same as
.vb
  KSPGetPC(ksp, &pc);
  PCSetOperators(pc, Amat, Pmat);
.ve
  and is equivalent to
.vb
  PCCreate(PetscObjectComm((PetscObject)ksp), &pc);
  PCSetOperators(pc, Amat, Pmat);
  KSPSetPC(ksp, pc);
.ve

  If you know the operator `Amat` has a null space you can use `MatSetNullSpace()` and `MatSetTransposeNullSpace()` to supply the null
  space to `Amat` and the `KSP` solvers will automatically use that null space as needed during the solution process.

  All future calls to `KSPSetOperators()` must use the same size matrices, unless `KSPReset()` is called!

  Passing a `NULL` for `Amat` or `Pmat` removes the matrix that is currently being used from the `KSP` context.

  If you wish to replace either `Amat` or `Pmat` but leave the other one untouched then
  first call `KSPGetOperators()` to get the one you wish to keep, call `PetscObjectReference()`
  on it and then pass it back in your call to `KSPSetOperators()`.

  Developer Notes:
  If the operators have NOT been set with `KSPSetOperators()` then the operators
  are created in the `PC` and returned to the user. In this case, if both operators
  mat and pmat are requested, two DIFFERENT operators will be returned. If
  only one is requested both operators in the `PC` will be the same (i.e. as
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
  of underlying objects (i.e. `SNES`, `KSP`, `PC`, `Mat`) and their lifespans can be completely
  managed by the top most level object (i.e. the `TS`, `SNES`, or `KSP`). Another way to look
  at this is when you create a `SNES` you do not NEED to create a `KSP` and attach it to
  the `SNES` object (the `SNES` object manages it for you). Similarly when you create a `KSP`
  you do not need to attach a `PC` to it (the `KSP` object manages the `PC` object for you).
  Thus, why should YOU have to create the `Mat` and attach it to the `SNES`/`KSP`/`PC`, when
  it can be created for you?

.seealso: [](ch_ksp), `KSP`, `Mat`, `KSPSolve()`, `KSPGetPC()`, `PCGetOperators()`, `PCSetOperators()`, `KSPGetOperators()`, `KSPSetComputeOperators()`, `KSPSetComputeInitialGuess()`, `KSPSetComputeRHS()`
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPGetOperators - Gets the matrix associated with the linear system
  and a (possibly) different one used to construct the preconditioner from the `KSP` context

  Collective

  Input Parameter:
. ksp - the `KSP` context

  Output Parameters:
+ Amat - the matrix that defines the linear system
- Pmat - the matrix to be used in constructing the preconditioner, usually the same as `Amat`.

  Level: intermediate

  Notes:
  If `KSPSetOperators()` has not been called then the `KSP` object will attempt to automatically create the matrix `Amat` and return it

  Use `KSPGetOperatorsSet()` to determine if matrices have been provided.

  DOES NOT increase the reference counts of the matrix, so you should NOT destroy them.

.seealso: [](ch_ksp), `KSP`, `KSPSolve()`, `KSPGetPC()`, `PCSetOperators()`, `KSPSetOperators()`, `KSPGetOperatorsSet()`
@*/
PetscErrorCode KSPGetOperators(KSP ksp, Mat *Amat, Mat *Pmat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  if (!ksp->pc) PetscCall(KSPGetPC(ksp, &ksp->pc));
  PetscCall(PCGetOperators(ksp->pc, Amat, Pmat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPGetOperatorsSet - Determines if the matrix associated with the linear system and
  possibly a different one from which the preconditioner will be built have been set in the `KSP` with `KSPSetOperators()`

  Not Collective, though the results on all processes will be the same

  Input Parameter:
. ksp - the `KSP` context

  Output Parameters:
+ mat  - the matrix associated with the linear system was set
- pmat - matrix from which the preconditioner will be built, usually the same as `mat` was set

  Level: intermediate

  Note:
  This routine exists because if you call `KSPGetOperators()` on a `KSP` that does not yet have operators they are
  automatically created in the call.

.seealso: [](ch_ksp), `KSP`, `PCSetOperators()`, `KSPGetOperators()`, `KSPSetOperators()`, `PCGetOperators()`, `PCGetOperatorsSet()`
@*/
PetscErrorCode KSPGetOperatorsSet(KSP ksp, PetscBool *mat, PetscBool *pmat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  if (!ksp->pc) PetscCall(KSPGetPC(ksp, &ksp->pc));
  PetscCall(PCGetOperatorsSet(ksp->pc, mat, pmat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  KSPSetPreSolve - Sets a function that is called at the beginning of each `KSPSolve()`. Used in conjunction with `KSPSetPostSolve()`.

  Logically Collective

  Input Parameters:
+ ksp      - the solver object
. presolve - the function to call before the solve, see` KSPPSolveFn`
- ctx      - an optional context needed by the function

  Level: developer

  Notes:
  The function provided here `presolve` is used to modify the right hand side, and possibly the matrix, of the linear system to be solved.
  The function provided with `KSPSetPostSolve()` then modifies the resulting solution of that linear system to obtain the correct solution
  to the initial linear system.

  The functions `PCPreSolve()` and `PCPostSolve()` provide a similar functionality and are used, for example with `PCEISENSTAT`.

.seealso: [](ch_ksp), `KSPPSolveFn`, `KSPSetUp()`, `KSPSolve()`, `KSPDestroy()`, `KSP`, `KSPSetPostSolve()`, `PCEISENSTAT`, `PCPreSolve()`, `PCPostSolve()`
@*/
PetscErrorCode KSPSetPreSolve(KSP ksp, KSPPSolveFn *presolve, void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  ksp->presolve = presolve;
  ksp->prectx   = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  KSPSetPostSolve - Sets a function that is called at the end of each `KSPSolve()` (whether it converges or not). Used in conjunction with `KSPSetPreSolve()`.

  Logically Collective

  Input Parameters:
+ ksp       - the solver object
. postsolve - the function to call after the solve, see` KSPPSolveFn`
- ctx       - an optional context needed by the function

  Level: developer

.seealso: [](ch_ksp), `KSPPSolveFn`, `KSPSetUp()`, `KSPSolve()`, `KSPDestroy()`, `KSP`, `KSPSetPreSolve()`, `PCEISENSTAT`
@*/
PetscErrorCode KSPSetPostSolve(KSP ksp, KSPPSolveFn *postsolve, void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  ksp->postsolve = postsolve;
  ksp->postctx   = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPSetNestLevel - sets the amount of nesting the `KSP` has. That is the number of levels of `KSP` above this `KSP` in a linear solve.

  Collective

  Input Parameters:
+ ksp   - the `KSP`
- level - the nest level

  Level: developer

  Note:
  For example, the `KSP` in each block of a `KSPBJACOBI` has a level of 1, while the outer `KSP` has a level of 0.

.seealso: [](ch_ksp), `KSPSetUp()`, `KSPSolve()`, `KSPDestroy()`, `KSP`, `KSPGMRES`, `KSPType`, `KSPGetNestLevel()`, `PCSetKSPNestLevel()`, `PCGetKSPNestLevel()`
@*/
PetscErrorCode KSPSetNestLevel(KSP ksp, PetscInt level)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidLogicalCollectiveInt(ksp, level, 2);
  ksp->nestlevel = level;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPGetNestLevel - gets the amount of nesting the `KSP` has

  Not Collective

  Input Parameter:
. ksp - the `KSP`

  Output Parameter:
. level - the nest level

  Level: developer

.seealso: [](ch_ksp), `KSPSetUp()`, `KSPSolve()`, `KSPDestroy()`, `KSP`, `KSPGMRES`, `KSPType`, `KSPSetNestLevel()`, `PCSetKSPNestLevel()`, `PCGetKSPNestLevel()`
@*/
PetscErrorCode KSPGetNestLevel(KSP ksp, PetscInt *level)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscAssertPointer(level, 2);
  *level = ksp->nestlevel;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPCreate - Creates the `KSP` context. This `KSP` context is used in PETSc to solve linear systems with `KSPSolve()`

  Collective

  Input Parameter:
. comm - MPI communicator

  Output Parameter:
. inksp - location to put the `KSP` context

  Level: beginner

  Note:
  The default `KSPType` is `KSPGMRES` with a restart of 30, using modified Gram-Schmidt orthogonalization. The `KSPType` may be
  changed with `KSPSetType()`

.seealso: [](ch_ksp), `KSPSetUp()`, `KSPSolve()`, `KSPDestroy()`, `KSP`, `KSPGMRES`, `KSPType`, `KSPSetType()`
@*/
PetscErrorCode KSPCreate(MPI_Comm comm, KSP *inksp)
{
  KSP   ksp;
  void *ctx;

  PetscFunctionBegin;
  PetscAssertPointer(inksp, 2);
  PetscCall(KSPInitializePackage());

  PetscCall(PetscHeaderCreate(ksp, KSP_CLASSID, "KSP", "Krylov Method", "KSP", comm, KSPDestroy, KSPView));
  ksp->default_max_it = ksp->max_it = 10000;
  ksp->pc_side = ksp->pc_side_set = PC_SIDE_DEFAULT;

  ksp->default_rtol = ksp->rtol = 1.e-5;
  ksp->default_abstol = ksp->abstol = PetscDefined(USE_REAL_SINGLE) ? 1.e-25 : 1.e-50;
  ksp->default_divtol = ksp->divtol = 1.e4;

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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPSetType - Sets the algorithm/method to be used to solve the linear system with the given `KSP`

  Logically Collective

  Input Parameters:
+ ksp  - the Krylov space context
- type - a known method

  Options Database Key:
. -ksp_type  <method> - Sets the method; see `KSPType` or use `-help` for a list  of available methods (for instance, cg or gmres)

  Level: intermediate

  Notes:
  See `KSPType` for available methods (for instance, `KSPCG` or `KSPGMRES`).

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

  Developer Note:
  `KSPRegister()` is used to add Krylov types to `KSPList` from which they are accessed by `KSPSetType()`.

.seealso: [](ch_ksp), `PCSetType()`, `KSPType`, `KSPRegister()`, `KSPCreate()`, `KSP`
@*/
PetscErrorCode KSPSetType(KSP ksp, KSPType type)
{
  PetscBool match;
  PetscErrorCode (*r)(KSP);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscAssertPointer(type, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)ksp, type, &match));
  if (match) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscFunctionListFind(KSPList, type, &r));
  PetscCheck(r, PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unable to find requested KSP type %s", type);
  /* Destroy the previous private KSP context */
  PetscTryTypeMethod(ksp, destroy);

  /* Reinitialize function pointers in KSPOps structure */
  PetscCall(PetscMemzero(ksp->ops, sizeof(struct _KSPOps)));
  ksp->ops->buildsolution = KSPBuildSolutionDefault;
  ksp->ops->buildresidual = KSPBuildResidualDefault;
  PetscCall(KSPNormSupportTableReset_Private(ksp));
  ksp->converged_neg_curve = PETSC_FALSE; // restore default
  ksp->setupnewmatrix      = PETSC_FALSE; // restore default (setup not called in case of new matrix)
  /* Call the KSPCreate_XXX routine for this particular Krylov solver */
  ksp->setupstage     = KSP_SETUP_NEW;
  ksp->guess_not_read = PETSC_FALSE; // restore default
  PetscCall((*r)(ksp));
  PetscCall(PetscObjectChangeTypeName((PetscObject)ksp, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPGetType - Gets the `KSP` type as a string from the `KSP` object.

  Not Collective

  Input Parameter:
. ksp - Krylov context

  Output Parameter:
. type - name of the `KSP` method

  Level: intermediate

.seealso: [](ch_ksp), `KSPType`, `KSP`, `KSPSetType()`
@*/
PetscErrorCode KSPGetType(KSP ksp, KSPType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscAssertPointer(type, 2);
  *type = ((PetscObject)ksp)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  KSPRegister -  Adds a method, `KSPType`, to the Krylov subspace solver package.

  Not Collective, No Fortran Support

  Input Parameters:
+ sname    - name of a new user-defined solver
- function - routine to create method

  Level: advanced

  Note:
  `KSPRegister()` may be called multiple times to add several user-defined solvers.

  Example Usage:
.vb
   KSPRegister("my_solver", MySolverCreate);
.ve

  Then, your solver can be chosen with the procedural interface via
.vb
  KSPSetType(ksp, "my_solver")
.ve
  or at runtime via the option `-ksp_type my_solver`

.seealso: [](ch_ksp), `KSP`, `KSPType`, `KSPSetType`, `KSPRegisterAll()`
@*/
PetscErrorCode KSPRegister(const char sname[], PetscErrorCode (*function)(KSP))
{
  PetscFunctionBegin;
  PetscCall(KSPInitializePackage());
  PetscCall(PetscFunctionListAdd(&KSPList, sname, function));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode KSPMonitorMakeKey_Internal(const char name[], PetscViewerType vtype, PetscViewerFormat format, char key[])
{
  PetscFunctionBegin;
  PetscCall(PetscStrncpy(key, name, PETSC_MAX_PATH_LEN));
  PetscCall(PetscStrlcat(key, ":", PETSC_MAX_PATH_LEN));
  PetscCall(PetscStrlcat(key, vtype, PETSC_MAX_PATH_LEN));
  PetscCall(PetscStrlcat(key, ":", PETSC_MAX_PATH_LEN));
  PetscCall(PetscStrlcat(key, PetscViewerFormats[format], PETSC_MAX_PATH_LEN));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  KSPMonitorRegister -  Registers a Krylov subspace solver monitor routine that may be accessed with `KSPMonitorSetFromOptions()`

  Not Collective

  Input Parameters:
+ name    - name of a new monitor type
. vtype   - A `PetscViewerType` for the output
. format  - A `PetscViewerFormat` for the output
. monitor - Monitor routine, see `KSPMonitorRegisterFn`
. create  - Creation routine, or `NULL`
- destroy - Destruction routine, or `NULL`

  Level: advanced

  Notes:
  `KSPMonitorRegister()` may be called multiple times to add several user-defined monitors.

  The calling sequence for the given function matches the calling sequence used by `KSPMonitorFn` functions passed to `KSPMonitorSet()` with the additional
  requirement that its final argument be a `PetscViewerAndFormat`.

  Example Usage:
.vb
  KSPMonitorRegister("my_monitor", PETSCVIEWERASCII, PETSC_VIEWER_ASCII_INFO_DETAIL, MyMonitor, NULL, NULL);
.ve

  Then, your monitor can be chosen with the procedural interface via
.vb
  KSPMonitorSetFromOptions(ksp, "-ksp_monitor_my_monitor", "my_monitor", NULL)
.ve
  or at runtime via the option `-ksp_monitor_my_monitor`

.seealso: [](ch_ksp), `KSP`, `KSPMonitorSet()`, `KSPMonitorRegisterAll()`, `KSPMonitorSetFromOptions()`
@*/
PetscErrorCode KSPMonitorRegister(const char name[], PetscViewerType vtype, PetscViewerFormat format, KSPMonitorRegisterFn *monitor, KSPMonitorRegisterCreateFn *create, KSPMonitorRegisterDestroyFn *destroy)
{
  char key[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  PetscCall(KSPInitializePackage());
  PetscCall(KSPMonitorMakeKey_Internal(name, vtype, format, key));
  PetscCall(PetscFunctionListAdd(&KSPMonitorList, key, monitor));
  if (create) PetscCall(PetscFunctionListAdd(&KSPMonitorCreateList, key, create));
  if (destroy) PetscCall(PetscFunctionListAdd(&KSPMonitorDestroyList, key, destroy));
  PetscFunctionReturn(PETSC_SUCCESS);
}
