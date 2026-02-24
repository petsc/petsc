#include <petsc/private/kspimpl.h> /*I "petscksp.h" I*/

/*@C
  KSPFlexibleSetModifyPC - Sets the routine used by flexible `KSP` methods to modify the preconditioner. [](sec_flexibleksp)

  Logically Collective

  Input Parameters:
+ ksp     - iterative context obtained from `KSPCreate()`
. fcn     - function to modify the `PC`, see `KSPFlexibleModifyPCFn`
. ctx     - optional context
- destroy - optional context destroy routine

  Level: intermediate

  Note:
  Several `fcn` routines are predefined, including `KSPFlexibleModifyPCNoChange()` and `KSPFlexibleModifyPCKSP()`

.seealso: [](ch_ksp), [](sec_flexibleksp), `KSPFGMRES`, `KSPFCG`, `KSPPIPEFCG`, `KSPGCR`, `KSPPIPEGCR`, `KSPFlexibleModifyPCFn`, `KSPFlexibleModifyPCNoChange()`, `KSPFlexibleModifyPCKSP()`
@*/
PetscErrorCode KSPFlexibleSetModifyPC(KSP ksp, KSPFlexibleModifyPCFn *fcn, PetscCtx ctx, PetscCtxDestroyFn *destroy)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscTryMethod(ksp, "KSPFlexibleSetModifyPC_C", (KSP, KSPFlexibleModifyPCFn *, PetscCtx, PetscCtxDestroyFn *), (ksp, fcn, ctx, destroy));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  KSPFlexibleModifyPCNoChange - this is the default used by the flexible Krylov methods - it doesn't change the preconditioner. [](sec_flexibleksp)

  Input Parameters:
+ ksp       - the ksp context being used.
. total_its - the total number of `KSP` iterations that have occurred.
. loc_its   - the number of `KSP` iterations since last restart.
. res_norm  - the current residual norm.
- ctx       - context variable, unused in this routine

  Level: intermediate

  Note:
  See `KSPFlexibleModifyPCKSP()` for a template for providing your own modification routines.

.seealso: [](ch_ksp), [](sec_flexibleksp), `KSPFGMRES`, `KSPFCG`, `KSPPIPEFCG`, `KSPGCR`, `KSPPIPEGCR`, `KSPFlexibleModifyPCFn`, `KSPFlexibleSetModifyPC()`, `KSPFlexibleModifyPCKSP()`
@*/
PetscErrorCode KSPFlexibleModifyPCNoChange(KSP ksp, PetscInt total_its, PetscInt loc_its, PetscReal res_norm, PetscCtx ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  KSPFlexibleModifyPCKSP - modifies the attributes of the `PCKSP` preconditioner, see [](sec_flexibleksp).

  Input Parameters:
+ ksp       - the ksp context being used.
. total_its - the total number of `KSP` iterations that have occurred.
. loc_its   - the number of `KSP` iterations since last restart.
. res_norm  - the current residual norm.
- ctx       - context, unused in this routine

  Level: intermediate

  Notes:
  You can use this as a template for writing a custom modification callback. See the source code for the change it makes.

  You can provide this to a `KSP` with `KSPFlexibleSetModifyPC()`

.seealso: [](ch_ksp), [](sec_flexibleksp), `KSPFGMRES`, `KSPFCG`, `KSPPIPEFCG`, `KSPGCR`, `KSPPIPEGCR`, `KSPFlexibleModifyPCFn`, `KSPFlexibleSetModifyPC()`
@*/
PetscErrorCode KSPFlexibleModifyPCKSP(KSP ksp, PetscInt total_its, PetscInt loc_its, PetscReal res_norm, PetscCtx ctx)
{
  PC        pc;
  PetscInt  maxits;
  KSP       sub_ksp;
  PetscReal rtol, abstol, dtol;
  PetscBool isksp;

  PetscFunctionBegin;
  PetscCall(KSPGetPC(ksp, &pc));

  PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCKSP, &isksp));
  if (isksp) {
    PetscCall(PCKSPGetKSP(pc, &sub_ksp));

    /* note that at this point you could check the type of KSP with KSPGetType() */

    /* Now we can use functions such as KSPGMRESSetRestart() or
      KSPGMRESSetOrthogonalization() or KSPSetTolerances() */

    PetscCall(KSPGetTolerances(sub_ksp, &rtol, &abstol, &dtol, &maxits));
    if (!loc_its) rtol = .1;
    else rtol *= .9;
    PetscCall(KSPSetTolerances(sub_ksp, rtol, abstol, dtol, maxits));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
