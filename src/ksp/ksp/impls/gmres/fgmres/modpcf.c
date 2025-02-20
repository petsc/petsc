#include <petsc/private/kspimpl.h> /*I "petscksp.h" I*/

/*@C
  KSPFGMRESSetModifyPC - Sets the routine used by `KSPFGMRES` to modify the preconditioner. [](sec_flexibleksp)

  Logically Collective

  Input Parameters:
+ ksp     - iterative context obtained from `KSPCreate()`
. fcn     - modifypc function
. ctx     - optional context
- destroy - optional context destroy routine

  Calling sequence of `fcn`:
+ ksp       - the ksp context being used.
. total_its - the total number of FGMRES iterations that have occurred.
. local_its - the number of FGMRES iterations since last restart.
. res_norm  - the current residual norm.
- ctx       - optional context variable

  Calling sequence of `destroy`:
. ctx - the function context

  Options Database Keys:
+ -ksp_fgmres_modifypcnochange - do not change the `PC`
- -ksp_fgmres_modifypcksp      - changes the inner KSP solver tolerances

  Level: intermediate

  Note:
  Several `fcn` routines are predefined, including `KSPFGMRESModifyPCNoChange()`, and  `KSPFGMRESModifyPCKSP()`

.seealso: [](ch_ksp), [](sec_flexibleksp), `KSPFGMRES`, `KSPFGMRESModifyPCNoChange()`, `KSPFGMRESModifyPCKSP()`
@*/
PetscErrorCode KSPFGMRESSetModifyPC(KSP ksp, PetscErrorCode (*fcn)(KSP ksp, PetscInt total_its, PetscInt local_its, PetscReal res_norm, void *ctx), void *ctx, PetscErrorCode (*destroy)(void *ctx))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscTryMethod(ksp, "KSPFGMRESSetModifyPC_C", (KSP, PetscErrorCode (*)(KSP, PetscInt, PetscInt, PetscReal, void *), void *, PetscErrorCode (*)(void *)), (ksp, fcn, ctx, destroy));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  KSPFGMRESModifyPCNoChange - this is the default used by `KSPFMGMRES` - it doesn't change the preconditioner. [](sec_flexibleksp)

  Input Parameters:
+ ksp       - the ksp context being used.
. total_its - the total number of `KSPFGMRES` iterations that have occurred.
. loc_its   - the number of `KSPFGMRES` iterations since last restart.
. res_norm  - the current residual norm.
- ctx       - context variable, unused in this routine

  Level: intermediate

.seealso: [](ch_ksp), [](sec_flexibleksp), `KSPFGMRES`, `KSPFGMRESSetModifyPC()`, `KSPFGMRESModifyPCKSP()`
@*/
PetscErrorCode KSPFGMRESModifyPCNoChange(KSP ksp, PetscInt total_its, PetscInt loc_its, PetscReal res_norm, void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  KSPFGMRESModifyPCKSP - modifies the attributes of the `KSPFGMRES` preconditioner, see [](sec_flexibleksp).

  Input Parameters:
+ ksp       - the ksp context being used.
. total_its - the total number of `KSPFGMRES` iterations that have occurred.
. loc_its   - the number of `KSPFGMRES` iterations since last restart.
. res_norm  - the current residual norm.
- ctx       - context, not used in this routine

  Level: intermediate

  Note:
  You can use this as a template for writing a custom modification callback

.seealso: [](ch_ksp), [](sec_flexibleksp), `KSPFGMRES`, `KSPFGMRESSetModifyPC()`
@*/
PetscErrorCode KSPFGMRESModifyPCKSP(KSP ksp, PetscInt total_its, PetscInt loc_its, PetscReal res_norm, void *ctx)
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
