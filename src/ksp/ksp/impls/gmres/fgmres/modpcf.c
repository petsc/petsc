
#include <petsc/private/kspimpl.h> /*I "petscksp.h" I*/

/*@C
   KSPFGMRESSetModifyPC - Sets the routine used by `KSPFGMRES` to modify the preconditioner. [](sec_flexibleksp)

   Logically Collective

   Input Parameters:
+  ksp - iterative context obtained from `KSPCreate()`
.  fcn - modifypc function
.  ctx - optional context
-  d - optional context destroy routine

   Calling Sequence of `function`:
$    PetscErrorCode fcn(KSP ksp, PetscInt total_its, PetscInt loc_its, PetscReal res_norm, void *ctx);
+    ksp - the ksp context being used.
.    total_its     - the total number of FGMRES iterations that have occurred.
.    loc_its       - the number of FGMRES iterations since last restart.
.    res_norm      - the current residual norm.
-    ctx           - optional context variable

   Calling Sequence of `d`:
$ PetscErrorCode d(void *ctx)

   Options Database Keys:
+   -ksp_fgmres_modifypcnochange - do not change the `PC`
-   -ksp_fgmres_modifypcksp - changes the inner KSP solver tolerances

   Level: intermediate

   Note:
   Several modifypc routines are predefined, including  `KSPFGMRESModifyPCNoChange()`, and  `KSPFGMRESModifyPCKSP()`

.seealso: [](chapter_ksp), [](sec_flexibleksp), `KSPFGMRES`, `KSPFGMRESModifyPCNoChange()`, `KSPFGMRESModifyPCKSP()`
@*/
PetscErrorCode KSPFGMRESSetModifyPC(KSP ksp, PetscErrorCode (*fcn)(KSP, PetscInt, PetscInt, PetscReal, void *), void *ctx, PetscErrorCode (*d)(void *))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscTryMethod(ksp, "KSPFGMRESSetModifyPC_C", (KSP, PetscErrorCode(*)(KSP, PetscInt, PetscInt, PetscReal, void *), void *, PetscErrorCode (*)(void *)), (ksp, fcn, ctx, d));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPFGMRESModifyPCNoChange - this is the default used by `KSPFMGMRES` - it doesn't change the preconditioner. [](sec_flexibleksp)

  Input Parameters:
+    ksp - the ksp context being used.
.    total_its     - the total number of `KSPFGMRES` iterations that have occurred.
.    loc_its       - the number of `KSPFGMRES` iterations since last restart.
                    a restart (so number of Krylov directions to be computed)
.    res_norm      - the current residual norm.
-    dummy         - context variable, unused in this routine

   Level: intermediate

.seealso: [](chapter_ksp), [](sec_flexibleksp), `KSPFGMRES`, `KSPFGMRESSetModifyPC()`, `KSPFGMRESModifyPCKSP()`
@*/
PetscErrorCode KSPFGMRESModifyPCNoChange(KSP ksp, PetscInt total_its, PetscInt loc_its, PetscReal res_norm, void *dummy)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
     KSPFGMRESModifyPCKSP - modifies the attributes of the `KSPFGMRES` preconditioner. [](sec_flexibleksp). It serves as an example (not as something useful in practice)

  Input Parameters:
+    ksp - the ksp context being used.
.    total_its     - the total number of `KSPFGMRES` iterations that have occurred.
.    loc_its       - the number of `KSPFGMRES` iterations since last restart.
.    res_norm      - the current residual norm.
-    dummy         - context, not used here

   Level: intermediate

   Note:
    You can use this as a template for writing a custom monification callback

.seealso: [](chapter_ksp), [](sec_flexibleksp), `KSPFGMRES`, `KSPFGMRESSetModifyPC()`, `KSPFGMRESModifyPCKSP()`
@*/
PetscErrorCode KSPFGMRESModifyPCKSP(KSP ksp, PetscInt total_its, PetscInt loc_its, PetscReal res_norm, void *dummy)
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
