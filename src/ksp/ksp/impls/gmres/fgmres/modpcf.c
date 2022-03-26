
#include <petsc/private/kspimpl.h>  /*I "petscksp.h" I*/

/*@C
   KSPFGMRESSetModifyPC - Sets the routine used by FGMRES to modify the preconditioner.

   Logically Collective on ksp

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate
.  fcn - modifypc function
.  ctx - optional context
-  d - optional context destroy routine

   Calling Sequence of function:
    PetscErrorCode fcn(KSP ksp,PetscInt total_its,PetscInt loc_its,PetscReal res_norm,void*ctx);

    ksp - the ksp context being used.
    total_its     - the total number of FGMRES iterations that have occurred.
    loc_its       - the number of FGMRES iterations since last restart.
    res_norm      - the current residual norm.
    ctx           - optional context variable

   Options Database Keys:
   -ksp_fgmres_modifypcnochange
   -ksp_fgmres_modifypcksp

   Level: intermediate

   Contributed by Allison Baker

   Notes:
   Several modifypc routines are predefined, including
    KSPFGMRESModifyPCNoChange()
    KSPFGMRESModifyPCKSP()

.seealso: KSPFGMRESModifyPCNoChange(), KSPFGMRESModifyPCKSP()

@*/
PetscErrorCode  KSPFGMRESSetModifyPC(KSP ksp,PetscErrorCode (*fcn)(KSP,PetscInt,PetscInt,PetscReal,void*),void *ctx,PetscErrorCode (*d)(void*))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscCall(PetscTryMethod(ksp,"KSPFGMRESSetModifyPC_C",(KSP,PetscErrorCode (*)(KSP,PetscInt,PetscInt,PetscReal,void*),void*,PetscErrorCode (*)(void*)),(ksp,fcn,ctx,d)));
  PetscFunctionReturn(0);
}

/* The following are different routines used to modify the preconditioner */

/*@

  KSPFGMRESModifyPCNoChange - this is the default used by fgmres - it doesn't change the preconditioner.

  Input Parameters:
+    ksp - the ksp context being used.
.    total_its     - the total number of FGMRES iterations that have occurred.
.    loc_its       - the number of FGMRES iterations since last restart.
                    a restart (so number of Krylov directions to be computed)
.    res_norm      - the current residual norm.
-    dummy         - context variable, unused in this routine

   Level: intermediate

   Contributed by Allison Baker

You can use this as a template!

.seealso: KSPFGMRESSetModifyPC(), KSPFGMRESModifyPCKSP()

@*/
PetscErrorCode  KSPFGMRESModifyPCNoChange(KSP ksp,PetscInt total_its,PetscInt loc_its,PetscReal res_norm,void *dummy)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/*@

 KSPFGMRESModifyPCKSP - modifies the attributes of the
     GMRES preconditioner.  It serves as an example (not as something
     useful!)

  Input Parameters:
+    ksp - the ksp context being used.
.    total_its     - the total number of FGMRES iterations that have occurred.
.    loc_its       - the number of FGMRES iterations since last restart.
.    res_norm      - the current residual norm.
-    dummy         - context, not used here

   Level: intermediate

   Contributed by Allison Baker

 This could be used as a template!

.seealso: KSPFGMRESSetModifyPC(), KSPFGMRESModifyPCKSP()

@*/
PetscErrorCode  KSPFGMRESModifyPCKSP(KSP ksp,PetscInt total_its,PetscInt loc_its,PetscReal res_norm,void *dummy)
{
  PC             pc;
  PetscInt       maxits;
  KSP            sub_ksp;
  PetscReal      rtol,abstol,dtol;
  PetscBool      isksp;

  PetscFunctionBegin;
  PetscCall(KSPGetPC(ksp,&pc));

  PetscCall(PetscObjectTypeCompare((PetscObject)pc,PCKSP,&isksp));
  if (isksp) {
    PetscCall(PCKSPGetKSP(pc,&sub_ksp));

    /* note that at this point you could check the type of KSP with KSPGetType() */

    /* Now we can use functions such as KSPGMRESSetRestart() or
      KSPGMRESSetOrthogonalization() or KSPSetTolerances() */

    PetscCall(KSPGetTolerances(sub_ksp,&rtol,&abstol,&dtol,&maxits));
    if (!loc_its) rtol = .1;
    else rtol *= .9;
    PetscCall(KSPSetTolerances(sub_ksp,rtol,abstol,dtol,maxits));
  }
  PetscFunctionReturn(0);
}
