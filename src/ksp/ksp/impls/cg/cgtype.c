
#include <../src/ksp/ksp/impls/cg/cgimpl.h>       /*I "petscksp.h" I*/

/*@
    KSPCGSetType - Sets the variant of the conjugate gradient method to
    use for solving a linear system with a complex coefficient matrix.
    This option is irrelevant when solving a real system.

    Logically Collective on ksp

    Input Parameters:
+   ksp - the iterative context
-   type - the variant of CG to use, one of
.vb
      KSP_CG_HERMITIAN - complex, Hermitian matrix (default)
      KSP_CG_SYMMETRIC - complex, symmetric matrix
.ve

    Level: intermediate

    Options Database Keys:
+   -ksp_cg_hermitian - Indicates Hermitian matrix
-   -ksp_cg_symmetric - Indicates symmetric matrix

    Note:
    By default, the matrix is assumed to be complex, Hermitian.

.seealso: KSP, KSPCG
@*/
PetscErrorCode  KSPCGSetType(KSP ksp,KSPCGType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscCall(PetscTryMethod(ksp,"KSPCGSetType_C",(KSP,KSPCGType),(ksp,type)));
  PetscFunctionReturn(0);
}

/*@
    KSPCGUseSingleReduction - Merge the two inner products needed in CG into a single MPI_Allreduce() call.

    Logically Collective on ksp

    Input Parameters:
+   ksp - the iterative context
-   flg - turn on or off the single reduction

    Options Database:
.   -ksp_cg_single_reduction <bool> - Merge inner products into single MPI_Allreduce

    Level: intermediate

     The algorithm used in this case is described as Method 1 in Lapack Working Note 56, "Conjugate Gradient Algorithms with Reduced Synchronization Overhead
     Distributed Memory Multiprocessors", by E. F. D'Azevedo, V. L. Eijkhout, and C. H. Romine, December 3, 1999. V. Eijkhout credits the algorithm
     initially to Chronopoulos and Gear.

     It requires two extra work vectors than the conventional implementation in PETSc.

     See also KSPPIPECG, KSPPIPECR, and KSPGROPPCG that use non-blocking reductions.

.seealso: KSP, KSPCG, KSPGMRES
@*/
PetscErrorCode  KSPCGUseSingleReduction(KSP ksp,PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveBool(ksp,flg,2);
  PetscCall(PetscTryMethod(ksp,"KSPCGUseSingleReduction_C",(KSP,PetscBool),(ksp,flg)));
  PetscFunctionReturn(0);
}

/*@
    KSPCGSetRadius - Sets the radius of the trust region.

    Logically Collective on ksp

    Input Parameters:
+   ksp    - the iterative context
-   radius - the trust region radius (Infinity is the default)

    Level: advanced

.seealso: KSP, KSPCG, KSPNASH, KSPSTCG, KSPGLTR
@*/
PetscErrorCode  KSPCGSetRadius(KSP ksp, PetscReal radius)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidLogicalCollectiveReal(ksp,radius,2);
  PetscCall(PetscTryMethod(ksp,"KSPCGSetRadius_C",(KSP,PetscReal),(ksp,radius)));
  PetscFunctionReturn(0);
}

/*@
    KSPCGGetNormD - Got norm of the direction.

    Collective on ksp

    Input Parameters:
+   ksp    - the iterative context
-   norm_d - the norm of the direction

    Level: advanced

.seealso: KSP, KSPCG, KSPNASH, KSPSTCG, KSPGLTR
@*/
PetscErrorCode  KSPCGGetNormD(KSP ksp, PetscReal *norm_d)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscCall(PetscUseMethod(ksp,"KSPCGGetNormD_C",(KSP,PetscReal*),(ksp,norm_d)));
  PetscFunctionReturn(0);
}

/*@
    KSPCGGetObjFcn - Get objective function value.

    Collective on ksp

    Input Parameters:
+   ksp   - the iterative context
-   o_fcn - the objective function value

    Level: advanced

.seealso: KSP, KSPCG, KSPNASH, KSPSTCG, KSPGLTR
@*/
PetscErrorCode  KSPCGGetObjFcn(KSP ksp, PetscReal *o_fcn)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscCall(PetscUseMethod(ksp,"KSPCGGetObjFcn_C",(KSP,PetscReal*),(ksp,o_fcn)));
  PetscFunctionReturn(0);
}
