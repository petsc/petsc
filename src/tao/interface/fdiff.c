#include <petsctao.h>         /*I  "petsctao.h"  I*/
#include <petsc/private/taoimpl.h>
#include <petscsnes.h>
#include <petscdmshell.h>

/*
   For finited difference computations of the Hessian, we use PETSc's SNESComputeJacobianDefault
*/
static PetscErrorCode Fsnes(SNES snes,Vec X,Vec G,void* ctx)
{
  Tao            tao = (Tao)ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,4);
  CHKERRQ(TaoComputeGradient(tao,X,G));
  PetscFunctionReturn(0);
}

/*@C
  TaoDefaultComputeGradient - computes the gradient using finite differences.

  Collective on Tao

  Input Parameters:
+ tao   - the Tao context
. X     - compute gradient at this point
- dummy - not used

  Output Parameter:
. G - Gradient Vector

  Options Database Key:
+ -tao_fd_gradient      - activates TaoDefaultComputeGradient()
- -tao_fd_delta <delta> - change in X used to calculate finite differences

  Level: advanced

  Notes:
  This routine is slow and expensive, and is not currently optimized
  to take advantage of sparsity in the problem.  Although
  TaoDefaultComputeGradient is not recommended for general use
  in large-scale applications, It can be useful in checking the
  correctness of a user-provided gradient.  Use the tao method TAOTEST
  to get an indication of whether your gradient is correct.
  This finite difference gradient evaluation can be set using the routine TaoSetGradient() or by using the command line option -tao_fd_gradient

.seealso: TaoSetGradient()
@*/
PetscErrorCode TaoDefaultComputeGradient(Tao tao,Vec Xin,Vec G,void *dummy)
{
  Vec            X;
  PetscScalar    *g;
  PetscReal      f, f2;
  PetscInt       low,high,N,i;
  PetscBool      flg;
  PetscReal      h=.5*PETSC_SQRT_MACHINE_EPSILON;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsGetReal(((PetscObject)tao)->options,((PetscObject)tao)->prefix,"-tao_fd_delta",&h,&flg));
  CHKERRQ(VecDuplicate(Xin,&X));
  CHKERRQ(VecCopy(Xin,X));
  CHKERRQ(VecGetSize(X,&N));
  CHKERRQ(VecGetOwnershipRange(X,&low,&high));
  CHKERRQ(VecSetOption(X,VEC_IGNORE_OFF_PROC_ENTRIES,PETSC_TRUE));
  CHKERRQ(VecGetArray(G,&g));
  for (i=0;i<N;i++) {
    CHKERRQ(VecSetValue(X,i,-h,ADD_VALUES));
    CHKERRQ(VecAssemblyBegin(X));
    CHKERRQ(VecAssemblyEnd(X));
    CHKERRQ(TaoComputeObjective(tao,X,&f));
    CHKERRQ(VecSetValue(X,i,2.0*h,ADD_VALUES));
    CHKERRQ(VecAssemblyBegin(X));
    CHKERRQ(VecAssemblyEnd(X));
    CHKERRQ(TaoComputeObjective(tao,X,&f2));
    CHKERRQ(VecSetValue(X,i,-h,ADD_VALUES));
    CHKERRQ(VecAssemblyBegin(X));
    CHKERRQ(VecAssemblyEnd(X));
    if (i>=low && i<high) {
      g[i-low]=(f2-f)/(2.0*h);
    }
  }
  CHKERRQ(VecRestoreArray(G,&g));
  CHKERRQ(VecDestroy(&X));
  PetscFunctionReturn(0);
}

/*@C
   TaoDefaultComputeHessian - Computes the Hessian using finite differences.

   Collective on Tao

   Input Parameters:
+  tao   - the Tao context
.  V     - compute Hessian at this point
-  dummy - not used

   Output Parameters:
+  H - Hessian matrix (not altered in this routine)
-  B - newly computed Hessian matrix to use with preconditioner (generally the same as H)

   Options Database Key:
.  -tao_fd_hessian - activates TaoDefaultComputeHessian()

   Level: advanced

   Notes:
   This routine is slow and expensive, and is not currently optimized
   to take advantage of sparsity in the problem.  Although
   TaoDefaultComputeHessian() is not recommended for general use
   in large-scale applications, It can be useful in checking the
   correctness of a user-provided Hessian.

.seealso: TaoSetHessian(), TaoDefaultComputeHessianColor(), SNESComputeJacobianDefault(), TaoSetGradient(), TaoDefaultComputeGradient()
@*/
PetscErrorCode TaoDefaultComputeHessian(Tao tao,Vec V,Mat H,Mat B,void *dummy)
{
  SNES           snes;
  DM             dm;

  PetscFunctionBegin;
  CHKERRQ(PetscInfo(tao,"TAO Using finite differences w/o coloring to compute Hessian matrix\n"));
  CHKERRQ(SNESCreate(PetscObjectComm((PetscObject)H),&snes));
  CHKERRQ(SNESSetFunction(snes,NULL,Fsnes,tao));
  CHKERRQ(SNESGetDM(snes,&dm));
  CHKERRQ(DMShellSetGlobalVector(dm,V));
  CHKERRQ(SNESSetUp(snes));
  if (H) {
    PetscInt n,N;

    CHKERRQ(VecGetSize(V,&N));
    CHKERRQ(VecGetLocalSize(V,&n));
    CHKERRQ(MatSetSizes(H,n,n,N,N));
    CHKERRQ(MatSetUp(H));
  }
  if (B && B != H) {
    PetscInt n,N;

    CHKERRQ(VecGetSize(V,&N));
    CHKERRQ(VecGetLocalSize(V,&n));
    CHKERRQ(MatSetSizes(B,n,n,N,N));
    CHKERRQ(MatSetUp(B));
  }
  CHKERRQ(SNESComputeJacobianDefault(snes,V,H,B,NULL));
  CHKERRQ(SNESDestroy(&snes));
  PetscFunctionReturn(0);
}

/*@C
   TaoDefaultComputeHessianColor - Computes the Hessian using colored finite differences.

   Collective on Tao

   Input Parameters:
+  tao - the Tao context
.  V   - compute Hessian at this point
-  ctx - the PetscColoring object (must be of type MatFDColoring)

   Output Parameters:
+  H - Hessian matrix (not altered in this routine)
-  B - newly computed Hessian matrix to use with preconditioner (generally the same as H)

   Level: advanced

.seealso: TaoSetHessian(), TaoDefaultComputeHessian(),SNESComputeJacobianDefaultColor(), TaoSetGradient()
@*/
PetscErrorCode TaoDefaultComputeHessianColor(Tao tao,Vec V,Mat H,Mat B,void *ctx)
{
  MatFDColoring       coloring = (MatFDColoring)ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(coloring,MAT_FDCOLORING_CLASSID,5);
  CHKERRQ(PetscInfo(tao,"TAO computing matrix using finite differences Hessian and coloring\n"));
  CHKERRQ(MatFDColoringApply(B,coloring,V,ctx));
  if (H != B) {
    CHKERRQ(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TaoDefaultComputeHessianMFFD(Tao tao,Vec X,Mat H,Mat B,void *ctx)
{
  PetscInt       n,N;
  PetscBool      assembled;

  PetscFunctionBegin;
  PetscCheck(!B || B == H,PetscObjectComm((PetscObject)tao),PETSC_ERR_SUP,"Preconditioning Hessian matrix");
  CHKERRQ(MatAssembled(H, &assembled));
  if (!assembled) {
    CHKERRQ(VecGetSize(X,&N));
    CHKERRQ(VecGetLocalSize(X,&n));
    CHKERRQ(MatSetSizes(H,n,n,N,N));
    CHKERRQ(MatSetType(H,MATMFFD));
    CHKERRQ(MatSetUp(H));
    CHKERRQ(MatMFFDSetFunction(H,(PetscErrorCode (*)(void*,Vec,Vec))TaoComputeGradient,tao));
  }
  CHKERRQ(MatMFFDSetBase(H,X,NULL));
  CHKERRQ(MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}
