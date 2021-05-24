#include <petsctao.h>         /*I  "petsctao.h"  I*/
#include <petsc/private/taoimpl.h>
#include <petscsnes.h>
#include <petscdmshell.h>

/*
   For finited difference computations of the Hessian, we use PETSc's SNESComputeJacobianDefault
*/
static PetscErrorCode Fsnes(SNES snes,Vec X,Vec G,void* ctx)
{
  PetscErrorCode ierr;
  Tao            tao = (Tao)ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,4);
  ierr = TaoComputeGradient(tao,X,G);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  TaoDefaultComputeGradient - computes the gradient using finite differences.

  Collective on Tao

  Input Parameters:
+ tao   - the Tao context
. X     - compute gradient at this point
- dummy - not used

  Output Parameters:
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
  This finite difference gradient evaluation can be set using the routine TaoSetGradientRoutine() or by using the command line option -tao_fd_gradient

.seealso: TaoSetGradientRoutine()
@*/
PetscErrorCode TaoDefaultComputeGradient(Tao tao,Vec Xin,Vec G,void *dummy)
{
  Vec            X;
  PetscScalar    *g;
  PetscReal      f, f2;
  PetscErrorCode ierr;
  PetscInt       low,high,N,i;
  PetscBool      flg;
  PetscReal      h=.5*PETSC_SQRT_MACHINE_EPSILON;

  PetscFunctionBegin;
  ierr = PetscOptionsGetReal(((PetscObject)tao)->options,((PetscObject)tao)->prefix,"-tao_fd_delta",&h,&flg);CHKERRQ(ierr);
  ierr = VecDuplicate(Xin,&X);CHKERRQ(ierr);
  ierr = VecCopy(Xin,X);CHKERRQ(ierr);
  ierr = VecGetSize(X,&N);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(X,&low,&high);CHKERRQ(ierr);
  ierr = VecSetOption(X,VEC_IGNORE_OFF_PROC_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
  ierr = VecGetArray(G,&g);CHKERRQ(ierr);
  for (i=0;i<N;i++) {
    ierr = VecSetValue(X,i,-h,ADD_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(X);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(X);CHKERRQ(ierr);
    ierr = TaoComputeObjective(tao,X,&f);CHKERRQ(ierr);
    ierr = VecSetValue(X,i,2.0*h,ADD_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(X);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(X);CHKERRQ(ierr);
    ierr = TaoComputeObjective(tao,X,&f2);CHKERRQ(ierr);
    ierr = VecSetValue(X,i,-h,ADD_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(X);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(X);CHKERRQ(ierr);
    if (i>=low && i<high) {
      g[i-low]=(f2-f)/(2.0*h);
    }
  }
  ierr = VecRestoreArray(G,&g);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
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

.seealso: TaoSetHessianRoutine(), TaoDefaultComputeHessianColor(), SNESComputeJacobianDefault(), TaoSetGradientRoutine(), TaoDefaultComputeGradient()
@*/
PetscErrorCode TaoDefaultComputeHessian(Tao tao,Vec V,Mat H,Mat B,void *dummy)
{
  PetscErrorCode ierr;
  SNES           snes;
  DM             dm;

  PetscFunctionBegin;
  ierr = PetscInfo(tao,"TAO Using finite differences w/o coloring to compute Hessian matrix\n");CHKERRQ(ierr);
  ierr = SNESCreate(PetscObjectComm((PetscObject)H),&snes);CHKERRQ(ierr);
  ierr = SNESSetFunction(snes,NULL,Fsnes,tao);CHKERRQ(ierr);
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMShellSetGlobalVector(dm,V);CHKERRQ(ierr);
  ierr = SNESSetUp(snes);CHKERRQ(ierr);
  if (H) {
    PetscInt n,N;

    ierr = VecGetSize(V,&N);CHKERRQ(ierr);
    ierr = VecGetLocalSize(V,&n);CHKERRQ(ierr);
    ierr = MatSetSizes(H,n,n,N,N);CHKERRQ(ierr);
    ierr = MatSetUp(H);CHKERRQ(ierr);
  }
  if (B && B != H) {
    PetscInt n,N;

    ierr = VecGetSize(V,&N);CHKERRQ(ierr);
    ierr = VecGetLocalSize(V,&n);CHKERRQ(ierr);
    ierr = MatSetSizes(B,n,n,N,N);CHKERRQ(ierr);
    ierr = MatSetUp(B);CHKERRQ(ierr);
  }
  ierr = SNESComputeJacobianDefault(snes,V,H,B,NULL);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
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

.seealso: TaoSetHessianRoutine(), TaoDefaultComputeHessian(),SNESComputeJacobianDefaultColor(), TaoSetGradientRoutine()
@*/
PetscErrorCode TaoDefaultComputeHessianColor(Tao tao,Vec V,Mat H,Mat B,void *ctx)
{
  PetscErrorCode      ierr;
  MatFDColoring       coloring = (MatFDColoring)ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(coloring,MAT_FDCOLORING_CLASSID,5);
  ierr = PetscInfo(tao,"TAO computing matrix using finite differences Hessian and coloring\n");CHKERRQ(ierr);
  ierr = MatFDColoringApply(B,coloring,V,ctx);CHKERRQ(ierr);
  if (H != B) {
    ierr = MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TaoDefaultComputeHessianMFFD(Tao tao,Vec X,Mat H,Mat B,void *ctx)
{
  PetscInt       n,N;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (B && B != H) SETERRQ(PetscObjectComm((PetscObject)tao),PETSC_ERR_SUP,"Preconditioning Hessian matrix");
  ierr = VecGetSize(X,&N);CHKERRQ(ierr);
  ierr = VecGetLocalSize(X,&n);CHKERRQ(ierr);
  ierr = MatSetSizes(H,n,n,N,N);CHKERRQ(ierr);
  ierr = MatSetType(H,MATMFFD);CHKERRQ(ierr);
  ierr = MatSetUp(H);CHKERRQ(ierr);
  ierr = MatMFFDSetBase(H,X,NULL);CHKERRQ(ierr);
  ierr = MatMFFDSetFunction(H,(PetscErrorCode (*)(void*,Vec,Vec))TaoComputeGradient,tao);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
