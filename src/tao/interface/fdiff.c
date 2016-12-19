#include <petsctao.h>         /*I  "petsctao.h"  I*/
#include <petsc/private/taoimpl.h>
#include <petscsnes.h>

/*
   For finited difference computations of the Hessian, we use PETSc's SNESComputeJacobianDefault
*/

static PetscErrorCode Fsnes(SNES snes ,Vec X,Vec G,void*ctx)
{
  PetscErrorCode ierr;
  Tao            tao = (Tao)ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ctx,TAO_CLASSID,4);
  ierr=TaoComputeGradient(tao,X,G);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  TaoDefaultComputeGradient - computes the gradient using finite differences.

  Collective on Tao

  Input Parameters:
+ tao - the Tao context
. X - compute gradient at this point
- dummy - not used

  Output Parameters:
. G - Gradient Vector

   Options Database Key:
+  -tao_fd_gradient - Activates TaoDefaultComputeGradient()
-  -tao_fd_delta <delta> - change in x used to calculate finite differences

   Level: advanced

   Note:
   This routine is slow and expensive, and is not currently optimized
   to take advantage of sparsity in the problem.  Although
   TaoAppDefaultComputeGradient is not recommended for general use
   in large-scale applications, It can be useful in checking the
   correctness of a user-provided gradient.  Use the tao method TAOTEST
   to get an indication of whether your gradient is correct.


   Note:
   This finite difference gradient evaluation can be set using the routine TaoSetGradientRoutine() or by using the command line option -tao_fd_gradient

.seealso: TaoSetGradientRoutine()

@*/
PetscErrorCode TaoDefaultComputeGradient(Tao tao,Vec X,Vec G,void *dummy)
{
  PetscScalar    *x,*g;
  PetscReal      f, f2;
  PetscErrorCode ierr;
  PetscInt       low,high,N,i;
  PetscBool      flg;
  PetscReal      h=.5*PETSC_SQRT_MACHINE_EPSILON;

  PetscFunctionBegin;
  ierr = PetscOptionsGetReal(((PetscObject)tao)->options,((PetscObject)tao)->prefix,"-tao_fd_delta",&h,&flg);CHKERRQ(ierr);
  ierr = VecGetSize(X,&N);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(X,&low,&high);CHKERRQ(ierr);
  ierr = VecGetArray(G,&g);CHKERRQ(ierr);
  for (i=0;i<N;i++) {
    if (i>=low && i<high) {
      ierr = VecGetArray(X,&x);CHKERRQ(ierr);
      x[i-low] -= h;
      ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
    }

    ierr = TaoComputeObjective(tao, X,&f);CHKERRQ(ierr);

    if (i>=low && i<high) {
      ierr = VecGetArray(X,&x);CHKERRQ(ierr);
      x[i-low] += 2*h;
      ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
    }

    ierr = TaoComputeObjective(tao,X,&f2);CHKERRQ(ierr);

    if (i>=low && i<high) {
      ierr = VecGetArray(X,&x);CHKERRQ(ierr);
      x[i-low] -= h;
      ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
    }
    if (i>=low && i<high) {
      g[i-low]=(f2-f)/(2.0*h);
    }
  }
  ierr = VecRestoreArray(G,&g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TaoDefaultComputeHessian - Computes the Hessian using finite differences.

   Collective on Tao

   Input Parameters:
+  tao - the Tao context
.  V - compute Hessian at this point
-  dummy - not used

   Output Parameters:
+  H - Hessian matrix (not altered in this routine)
-  B - newly computed Hessian matrix to use with preconditioner (generally the same as H)

   Options Database Key:
+  -tao_fd - Activates TaoDefaultComputeHessian()
-  -tao_view_hessian - view the hessian after each evaluation using PETSC_VIEWER_STDOUT_WORLD

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
  PetscErrorCode       ierr;
  MPI_Comm             comm;
  Vec                  G;
  SNES                 snes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(V,VEC_CLASSID,2);
  ierr = VecDuplicate(V,&G);CHKERRQ(ierr);

  ierr = PetscInfo(tao,"TAO Using finite differences w/o coloring to compute Hessian matrix\n");CHKERRQ(ierr);

  ierr = TaoComputeGradient(tao,V,G);CHKERRQ(ierr);

  ierr = PetscObjectGetComm((PetscObject)H,&comm);CHKERRQ(ierr);
  ierr = SNESCreate(comm,&snes);CHKERRQ(ierr);

  ierr = SNESSetFunction(snes,G,Fsnes,tao);CHKERRQ(ierr);
  ierr = SNESComputeJacobianDefault(snes,V,H,B,tao);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = VecDestroy(&G);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TaoDefaultComputeHessianColor - Computes the Hessian using colored finite differences.

   Collective on Tao

   Input Parameters:
+  tao - the Tao context
.  V - compute Hessian at this point
-  ctx - the PetscColoring object (must be of type MatFDColoring)

   Output Parameters:
+  H - Hessian matrix (not altered in this routine)
-  B - newly computed Hessian matrix to use with preconditioner (generally the same as H)

   Level: advanced


.seealso: TaoSetHessianRoutine(), TaoDefaultComputeHessian(),SNESComputeJacobianDefaultColor(), TaoSetGradientRoutine()

@*/
PetscErrorCode TaoDefaultComputeHessianColor(Tao tao, Vec V, Mat H,Mat B,void *ctx)
{
  PetscErrorCode      ierr;
  MatFDColoring       coloring = (MatFDColoring)ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ctx,MAT_FDCOLORING_CLASSID,6);
  ierr=PetscInfo(tao,"TAO computing matrix using finite differences Hessian and coloring\n");CHKERRQ(ierr);
  ierr = MatFDColoringApply(B,coloring,V,ctx);CHKERRQ(ierr);
  if (H != B) {
    ierr = MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


