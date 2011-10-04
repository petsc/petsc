#include "taosolver.h"         /*I  "taosolver.h"  I*/
#include "private/taosolver_impl.h"    
#include "petscsnes.h"


/* 
   For finited difference computations of the Hessian, we use PETSc's SNESDefaultComputeJacobian 
*/

#undef __FUNCT__  
#define __FUNCT__ "Fsnes"
static PetscErrorCode Fsnes(SNES snes ,Vec X,Vec G,void*ctx){
  PetscErrorCode ierr;
  TaoSolver tao = (TaoSolver)ctx;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ctx,TAOSOLVER_CLASSID,4);
  ierr=TaoComputeGradient(tao,X,G); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TaoDefaultComputeGradient"
/*@C
  TaoDefaultComputeGradient - computes the gradient using finite differences.
 
  Collective on TaoSolver

  Input Parameters:
+ tao - the TaoSolver context
. X - compute gradient at this point
- dummy - not used

  Output Parameters:
. G - Gradient Vector

   Options Database Key:
+  -tao_fd_gradient - Activates TaoDefaultComputeGradient()
-  -tao_fd_delta <delta> - change in x used to calculate finite differences




   Level: intermediate

   Note:
   This routine is slow and expensive, and is not currently optimized
   to take advantage of sparsity in the problem.  Although
   TaoAppDefaultComputeGradient is not recommended for general use
   in large-scale applications, It can be useful in checking the
   correctness of a user-provided gradient.  Use the tao method "tao_fd_test"
   to get an indication of whether your gradient is correct.


   Note:
   The gradient evaluation must be set using the routine TaoSetGradientRoutine().

.keywords: finite differences, Hessian

.seealso: TaoSetGradientRoutine()

@*/
PetscErrorCode TaoDefaultComputeGradient(TaoSolver tao,Vec X,Vec G,void *dummy) 
{
  PetscReal *g;
  PetscReal f, f2;
  PetscErrorCode ierr;
  PetscInt low,high,N,i;
  PetscBool flg;
  PetscReal h=PETSC_SQRT_MACHINE_EPSILON;
  PetscFunctionBegin;
  ierr = TaoComputeObjective(tao, X,&f); CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-tao_fd_delta",&h,&flg); CHKERRQ(ierr);
  ierr = VecGetSize(X,&N); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(X,&low,&high); CHKERRQ(ierr);
  ierr = VecGetArray(G,&g); CHKERRQ(ierr);
  for (i=0;i<N;i++) {
    printf("i=%d\n",i);
      ierr = VecSetValue(X,i,h,ADD_VALUES); CHKERRQ(ierr);
      ierr = VecAssemblyBegin(X); CHKERRQ(ierr);
      ierr = VecAssemblyEnd(X); CHKERRQ(ierr);

      ierr = TaoComputeObjective(tao,X,&f2); CHKERRQ(ierr);

      ierr = VecSetValue(X,i,-h,ADD_VALUES);
      ierr = VecAssemblyBegin(X); CHKERRQ(ierr);
      ierr = VecAssemblyEnd(X); CHKERRQ(ierr);
      
      if (i>=low && i<high) {
	  g[i-low]=(f2-f)/h;
      }
  }
  ierr = VecRestoreArray(G,&g); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TaoDefaultComputeHessian"
/*@C
   TaoDefaultComputeHessian - Computes the Hessian using finite differences. 

   Collective on TaoSolver

   Input Parameters:
+  tao - the TaoSolver context 
.  V - compute Hessian at this point
-  dummy - not used

   Output Parameters:
+  H - Hessian matrix (not altered in this routine)
.  B - newly computed Hessian matrix to use with preconditioner (generally the same as H)
-  flag - flag indicating whether the matrix sparsity structure has changed

   Options Database Key:
+  -tao_fd - Activates TaoDefaultComputeHessian()
-  -tao_view_hessian - view the hessian after each evaluation using PETSC_VIEWER_STDOUT_WORLD

   Level: intermediate

   Notes:
   This routine is slow and expensive, and is not currently optimized
   to take advantage of sparsity in the problem.  Although
   TaoAppDefaultComputeHessian() is not recommended for general use
   in large-scale applications, It can be useful in checking the
   correctness of a user-provided Hessian.

   Note:
   The gradient evaluation must be set using the routine TaoSetGradientRoutine().

.keywords: finite differences, Hessian

.seealso: TaoSetHessianRoutine(), TaoDefaultComputeHessianColor(), SNESDefaultComputeJacobian(),
          TaoSetGradientRoutine(), TaoDefaultComputeGradient()

@*/
PetscErrorCode TaoDefaultComputeHessian(TaoSolver tao,Vec V,Mat *H,Mat *B,
			     MatStructure *flag,void *dummy){
  PetscErrorCode       ierr;
  MPI_Comm             comm;
  Vec                  G;
  SNES                 snes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(V,VEC_CLASSID,2);
  ierr = VecDuplicate(V,&G);CHKERRQ(ierr);

  ierr = PetscInfo(tao,"TAO Using finite differences w/o coloring to compute Hessian matrix\n"); CHKERRQ(ierr);

  ierr = TaoComputeGradient(tao,V,G); CHKERRQ(ierr);

  ierr = PetscObjectGetComm((PetscObject)(*H),&comm);CHKERRQ(ierr);
  ierr = SNESCreate(comm,&snes);CHKERRQ(ierr);

  ierr = SNESSetFunction(snes,G,Fsnes,tao);CHKERRQ(ierr);
  ierr = SNESDefaultComputeJacobian(snes,V,H,B,flag,tao);CHKERRQ(ierr);

  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  
  ierr = VecDestroy(&G);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}




#undef __FUNCT__  
#define __FUNCT__ "TaoDefaultComputeHessianColor"
/*@C
   TaoDefaultComputeHessianColor - Computes the Hessian using colored finite differences. 

   Collective on TaoSolver

   Input Parameters:
+  tao - the TaoSolver context
.  V - compute Hessian at this point
-  ctx - the PetscColoring object

   Output Parameters:
+  H - Hessian matrix (not altered in this routine)
.  B - newly computed Hessian matrix to use with preconditioner (generally the same as H)
-  flag - flag indicating whether the matrix sparsity structure has changed

   Options Database Keys:
+  -mat_fd_coloring_freq <freq>
-  -tao_view_hessian - view the hessian after each evaluation using PETSC_VIEWER_STDOUT_WORLD

   Level: intermediate

   Note:
   The gradient evaluation must be set using the routine TaoSetPetscGradient().

 .keywords: finite differences, Hessian, coloring, sparse

.seealso: TaoSetHessianRoutine(), TaoDefaultComputeHessian(),SNESDefaultComputeJacobianColor(), 
          TaoSetGradientRoutine()

@*/
PetscErrorCode TaoDefaultComputeHessianColor(TaoSolver tao, Vec V, Mat *HH,Mat *BB,MatStructure *flag,void *ctx){
  PetscErrorCode      ierr;
  Mat                 H=*HH,B=*BB;
  MatFDColoring       coloring = (MatFDColoring)ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ctx,MAT_FDCOLORING_CLASSID,6);

  
  
  *flag = SAME_NONZERO_PATTERN;

  ierr=PetscInfo(tao,"TAO computing matrix using finite differences Hessian and coloring\n"); CHKERRQ(ierr);
  ierr = MatFDColoringApply(B,coloring,V,flag,ctx); CHKERRQ(ierr);

  if (H != B) {
      ierr = MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
      ierr = MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


