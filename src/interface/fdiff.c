#include "taosolver.h"         /*I  "taosolver.h"  I*/
#include "private/taosolver_impl.h"      /*I "private/taosolver_impl.h"  I*/
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
  ierr=TaoSolverComputeGradient(tao,X,G); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TaoSolverDefaultComputeGradient"
/*@C
  TaoSolverDefaultComputeGradient - computes the gradient using finite differences.
 
  Collective on TAO_APPLICATION

  Input Parameters:
+ taoapp - the TAO_APPLICATION context
. X - compute gradient at this point
- ctx - the TAO_APPLICATION structure, cast to (void*)

  Output Parameters:
. G - Gradient Vector

   Options Database Key:
+  -tao_fd_gradient - Activates TaoAppDefaultComputeGradient()
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
   The gradient evaluation must be set using the routine TaoAppSetGradientRoutine().

.keywords: TAO_APPLICATION, finite differences, Hessian

.seealso: TaoAppDefaultComputeGradient(),  TaoAppSetGradientRoutine()

@*/
PetscErrorCode TaoSolverDefaultComputeGradient(TaoSolver tao,Vec X,Vec G,void *dummy) 
{
  Vec TempX;
  PetscReal *g;
  PetscReal f, f2;
  PetscErrorCode ierr;
  PetscInt low,high,N,i;
  PetscBool flg;
  PetscReal h=1.0e-6;
  PetscFunctionBegin;
  ierr = TaoSolverComputeObjective(tao, X,&f); CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-tao_fd_delta",&h,&flg); CHKERRQ(ierr);
  ierr = VecDuplicate(X,&TempX); CHKERRQ(ierr);
  ierr = VecCopy(X,TempX); CHKERRQ(ierr);
  ierr = VecGetSize(X,&N); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(TempX,&low,&high); CHKERRQ(ierr);
  ierr = VecGetArray(G,&g); CHKERRQ(ierr);
  for (i=0;i<N;i++) {
      ierr = VecSetValue(TempX,i,h,ADD_VALUES); CHKERRQ(ierr);
      ierr = VecAssemblyBegin(TempX); CHKERRQ(ierr);
      ierr = VecAssemblyEnd(TempX); CHKERRQ(ierr);

      ierr = TaoSolverComputeObjective(tao,TempX,&f2); CHKERRQ(ierr);

      ierr = VecSetValue(TempX,i,-h,ADD_VALUES);
      ierr = VecAssemblyBegin(TempX); CHKERRQ(ierr);
      ierr = VecAssemblyEnd(TempX); CHKERRQ(ierr);
      
      if (i>=low && i<high) {
	  g[i-low]=(f2-f)/h;
      }
  }
  ierr = VecRestoreArray(G,&g); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TaoSolverDefaultComputeHessian"
/*@C
   TaoSolverDefaultComputeHessian - Computes the Hessian using finite differences. 

   Collective on TaoSolver

   Input Parameters:
+  tao - the TaoSolver context 
.  V - compute Hessian at this point
-  ctx - the TAO_APPLICATION structure, cast to (void*)

   Output Parameters:
+  H - Hessian matrix (not altered in this routine)
.  B - newly computed Hessian matrix to use with preconditioner (generally the same as H)
-  flag - flag indicating whether the matrix sparsity structure has changed

   Options Database Key:
+  -tao_fd - Activates TaoAppDefaultComputeHessian()
-  -tao_view_hessian - view the hessian after each evaluation using PETSC_VIEWER_STDOUT_WORLD

   Level: intermediate

   Notes:
   This routine is slow and expensive, and is not currently optimized
   to take advantage of sparsity in the problem.  Although
   TaoAppDefaultComputeHessian() is not recommended for general use
   in large-scale applications, It can be useful in checking the
   correctness of a user-provided Hessian.

   Note:
   The gradient evaluation must be set using the routine TaoAppSetGradientRoutine().

.keywords: TAO_APPLICATION, finite differences, Hessian

.seealso: TaoAppSetHessianRoutine(), TaoAppDefaultComputeHessianColor(), SNESDefaultComputeJacobian(),
          TaoAppSetGradientRoutine(), TaoAppDefaultComputeGradient()

@*/
PetscErrorCode TaoSolverDefaultComputeHessian(TaoSolver tao,Vec V,Mat *H,Mat *B,
			     MatStructure *flag,void *ctx){
  PetscErrorCode       ierr;
  MPI_Comm             comm;
  Vec                  G;
  SNES                 snes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(V,VEC_CLASSID,2);
  ierr = VecDuplicate(V,&G);CHKERRQ(ierr);

  ierr = PetscInfo(tao,"TAO Using finite differences w/o coloring to compute Hessian matrix\n"); CHKERRQ(ierr);

  ierr = TaoSolverComputeGradient(tao,V,G); CHKERRQ(ierr);

  ierr = PetscObjectGetComm((PetscObject)(*H),&comm);CHKERRQ(ierr);
  ierr = SNESCreate(comm,&snes);CHKERRQ(ierr);

  ierr = SNESSetFunction(snes,G,Fsnes,tao);CHKERRQ(ierr);
  ierr = SNESDefaultComputeJacobian(snes,V,H,B,flag,tao);CHKERRQ(ierr);

  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  
  ierr = VecDestroy(&G);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}




#undef __FUNCT__  
#define __FUNCT__ "TaoSolverDefaultComputeHessianColor"
/*@C
   TaoSolverDefaultComputeHessianColor - Computes the Hessian using colored finite differences. 

   Collective on TaoSolver

   Input Parameters:
+  tao - the TaoSolver context
.  V - compute Hessian at this point

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

 .keywords: TAO_APPLICATION, finite differences, Hessian, coloring, sparse

.seealso: TaoAppSetHessianRoutine(), TaoAppDefaultComputeHessian(),SNESDefaultComputeJacobianColor(), 
          TaoAppSetGradientRoutine(), TaoAppSetColoring()

@*/
PetscErrorCode TaoSolverDefaultComputeHessianColor(TaoSolver tao, Vec V, Mat *HH,Mat *BB,MatStructure *flag,void *ctx){
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



/*
#undef __FUNCT__  
#define __FUNCT__ "TaoSolverSetFiniteDifferencesOptions"
@C
  TaoSolverSetFiniteDifferencesOptions - Sets various TAO parameters from user options

   Collective on TAO_SOLVER

   Input Parameters:
+  tao - the TAO Solver

   Level: beginner

.keywords:  options, finite differences


 PetscErrorCode TaoSolverSetFiniteDifferencesOptions(TaoSolver tao){
  PetscErrorCode ierr;
  PetscBool flg;

  PetscFunctionBegin;

  PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);

  flg=PETSC_FALSE;
  ierr = PetscOptionsName("-tao_fd","use finite differences for Hessian","TaoSolverDefaultComputeHessian",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = TaoSolverSetHessianRoutine(tao,TaoSolverDefaultComputeHessian,(void*)tao);CHKERRQ(ierr);
    ierr = PetscInfo(tao,"Setting default finite difference Hessian matrix\n"); CHKERRQ(ierr);
  }

  flg=PETSC_FALSE;
.  ierr = PetscOptionsName("-tao_fdgrad","use finite differences for gradient","TaoSolverDefaultComputeGradient",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = TaoSolverSetGradientRoutine(tao,TaoSolverDefaultComputeGradient,(void*)tao);CHKERRQ(ierr);
    ierr = PetscInfo(tao,"Setting default finite difference gradient routine\n"); CHKERRQ(ierr);
  }


  flg=PETSC_FALSE;
  ierr = PetscOptionsName("-tao_fd_coloring","use finite differences with coloring to compute Hessian","TaoSolverDefaultComputeHessianColor",&flg);CHKERRQ(ierr);
  if (flg) {
    
    ierr = TaoSolverSetHessianRoutine(tao,TaoSolverDefaultComputeHessianColor,(void*)tao);CHKERRQ(ierr);
    ierr = PetscInfo(tao,"Use finite differencing with coloring to compute Hessian \n"); CHKERRQ(ierr);
  }
    
  PetscFunctionReturn(0);
}


static char TaoColoringXXX[] = "TaoColoring";

typedef struct {
  ISColoring coloring;
} TaoColorit;

#undef __FUNCT__  
#define __FUNCT__ "TaoAppDestroyColoringXXX"
static int TaoAppDestroyColoringXXX(void*ctx){
  PetscErrorCode ierr;
  TaoColorit *mctx=(TaoColorit*)ctx;
  PetscFunctionBegin;
  if (mctx){
//    if (mctx->coloring){  
//      ierr = ISColoringDestroy(mctx->coloring);CHKERRQ(ierr);
//    }
    ierr = PetscFree(mctx); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
*/
