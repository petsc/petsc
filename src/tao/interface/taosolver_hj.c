#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/

/*@C
   TaoSetHessian - Sets the function to compute the Hessian as well as the location to store the matrix.

   Logically collective on tao

   Input Parameters:
+  tao  - the Tao context
.  H    - Matrix used for the hessian
.  Hpre - Matrix that will be used operated on by preconditioner, can be same as H
.  func - Hessian evaluation routine
-  ctx  - [optional] user-defined context for private data for the
         Hessian evaluation routine (may be NULL)

   Calling sequence of func:
$    func(Tao tao,Vec x,Mat H,Mat Hpre,void *ctx);

+  tao  - the Tao  context
.  x    - input vector
.  H    - Hessian matrix
.  Hpre - preconditioner matrix, usually the same as H
-  ctx  - [optional] user-defined Hessian context

   Level: beginner

.seealso: `Tao`, `TaoTypes`, `TaoSetObjective()`, `TaoSetGradient()`, `TaoSetObjectiveAndGradient()`, `TaoGetHessian()`
@*/
PetscErrorCode TaoSetHessian(Tao tao, Mat H, Mat Hpre, PetscErrorCode (*func)(Tao, Vec, Mat, Mat, void*), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (H) {
    PetscValidHeaderSpecific(H,MAT_CLASSID,2);
    PetscCheckSameComm(tao,1,H,2);
  }
  if (Hpre) {
    PetscValidHeaderSpecific(Hpre,MAT_CLASSID,3);
    PetscCheckSameComm(tao,1,Hpre,3);
  }
  if (ctx) tao->user_hessP = ctx;
  if (func) tao->ops->computehessian = func;
  if (H) {
    PetscCall(PetscObjectReference((PetscObject)H));
    PetscCall(MatDestroy(&tao->hessian));
    tao->hessian = H;
  }
  if (Hpre) {
    PetscCall(PetscObjectReference((PetscObject)Hpre));
    PetscCall(MatDestroy(&tao->hessian_pre));
    tao->hessian_pre = Hpre;
  }
  PetscFunctionReturn(0);
}

/*@C
   TaoGetHessian - Gets the function to compute the Hessian as well as the location to store the matrix.

   Not collective

   Input Parameter:
.  tao  - the Tao context

   OutputParameters:
+  H    - Matrix used for the hessian
.  Hpre - Matrix that will be used operated on by preconditioner, can be the same as H
.  func - Hessian evaluation routine
-  ctx  - user-defined context for private data for the Hessian evaluation routine

   Calling sequence of func:
$    func(Tao tao,Vec x,Mat H,Mat Hpre,void *ctx);

+  tao  - the Tao  context
.  x    - input vector
.  H    - Hessian matrix
.  Hpre - preconditioner matrix, usually the same as H
-  ctx  - [optional] user-defined Hessian context

   Level: beginner

.seealso: `Tao`, TaoType`, `TaoGetObjective()`, `TaoGetGradient()`, `TaoGetObjectiveAndGradient()`, `TaoSetHessian()`
@*/
PetscErrorCode TaoGetHessian(Tao tao, Mat *H, Mat *Hpre, PetscErrorCode (**func)(Tao, Vec, Mat, Mat, void*), void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (H) *H = tao->hessian;
  if (Hpre) *Hpre = tao->hessian_pre;
  if (ctx) *ctx = tao->user_hessP;
  if (func) *func = tao->ops->computehessian;
  PetscFunctionReturn(0);
}

PetscErrorCode TaoTestHessian(Tao tao)
{
  Mat               A,B,C,D,hessian;
  Vec               x = tao->solution;
  PetscReal         nrm,gnorm;
  PetscReal         threshold = 1.e-5;
  PetscInt          m,n,M,N;
  PetscBool         complete_print = PETSC_FALSE,test = PETSC_FALSE,flg;
  PetscViewer       viewer,mviewer;
  MPI_Comm          comm;
  PetscInt          tabs;
  static PetscBool  directionsprinted = PETSC_FALSE;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscObjectOptionsBegin((PetscObject)tao);
  PetscCall(PetscOptionsName("-tao_test_hessian","Compare hand-coded and finite difference Hessians","None",&test));
  PetscCall(PetscOptionsReal("-tao_test_hessian", "Threshold for element difference between hand-coded and finite difference being meaningful","None",threshold,&threshold,NULL));
  PetscCall(PetscOptionsViewer("-tao_test_hessian_view","View difference between hand-coded and finite difference Hessians element entries","None",&mviewer,&format,&complete_print));
  PetscOptionsEnd();
  if (!test) PetscFunctionReturn(0);

  PetscCall(PetscObjectGetComm((PetscObject)tao,&comm));
  PetscCall(PetscViewerASCIIGetStdout(comm,&viewer));
  PetscCall(PetscViewerASCIIGetTab(viewer, &tabs));
  PetscCall(PetscViewerASCIISetTab(viewer, ((PetscObject)tao)->tablevel));
  PetscCall(PetscViewerASCIIPrintf(viewer,"  ---------- Testing Hessian -------------\n"));
  if (!complete_print && !directionsprinted) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Run with -tao_test_hessian_view and optionally -tao_test_hessian <threshold> to show difference\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer,"    of hand-coded and finite difference Hessian entries greater than <threshold>.\n"));
  }
  if (!directionsprinted) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Testing hand-coded Hessian, if (for double precision runs) ||J - Jfd||_F/||J||_F is\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer,"    O(1.e-8), the hand-coded Hessian is probably correct.\n"));
    directionsprinted = PETSC_TRUE;
  }
  if (complete_print) PetscCall(PetscViewerPushFormat(mviewer,format));

  PetscCall(PetscObjectTypeCompare((PetscObject)tao->hessian,MATMFFD,&flg));
  if (!flg) hessian = tao->hessian;
  else hessian = tao->hessian_pre;

  while (hessian) {
    PetscCall(PetscObjectBaseTypeCompareAny((PetscObject)hessian,&flg,MATSEQAIJ,MATMPIAIJ,MATSEQDENSE,MATMPIDENSE,MATSEQBAIJ,MATMPIBAIJ,MATSEQSBAIJ,MATMPIBAIJ,""));
    if (flg) {
      A    = hessian;
      PetscCall(PetscObjectReference((PetscObject)A));
    } else {
      PetscCall(MatComputeOperator(hessian,MATAIJ,&A));
    }

    PetscCall(MatCreate(PetscObjectComm((PetscObject)A),&B));
    PetscCall(MatGetSize(A,&M,&N));
    PetscCall(MatGetLocalSize(A,&m,&n));
    PetscCall(MatSetSizes(B,m,n,M,N));
    PetscCall(MatSetType(B,((PetscObject)A)->type_name));
    PetscCall(MatSetUp(B));
    PetscCall(MatSetOption(B,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));

    PetscCall(TaoDefaultComputeHessian(tao,x,B,B,NULL));

    PetscCall(MatDuplicate(B,MAT_COPY_VALUES,&D));
    PetscCall(MatAYPX(D,-1.0,A,DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatNorm(D,NORM_FROBENIUS,&nrm));
    PetscCall(MatNorm(A,NORM_FROBENIUS,&gnorm));
    PetscCall(MatDestroy(&D));
    if (!gnorm) gnorm = 1; /* just in case */
    PetscCall(PetscViewerASCIIPrintf(viewer,"  ||H - Hfd||_F/||H||_F = %g, ||H - Hfd||_F = %g\n",(double)(nrm/gnorm),(double)nrm));

    if (complete_print) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  Hand-coded Hessian ----------\n"));
      PetscCall(MatView(A,mviewer));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  Finite difference Hessian ----------\n"));
      PetscCall(MatView(B,mviewer));
    }

    if (complete_print) {
      PetscInt          Istart, Iend, *ccols, bncols, cncols, j, row;
      PetscScalar       *cvals;
      const PetscInt    *bcols;
      const PetscScalar *bvals;

      PetscCall(MatAYPX(B,-1.0,A,DIFFERENT_NONZERO_PATTERN));
      PetscCall(MatCreate(PetscObjectComm((PetscObject)A),&C));
      PetscCall(MatSetSizes(C,m,n,M,N));
      PetscCall(MatSetType(C,((PetscObject)A)->type_name));
      PetscCall(MatSetUp(C));
      PetscCall(MatSetOption(C,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));
      PetscCall(MatGetOwnershipRange(B,&Istart,&Iend));

      for (row = Istart; row < Iend; row++) {
        PetscCall(MatGetRow(B,row,&bncols,&bcols,&bvals));
        PetscCall(PetscMalloc2(bncols,&ccols,bncols,&cvals));
        for (j = 0, cncols = 0; j < bncols; j++) {
          if (PetscAbsScalar(bvals[j]) > threshold) {
            ccols[cncols] = bcols[j];
            cvals[cncols] = bvals[j];
            cncols += 1;
          }
        }
        if (cncols) {
          PetscCall(MatSetValues(C,1,&row,cncols,ccols,cvals,INSERT_VALUES));
        }
        PetscCall(MatRestoreRow(B,row,&bncols,&bcols,&bvals));
        PetscCall(PetscFree2(ccols,cvals));
      }
      PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  Finite-difference minus hand-coded Hessian with tolerance %g ----------\n",(double)threshold));
      PetscCall(MatView(C,mviewer));
      PetscCall(MatDestroy(&C));
    }
    PetscCall(MatDestroy(&A));
    PetscCall(MatDestroy(&B));

    if (hessian != tao->hessian_pre) {
      hessian = tao->hessian_pre;
      PetscCall(PetscViewerASCIIPrintf(viewer,"  ---------- Testing Hessian for preconditioner -------------\n"));
    } else hessian = NULL;
  }
  if (complete_print) {
    PetscCall(PetscViewerPopFormat(mviewer));
    PetscCall(PetscViewerDestroy(&mviewer));
  }
  PetscCall(PetscViewerASCIISetTab(viewer,tabs));
  PetscFunctionReturn(0);
}

/*@C
   TaoComputeHessian - Computes the Hessian matrix that has been
   set with `TaoSetHessian()`.

   Collective on tao

   Input Parameters:
+  tao - the Tao solver context
-  X   - input vector

   Output Parameters:
+  H    - Hessian matrix
-  Hpre - Preconditioning matrix

   Options Database Keys:
+     -tao_test_hessian - compare the user provided Hessian with one compute via finite differences to check for errors
.     -tao_test_hessian <numerical value>  - display entries in the difference between the user provided Hessian and finite difference Hessian that are greater than a certain value to help users detect errors
-     -tao_test_hessian_view - display the user provided Hessian, the finite difference Hessian and the difference between them to help users detect the location of errors in the user provided Hessian

   Notes:
   Most users should not need to explicitly call this routine, as it
   is used internally within the minimization solvers.

   `TaoComputeHessian()` is typically used within optimization algorithms,
   so most users would not generally call this routine
   themselves.

   Developer Note:
   The Hessian test mechanism follows `SNESTestJacobian()`.

   Level: developer

.seealso: `Tao`, `TaoComputeObjective()`, `TaoComputeObjectiveAndGradient()`, `TaoSetHessian()`
@*/
PetscErrorCode TaoComputeHessian(Tao tao, Vec X, Mat H, Mat Hpre)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidHeaderSpecific(X, VEC_CLASSID,2);
  PetscCheckSameComm(tao,1,X,2);
  PetscCheck(tao->ops->computehessian,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call TaoSetHessian() first");
  ++tao->nhess;
  PetscCall(VecLockReadPush(X));
  PetscCall(PetscLogEventBegin(TAO_HessianEval,tao,X,H,Hpre));
  PetscStackPush("Tao user Hessian function");
  PetscCall((*tao->ops->computehessian)(tao,X,H,Hpre,tao->user_hessP));
  PetscStackPop;
  PetscCall(PetscLogEventEnd(TAO_HessianEval,tao,X,H,Hpre));
  PetscCall(VecLockReadPop(X));

  PetscCall(TaoTestHessian(tao));
  PetscFunctionReturn(0);
}

/*@C
   TaoComputeJacobian - Computes the Jacobian matrix that has been
   set with TaoSetJacobianRoutine().

   Collective on tao

   Input Parameters:
+  tao - the Tao solver context
-  X   - input vector

   Output Parameters:
+  J    - Jacobian matrix
-  Jpre - Preconditioning matrix

   Notes:
   Most users should not need to explicitly call this routine, as it
   is used internally within the minimization solvers.

   `TaoComputeJacobian()` is typically used within minimization
   implementations, so most users would not generally call this routine
   themselves.

   Level: developer

.seealso: `TaoComputeObjective()`, `TaoComputeObjectiveAndGradient()`, `TaoSetJacobianRoutine()`
@*/
PetscErrorCode TaoComputeJacobian(Tao tao, Vec X, Mat J, Mat Jpre)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidHeaderSpecific(X, VEC_CLASSID,2);
  PetscCheckSameComm(tao,1,X,2);
  PetscCheck(tao->ops->computejacobian,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call TaoSetJacobian() first");
  ++tao->njac;
  PetscCall(VecLockReadPush(X));
  PetscCall(PetscLogEventBegin(TAO_JacobianEval,tao,X,J,Jpre));
  PetscStackPush("Tao user Jacobian function");
  PetscCall((*tao->ops->computejacobian)(tao,X,J,Jpre,tao->user_jacP));
  PetscStackPop;
  PetscCall(PetscLogEventEnd(TAO_JacobianEval,tao,X,J,Jpre));
  PetscCall(VecLockReadPop(X));
  PetscFunctionReturn(0);
}

/*@C
   TaoComputeResidualJacobian - Computes the least-squares residual Jacobian matrix that has been
   set with `TaoSetJacobianResidual()`.

   Collective on tao

   Input Parameters:
+  tao - the Tao solver context
-  X   - input vector

   Output Parameters:
+  J    - Jacobian matrix
-  Jpre - Preconditioning matrix

   Notes:
   Most users should not need to explicitly call this routine, as it
   is used internally within the minimization solvers.

   `TaoComputeResidualJacobian()` is typically used within least-squares
   implementations, so most users would not generally call this routine
   themselves.

   Level: developer

.seealso: `Tao`, `TaoComputeResidual()`, `TaoSetJacobianResidual()`
@*/
PetscErrorCode TaoComputeResidualJacobian(Tao tao, Vec X, Mat J, Mat Jpre)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidHeaderSpecific(X, VEC_CLASSID,2);
  PetscCheckSameComm(tao,1,X,2);
  PetscCheck(tao->ops->computeresidualjacobian,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call TaoSetResidualJacobian() first");
  ++tao->njac;
  PetscCall(VecLockReadPush(X));
  PetscCall(PetscLogEventBegin(TAO_JacobianEval,tao,X,J,Jpre));
  PetscStackPush("Tao user least-squares residual Jacobian function");
  PetscCall((*tao->ops->computeresidualjacobian)(tao,X,J,Jpre,tao->user_lsjacP));
  PetscStackPop;
  PetscCall(PetscLogEventEnd(TAO_JacobianEval,tao,X,J,Jpre));
  PetscCall(VecLockReadPop(X));
  PetscFunctionReturn(0);
}

/*@C
   TaoComputeJacobianState - Computes the Jacobian matrix that has been
   set with `TaoSetJacobianStateRoutine()`.

   Collective on tao

   Input Parameters:
+  tao - the Tao solver context
-  X   - input vector

   Output Parameters:
+  J    - Jacobian matrix
.  Jpre - Preconditioning matrix
-  Jinv - unknown

   Notes:
   Most users should not need to explicitly call this routine, as it
   is used internally within the optimization algorithms.

   Level: developer

.seealso: `Tao`, `TaoComputeObjective()`, `TaoComputeObjectiveAndGradient()`, `TaoSetJacobianStateRoutine()`, `TaoComputeJacobianDesign()`, `TaoSetStateDesignIS()`
@*/
PetscErrorCode TaoComputeJacobianState(Tao tao, Vec X, Mat J, Mat Jpre, Mat Jinv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidHeaderSpecific(X, VEC_CLASSID,2);
  PetscCheckSameComm(tao,1,X,2);
  PetscCheck(tao->ops->computejacobianstate,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call TaoSetJacobianState() first");
  ++tao->njac_state;
  PetscCall(VecLockReadPush(X));
  PetscCall(PetscLogEventBegin(TAO_JacobianEval,tao,X,J,Jpre));
  PetscStackPush("Tao user Jacobian(state) function");
  PetscCall((*tao->ops->computejacobianstate)(tao,X,J,Jpre,Jinv,tao->user_jac_stateP));
  PetscStackPop;
  PetscCall(PetscLogEventEnd(TAO_JacobianEval,tao,X,J,Jpre));
  PetscCall(VecLockReadPop(X));
  PetscFunctionReturn(0);
}

/*@C
   TaoComputeJacobianDesign - Computes the Jacobian matrix that has been
   set with `TaoSetJacobianDesignRoutine()`.

   Collective on tao

   Input Parameters:
+  tao - the Tao solver context
-  X   - input vector

   Output Parameters:
.  J - Jacobian matrix

   Notes:
   Most users should not need to explicitly call this routine, as it
   is used internally within the optimization algorithms.

   Level: developer

.seealso: `Tao`, `TaoComputeObjective()`, `TaoComputeObjectiveAndGradient()`, `TaoSetJacobianDesignRoutine()`, `TaoComputeJacobianDesign()`, `TaoSetStateDesignIS()`
@*/
PetscErrorCode TaoComputeJacobianDesign(Tao tao, Vec X, Mat J)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidHeaderSpecific(X, VEC_CLASSID,2);
  PetscCheckSameComm(tao,1,X,2);
  PetscCheck(tao->ops->computejacobiandesign,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call TaoSetJacobianDesign() first");
  ++tao->njac_design;
  PetscCall(VecLockReadPush(X));
  PetscCall(PetscLogEventBegin(TAO_JacobianEval,tao,X,J,NULL));
  PetscStackPush("Tao user Jacobian(design) function");
  PetscCall((*tao->ops->computejacobiandesign)(tao,X,J,tao->user_jac_designP));
  PetscStackPop;
  PetscCall(PetscLogEventEnd(TAO_JacobianEval,tao,X,J,NULL));
  PetscCall(VecLockReadPop(X));
  PetscFunctionReturn(0);
}

/*@C
   TaoSetJacobianRoutine - Sets the function to compute the Jacobian as well as the location to store the matrix.

   Logically collective on tao

   Input Parameters:
+  tao  - the Tao context
.  J    - Matrix used for the jacobian
.  Jpre - Matrix that will be used operated on by preconditioner, can be same as J
.  func - Jacobian evaluation routine
-  ctx  - [optional] user-defined context for private data for the
          Jacobian evaluation routine (may be NULL)

   Calling sequence of func:
$    func(Tao tao,Vec x,Mat J,Mat Jpre,void *ctx);

+  tao  - the Tao  context
.  x    - input vector
.  J    - Jacobian matrix
.  Jpre - preconditioning matrix, usually the same as J
-  ctx  - [optional] user-defined Jacobian context

   Level: intermediate

.seealso: `Tao`, `TaoSetGradient()`, `TaoSetObjective()`
@*/
PetscErrorCode TaoSetJacobianRoutine(Tao tao, Mat J, Mat Jpre, PetscErrorCode (*func)(Tao, Vec, Mat, Mat, void*), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (J) {
    PetscValidHeaderSpecific(J,MAT_CLASSID,2);
    PetscCheckSameComm(tao,1,J,2);
  }
  if (Jpre) {
    PetscValidHeaderSpecific(Jpre,MAT_CLASSID,3);
    PetscCheckSameComm(tao,1,Jpre,3);
  }
  if (ctx) {
    tao->user_jacP = ctx;
  }
  if (func) {
    tao->ops->computejacobian = func;
  }
  if (J) {
    PetscCall(PetscObjectReference((PetscObject)J));
    PetscCall(MatDestroy(&tao->jacobian));
    tao->jacobian = J;
  }
  if (Jpre) {
    PetscCall(PetscObjectReference((PetscObject)Jpre));
    PetscCall(MatDestroy(&tao->jacobian_pre));
    tao->jacobian_pre=Jpre;
  }
  PetscFunctionReturn(0);
}

/*@C
   TaoSetJacobianResidualRoutine - Sets the function to compute the least-squares residual Jacobian as well as the
   location to store the matrix.

   Logically collective on tao

   Input Parameters:
+  tao  - the Tao context
.  J    - Matrix used for the jacobian
.  Jpre - Matrix that will be used operated on by preconditioner, can be same as J
.  func - Jacobian evaluation routine
-  ctx  - [optional] user-defined context for private data for the
          Jacobian evaluation routine (may be NULL)

   Calling sequence of func:
$    func(Tao tao,Vec x,Mat J,Mat Jpre,void *ctx);

+  tao  - the Tao  context
.  x    - input vector
.  J    - Jacobian matrix
.  Jpre - preconditioning matrix, usually the same as J
-  ctx  - [optional] user-defined Jacobian context

   Level: intermediate

.seealso: `Tao`, `TaoSetGradient()`, `TaoSetObjective()`
@*/
PetscErrorCode TaoSetJacobianResidualRoutine(Tao tao, Mat J, Mat Jpre, PetscErrorCode (*func)(Tao, Vec, Mat, Mat, void*), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (J) {
    PetscValidHeaderSpecific(J,MAT_CLASSID,2);
    PetscCheckSameComm(tao,1,J,2);
  }
  if (Jpre) {
    PetscValidHeaderSpecific(Jpre,MAT_CLASSID,3);
    PetscCheckSameComm(tao,1,Jpre,3);
  }
  if (ctx) {
    tao->user_lsjacP = ctx;
  }
  if (func) {
    tao->ops->computeresidualjacobian = func;
  }
  if (J) {
    PetscCall(PetscObjectReference((PetscObject)J));
    PetscCall(MatDestroy(&tao->ls_jac));
    tao->ls_jac = J;
  }
  if (Jpre) {
    PetscCall(PetscObjectReference((PetscObject)Jpre));
    PetscCall(MatDestroy(&tao->ls_jac_pre));
    tao->ls_jac_pre=Jpre;
  }
  PetscFunctionReturn(0);
}

/*@C
   TaoSetJacobianStateRoutine - Sets the function to compute the Jacobian
   (and its inverse) of the constraint function with respect to the state variables.
   Used only for PDE-constrained optimization.

   Logically collective on tao

   Input Parameters:
+  tao  - the Tao context
.  J    - Matrix used for the jacobian
.  Jpre - Matrix that will be used operated on by PETSc preconditioner, can be same as J.  Only used if Jinv is NULL
.  Jinv - [optional] Matrix used to apply the inverse of the state jacobian. Use NULL to default to PETSc KSP solvers to apply the inverse.
.  func - Jacobian evaluation routine
-  ctx  - [optional] user-defined context for private data for the
          Jacobian evaluation routine (may be NULL)

   Calling sequence of func:
$    func(Tao tao,Vec x,Mat J,Mat Jpre,Mat Jinv,void *ctx);

+  tao  - the Tao  context
.  x    - input vector
.  J    - Jacobian matrix
.  Jpre - preconditioner matrix, usually the same as J
.  Jinv - inverse of J
-  ctx  - [optional] user-defined Jacobian context

   Level: intermediate

.seealso: `Tao`, `TaoComputeJacobianState()`, `TaoSetJacobianDesignRoutine()`, `TaoSetStateDesignIS()`
@*/
PetscErrorCode TaoSetJacobianStateRoutine(Tao tao, Mat J, Mat Jpre, Mat Jinv, PetscErrorCode (*func)(Tao, Vec, Mat, Mat, Mat, void*), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (J) {
    PetscValidHeaderSpecific(J,MAT_CLASSID,2);
    PetscCheckSameComm(tao,1,J,2);
  }
  if (Jpre) {
    PetscValidHeaderSpecific(Jpre,MAT_CLASSID,3);
    PetscCheckSameComm(tao,1,Jpre,3);
  }
  if (Jinv) {
    PetscValidHeaderSpecific(Jinv,MAT_CLASSID,4);
    PetscCheckSameComm(tao,1,Jinv,4);
  }
  if (ctx) {
    tao->user_jac_stateP = ctx;
  }
  if (func) {
    tao->ops->computejacobianstate = func;
  }
  if (J) {
    PetscCall(PetscObjectReference((PetscObject)J));
    PetscCall(MatDestroy(&tao->jacobian_state));
    tao->jacobian_state = J;
  }
  if (Jpre) {
    PetscCall(PetscObjectReference((PetscObject)Jpre));
    PetscCall(MatDestroy(&tao->jacobian_state_pre));
    tao->jacobian_state_pre=Jpre;
  }
  if (Jinv) {
    PetscCall(PetscObjectReference((PetscObject)Jinv));
    PetscCall(MatDestroy(&tao->jacobian_state_inv));
    tao->jacobian_state_inv=Jinv;
  }
  PetscFunctionReturn(0);
}

/*@C
   TaoSetJacobianDesignRoutine - Sets the function to compute the Jacobian of
   the constraint function with respect to the design variables.  Used only for
   PDE-constrained optimization.

   Logically collective on tao

   Input Parameters:
+  tao  - the Tao context
.  J    - Matrix used for the jacobian
.  func - Jacobian evaluation routine
-  ctx  - [optional] user-defined context for private data for the
          Jacobian evaluation routine (may be NULL)

   Calling sequence of func:
$    func(Tao tao,Vec x,Mat J,void *ctx);

+  tao - the Tao  context
.  x   - input vector
.  J   - Jacobian matrix
-  ctx - [optional] user-defined Jacobian context

   Level: intermediate

.seealso: `Tao`, `TaoComputeJacobianDesign()`, `TaoSetJacobianStateRoutine()`, `TaoSetStateDesignIS()`
@*/
PetscErrorCode TaoSetJacobianDesignRoutine(Tao tao, Mat J, PetscErrorCode (*func)(Tao, Vec, Mat, void*), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (J) {
    PetscValidHeaderSpecific(J,MAT_CLASSID,2);
    PetscCheckSameComm(tao,1,J,2);
  }
  if (ctx) {
    tao->user_jac_designP = ctx;
  }
  if (func) {
    tao->ops->computejacobiandesign = func;
  }
  if (J) {
    PetscCall(PetscObjectReference((PetscObject)J));
    PetscCall(MatDestroy(&tao->jacobian_design));
    tao->jacobian_design = J;
  }
  PetscFunctionReturn(0);
}

/*@
   TaoSetStateDesignIS - Indicate to the Tao which variables in the
   solution vector are state variables and which are design.  Only applies to
   PDE-constrained optimization.

   Logically Collective on Tao

   Input Parameters:
+  tao  - The Tao context
.  s_is - the index set corresponding to the state variables
-  d_is - the index set corresponding to the design variables

   Level: intermediate

.seealso: `Tao`, `TaoSetJacobianStateRoutine()`, `TaoSetJacobianDesignRoutine()`
@*/
PetscErrorCode TaoSetStateDesignIS(Tao tao, IS s_is, IS d_is)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectReference((PetscObject)s_is));
  PetscCall(ISDestroy(&tao->state_is));
  tao->state_is = s_is;
  PetscCall(PetscObjectReference((PetscObject)(d_is)));
  PetscCall(ISDestroy(&tao->design_is));
  tao->design_is = d_is;
  PetscFunctionReturn(0);
}

/*@C
   TaoComputeJacobianEquality - Computes the Jacobian matrix that has been
   set with `TaoSetJacobianEqualityRoutine()`.

   Collective on tao

   Input Parameters:
+  tao - the Tao solver context
-  X   - input vector

   Output Parameters:
+  J    - Jacobian matrix
-  Jpre - Preconditioning matrix

   Notes:
   Most users should not need to explicitly call this routine, as it
   is used internally within the optimization algorithms.

   Level: developer

.seealso: `TaoComputeObjective()`, `TaoComputeObjectiveAndGradient()`, `TaoSetJacobianStateRoutine()`, `TaoComputeJacobianDesign()`, `TaoSetStateDesignIS()`
@*/
PetscErrorCode TaoComputeJacobianEquality(Tao tao, Vec X, Mat J, Mat Jpre)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidHeaderSpecific(X, VEC_CLASSID,2);
  PetscCheckSameComm(tao,1,X,2);
  PetscCheck(tao->ops->computejacobianequality,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call TaoSetJacobianEquality() first");
  ++tao->njac_equality;
  PetscCall(VecLockReadPush(X));
  PetscCall(PetscLogEventBegin(TAO_JacobianEval,tao,X,J,Jpre));
  PetscStackPush("Tao user Jacobian(equality) function");
  PetscCall((*tao->ops->computejacobianequality)(tao,X,J,Jpre,tao->user_jac_equalityP));
  PetscStackPop;
  PetscCall(PetscLogEventEnd(TAO_JacobianEval,tao,X,J,Jpre));
  PetscCall(VecLockReadPop(X));
  PetscFunctionReturn(0);
}

/*@C
   TaoComputeJacobianInequality - Computes the Jacobian matrix that has been
   set with `TaoSetJacobianInequalityRoutine()`.

   Collective on tao

   Input Parameters:
+  tao - the Tao solver context
-  X   - input vector

   Output Parameters:
+  J    - Jacobian matrix
-  Jpre - Preconditioning matrix

   Notes:
   Most users should not need to explicitly call this routine, as it
   is used internally within the minimization solvers.

   Level: developer

.seealso: `Tao`, `TaoComputeObjective()`, `TaoComputeObjectiveAndGradient()`, `TaoSetJacobianStateRoutine()`, `TaoComputeJacobianDesign()`, `TaoSetStateDesignIS()`
@*/
PetscErrorCode TaoComputeJacobianInequality(Tao tao, Vec X, Mat J, Mat Jpre)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidHeaderSpecific(X, VEC_CLASSID,2);
  PetscCheckSameComm(tao,1,X,2);
  PetscCheck(tao->ops->computejacobianinequality,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call TaoSetJacobianInequality() first");
  ++tao->njac_inequality;
  PetscCall(VecLockReadPush(X));
  PetscCall(PetscLogEventBegin(TAO_JacobianEval,tao,X,J,Jpre));
  PetscStackPush("Tao user Jacobian(inequality) function");
  PetscCall((*tao->ops->computejacobianinequality)(tao,X,J,Jpre,tao->user_jac_inequalityP));
  PetscStackPop;
  PetscCall(PetscLogEventEnd(TAO_JacobianEval,tao,X,J,Jpre));
  PetscCall(VecLockReadPop(X));
  PetscFunctionReturn(0);
}

/*@C
   TaoSetJacobianEqualityRoutine - Sets the function to compute the Jacobian
   (and its inverse) of the constraint function with respect to the equality variables.
   Used only for PDE-constrained optimization.

   Logically collective on tao

   Input Parameters:
+  tao  - the Tao context
.  J    - Matrix used for the jacobian
.  Jpre - Matrix that will be used operated on by PETSc preconditioner, can be same as J.
.  func - Jacobian evaluation routine
-  ctx  - [optional] user-defined context for private data for the
          Jacobian evaluation routine (may be NULL)

   Calling sequence of func:
$    func(Tao tao,Vec x,Mat J,Mat Jpre,void *ctx);

+  tao  - the Tao  context
.  x    - input vector
.  J    - Jacobian matrix
.  Jpre - preconditioner matrix, usually the same as J
-  ctx  - [optional] user-defined Jacobian context

   Level: intermediate

.seealso: `Tao`, `TaoComputeJacobianEquality()`, `TaoSetJacobianDesignRoutine()`, `TaoSetEqualityDesignIS()`
@*/
PetscErrorCode TaoSetJacobianEqualityRoutine(Tao tao, Mat J, Mat Jpre, PetscErrorCode (*func)(Tao, Vec, Mat, Mat, void*), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (J) {
    PetscValidHeaderSpecific(J,MAT_CLASSID,2);
    PetscCheckSameComm(tao,1,J,2);
  }
  if (Jpre) {
    PetscValidHeaderSpecific(Jpre,MAT_CLASSID,3);
    PetscCheckSameComm(tao,1,Jpre,3);
  }
  if (ctx) {
    tao->user_jac_equalityP = ctx;
  }
  if (func) {
    tao->ops->computejacobianequality = func;
  }
  if (J) {
    PetscCall(PetscObjectReference((PetscObject)J));
    PetscCall(MatDestroy(&tao->jacobian_equality));
    tao->jacobian_equality = J;
  }
  if (Jpre) {
    PetscCall(PetscObjectReference((PetscObject)Jpre));
    PetscCall(MatDestroy(&tao->jacobian_equality_pre));
    tao->jacobian_equality_pre=Jpre;
  }
  PetscFunctionReturn(0);
}

/*@C
   TaoSetJacobianInequalityRoutine - Sets the function to compute the Jacobian
   (and its inverse) of the constraint function with respect to the inequality variables.
   Used only for PDE-constrained optimization.

   Logically collective on tao

   Input Parameters:
+  tao  - the Tao context
.  J    - Matrix used for the jacobian
.  Jpre - Matrix that will be used operated on by PETSc preconditioner, can be same as J.
.  func - Jacobian evaluation routine
-  ctx  - [optional] user-defined context for private data for the
          Jacobian evaluation routine (may be NULL)

   Calling sequence of func:
$    func(Tao tao,Vec x,Mat J,Mat Jpre,void *ctx);

+  tao  - the Tao  context
.  x    - input vector
.  J    - Jacobian matrix
.  Jpre - preconditioner matrix, usually the same as J
-  ctx  - [optional] user-defined Jacobian context

   Level: intermediate

.seealso: `Tao`, `TaoComputeJacobianInequality()`, `TaoSetJacobianDesignRoutine()`, `TaoSetInequalityDesignIS()`
@*/
PetscErrorCode TaoSetJacobianInequalityRoutine(Tao tao, Mat J, Mat Jpre, PetscErrorCode (*func)(Tao, Vec, Mat, Mat,void*), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (J) {
    PetscValidHeaderSpecific(J,MAT_CLASSID,2);
    PetscCheckSameComm(tao,1,J,2);
  }
  if (Jpre) {
    PetscValidHeaderSpecific(Jpre,MAT_CLASSID,3);
    PetscCheckSameComm(tao,1,Jpre,3);
  }
  if (ctx) {
    tao->user_jac_inequalityP = ctx;
  }
  if (func) {
    tao->ops->computejacobianinequality = func;
  }
  if (J) {
    PetscCall(PetscObjectReference((PetscObject)J));
    PetscCall(MatDestroy(&tao->jacobian_inequality));
    tao->jacobian_inequality = J;
  }
  if (Jpre) {
    PetscCall(PetscObjectReference((PetscObject)Jpre));
    PetscCall(MatDestroy(&tao->jacobian_inequality_pre));
    tao->jacobian_inequality_pre=Jpre;
  }
  PetscFunctionReturn(0);
}
