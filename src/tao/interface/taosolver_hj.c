#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/


/*@C
   TaoSetHessianRoutine - Sets the function to compute the Hessian as well as the location to store the matrix.

   Logically collective on Tao

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
@*/
PetscErrorCode TaoSetHessianRoutine(Tao tao, Mat H, Mat Hpre, PetscErrorCode (*func)(Tao, Vec, Mat, Mat, void*), void *ctx)
{
  PetscErrorCode ierr;

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
  if (ctx) {
    tao->user_hessP = ctx;
  }
  if (func) {
    tao->ops->computehessian = func;
  }
  if (H) {
    ierr = PetscObjectReference((PetscObject)H);CHKERRQ(ierr);
    ierr = MatDestroy(&tao->hessian);CHKERRQ(ierr);
    tao->hessian = H;
  }
  if (Hpre) {
    ierr = PetscObjectReference((PetscObject)Hpre);CHKERRQ(ierr);
    ierr = MatDestroy(&tao->hessian_pre);CHKERRQ(ierr);
    tao->hessian_pre = Hpre;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TaoTestHessian(Tao tao)
{
  Mat               A,B,C,D,hessian;
  Vec               x = tao->solution;
  PetscErrorCode    ierr;
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
  ierr = PetscObjectOptionsBegin((PetscObject)tao);CHKERRQ(ierr);
  ierr = PetscOptionsName("-tao_test_hessian","Compare hand-coded and finite difference Hessians","None",&test);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_test_hessian", "Threshold for element difference between hand-coded and finite difference being meaningful","None",threshold,&threshold,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsViewer("-tao_test_hessian_view","View difference between hand-coded and finite difference Hessians element entries","None",&mviewer,&format,&complete_print);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (!test) PetscFunctionReturn(0);

  ierr = PetscObjectGetComm((PetscObject)tao,&comm);CHKERRQ(ierr);
  ierr = PetscViewerASCIIGetStdout(comm,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIGetTab(viewer, &tabs);CHKERRQ(ierr);
  ierr = PetscViewerASCIISetTab(viewer, ((PetscObject)tao)->tablevel);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  ---------- Testing Hessian -------------\n");CHKERRQ(ierr);
  if (!complete_print && !directionsprinted) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Run with -tao_test_hessian_view and optionally -tao_test_hessian <threshold> to show difference\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"    of hand-coded and finite difference Hessian entries greater than <threshold>.\n");CHKERRQ(ierr);
  }
  if (!directionsprinted) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Testing hand-coded Hessian, if (for double precision runs) ||J - Jfd||_F/||J||_F is\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"    O(1.e-8), the hand-coded Hessian is probably correct.\n");CHKERRQ(ierr);
    directionsprinted = PETSC_TRUE;
  }
  if (complete_print) {
    ierr = PetscViewerPushFormat(mviewer,format);CHKERRQ(ierr);
  }

  ierr = PetscObjectTypeCompare((PetscObject)tao->hessian,MATMFFD,&flg);CHKERRQ(ierr);
  if (!flg) hessian = tao->hessian;
  else hessian = tao->hessian_pre;

  while (hessian) {
    ierr = PetscObjectBaseTypeCompareAny((PetscObject)hessian,&flg,MATSEQAIJ,MATMPIAIJ,MATSEQDENSE,MATMPIDENSE,MATSEQBAIJ,MATMPIBAIJ,MATSEQSBAIJ,MATMPIBAIJ,"");CHKERRQ(ierr);
    if (flg) {
      A    = hessian;
      ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
    } else {
      ierr = MatComputeOperator(hessian,MATAIJ,&A);CHKERRQ(ierr);
    }

    ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
    ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
    ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
    ierr = MatSetSizes(B,m,n,M,N);CHKERRQ(ierr);
    ierr = MatSetType(B,((PetscObject)A)->type_name);CHKERRQ(ierr);
    ierr = MatSetUp(B);CHKERRQ(ierr);
    ierr = MatSetOption(B,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);

    ierr = TaoDefaultComputeHessian(tao,x,B,B,NULL);CHKERRQ(ierr);

    ierr = MatDuplicate(B,MAT_COPY_VALUES,&D);CHKERRQ(ierr);
    ierr = MatAYPX(D,-1.0,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatNorm(D,NORM_FROBENIUS,&nrm);CHKERRQ(ierr);
    ierr = MatNorm(A,NORM_FROBENIUS,&gnorm);CHKERRQ(ierr);
    ierr = MatDestroy(&D);CHKERRQ(ierr);
    if (!gnorm) gnorm = 1; /* just in case */
    ierr = PetscViewerASCIIPrintf(viewer,"  ||H - Hfd||_F/||H||_F = %g, ||H - Hfd||_F = %g\n",(double)(nrm/gnorm),(double)nrm);CHKERRQ(ierr);

    if (complete_print) {
      ierr = PetscViewerASCIIPrintf(viewer,"  Hand-coded Hessian ----------\n");CHKERRQ(ierr);
      ierr = MatView(A,mviewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  Finite difference Hessian ----------\n");CHKERRQ(ierr);
      ierr = MatView(B,mviewer);CHKERRQ(ierr);
    }

    if (complete_print) {
      PetscInt          Istart, Iend, *ccols, bncols, cncols, j, row;
      PetscScalar       *cvals;
      const PetscInt    *bcols;
      const PetscScalar *bvals;

      ierr = MatAYPX(B,-1.0,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatCreate(PetscObjectComm((PetscObject)A),&C);CHKERRQ(ierr);
      ierr = MatSetSizes(C,m,n,M,N);CHKERRQ(ierr);
      ierr = MatSetType(C,((PetscObject)A)->type_name);CHKERRQ(ierr);
      ierr = MatSetUp(C);CHKERRQ(ierr);
      ierr = MatSetOption(C,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
      ierr = MatGetOwnershipRange(B,&Istart,&Iend);CHKERRQ(ierr);

      for (row = Istart; row < Iend; row++) {
        ierr = MatGetRow(B,row,&bncols,&bcols,&bvals);CHKERRQ(ierr);
        ierr = PetscMalloc2(bncols,&ccols,bncols,&cvals);CHKERRQ(ierr);
        for (j = 0, cncols = 0; j < bncols; j++) {
          if (PetscAbsScalar(bvals[j]) > threshold) {
            ccols[cncols] = bcols[j];
            cvals[cncols] = bvals[j];
            cncols += 1;
          }
        }
        if (cncols) {
          ierr = MatSetValues(C,1,&row,cncols,ccols,cvals,INSERT_VALUES);CHKERRQ(ierr);
        }
        ierr = MatRestoreRow(B,row,&bncols,&bcols,&bvals);CHKERRQ(ierr);
        ierr = PetscFree2(ccols,cvals);CHKERRQ(ierr);
      }
      ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  Finite-difference minus hand-coded Hessian with tolerance %g ----------\n",(double)threshold);CHKERRQ(ierr);
      ierr = MatView(C,mviewer);CHKERRQ(ierr);
      ierr = MatDestroy(&C);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);

    if (hessian != tao->hessian_pre) {
      hessian = tao->hessian_pre;
      ierr = PetscViewerASCIIPrintf(viewer,"  ---------- Testing Hessian for preconditioner -------------\n");CHKERRQ(ierr);
    } else hessian = NULL;
  }
  if (complete_print) {
    ierr = PetscViewerPopFormat(mviewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&mviewer);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIISetTab(viewer,tabs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TaoComputeHessian - Computes the Hessian matrix that has been
   set with TaoSetHessianRoutine().

   Collective on Tao

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

   TaoComputeHessian() is typically used within minimization
   implementations, so most users would not generally call this routine
   themselves.

   Developer Notes:
   The Hessian test mechanism follows SNESTestJacobian().

   Level: developer

.seealso: TaoComputeObjective(), TaoComputeObjectiveAndGradient(), TaoSetHessianRoutine()
@*/
PetscErrorCode TaoComputeHessian(Tao tao, Vec X, Mat H, Mat Hpre)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidHeaderSpecific(X, VEC_CLASSID,2);
  PetscCheckSameComm(tao,1,X,2);
  if (!tao->ops->computehessian) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call TaoSetHessianRoutine() first");
  ++tao->nhess;
  ierr = VecLockReadPush(X);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(TAO_HessianEval,tao,X,H,Hpre);CHKERRQ(ierr);
  PetscStackPush("Tao user Hessian function");
  ierr = (*tao->ops->computehessian)(tao,X,H,Hpre,tao->user_hessP);CHKERRQ(ierr);
  PetscStackPop;
  ierr = PetscLogEventEnd(TAO_HessianEval,tao,X,H,Hpre);CHKERRQ(ierr);
  ierr = VecLockReadPop(X);CHKERRQ(ierr);

  ierr = TaoTestHessian(tao);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TaoComputeJacobian - Computes the Jacobian matrix that has been
   set with TaoSetJacobianRoutine().

   Collective on Tao

   Input Parameters:
+  tao - the Tao solver context
-  X   - input vector

   Output Parameters:
+  J    - Jacobian matrix
-  Jpre - Preconditioning matrix

   Notes:
   Most users should not need to explicitly call this routine, as it
   is used internally within the minimization solvers.

   TaoComputeJacobian() is typically used within minimization
   implementations, so most users would not generally call this routine
   themselves.

   Level: developer

.seealso: TaoComputeObjective(), TaoComputeObjectiveAndGradient(), TaoSetJacobianRoutine()
@*/
PetscErrorCode TaoComputeJacobian(Tao tao, Vec X, Mat J, Mat Jpre)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidHeaderSpecific(X, VEC_CLASSID,2);
  PetscCheckSameComm(tao,1,X,2);
  if (!tao->ops->computejacobian) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call TaoSetJacobian() first");
  ++tao->njac;
  ierr = VecLockReadPush(X);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(TAO_JacobianEval,tao,X,J,Jpre);CHKERRQ(ierr);
  PetscStackPush("Tao user Jacobian function");
  ierr = (*tao->ops->computejacobian)(tao,X,J,Jpre,tao->user_jacP);CHKERRQ(ierr);
  PetscStackPop;
  ierr = PetscLogEventEnd(TAO_JacobianEval,tao,X,J,Jpre);CHKERRQ(ierr);
  ierr = VecLockReadPop(X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TaoComputeResidualJacobian - Computes the least-squares residual Jacobian matrix that has been
   set with TaoSetJacobianResidual().

   Collective on Tao

   Input Parameters:
+  tao - the Tao solver context
-  X   - input vector

   Output Parameters:
+  J    - Jacobian matrix
-  Jpre - Preconditioning matrix

   Notes:
   Most users should not need to explicitly call this routine, as it
   is used internally within the minimization solvers.

   TaoComputeResidualJacobian() is typically used within least-squares
   implementations, so most users would not generally call this routine
   themselves.

   Level: developer

.seealso: TaoComputeResidual(), TaoSetJacobianResidual()
@*/
PetscErrorCode TaoComputeResidualJacobian(Tao tao, Vec X, Mat J, Mat Jpre)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidHeaderSpecific(X, VEC_CLASSID,2);
  PetscCheckSameComm(tao,1,X,2);
  if (!tao->ops->computeresidualjacobian) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call TaoSetResidualJacobian() first");
  ++tao->njac;
  ierr = VecLockReadPush(X);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(TAO_JacobianEval,tao,X,J,Jpre);CHKERRQ(ierr);
  PetscStackPush("Tao user least-squares residual Jacobian function");
  ierr = (*tao->ops->computeresidualjacobian)(tao,X,J,Jpre,tao->user_lsjacP);CHKERRQ(ierr);
  PetscStackPop;
  ierr = PetscLogEventEnd(TAO_JacobianEval,tao,X,J,Jpre);CHKERRQ(ierr);
  ierr = VecLockReadPop(X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TaoComputeJacobianState - Computes the Jacobian matrix that has been
   set with TaoSetJacobianStateRoutine().

   Collective on Tao

   Input Parameters:
+  tao - the Tao solver context
-  X   - input vector

   Output Parameters:
+  Jpre - Jacobian matrix
-  Jinv - Preconditioning matrix

   Notes:
   Most users should not need to explicitly call this routine, as it
   is used internally within the minimization solvers.

   TaoComputeJacobianState() is typically used within minimization
   implementations, so most users would not generally call this routine
   themselves.

   Level: developer

.seealso: TaoComputeObjective(), TaoComputeObjectiveAndGradient(), TaoSetJacobianStateRoutine(), TaoComputeJacobianDesign(), TaoSetStateDesignIS()
@*/
PetscErrorCode TaoComputeJacobianState(Tao tao, Vec X, Mat J, Mat Jpre, Mat Jinv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidHeaderSpecific(X, VEC_CLASSID,2);
  PetscCheckSameComm(tao,1,X,2);
  if (!tao->ops->computejacobianstate) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call TaoSetJacobianState() first");
  ++tao->njac_state;
  ierr = VecLockReadPush(X);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(TAO_JacobianEval,tao,X,J,Jpre);CHKERRQ(ierr);
  PetscStackPush("Tao user Jacobian(state) function");
  ierr = (*tao->ops->computejacobianstate)(tao,X,J,Jpre,Jinv,tao->user_jac_stateP);CHKERRQ(ierr);
  PetscStackPop;
  ierr = PetscLogEventEnd(TAO_JacobianEval,tao,X,J,Jpre);CHKERRQ(ierr);
  ierr = VecLockReadPop(X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TaoComputeJacobianDesign - Computes the Jacobian matrix that has been
   set with TaoSetJacobianDesignRoutine().

   Collective on Tao

   Input Parameters:
+  tao - the Tao solver context
-  X   - input vector

   Output Parameters:
.  J - Jacobian matrix

   Notes:
   Most users should not need to explicitly call this routine, as it
   is used internally within the minimization solvers.

   TaoComputeJacobianDesign() is typically used within minimization
   implementations, so most users would not generally call this routine
   themselves.

   Level: developer

.seealso:  TaoComputeObjective(), TaoComputeObjectiveAndGradient(), TaoSetJacobianDesignRoutine(), TaoComputeJacobianDesign(), TaoSetStateDesignIS()
@*/
PetscErrorCode TaoComputeJacobianDesign(Tao tao, Vec X, Mat J)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidHeaderSpecific(X, VEC_CLASSID,2);
  PetscCheckSameComm(tao,1,X,2);
  if (!tao->ops->computejacobiandesign) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call TaoSetJacobianDesign() first");
  ++tao->njac_design;
  ierr = VecLockReadPush(X);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(TAO_JacobianEval,tao,X,J,NULL);CHKERRQ(ierr);
  PetscStackPush("Tao user Jacobian(design) function");
  ierr = (*tao->ops->computejacobiandesign)(tao,X,J,tao->user_jac_designP);CHKERRQ(ierr);
  PetscStackPop;
  ierr = PetscLogEventEnd(TAO_JacobianEval,tao,X,J,NULL);CHKERRQ(ierr);
  ierr = VecLockReadPop(X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TaoSetJacobianRoutine - Sets the function to compute the Jacobian as well as the location to store the matrix.

   Logically collective on Tao

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
@*/
PetscErrorCode TaoSetJacobianRoutine(Tao tao, Mat J, Mat Jpre, PetscErrorCode (*func)(Tao, Vec, Mat, Mat, void*), void *ctx)
{
  PetscErrorCode ierr;

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
    ierr = PetscObjectReference((PetscObject)J);CHKERRQ(ierr);
    ierr = MatDestroy(&tao->jacobian);CHKERRQ(ierr);
    tao->jacobian = J;
  }
  if (Jpre) {
    ierr = PetscObjectReference((PetscObject)Jpre);CHKERRQ(ierr);
    ierr = MatDestroy(&tao->jacobian_pre);CHKERRQ(ierr);
    tao->jacobian_pre=Jpre;
  }
  PetscFunctionReturn(0);
}

/*@C
   TaoSetJacobianResidualRoutine - Sets the function to compute the least-squares residual Jacobian as well as the
   location to store the matrix.

   Logically collective on Tao

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
@*/
PetscErrorCode TaoSetJacobianResidualRoutine(Tao tao, Mat J, Mat Jpre, PetscErrorCode (*func)(Tao, Vec, Mat, Mat, void*), void *ctx)
{
  PetscErrorCode ierr;

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
    ierr = PetscObjectReference((PetscObject)J);CHKERRQ(ierr);
    ierr = MatDestroy(&tao->ls_jac);CHKERRQ(ierr);
    tao->ls_jac = J;
  }
  if (Jpre) {
    ierr = PetscObjectReference((PetscObject)Jpre);CHKERRQ(ierr);
    ierr = MatDestroy(&tao->ls_jac_pre);CHKERRQ(ierr);
    tao->ls_jac_pre=Jpre;
  }
  PetscFunctionReturn(0);
}

/*@C
   TaoSetJacobianStateRoutine - Sets the function to compute the Jacobian
   (and its inverse) of the constraint function with respect to the state variables.
   Used only for pde-constrained optimization.

   Logically collective on Tao

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
.seealso: TaoComputeJacobianState(), TaoSetJacobianDesignRoutine(), TaoSetStateDesignIS()
@*/
PetscErrorCode TaoSetJacobianStateRoutine(Tao tao, Mat J, Mat Jpre, Mat Jinv, PetscErrorCode (*func)(Tao, Vec, Mat, Mat, Mat, void*), void *ctx)
{
  PetscErrorCode ierr;

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
    ierr = PetscObjectReference((PetscObject)J);CHKERRQ(ierr);
    ierr = MatDestroy(&tao->jacobian_state);CHKERRQ(ierr);
    tao->jacobian_state = J;
  }
  if (Jpre) {
    ierr = PetscObjectReference((PetscObject)Jpre);CHKERRQ(ierr);
    ierr = MatDestroy(&tao->jacobian_state_pre);CHKERRQ(ierr);
    tao->jacobian_state_pre=Jpre;
  }
  if (Jinv) {
    ierr = PetscObjectReference((PetscObject)Jinv);CHKERRQ(ierr);
    ierr = MatDestroy(&tao->jacobian_state_inv);CHKERRQ(ierr);
    tao->jacobian_state_inv=Jinv;
  }
  PetscFunctionReturn(0);
}

/*@C
   TaoSetJacobianDesignRoutine - Sets the function to compute the Jacobian of
   the constraint function with respect to the design variables.  Used only for
   pde-constrained optimization.

   Logically collective on Tao

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

.seealso: TaoComputeJacobianDesign(), TaoSetJacobianStateRoutine(), TaoSetStateDesignIS()
@*/
PetscErrorCode TaoSetJacobianDesignRoutine(Tao tao, Mat J, PetscErrorCode (*func)(Tao, Vec, Mat, void*), void *ctx)
{
  PetscErrorCode ierr;

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
    ierr = PetscObjectReference((PetscObject)J);CHKERRQ(ierr);
    ierr = MatDestroy(&tao->jacobian_design);CHKERRQ(ierr);
    tao->jacobian_design = J;
  }
  PetscFunctionReturn(0);
}

/*@
   TaoSetStateDesignIS - Indicate to the Tao which variables in the
   solution vector are state variables and which are design.  Only applies to
   pde-constrained optimization.

   Logically Collective on Tao

   Input Parameters:
+  tao  - The Tao context
.  s_is - the index set corresponding to the state variables
-  d_is - the index set corresponding to the design variables

   Level: intermediate

.seealso: TaoSetJacobianStateRoutine(), TaoSetJacobianDesignRoutine()
@*/
PetscErrorCode TaoSetStateDesignIS(Tao tao, IS s_is, IS d_is)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectReference((PetscObject)s_is);CHKERRQ(ierr);
  ierr = ISDestroy(&tao->state_is);CHKERRQ(ierr);
  tao->state_is = s_is;
  ierr = PetscObjectReference((PetscObject)(d_is));CHKERRQ(ierr);
  ierr = ISDestroy(&tao->design_is);CHKERRQ(ierr);
  tao->design_is = d_is;
  PetscFunctionReturn(0);
}

/*@C
   TaoComputeJacobianEquality - Computes the Jacobian matrix that has been
   set with TaoSetJacobianEqualityRoutine().

   Collective on Tao

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

.seealso:  TaoComputeObjective(), TaoComputeObjectiveAndGradient(), TaoSetJacobianStateRoutine(), TaoComputeJacobianDesign(), TaoSetStateDesignIS()
@*/
PetscErrorCode TaoComputeJacobianEquality(Tao tao, Vec X, Mat J, Mat Jpre)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidHeaderSpecific(X, VEC_CLASSID,2);
  PetscCheckSameComm(tao,1,X,2);
  if (!tao->ops->computejacobianequality) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call TaoSetJacobianEquality() first");
  ++tao->njac_equality;
  ierr = VecLockReadPush(X);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(TAO_JacobianEval,tao,X,J,Jpre);CHKERRQ(ierr);
  PetscStackPush("Tao user Jacobian(equality) function");
  ierr = (*tao->ops->computejacobianequality)(tao,X,J,Jpre,tao->user_jac_equalityP);CHKERRQ(ierr);
  PetscStackPop;
  ierr = PetscLogEventEnd(TAO_JacobianEval,tao,X,J,Jpre);CHKERRQ(ierr);
  ierr = VecLockReadPop(X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TaoComputeJacobianInequality - Computes the Jacobian matrix that has been
   set with TaoSetJacobianInequalityRoutine().

   Collective on Tao

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

.seealso:  TaoComputeObjective(), TaoComputeObjectiveAndGradient(), TaoSetJacobianStateRoutine(), TaoComputeJacobianDesign(), TaoSetStateDesignIS()
@*/
PetscErrorCode TaoComputeJacobianInequality(Tao tao, Vec X, Mat J, Mat Jpre)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidHeaderSpecific(X, VEC_CLASSID,2);
  PetscCheckSameComm(tao,1,X,2);
  if (!tao->ops->computejacobianinequality) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call TaoSetJacobianInequality() first");
  ++tao->njac_inequality;
  ierr = VecLockReadPush(X);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(TAO_JacobianEval,tao,X,J,Jpre);CHKERRQ(ierr);
  PetscStackPush("Tao user Jacobian(inequality) function");
  ierr = (*tao->ops->computejacobianinequality)(tao,X,J,Jpre,tao->user_jac_inequalityP);CHKERRQ(ierr);
  PetscStackPop;
  ierr = PetscLogEventEnd(TAO_JacobianEval,tao,X,J,Jpre);CHKERRQ(ierr);
  ierr = VecLockReadPop(X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TaoSetJacobianEqualityRoutine - Sets the function to compute the Jacobian
   (and its inverse) of the constraint function with respect to the equality variables.
   Used only for pde-constrained optimization.

   Logically collective on Tao

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

.seealso: TaoComputeJacobianEquality(), TaoSetJacobianDesignRoutine(), TaoSetEqualityDesignIS()
@*/
PetscErrorCode TaoSetJacobianEqualityRoutine(Tao tao, Mat J, Mat Jpre, PetscErrorCode (*func)(Tao, Vec, Mat, Mat,void*), void *ctx)
{
  PetscErrorCode ierr;

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
    ierr = PetscObjectReference((PetscObject)J);CHKERRQ(ierr);
    ierr = MatDestroy(&tao->jacobian_equality);CHKERRQ(ierr);
    tao->jacobian_equality = J;
  }
  if (Jpre) {
    ierr = PetscObjectReference((PetscObject)Jpre);CHKERRQ(ierr);
    ierr = MatDestroy(&tao->jacobian_equality_pre);CHKERRQ(ierr);
    tao->jacobian_equality_pre=Jpre;
  }
  PetscFunctionReturn(0);
}

/*@C
   TaoSetJacobianInequalityRoutine - Sets the function to compute the Jacobian
   (and its inverse) of the constraint function with respect to the inequality variables.
   Used only for pde-constrained optimization.

   Logically collective on Tao

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

.seealso: TaoComputeJacobianInequality(), TaoSetJacobianDesignRoutine(), TaoSetInequalityDesignIS()
@*/
PetscErrorCode TaoSetJacobianInequalityRoutine(Tao tao, Mat J, Mat Jpre, PetscErrorCode (*func)(Tao, Vec, Mat, Mat,void*), void *ctx)
{
  PetscErrorCode ierr;

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
    ierr = PetscObjectReference((PetscObject)J);CHKERRQ(ierr);
    ierr = MatDestroy(&tao->jacobian_inequality);CHKERRQ(ierr);
    tao->jacobian_inequality = J;
  }
  if (Jpre) {
    ierr = PetscObjectReference((PetscObject)Jpre);CHKERRQ(ierr);
    ierr = MatDestroy(&tao->jacobian_inequality_pre);CHKERRQ(ierr);
    tao->jacobian_inequality_pre=Jpre;
  }
  PetscFunctionReturn(0);
}
