
#include <petsc/private/snesimpl.h>

typedef struct {
  PetscBool     complete_print;
  PetscBool     threshold_print;
  PetscScalar   threshold;
} SNES_Test;


PetscErrorCode SNESSolve_Test(SNES snes)
{
  Mat            A = snes->jacobian,B,C;
  Vec            x = snes->vec_sol,f = snes->vec_func,f1 = snes->vec_sol_update;
  PetscErrorCode ierr;
  PetscInt       i;
  PetscReal      nrm,gnorm;
  SNES_Test      *neP = (SNES_Test*)snes->data;
  PetscErrorCode (*objective)(SNES,Vec,PetscReal*,void*);
  void           *ctx;
  PetscReal      fnorm,f1norm,dnorm;

  PetscFunctionBegin;
  if (A != snes->jacobian_pre) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot test with alternative preconditioner");

  ierr = PetscPrintf(PetscObjectComm((PetscObject)snes),"Testing hand-coded Jacobian, if the ratio is\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PetscObjectComm((PetscObject)snes),"O(1.e-8), the hand-coded Jacobian is probably correct.\n");CHKERRQ(ierr);
  if (!neP->complete_print) {
    ierr = PetscPrintf(PetscObjectComm((PetscObject)snes),"Run with -snes_test_display to show difference\n");CHKERRQ(ierr);
    ierr = PetscPrintf(PetscObjectComm((PetscObject)snes),"of hand-coded and finite difference Jacobian.\n");CHKERRQ(ierr);
  }


  for (i=0; i<3; i++) {
    void                     *functx;
    static const char *const loc[] = {"user-defined state","constant state -1.0","constant state 1.0"};
    PetscInt                 m,n,M,N;

    if (i == 1) {
      ierr = VecSet(x,-1.0);CHKERRQ(ierr);
    } else if (i == 2) {
      ierr = VecSet(x,1.0);CHKERRQ(ierr);
    }

    /* evaluate the function at this point because SNESComputeJacobianDefaultColor() assumes that the function has been evaluated and put into snes->vec_func */
    ierr = SNESComputeFunction(snes,x,f);CHKERRQ(ierr);
    if (snes->domainerror) {
      ierr              = PetscPrintf(PetscObjectComm((PetscObject)snes),"Domain error at %s\n",loc[i]);CHKERRQ(ierr);
      snes->domainerror = PETSC_FALSE;
      continue;
    }

    /* compute both versions of Jacobian */
    ierr = SNESComputeJacobian(snes,x,A,A);CHKERRQ(ierr);

    ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
    ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
    ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
    ierr = MatSetSizes(B,m,n,M,N);CHKERRQ(ierr);
    ierr = MatSetType(B,((PetscObject)A)->type_name);CHKERRQ(ierr);
    ierr = MatSetUp(B);CHKERRQ(ierr);
    ierr = MatSetOption(B,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);

    ierr = SNESGetFunction(snes,NULL,NULL,&functx);CHKERRQ(ierr);
    ierr = SNESComputeJacobianDefault(snes,x,B,B,functx);CHKERRQ(ierr);
    if (neP->complete_print) {
      MPI_Comm    comm;
      PetscViewer viewer;
      ierr = PetscPrintf(PetscObjectComm((PetscObject)snes),"Finite difference Jacobian (%s)\n",loc[i]);CHKERRQ(ierr);
      ierr = PetscObjectGetComm((PetscObject)B,&comm);CHKERRQ(ierr);
      ierr = PetscViewerASCIIGetStdout(comm,&viewer);CHKERRQ(ierr);
      ierr = MatView(B,viewer);CHKERRQ(ierr);
    }
    /* compare */
    ierr = MatAYPX(B,-1.0,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatNorm(B,NORM_FROBENIUS,&nrm);CHKERRQ(ierr);
    ierr = MatNorm(A,NORM_FROBENIUS,&gnorm);CHKERRQ(ierr);
    if (neP->complete_print) {
      MPI_Comm    comm;
      PetscViewer viewer;
      ierr = PetscPrintf(PetscObjectComm((PetscObject)snes),"Hand-coded Jacobian (%s)\n",loc[i]);CHKERRQ(ierr);
      ierr = PetscObjectGetComm((PetscObject)B,&comm);CHKERRQ(ierr);
      ierr = PetscViewerASCIIGetStdout(comm,&viewer);CHKERRQ(ierr);
      ierr = MatView(A,viewer);CHKERRQ(ierr);
      ierr = PetscPrintf(PetscObjectComm((PetscObject)snes),"Hand-coded minus finite-difference Jacobian (%s)\n",loc[i]);CHKERRQ(ierr);
      ierr = MatView(B,viewer);CHKERRQ(ierr);
    }


    if (neP->threshold_print) {
      MPI_Comm          comm;
      PetscViewer       viewer;
      PetscInt          Istart, Iend, *ccols, bncols, cncols, j, row;
      PetscScalar       *cvals;
      const PetscInt    *bcols;
      const PetscScalar *bvals;
      
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
          if (PetscAbsScalar(bvals[j]) > PetscAbsScalar(neP->threshold)) {
            ccols[cncols] = bcols[j];
            cvals[cncols] = bvals[j];
            cncols += 1;
          }
        }
	if(cncols) {
	  ierr = MatSetValues(C,1,&row,cncols,ccols,cvals,INSERT_VALUES);CHKERRQ(ierr);
	}
        ierr = MatRestoreRow(B,row,&bncols,&bcols,&bvals);CHKERRQ(ierr);
        ierr = PetscFree2(ccols,cvals);CHKERRQ(ierr); 
      }
      
      ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      
      ierr = PetscPrintf(PetscObjectComm((PetscObject)snes),"Entries where difference is over threshold (%s)\n",loc[i]);CHKERRQ(ierr);
      ierr = PetscObjectGetComm((PetscObject)C,&comm);CHKERRQ(ierr);
      ierr = PetscViewerASCIIGetStdout(comm,&viewer);CHKERRQ(ierr);

      ierr = MatView(C,viewer);CHKERRQ(ierr);
      ierr = MatDestroy(&C);CHKERRQ(ierr);
    }

    if (!gnorm) gnorm = 1; /* just in case */
    ierr = PetscPrintf(PetscObjectComm((PetscObject)snes),"Norm of matrix ratio %g, difference %g (%s)\n",(double)(nrm/gnorm),(double)nrm,loc[i]);CHKERRQ(ierr);

    ierr = SNESGetObjective(snes,&objective,&ctx);CHKERRQ(ierr);
    if (objective) {
      ierr = SNESComputeFunction(snes,x,f);CHKERRQ(ierr);
      ierr = VecNorm(f,NORM_2,&fnorm);CHKERRQ(ierr);
      if (neP->complete_print) {
        PetscViewer viewer;
        ierr = PetscPrintf(PetscObjectComm((PetscObject)snes),"Hand-coded Function (%s)\n",loc[i]);CHKERRQ(ierr);
        ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)snes),&viewer);CHKERRQ(ierr);
        ierr = VecView(f,viewer);CHKERRQ(ierr);
      }
      ierr = SNESObjectiveComputeFunctionDefaultFD(snes,x,f1,NULL);CHKERRQ(ierr);
      ierr = VecNorm(f1,NORM_2,&f1norm);CHKERRQ(ierr);
      if (neP->complete_print) {
        PetscViewer viewer;
        ierr = PetscPrintf(PetscObjectComm((PetscObject)snes),"Finite-difference Function (%s)\n",loc[i]);CHKERRQ(ierr);
        ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)snes),&viewer);CHKERRQ(ierr);
        ierr = VecView(f1,viewer);CHKERRQ(ierr);
      }
      /* compare the two */
      ierr = VecAXPY(f,-1.0,f1);CHKERRQ(ierr);
      ierr = VecNorm(f,NORM_2,&dnorm);CHKERRQ(ierr);
      if (!fnorm) fnorm = 1.;
      ierr = PetscPrintf(PetscObjectComm((PetscObject)snes),"Norm of function ratio %g, difference %g (%s)\n",dnorm/fnorm,dnorm,loc[i]);CHKERRQ(ierr);
      if (neP->complete_print) {
        PetscViewer viewer;
        ierr = PetscPrintf(PetscObjectComm((PetscObject)snes),"Difference (%s)\n",loc[i]);CHKERRQ(ierr);
        ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)snes),&viewer);CHKERRQ(ierr);
        ierr = VecView(f,viewer);CHKERRQ(ierr);
      }
    }
    ierr = MatDestroy(&B);CHKERRQ(ierr);
  }

  /*
   Abort after the first iteration due to the jacobian not being valid.
  */

  SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE,"SNESTest aborts after Jacobian test: it is NORMAL behavior.");
  PetscFunctionReturn(0);
}


/* ------------------------------------------------------------ */
PetscErrorCode SNESDestroy_Test(SNES snes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(snes->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESSetFromOptions_Test(PetscOptionItems *PetscOptionsObject,SNES snes)
{
  SNES_Test      *ls = (SNES_Test*)snes->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Hand-coded Jacobian tester options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-snes_test_display","Display difference between hand-coded and finite difference Jacobians","None",ls->complete_print,&ls->complete_print,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-snes_test_display_threshold", "Display difference between hand-coded and finite difference Jacobians which exceed input threshold", "None", ls->threshold, &ls->threshold, &ls->threshold_print);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SNESSetUp_Test(SNES snes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESSetUpMatrices(snes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */
/*MC
      SNESTEST - Test hand-coded Jacobian against finite difference Jacobian

   Options Database:
+  -snes_type test    - use a SNES solver that evaluates the difference between hand-code and finite-difference Jacobians
-  -snes_test_display - display the elements of the matrix, the difference between the Jacobian approximated by finite-differencing and hand-coded Jacobian

   Level: intermediate

   Notes: This solver is not a solver and does not converge to a solution.  SNESTEST checks the Jacobian at three
   points: the 0, 1, and -1 solution vectors.  At each point the following is reported.

   Output:
+  difference - ||J - Jd||, the norm of the difference of the hand-coded Jacobian J and the approximate Jacobian Jd obtained by finite-differencing
   the residual,
-  ratio      - ||J - Jd||/||J||, the ratio of the norms of the above difference and the hand-coded Jacobian.

   Frobenius norm is used in the above throughout. After doing these three tests, it always aborts with the error message
   "SNESTest aborts after Jacobian test".  No other behavior is to be expected.  It may be similarly used to check if a
   SNES function is the gradient of an objective function set with SNESSetObjective().

.seealso:  SNESCreate(), SNES, SNESSetType(), SNESUpdateCheckJacobian(), SNESNEWTONLS, SNESNEWTONTR

M*/
PETSC_EXTERN PetscErrorCode SNESCreate_Test(SNES snes)
{
  SNES_Test      *neP;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  snes->ops->solve          = SNESSolve_Test;
  snes->ops->destroy        = SNESDestroy_Test;
  snes->ops->setfromoptions = SNESSetFromOptions_Test;
  snes->ops->view           = 0;
  snes->ops->setup          = SNESSetUp_Test;
  snes->ops->reset          = 0;

  snes->usesksp = PETSC_FALSE;

  snes->alwayscomputesfinalresidual = PETSC_FALSE;

  ierr                = PetscNewLog(snes,&neP);CHKERRQ(ierr);
  snes->data          = (void*)neP;
  neP->complete_print = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
    SNESUpdateCheckJacobian - Checks each Jacobian computed by the nonlinear solver comparing the users function with a finite difference computation.

   Options Database:
+    -snes_check_jacobian - use this every time SNESSolve() is called
-    -snes_check_jacobian_view -  Display difference between Jacobian approximated by finite-differencing and the hand-coded Jacobian

   Output:
+  difference - ||J - Jd||, the norm of the difference of the hand-coded Jacobian J and the approximate Jacobian Jd obtained by finite-differencing
   the residual,
-  ratio      - ||J - Jd||/||J||, the ratio of the norms of the above difference and the hand-coded Jacobian.

   Notes:
   Frobenius norm is used in the above throughout.  This check is carried out every SNES iteration.

   Level: intermediate

.seealso:  SNESTEST, SNESCreate(), SNES, SNESSetType(), SNESNEWTONLS, SNESNEWTONTR, SNESSolve()

@*/
PetscErrorCode SNESUpdateCheckJacobian(SNES snes,PetscInt it)
{
  Mat            A = snes->jacobian,B;
  Vec            x = snes->vec_sol,f = snes->vec_func,f1 = snes->vec_sol_update;
  PetscErrorCode ierr;
  PetscReal      nrm,gnorm;
  PetscErrorCode (*objective)(SNES,Vec,PetscReal*,void*);
  void           *ctx;
  PetscReal      fnorm,f1norm,dnorm;
  PetscInt       m,n,M,N;
  PetscBool      complete_print = PETSC_FALSE;
  void           *functx;
  PetscViewer    viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)snes));

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(((PetscObject)snes)->options,((PetscObject)snes)->prefix,"-snes_check_jacobian_view",&complete_print);CHKERRQ(ierr);
  if (A != snes->jacobian_pre) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot check Jacobian with alternative preconditioner");

  ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"      Testing hand-coded Jacobian, if the ratio is O(1.e-8), the hand-coded Jacobian is probably correct.\n");CHKERRQ(ierr);
  if (!complete_print) {
    ierr = PetscViewerASCIIPrintf(viewer,"      Run with -snes_check_jacobian_view [viewer][:filename][:format] to show difference of hand-coded and finite difference Jacobian.\n");CHKERRQ(ierr);
  }

  /* compute both versions of Jacobian */
  ierr = SNESComputeJacobian(snes,x,A,A);CHKERRQ(ierr);

  ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  ierr = MatSetSizes(B,m,n,M,N);CHKERRQ(ierr);
  ierr = MatSetType(B,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);
  ierr = MatSetOption(B,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
  ierr = SNESGetFunction(snes,NULL,NULL,&functx);CHKERRQ(ierr);
  ierr = SNESComputeJacobianDefault(snes,x,B,B,functx);CHKERRQ(ierr);

  if (complete_print) {
    ierr = PetscViewerASCIIPrintf(viewer,"    Finite difference Jacobian\n");CHKERRQ(ierr);
    ierr = MatViewFromOptions(B,(PetscObject)snes,"-snes_check_jacobian_view");CHKERRQ(ierr);
  }
  /* compare */
  ierr = MatAYPX(B,-1.0,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatNorm(B,NORM_FROBENIUS,&nrm);CHKERRQ(ierr);
  ierr = MatNorm(A,NORM_FROBENIUS,&gnorm);CHKERRQ(ierr);
  if (complete_print) {
    ierr = PetscViewerASCIIPrintf(viewer,"    Hand-coded Jacobian\n");CHKERRQ(ierr);
    ierr = MatViewFromOptions(A,(PetscObject)snes,"-snes_check_jacobian_view");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"    Hand-coded minus finite difference Jacobian\n");CHKERRQ(ierr);
    ierr = MatViewFromOptions(B,(PetscObject)snes,"-snes_check_jacobian_view");CHKERRQ(ierr);
  }
  if (!gnorm) gnorm = 1; /* just in case */
  ierr = PetscViewerASCIIPrintf(viewer,"    %g = ||J - Jfd||/||J|| %g  = ||J - Jfd||\n",(double)(nrm/gnorm),(double)nrm);CHKERRQ(ierr);

  ierr = SNESGetObjective(snes,&objective,&ctx);CHKERRQ(ierr);
  if (objective) {
    ierr = SNESComputeFunction(snes,x,f);CHKERRQ(ierr);
    ierr = VecNorm(f,NORM_2,&fnorm);CHKERRQ(ierr);
    if (complete_print) {
      ierr = PetscViewerASCIIPrintf(viewer,"    Hand-coded Objective Function \n");CHKERRQ(ierr);
      ierr = VecView(f,viewer);CHKERRQ(ierr);
    }
    ierr = SNESObjectiveComputeFunctionDefaultFD(snes,x,f1,NULL);CHKERRQ(ierr);
    ierr = VecNorm(f1,NORM_2,&f1norm);CHKERRQ(ierr);
    if (complete_print) {
      ierr = PetscViewerASCIIPrintf(viewer,"    Finite-Difference Objective Function\n");CHKERRQ(ierr);
      ierr = VecView(f1,viewer);CHKERRQ(ierr);
    }
    /* compare the two */
    ierr = VecAXPY(f,-1.0,f1);CHKERRQ(ierr);
    ierr = VecNorm(f,NORM_2,&dnorm);CHKERRQ(ierr);
    if (!fnorm) fnorm = 1.;
    ierr = PetscViewerASCIIPrintf(viewer,"    %g = Norm of objective function ratio %g = difference\n",dnorm/fnorm,dnorm);CHKERRQ(ierr);
    if (complete_print) {
      ierr = PetscViewerASCIIPrintf(viewer,"    Difference\n");CHKERRQ(ierr);
      ierr = VecView(f,viewer);CHKERRQ(ierr);
    }
  }
  ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)snes)->tablevel);CHKERRQ(ierr);

  ierr = MatDestroy(&B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
