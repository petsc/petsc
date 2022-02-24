#include <petsc/private/snesimpl.h>  /*I "petscsnes.h" I*/
#include <petscdm.h>

/*@C
   SNESVISetComputeVariableBounds - Sets a function that is called to compute the variable bounds

   Input parameter:
+  snes - the SNES context
-  compute - computes the bounds

   Level: advanced

.seealso:   SNESVISetVariableBounds()

@*/
PetscErrorCode SNESVISetComputeVariableBounds(SNES snes, PetscErrorCode (*compute)(SNES,Vec,Vec))
{
  PetscErrorCode (*f)(SNES,PetscErrorCode (*)(SNES,Vec,Vec));

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  CHKERRQ(PetscObjectQueryFunction((PetscObject)snes,"SNESVISetComputeVariableBounds_C",&f));
  if (f) CHKERRQ(PetscUseMethod(snes,"SNESVISetComputeVariableBounds_C",(SNES,PetscErrorCode (*)(SNES,Vec,Vec)),(snes,compute)));
  else CHKERRQ(SNESVISetComputeVariableBounds_VI(snes,compute));
  PetscFunctionReturn(0);
}

PetscErrorCode SNESVISetComputeVariableBounds_VI(SNES snes,SNESVIComputeVariableBoundsFunction compute)
{
  PetscFunctionBegin;
  snes->ops->computevariablebounds = compute;
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------------------------*/

PetscErrorCode  SNESVIMonitorResidual(SNES snes,PetscInt its,PetscReal fgnorm,void *dummy)
{
  Vec            X, F, Finactive;
  IS             isactive;
  PetscViewer    viewer = (PetscViewer) dummy;

  PetscFunctionBegin;
  CHKERRQ(SNESGetFunction(snes,&F,NULL,NULL));
  CHKERRQ(SNESGetSolution(snes,&X));
  CHKERRQ(SNESVIGetActiveSetIS(snes,X,F,&isactive));
  CHKERRQ(VecDuplicate(F,&Finactive));
  CHKERRQ(VecCopy(F,Finactive));
  CHKERRQ(VecISSet(Finactive,isactive,0.0));
  CHKERRQ(ISDestroy(&isactive));
  CHKERRQ(VecView(Finactive,viewer));
  CHKERRQ(VecDestroy(&Finactive));
  PetscFunctionReturn(0);
}

PetscErrorCode  SNESMonitorVI(SNES snes,PetscInt its,PetscReal fgnorm,void *dummy)
{
  PetscViewer       viewer = (PetscViewer) dummy;
  const PetscScalar *x,*xl,*xu,*f;
  PetscInt          i,n,act[2] = {0,0},fact[2],N;
  /* Number of components that actually hit the bounds (c.f. active variables) */
  PetscInt          act_bound[2] = {0,0},fact_bound[2];
  PetscReal         rnorm,fnorm,zerotolerance = snes->vizerotolerance;
  double            tmp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,4);
  CHKERRQ(VecGetLocalSize(snes->vec_sol,&n));
  CHKERRQ(VecGetSize(snes->vec_sol,&N));
  CHKERRQ(VecGetArrayRead(snes->xl,&xl));
  CHKERRQ(VecGetArrayRead(snes->xu,&xu));
  CHKERRQ(VecGetArrayRead(snes->vec_sol,&x));
  CHKERRQ(VecGetArrayRead(snes->vec_func,&f));

  rnorm = 0.0;
  for (i=0; i<n; i++) {
    if (((PetscRealPart(x[i]) > PetscRealPart(xl[i]) + zerotolerance || (PetscRealPart(f[i]) <= 0.0)) && ((PetscRealPart(x[i]) < PetscRealPart(xu[i]) - zerotolerance) || PetscRealPart(f[i]) >= 0.0))) rnorm += PetscRealPart(PetscConj(f[i])*f[i]);
    else if (PetscRealPart(x[i]) <= PetscRealPart(xl[i]) + zerotolerance && PetscRealPart(f[i]) > 0.0) act[0]++;
    else if (PetscRealPart(x[i]) >= PetscRealPart(xu[i]) - zerotolerance && PetscRealPart(f[i]) < 0.0) act[1]++;
    else SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_PLIB,"Can never get here");
  }

  for (i=0; i<n; i++) {
    if (PetscRealPart(x[i]) <= PetscRealPart(xl[i]) + zerotolerance) act_bound[0]++;
    else if (PetscRealPart(x[i]) >= PetscRealPart(xu[i]) - zerotolerance) act_bound[1]++;
  }
  CHKERRQ(VecRestoreArrayRead(snes->vec_func,&f));
  CHKERRQ(VecRestoreArrayRead(snes->xl,&xl));
  CHKERRQ(VecRestoreArrayRead(snes->xu,&xu));
  CHKERRQ(VecRestoreArrayRead(snes->vec_sol,&x));
  CHKERRMPI(MPIU_Allreduce(&rnorm,&fnorm,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)snes)));
  CHKERRMPI(MPIU_Allreduce(act,fact,2,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)snes)));
  CHKERRMPI(MPIU_Allreduce(act_bound,fact_bound,2,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)snes)));
  fnorm = PetscSqrtReal(fnorm);

  CHKERRQ(PetscViewerASCIIAddTab(viewer,((PetscObject)snes)->tablevel));
  if (snes->ntruebounds) tmp = ((double)(fact[0]+fact[1]))/((double)snes->ntruebounds);
  else tmp = 0.0;
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"%3D SNES VI Function norm %g Active lower constraints %D/%D upper constraints %D/%D Percent of total %g Percent of bounded %g\n",its,(double)fnorm,fact[0],fact_bound[0],fact[1],fact_bound[1],((double)(fact[0]+fact[1]))/((double)N),tmp));

  CHKERRQ(PetscViewerASCIISubtractTab(viewer,((PetscObject)snes)->tablevel));
  PetscFunctionReturn(0);
}

/*
     Checks if J^T F = 0 which implies we've found a local minimum of the norm of the function,
    || F(u) ||_2 but not a zero, F(u) = 0. In the case when one cannot compute J^T F we use the fact that
    0 = (J^T F)^T W = F^T J W iff W not in the null space of J. Thanks for Jorge More
    for this trick. One assumes that the probability that W is in the null space of J is very, very small.
*/
PetscErrorCode SNESVICheckLocalMin_Private(SNES snes,Mat A,Vec F,Vec W,PetscReal fnorm,PetscBool *ismin)
{
  PetscReal      a1;
  PetscBool      hastranspose;

  PetscFunctionBegin;
  *ismin = PETSC_FALSE;
  CHKERRQ(MatHasOperation(A,MATOP_MULT_TRANSPOSE,&hastranspose));
  if (hastranspose) {
    /* Compute || J^T F|| */
    CHKERRQ(MatMultTranspose(A,F,W));
    CHKERRQ(VecNorm(W,NORM_2,&a1));
    CHKERRQ(PetscInfo(snes,"|| J^T F|| %g near zero implies found a local minimum\n",(double)(a1/fnorm)));
    if (a1/fnorm < 1.e-4) *ismin = PETSC_TRUE;
  } else {
    Vec         work;
    PetscScalar result;
    PetscReal   wnorm;

    CHKERRQ(VecSetRandom(W,NULL));
    CHKERRQ(VecNorm(W,NORM_2,&wnorm));
    CHKERRQ(VecDuplicate(W,&work));
    CHKERRQ(MatMult(A,W,work));
    CHKERRQ(VecDot(F,work,&result));
    CHKERRQ(VecDestroy(&work));
    a1   = PetscAbsScalar(result)/(fnorm*wnorm);
    CHKERRQ(PetscInfo(snes,"(F^T J random)/(|| F ||*||random|| %g near zero implies found a local minimum\n",(double)a1));
    if (a1 < 1.e-4) *ismin = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*
     Checks if J^T(F - J*X) = 0
*/
PetscErrorCode SNESVICheckResidual_Private(SNES snes,Mat A,Vec F,Vec X,Vec W1,Vec W2)
{
  PetscReal      a1,a2;
  PetscBool      hastranspose;

  PetscFunctionBegin;
  CHKERRQ(MatHasOperation(A,MATOP_MULT_TRANSPOSE,&hastranspose));
  if (hastranspose) {
    CHKERRQ(MatMult(A,X,W1));
    CHKERRQ(VecAXPY(W1,-1.0,F));

    /* Compute || J^T W|| */
    CHKERRQ(MatMultTranspose(A,W1,W2));
    CHKERRQ(VecNorm(W1,NORM_2,&a1));
    CHKERRQ(VecNorm(W2,NORM_2,&a2));
    if (a1 != 0.0) {
      CHKERRQ(PetscInfo(snes,"||J^T(F-Ax)||/||F-AX|| %g near zero implies inconsistent rhs\n",(double)(a2/a1)));
    }
  }
  PetscFunctionReturn(0);
}

/*
  SNESConvergedDefault_VI - Checks the convergence of the semismooth newton algorithm.

  Notes:
  The convergence criterion currently implemented is
  merit < abstol
  merit < rtol*merit_initial
*/
PetscErrorCode SNESConvergedDefault_VI(SNES snes,PetscInt it,PetscReal xnorm,PetscReal gradnorm,PetscReal fnorm,SNESConvergedReason *reason,void *dummy)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidPointer(reason,6);

  *reason = SNES_CONVERGED_ITERATING;

  if (!it) {
    /* set parameter for default relative tolerance convergence test */
    snes->ttol = fnorm*snes->rtol;
  }
  if (fnorm != fnorm) {
    CHKERRQ(PetscInfo(snes,"Failed to converged, function norm is NaN\n"));
    *reason = SNES_DIVERGED_FNORM_NAN;
  } else if (fnorm < snes->abstol && (it || !snes->forceiteration)) {
    CHKERRQ(PetscInfo(snes,"Converged due to function norm %g < %g\n",(double)fnorm,(double)snes->abstol));
    *reason = SNES_CONVERGED_FNORM_ABS;
  } else if (snes->nfuncs >= snes->max_funcs && snes->max_funcs >= 0) {
    CHKERRQ(PetscInfo(snes,"Exceeded maximum number of function evaluations: %D > %D\n",snes->nfuncs,snes->max_funcs));
    *reason = SNES_DIVERGED_FUNCTION_COUNT;
  }

  if (it && !*reason) {
    if (fnorm < snes->ttol) {
      CHKERRQ(PetscInfo(snes,"Converged due to function norm %g < %g (relative tolerance)\n",(double)fnorm,(double)snes->ttol));
      *reason = SNES_CONVERGED_FNORM_RELATIVE;
    }
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   SNESVIProjectOntoBounds - Projects X onto the feasible region so that Xl[i] <= X[i] <= Xu[i] for i = 1...n.

   Input Parameters:
.  SNES - nonlinear solver context

   Output Parameters:
.  X - Bound projected X

*/

PetscErrorCode SNESVIProjectOntoBounds(SNES snes,Vec X)
{
  const PetscScalar *xl,*xu;
  PetscScalar       *x;
  PetscInt          i,n;

  PetscFunctionBegin;
  CHKERRQ(VecGetLocalSize(X,&n));
  CHKERRQ(VecGetArray(X,&x));
  CHKERRQ(VecGetArrayRead(snes->xl,&xl));
  CHKERRQ(VecGetArrayRead(snes->xu,&xu));

  for (i = 0; i<n; i++) {
    if (PetscRealPart(x[i]) < PetscRealPart(xl[i])) x[i] = xl[i];
    else if (PetscRealPart(x[i]) > PetscRealPart(xu[i])) x[i] = xu[i];
  }
  CHKERRQ(VecRestoreArray(X,&x));
  CHKERRQ(VecRestoreArrayRead(snes->xl,&xl));
  CHKERRQ(VecRestoreArrayRead(snes->xu,&xu));
  PetscFunctionReturn(0);
}

/*
   SNESVIGetActiveSetIndices - Gets the global indices for the active set variables

   Input parameter:
.  snes - the SNES context
.  X    - the snes solution vector
.  F    - the nonlinear function vector

   Output parameter:
.  ISact - active set index set
 */
PetscErrorCode SNESVIGetActiveSetIS(SNES snes,Vec X,Vec F,IS *ISact)
{
  Vec               Xl=snes->xl,Xu=snes->xu;
  const PetscScalar *x,*f,*xl,*xu;
  PetscInt          *idx_act,i,nlocal,nloc_isact=0,ilow,ihigh,i1=0;
  PetscReal         zerotolerance = snes->vizerotolerance;

  PetscFunctionBegin;
  CHKERRQ(VecGetLocalSize(X,&nlocal));
  CHKERRQ(VecGetOwnershipRange(X,&ilow,&ihigh));
  CHKERRQ(VecGetArrayRead(X,&x));
  CHKERRQ(VecGetArrayRead(Xl,&xl));
  CHKERRQ(VecGetArrayRead(Xu,&xu));
  CHKERRQ(VecGetArrayRead(F,&f));
  /* Compute active set size */
  for (i=0; i < nlocal;i++) {
    if (!((PetscRealPart(x[i]) > PetscRealPart(xl[i]) + zerotolerance || (PetscRealPart(f[i]) <= 0.0)) && ((PetscRealPart(x[i]) < PetscRealPart(xu[i]) - zerotolerance) || PetscRealPart(f[i]) >= 0.0))) nloc_isact++;
  }

  CHKERRQ(PetscMalloc1(nloc_isact,&idx_act));

  /* Set active set indices */
  for (i=0; i < nlocal; i++) {
    if (!((PetscRealPart(x[i]) > PetscRealPart(xl[i]) + zerotolerance || (PetscRealPart(f[i]) <= 0.0)) && ((PetscRealPart(x[i]) < PetscRealPart(xu[i]) - zerotolerance) || PetscRealPart(f[i]) >= 0.0))) idx_act[i1++] = ilow+i;
  }

  /* Create active set IS */
  CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)snes),nloc_isact,idx_act,PETSC_OWN_POINTER,ISact));

  CHKERRQ(VecRestoreArrayRead(X,&x));
  CHKERRQ(VecRestoreArrayRead(Xl,&xl));
  CHKERRQ(VecRestoreArrayRead(Xu,&xu));
  CHKERRQ(VecRestoreArrayRead(F,&f));
  PetscFunctionReturn(0);
}

PetscErrorCode SNESVICreateIndexSets_RS(SNES snes,Vec X,Vec F,IS *ISact,IS *ISinact)
{
  PetscInt       rstart,rend;

  PetscFunctionBegin;
  CHKERRQ(SNESVIGetActiveSetIS(snes,X,F,ISact));
  CHKERRQ(VecGetOwnershipRange(X,&rstart,&rend));
  CHKERRQ(ISComplement(*ISact,rstart,rend,ISinact));
  PetscFunctionReturn(0);
}

PetscErrorCode SNESVIComputeInactiveSetFnorm(SNES snes,Vec F,Vec X, PetscReal *fnorm)
{
  const PetscScalar *x,*xl,*xu,*f;
  PetscInt          i,n;
  PetscReal         rnorm,zerotolerance = snes->vizerotolerance;

  PetscFunctionBegin;
  CHKERRQ(VecGetLocalSize(X,&n));
  CHKERRQ(VecGetArrayRead(snes->xl,&xl));
  CHKERRQ(VecGetArrayRead(snes->xu,&xu));
  CHKERRQ(VecGetArrayRead(X,&x));
  CHKERRQ(VecGetArrayRead(F,&f));
  rnorm = 0.0;
  for (i=0; i<n; i++) {
    if (((PetscRealPart(x[i]) > PetscRealPart(xl[i]) + zerotolerance || (PetscRealPart(f[i]) <= 0.0)) && ((PetscRealPart(x[i]) < PetscRealPart(xu[i]) - zerotolerance) || PetscRealPart(f[i]) >= 0.0))) rnorm += PetscRealPart(PetscConj(f[i])*f[i]);
  }
  CHKERRQ(VecRestoreArrayRead(F,&f));
  CHKERRQ(VecRestoreArrayRead(snes->xl,&xl));
  CHKERRQ(VecRestoreArrayRead(snes->xu,&xu));
  CHKERRQ(VecRestoreArrayRead(X,&x));
  CHKERRMPI(MPIU_Allreduce(&rnorm,fnorm,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)snes)));
  *fnorm = PetscSqrtReal(*fnorm);
  PetscFunctionReturn(0);
}

PetscErrorCode SNESVIDMComputeVariableBounds(SNES snes,Vec xl, Vec xu)
{
  PetscFunctionBegin;
  CHKERRQ(DMComputeVariableBounds(snes->dm, xl, xu));
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   SNESSetUp_VI - Does setup common to all VI solvers -- basically makes sure bounds have been properly set up
   of the SNESVI nonlinear solver.

   Input Parameter:
.  snes - the SNES context

   Application Interface Routine: SNESSetUp()

   Notes:
   For basic use of the SNES solvers, the user need not explicitly call
   SNESSetUp(), since these actions will automatically occur during
   the call to SNESSolve().
 */
PetscErrorCode SNESSetUp_VI(SNES snes)
{
  PetscInt       i_start[3],i_end[3];

  PetscFunctionBegin;
  CHKERRQ(SNESSetWorkVecs(snes,1));
  CHKERRQ(SNESSetUpMatrices(snes));

  if (!snes->ops->computevariablebounds && snes->dm) {
    PetscBool flag;
    CHKERRQ(DMHasVariableBounds(snes->dm, &flag));
    if (flag) {
      snes->ops->computevariablebounds = SNESVIDMComputeVariableBounds;
    }
  }
  if (!snes->usersetbounds) {
    if (snes->ops->computevariablebounds) {
      if (!snes->xl) CHKERRQ(VecDuplicate(snes->vec_sol,&snes->xl));
      if (!snes->xu) CHKERRQ(VecDuplicate(snes->vec_sol,&snes->xu));
      CHKERRQ((*snes->ops->computevariablebounds)(snes,snes->xl,snes->xu));
    } else if (!snes->xl && !snes->xu) {
      /* If the lower and upper bound on variables are not set, set it to -Inf and Inf */
      CHKERRQ(VecDuplicate(snes->vec_sol, &snes->xl));
      CHKERRQ(VecSet(snes->xl,PETSC_NINFINITY));
      CHKERRQ(VecDuplicate(snes->vec_sol, &snes->xu));
      CHKERRQ(VecSet(snes->xu,PETSC_INFINITY));
    } else {
      /* Check if lower bound, upper bound and solution vector distribution across the processors is identical */
      CHKERRQ(VecGetOwnershipRange(snes->vec_sol,i_start,i_end));
      CHKERRQ(VecGetOwnershipRange(snes->xl,i_start+1,i_end+1));
      CHKERRQ(VecGetOwnershipRange(snes->xu,i_start+2,i_end+2));
      if ((i_start[0] != i_start[1]) || (i_start[0] != i_start[2]) || (i_end[0] != i_end[1]) || (i_end[0] != i_end[2]))
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Distribution of lower bound, upper bound and the solution vector should be identical across all the processors.");
    }
  }
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
PetscErrorCode SNESReset_VI(SNES snes)
{
  PetscFunctionBegin;
  CHKERRQ(VecDestroy(&snes->xl));
  CHKERRQ(VecDestroy(&snes->xu));
  snes->usersetbounds = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*
   SNESDestroy_VI - Destroys the private SNES_VI context that was created
   with SNESCreate_VI().

   Input Parameter:
.  snes - the SNES context

   Application Interface Routine: SNESDestroy()
 */
PetscErrorCode SNESDestroy_VI(SNES snes)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFree(snes->data));

  /* clear composed functions */
  CHKERRQ(PetscObjectComposeFunction((PetscObject)snes,"SNESLineSearchSet_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)snes,"SNESLineSearchSetDefaultMonitor_C",NULL));
  PetscFunctionReturn(0);
}

/*@
   SNESVISetVariableBounds - Sets the lower and upper bounds for the solution vector. xl <= x <= xu.

   Input Parameters:
+  snes - the SNES context.
.  xl   - lower bound.
-  xu   - upper bound.

   Notes:
   If this routine is not called then the lower and upper bounds are set to
   PETSC_NINFINITY and PETSC_INFINITY respectively during SNESSetUp().

   Level: advanced

@*/
PetscErrorCode SNESVISetVariableBounds(SNES snes, Vec xl, Vec xu)
{
  PetscErrorCode (*f)(SNES,Vec,Vec);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidHeaderSpecific(xl,VEC_CLASSID,2);
  PetscValidHeaderSpecific(xu,VEC_CLASSID,3);
  CHKERRQ(PetscObjectQueryFunction((PetscObject)snes,"SNESVISetVariableBounds_C",&f));
  if (f) CHKERRQ(PetscUseMethod(snes,"SNESVISetVariableBounds_C",(SNES,Vec,Vec),(snes,xl,xu)));
  else CHKERRQ(SNESVISetVariableBounds_VI(snes, xl, xu));
  snes->usersetbounds = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode SNESVISetVariableBounds_VI(SNES snes,Vec xl,Vec xu)
{
  const PetscScalar *xxl,*xxu;
  PetscInt          i,n, cnt = 0;

  PetscFunctionBegin;
  CHKERRQ(SNESGetFunction(snes,&snes->vec_func,NULL,NULL));
  PetscCheck(snes->vec_func,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call SNESSetFunction() or SNESSetDM() first");
  {
    PetscInt xlN,xuN,N;
    CHKERRQ(VecGetSize(xl,&xlN));
    CHKERRQ(VecGetSize(xu,&xuN));
    CHKERRQ(VecGetSize(snes->vec_func,&N));
    PetscCheck(xlN == N,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Incompatible vector lengths lower bound = %D solution vector = %D",xlN,N);
    PetscCheck(xuN == N,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Incompatible vector lengths: upper bound = %D solution vector = %D",xuN,N);
  }
  CHKERRQ(PetscObjectReference((PetscObject)xl));
  CHKERRQ(PetscObjectReference((PetscObject)xu));
  CHKERRQ(VecDestroy(&snes->xl));
  CHKERRQ(VecDestroy(&snes->xu));
  snes->xl = xl;
  snes->xu = xu;
  CHKERRQ(VecGetLocalSize(xl,&n));
  CHKERRQ(VecGetArrayRead(xl,&xxl));
  CHKERRQ(VecGetArrayRead(xu,&xxu));
  for (i=0; i<n; i++) cnt += ((xxl[i] != PETSC_NINFINITY) || (xxu[i] != PETSC_INFINITY));

  CHKERRMPI(MPIU_Allreduce(&cnt,&snes->ntruebounds,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)snes)));
  CHKERRQ(VecRestoreArrayRead(xl,&xxl));
  CHKERRQ(VecRestoreArrayRead(xu,&xxu));
  PetscFunctionReturn(0);
}

PetscErrorCode SNESSetFromOptions_VI(PetscOptionItems *PetscOptionsObject,SNES snes)
{
  PetscBool      flg = PETSC_FALSE;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"SNES VI options"));
  CHKERRQ(PetscOptionsReal("-snes_vi_zero_tolerance","Tolerance for considering x[] value to be on a bound","None",snes->vizerotolerance,&snes->vizerotolerance,NULL));
  CHKERRQ(PetscOptionsBool("-snes_vi_monitor","Monitor all non-active variables","SNESMonitorResidual",flg,&flg,NULL));
  if (flg) {
    CHKERRQ(SNESMonitorSet(snes,SNESMonitorVI,PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)snes)),NULL));
  }
  flg = PETSC_FALSE;
  CHKERRQ(PetscOptionsBool("-snes_vi_monitor_residual","Monitor residual all non-active variables; using zero for active constraints","SNESMonitorVIResidual",flg,&flg,NULL));
  if (flg) {
    CHKERRQ(SNESMonitorSet(snes,SNESVIMonitorResidual,PETSC_VIEWER_DRAW_(PetscObjectComm((PetscObject)snes)),NULL));
  }
  CHKERRQ(PetscOptionsTail());
  PetscFunctionReturn(0);
}
