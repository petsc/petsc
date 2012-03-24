#include "viimpl.h"  /*I "petscsnes.h" I*/

#undef __FUNCT__
#define __FUNCT__ "SNESVISetComputeVariableBounds"
/*@C
   SNESVISetComputeVariableBounds - Sets a function  that is called to compute the variable bounds

   Input parameter
+  snes - the SNES context
-  compute - computes the bounds

   Level: advanced

@*/
PetscErrorCode SNESVISetComputeVariableBounds(SNES snes, PetscErrorCode (*compute)(SNES,Vec,Vec))
{
  PetscErrorCode   ierr,(*f)(SNES,PetscErrorCode(*)(SNES,Vec,Vec));

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  ierr = PetscObjectQueryFunction((PetscObject)snes,"SNESVISetComputeVariableBounds_C",(PetscVoidStarFunction)&f);CHKERRQ(ierr);
  if (!f) {ierr = SNESSetType(snes,SNESVIRS);CHKERRQ(ierr);}
  ierr = PetscUseMethod(snes,"SNESVISetComputeVariableBounds_C",(SNES,PetscErrorCode(*)(SNES,Vec,Vec)),(snes,compute));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "SNESVISetComputeVariableBounds_VI"
PetscErrorCode SNESVISetComputeVariableBounds_VI(SNES snes,SNESVIComputeVariableBoundsFunction compute)
{
  PetscFunctionBegin;
  snes->ops->computevariablebounds = compute;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "SNESVIComputeInactiveSetIS"
/*
   SNESVIComputeInactiveSetIS - Gets the global indices for the bogus inactive set variables

   Input parameter
.  snes - the SNES context
.  X    - the snes solution vector

   Output parameter
.  ISact - active set index set

 */
PetscErrorCode SNESVIComputeInactiveSetIS(Vec upper,Vec lower,Vec X,Vec F,IS* inact)
{
  PetscErrorCode    ierr;
  const PetscScalar *x,*xl,*xu,*f;
  PetscInt          *idx_act,i,nlocal,nloc_isact=0,ilow,ihigh,i1=0;
  
  PetscFunctionBegin;
  ierr = VecGetLocalSize(X,&nlocal);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(X,&ilow,&ihigh);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(lower,&xl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(upper,&xu);CHKERRQ(ierr);
  ierr = VecGetArrayRead(F,&f);CHKERRQ(ierr);
  /* Compute inactive set size */
  for (i=0; i < nlocal;i++) {
    if (((PetscRealPart(x[i]) > PetscRealPart(xl[i]) + 1.e-8 || (PetscRealPart(f[i]) < 0.0)) && ((PetscRealPart(x[i]) < PetscRealPart(xu[i]) - 1.e-8) || PetscRealPart(f[i]) > 0.0))) nloc_isact++;
  }

  ierr = PetscMalloc(nloc_isact*sizeof(PetscInt),&idx_act);CHKERRQ(ierr);

  /* Set inactive set indices */
  for(i=0; i < nlocal; i++) {
    if (((PetscRealPart(x[i]) > PetscRealPart(xl[i]) + 1.e-8 || (PetscRealPart(f[i]) < 0.0)) && ((PetscRealPart(x[i]) < PetscRealPart(xu[i]) - 1.e-8) || PetscRealPart(f[i]) > 0.0))) idx_act[i1++] = ilow+i;
  }

   /* Create inactive set IS */
  ierr = ISCreateGeneral(((PetscObject)upper)->comm,nloc_isact,idx_act,PETSC_OWN_POINTER,inact);CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(lower,&xl);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(upper,&xu);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "SNESMonitorVI"
PetscErrorCode  SNESMonitorVI(SNES snes,PetscInt its,PetscReal fgnorm,void *dummy)
{
  PetscErrorCode    ierr;
  PetscViewer        viewer = dummy ? (PetscViewer) dummy : PETSC_VIEWER_STDOUT_(((PetscObject)snes)->comm);
  const PetscScalar  *x,*xl,*xu,*f;
  PetscInt           i,n,act[2] = {0,0},fact[2],N;
  /* Number of components that actually hit the bounds (c.f. active variables) */
  PetscInt           act_bound[2] = {0,0},fact_bound[2];
  PetscReal          rnorm,fnorm;
  double             tmp;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(snes->vec_sol,&n);CHKERRQ(ierr);
  ierr = VecGetSize(snes->vec_sol,&N);CHKERRQ(ierr);
  ierr = VecGetArrayRead(snes->xl,&xl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(snes->xu,&xu);CHKERRQ(ierr);
  ierr = VecGetArrayRead(snes->vec_sol,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(snes->vec_func,&f);CHKERRQ(ierr);
  
  rnorm = 0.0;
  for (i=0; i<n; i++) {
    if (((PetscRealPart(x[i]) > PetscRealPart(xl[i]) + 1.e-8 || (PetscRealPart(f[i]) < 0.0)) && ((PetscRealPart(x[i]) < PetscRealPart(xu[i]) - 1.e-8) || PetscRealPart(f[i]) > 0.0))) rnorm += PetscRealPart(PetscConj(f[i])*f[i]);
    else if (PetscRealPart(x[i]) <= PetscRealPart(xl[i]) + 1.e-8 && PetscRealPart(f[i]) >= 0.0) act[0]++;
    else if (PetscRealPart(x[i]) >= PetscRealPart(xu[i]) - 1.e-8 && PetscRealPart(f[i]) <= 0.0) act[1]++;
    else SETERRQ(((PetscObject)snes)->comm,PETSC_ERR_PLIB,"Can never get here");
  }

  for (i=0; i<n; i++) {
    if (PetscRealPart(x[i]) <= PetscRealPart(xl[i]) + 1.e-8) act_bound[0]++; 
    else if (PetscRealPart(x[i]) >= PetscRealPart(xu[i]) - 1.e-8) act_bound[1]++;
  }
  ierr = VecRestoreArrayRead(snes->vec_func,&f);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(snes->xl,&xl);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(snes->xu,&xu);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(snes->vec_sol,&x);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&rnorm,&fnorm,1,MPIU_REAL,MPIU_SUM,((PetscObject)snes)->comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(act,fact,2,MPIU_INT,MPIU_SUM,((PetscObject)snes)->comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(act_bound,fact_bound,2,MPIU_INT,MPIU_SUM,((PetscObject)snes)->comm);CHKERRQ(ierr);
  fnorm = PetscSqrtReal(fnorm);
  
  ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
  if(snes->ntruebounds) tmp = ((double)(fact[0]+fact[1]))/((double)snes->ntruebounds);
  else tmp = 0.0;
  ierr = PetscViewerASCIIPrintf(viewer,"%3D SNES VI Function norm %14.12e Active lower constraints %D/%D upper constraints %D/%D Percent of total %g Percent of bounded %g\n",its,(double)fnorm,fact[0],fact_bound[0],fact[1],fact_bound[1],((double)(fact[0]+fact[1]))/((double)N),tmp);CHKERRQ(ierr);
  
  ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     Checks if J^T F = 0 which implies we've found a local minimum of the norm of the function,
    || F(u) ||_2 but not a zero, F(u) = 0. In the case when one cannot compute J^T F we use the fact that
    0 = (J^T F)^T W = F^T J W iff W not in the null space of J. Thanks for Jorge More 
    for this trick. One assumes that the probability that W is in the null space of J is very, very small.
*/ 
#undef __FUNCT__  
#define __FUNCT__ "SNESVICheckLocalMin_Private"
PetscErrorCode SNESVICheckLocalMin_Private(SNES snes,Mat A,Vec F,Vec W,PetscReal fnorm,PetscBool *ismin)
{
  PetscReal      a1;
  PetscErrorCode ierr;
  PetscBool     hastranspose;

  PetscFunctionBegin;
  *ismin = PETSC_FALSE;
  ierr = MatHasOperation(A,MATOP_MULT_TRANSPOSE,&hastranspose);CHKERRQ(ierr);
  if (hastranspose) {
    /* Compute || J^T F|| */
    ierr = MatMultTranspose(A,F,W);CHKERRQ(ierr);
    ierr = VecNorm(W,NORM_2,&a1);CHKERRQ(ierr);
    ierr = PetscInfo1(snes,"|| J^T F|| %g near zero implies found a local minimum\n",(double)(a1/fnorm));CHKERRQ(ierr);
    if (a1/fnorm < 1.e-4) *ismin = PETSC_TRUE;
  } else {
    Vec         work;
    PetscScalar result;
    PetscReal   wnorm;

    ierr = VecSetRandom(W,PETSC_NULL);CHKERRQ(ierr);
    ierr = VecNorm(W,NORM_2,&wnorm);CHKERRQ(ierr);
    ierr = VecDuplicate(W,&work);CHKERRQ(ierr);
    ierr = MatMult(A,W,work);CHKERRQ(ierr);
    ierr = VecDot(F,work,&result);CHKERRQ(ierr);
    ierr = VecDestroy(&work);CHKERRQ(ierr);
    a1   = PetscAbsScalar(result)/(fnorm*wnorm);
    ierr = PetscInfo1(snes,"(F^T J random)/(|| F ||*||random|| %g near zero implies found a local minimum\n",(double)a1);CHKERRQ(ierr);
    if (a1 < 1.e-4) *ismin = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*
     Checks if J^T(F - J*X) = 0 
*/ 
#undef __FUNCT__  
#define __FUNCT__ "SNESVICheckResidual_Private"
PetscErrorCode SNESVICheckResidual_Private(SNES snes,Mat A,Vec F,Vec X,Vec W1,Vec W2)
{
  PetscReal      a1,a2;
  PetscErrorCode ierr;
  PetscBool     hastranspose;

  PetscFunctionBegin;
  ierr = MatHasOperation(A,MATOP_MULT_TRANSPOSE,&hastranspose);CHKERRQ(ierr);
  if (hastranspose) {
    ierr = MatMult(A,X,W1);CHKERRQ(ierr);
    ierr = VecAXPY(W1,-1.0,F);CHKERRQ(ierr);

    /* Compute || J^T W|| */
    ierr = MatMultTranspose(A,W1,W2);CHKERRQ(ierr);
    ierr = VecNorm(W1,NORM_2,&a1);CHKERRQ(ierr);
    ierr = VecNorm(W2,NORM_2,&a2);CHKERRQ(ierr);
    if (a1 != 0.0) {
      ierr = PetscInfo1(snes,"||J^T(F-Ax)||/||F-AX|| %g near zero implies inconsistent rhs\n",(double)(a2/a1));CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*
  SNESDefaultConverged_VI - Checks the convergence of the semismooth newton algorithm.

  Notes:
  The convergence criterion currently implemented is
  merit < abstol
  merit < rtol*merit_initial
*/
#undef __FUNCT__
#define __FUNCT__ "SNESDefaultConverged_VI"
PetscErrorCode SNESDefaultConverged_VI(SNES snes,PetscInt it,PetscReal xnorm,PetscReal gradnorm,PetscReal fnorm,SNESConvergedReason *reason,void *dummy)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidPointer(reason,6);
  
  *reason = SNES_CONVERGED_ITERATING;

  if (!it) {
    /* set parameter for default relative tolerance convergence test */
    snes->ttol = fnorm*snes->rtol;
  }
  if (fnorm != fnorm) {
    ierr = PetscInfo(snes,"Failed to converged, function norm is NaN\n");CHKERRQ(ierr);
    *reason = SNES_DIVERGED_FNORM_NAN;
  } else if (fnorm < snes->abstol) {
    ierr = PetscInfo2(snes,"Converged due to function norm %g < %g\n",(double)fnorm,(double)snes->abstol);CHKERRQ(ierr);
    *reason = SNES_CONVERGED_FNORM_ABS;
  } else if (snes->nfuncs >= snes->max_funcs) {
    ierr = PetscInfo2(snes,"Exceeded maximum number of function evaluations: %D > %D\n",snes->nfuncs,snes->max_funcs);CHKERRQ(ierr);
    *reason = SNES_DIVERGED_FUNCTION_COUNT;
  }

  if (it && !*reason) {
    if (fnorm < snes->ttol) {
      ierr = PetscInfo2(snes,"Converged due to function norm %g < %g (relative tolerance)\n",(double)fnorm,(double)snes->ttol);CHKERRQ(ierr);
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

#undef __FUNCT__
#define __FUNCT__ "SNESVIProjectOntoBounds"
PetscErrorCode SNESVIProjectOntoBounds(SNES snes,Vec X)
{
  PetscErrorCode    ierr;
  const PetscScalar *xl,*xu;
  PetscScalar       *x;
  PetscInt          i,n;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(X,&n);CHKERRQ(ierr);
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(snes->xl,&xl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(snes->xu,&xu);CHKERRQ(ierr);

  for(i = 0;i<n;i++) {
    if (PetscRealPart(x[i]) < PetscRealPart(xl[i])) x[i] = xl[i];
    else if (PetscRealPart(x[i]) > PetscRealPart(xu[i])) x[i] = xu[i];
  }
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(snes->xl,&xl);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(snes->xu,&xu);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESVIGetActiveSetIS"
/*
   SNESVIGetActiveSetIndices - Gets the global indices for the active set variables

   Input parameter
.  snes - the SNES context
.  X    - the snes solution vector
.  F    - the nonlinear function vector

   Output parameter
.  ISact - active set index set
 */
PetscErrorCode SNESVIGetActiveSetIS(SNES snes,Vec X,Vec F,IS* ISact)
{
  PetscErrorCode   ierr;
  Vec               Xl=snes->xl,Xu=snes->xu;
  const PetscScalar *x,*f,*xl,*xu;
  PetscInt          *idx_act,i,nlocal,nloc_isact=0,ilow,ihigh,i1=0;
  
  PetscFunctionBegin;
  ierr = VecGetLocalSize(X,&nlocal);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(X,&ilow,&ihigh);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Xl,&xl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Xu,&xu);CHKERRQ(ierr);
  ierr = VecGetArrayRead(F,&f);CHKERRQ(ierr);
  /* Compute active set size */
  for (i=0; i < nlocal;i++) {
    if (!((PetscRealPart(x[i]) > PetscRealPart(xl[i]) + 1.e-8 || (PetscRealPart(f[i]) < 0.0)) && ((PetscRealPart(x[i]) < PetscRealPart(xu[i]) - 1.e-8) || PetscRealPart(f[i]) > 0.0))) nloc_isact++;
  }

  ierr = PetscMalloc(nloc_isact*sizeof(PetscInt),&idx_act);CHKERRQ(ierr);

  /* Set active set indices */
  for(i=0; i < nlocal; i++) {
    if (!((PetscRealPart(x[i]) > PetscRealPart(xl[i]) + 1.e-8 || (PetscRealPart(f[i]) < 0.0)) && ((PetscRealPart(x[i]) < PetscRealPart(xu[i]) - 1.e-8) || PetscRealPart(f[i]) > 0.0))) idx_act[i1++] = ilow+i;
  }

   /* Create active set IS */
  ierr = ISCreateGeneral(((PetscObject)snes)->comm,nloc_isact,idx_act,PETSC_OWN_POINTER,ISact);CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Xl,&xl);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Xu,&xu);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESVICreateIndexSets_RS"
PetscErrorCode SNESVICreateIndexSets_RS(SNES snes,Vec X,Vec F,IS* ISact,IS* ISinact)
{
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = SNESVIGetActiveSetIS(snes,X,F,ISact);CHKERRQ(ierr);
  ierr = ISComplement(*ISact,X->map->rstart,X->map->rend,ISinact);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESVIComputeInactiveSetFnorm"
PetscErrorCode SNESVIComputeInactiveSetFnorm(SNES snes,Vec F,Vec X, PetscReal *fnorm)
{
  PetscErrorCode    ierr;
  const PetscScalar *x,*xl,*xu,*f;
  PetscInt          i,n;
  PetscReal         rnorm;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(X,&n);CHKERRQ(ierr);
  ierr = VecGetArrayRead(snes->xl,&xl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(snes->xu,&xu);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(F,&f);CHKERRQ(ierr);
  rnorm = 0.0;
  for (i=0; i<n; i++) {
    if (((PetscRealPart(x[i]) > PetscRealPart(xl[i]) + 1.e-8 || (PetscRealPart(f[i]) < 0.0)) && ((PetscRealPart(x[i]) < PetscRealPart(xu[i]) - 1.e-8) || PetscRealPart(f[i]) > 0.0))) rnorm += PetscRealPart(PetscConj(f[i])*f[i]);
  }
  ierr = VecRestoreArrayRead(F,&f);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(snes->xl,&xl);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(snes->xu,&xu);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&rnorm,fnorm,1,MPIU_REAL,MPIU_SUM,((PetscObject)snes)->comm);CHKERRQ(ierr);
  *fnorm = PetscSqrtReal(*fnorm);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESVIDMComputeVariableBounds"
PetscErrorCode SNESVIDMComputeVariableBounds(SNES snes,Vec xl, Vec xu)
{
  PetscErrorCode     ierr;
  PetscFunctionBegin;
  ierr = DMComputeVariableBounds(snes->dm, xl, xu); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* -------------------------------------------------------------------------- */
/*
   SNESSetUp_VI - Does setup common to all VI solvers -- basically makes sure bounds have been properly set up
   of the SNESVI nonlinear solver.

   Input Parameter:
.  snes - the SNES context
.  x - the solution vector

   Application Interface Routine: SNESSetUp()

   Notes:
   For basic use of the SNES solvers, the user need not explicitly call
   SNESSetUp(), since these actions will automatically occur during
   the call to SNESSolve().
 */
#undef __FUNCT__  
#define __FUNCT__ "SNESSetUp_VI"
PetscErrorCode SNESSetUp_VI(SNES snes)
{
  PetscErrorCode ierr;
  PetscInt       i_start[3],i_end[3];

  PetscFunctionBegin;

  ierr = SNESDefaultGetWork(snes,1);CHKERRQ(ierr);
  ierr = SNESSetUpMatrices(snes);CHKERRQ(ierr);

  if(!snes->ops->computevariablebounds && snes->dm) {
    snes->ops->computevariablebounds = SNESVIDMComputeVariableBounds;
  }
  if (snes->ops->computevariablebounds) {
    if (!snes->xl) {ierr = VecDuplicate(snes->vec_sol,&snes->xl);CHKERRQ(ierr);}
    if (!snes->xu) {ierr = VecDuplicate(snes->vec_sol,&snes->xu);CHKERRQ(ierr);}
    ierr = (*snes->ops->computevariablebounds)(snes,snes->xl,snes->xu);CHKERRQ(ierr);
  } else if (!snes->xl && !snes->xu) {
    /* If the lower and upper bound on variables are not set, set it to -Inf and Inf */
    ierr = VecDuplicate(snes->vec_sol, &snes->xl);CHKERRQ(ierr);
    ierr = VecSet(snes->xl,SNES_VI_NINF);CHKERRQ(ierr);
    ierr = VecDuplicate(snes->vec_sol, &snes->xu);CHKERRQ(ierr);
    ierr = VecSet(snes->xu,SNES_VI_INF);CHKERRQ(ierr);
  } else {
    /* Check if lower bound, upper bound and solution vector distribution across the processors is identical */
    ierr = VecGetOwnershipRange(snes->vec_sol,i_start,i_end);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(snes->xl,i_start+1,i_end+1);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(snes->xu,i_start+2,i_end+2);CHKERRQ(ierr);
    if ((i_start[0] != i_start[1]) || (i_start[0] != i_start[2]) || (i_end[0] != i_end[1]) || (i_end[0] != i_end[2]))
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Distribution of lower bound, upper bound and the solution vector should be identical across all the processors.");
  }

  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "SNESReset_VI"
PetscErrorCode SNESReset_VI(SNES snes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&snes->xl);CHKERRQ(ierr);
  ierr = VecDestroy(&snes->xu);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   SNESDestroy_VI - Destroys the private SNES_VI context that was created
   with SNESCreate_VI().

   Input Parameter:
.  snes - the SNES context

   Application Interface Routine: SNESDestroy()
 */
#undef __FUNCT__  
#define __FUNCT__ "SNESDestroy_VI"
PetscErrorCode SNESDestroy_VI(SNES snes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(snes->data);CHKERRQ(ierr);

  /* clear composed functions */
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)snes,"SNESLineSearchSet_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)snes,"SNESLineSearchSetMonitor_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESVISetVariableBounds"
/*@
   SNESVISetVariableBounds - Sets the lower and upper bounds for the solution vector. xl <= x <= xu.

   Input Parameters:
.  snes - the SNES context.
.  xl   - lower bound.
.  xu   - upper bound.

   Notes:
   If this routine is not called then the lower and upper bounds are set to 
   SNES_VI_INF and SNES_VI_NINF respectively during SNESSetUp().

   Level: advanced

@*/
PetscErrorCode SNESVISetVariableBounds(SNES snes, Vec xl, Vec xu)
{
  PetscErrorCode   ierr,(*f)(SNES,Vec,Vec);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidHeaderSpecific(xl,VEC_CLASSID,2);
  PetscValidHeaderSpecific(xu,VEC_CLASSID,3);
  ierr = PetscObjectQueryFunction((PetscObject)snes,"SNESVISetVariableBounds_C",(PetscVoidStarFunction)&f);CHKERRQ(ierr);
  if (!f) {ierr = SNESSetType(snes,SNESVIRS);CHKERRQ(ierr);}
  ierr = PetscUseMethod(snes,"SNESVISetVariableBounds_C",(SNES,Vec,Vec),(snes,xl,xu));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "SNESVISetVariableBounds_VI"
PetscErrorCode SNESVISetVariableBounds_VI(SNES snes,Vec xl,Vec xu)
{
  PetscErrorCode    ierr;
  const PetscScalar *xxl,*xxu;
  PetscInt          i,n, cnt = 0;

  PetscFunctionBegin;
  ierr = SNESGetFunction(snes,&snes->vec_func,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  if (!snes->vec_func) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call SNESSetFunction() or SNESSetDM() first");
  if (xl->map->N != snes->vec_func->map->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Incompatible vector lengths lower bound = %D solution vector = %D",xl->map->N,snes->vec_func->map->N);
  if (xu->map->N != snes->vec_func->map->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Incompatible vector lengths: upper bound = %D solution vector = %D",xu->map->N,snes->vec_func->map->N);
  ierr = SNESSetType(snes,SNESVIRS);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)xl);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)xu);CHKERRQ(ierr);
  ierr = VecDestroy(&snes->xl);CHKERRQ(ierr);
  ierr = VecDestroy(&snes->xu);CHKERRQ(ierr);
  snes->xl = xl;
  snes->xu = xu;
  ierr = VecGetLocalSize(xl,&n);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xl,&xxl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xu,&xxu);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    cnt += ((xxl[i] != SNES_VI_NINF) || (xxu[i] != SNES_VI_INF));
  }
  ierr = MPI_Allreduce(&cnt,&snes->ntruebounds,1,MPIU_INT,MPI_SUM,((PetscObject)snes)->comm);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xl,&xxl);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xu,&xxu);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "SNESSetFromOptions_VI"
PetscErrorCode SNESSetFromOptions_VI(SNES snes)
{
  PetscErrorCode  ierr;
  PetscBool       flg; 

  PetscFunctionBegin;
  ierr = PetscOptionsHead("SNES VI options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-snes_vi_monitor","Monitor all non-active variables","None",PETSC_FALSE,&flg,0);CHKERRQ(ierr);
  if (flg) {
    ierr = SNESMonitorSet(snes,SNESMonitorVI,0,0);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
