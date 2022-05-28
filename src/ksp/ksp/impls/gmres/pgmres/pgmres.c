
/*
    This file implements PGMRES (a Pipelined Generalized Minimal Residual method)
*/

#include <../src/ksp/ksp/impls/gmres/pgmres/pgmresimpl.h>       /*I  "petscksp.h"  I*/
#define PGMRES_DELTA_DIRECTIONS 10
#define PGMRES_DEFAULT_MAXK     30

static PetscErrorCode KSPPGMRESUpdateHessenberg(KSP,PetscInt,PetscBool*,PetscReal*);
static PetscErrorCode KSPPGMRESBuildSoln(PetscScalar*,Vec,Vec,KSP,PetscInt);

/*

    KSPSetUp_PGMRES - Sets up the workspace needed by pgmres.

    This is called once, usually automatically by KSPSolve() or KSPSetUp(),
    but can be called directly by KSPSetUp().

*/
static PetscErrorCode KSPSetUp_PGMRES(KSP ksp)
{
  PetscFunctionBegin;
  PetscCall(KSPSetUp_GMRES(ksp));
  PetscFunctionReturn(0);
}

/*

    KSPPGMRESCycle - Run pgmres, possibly with restart.  Return residual
                  history if requested.

    input parameters:
.        pgmres  - structure containing parameters and work areas

    output parameters:
.        itcount - number of iterations used.  If null, ignored.
.        converged - 0 if not converged

    Notes:
    On entry, the value in vector VEC_VV(0) should be
    the initial residual.

 */
static PetscErrorCode KSPPGMRESCycle(PetscInt *itcount,KSP ksp)
{
  KSP_PGMRES     *pgmres = (KSP_PGMRES*)(ksp->data);
  PetscReal      res_norm,res,newnorm;
  PetscInt       it     = 0,j,k;
  PetscBool      hapend = PETSC_FALSE;

  PetscFunctionBegin;
  if (itcount) *itcount = 0;
  PetscCall(VecNormalize(VEC_VV(0),&res_norm));
  KSPCheckNorm(ksp,res_norm);
  res    = res_norm;
  *RS(0) = res_norm;

  /* check for the convergence */
  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)ksp));
  if (ksp->normtype != KSP_NORM_NONE) ksp->rnorm = res;
  else ksp->rnorm = 0;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)ksp));
  pgmres->it = it-2;
  PetscCall(KSPLogResidualHistory(ksp,ksp->rnorm));
  PetscCall(KSPMonitor(ksp,ksp->its,ksp->rnorm));
  if (!res) {
    ksp->reason = KSP_CONVERGED_ATOL;
    PetscCall(PetscInfo(ksp,"Converged due to zero residual norm on entry\n"));
    PetscFunctionReturn(0);
  }

  PetscCall((*ksp->converged)(ksp,ksp->its,ksp->rnorm,&ksp->reason,ksp->cnvP));
  for (; !ksp->reason; it++) {
    Vec Zcur,Znext;
    if (pgmres->vv_allocated <= it + VEC_OFFSET + 1) {
      PetscCall(KSPGMRESGetNewVectors(ksp,it+1));
    }
    /* VEC_VV(it-1) is orthogonal, it will be normalized once the VecNorm arrives. */
    Zcur  = VEC_VV(it);         /* Zcur is not yet orthogonal, but the VecMDot to orthogonalize it has been started. */
    Znext = VEC_VV(it+1);       /* This iteration will compute Znext, update with a deferred correction once we know how
                                 * Zcur relates to the previous vectors, and start the reduction to orthogonalize it. */

    if (it < pgmres->max_k+1 && ksp->its+1 < PetscMax(2,ksp->max_it)) { /* We don't know whether what we have computed is enough, so apply the matrix. */
      PetscCall(KSP_PCApplyBAorAB(ksp,Zcur,Znext,VEC_TEMP_MATOP));
    }

    if (it > 1) {               /* Complete the pending reduction */
      PetscCall(VecNormEnd(VEC_VV(it-1),NORM_2,&newnorm));
      *HH(it-1,it-2) = newnorm;
    }
    if (it > 0) {               /* Finish the reduction computing the latest column of H */
      PetscCall(VecMDotEnd(Zcur,it,&(VEC_VV(0)),HH(0,it-1)));
    }

    if (it > 1) {
      /* normalize the base vector from two iterations ago, basis is complete up to here */
      PetscCall(VecScale(VEC_VV(it-1),1./ *HH(it-1,it-2)));

      PetscCall(KSPPGMRESUpdateHessenberg(ksp,it-2,&hapend,&res));
      pgmres->it = it-2;
      ksp->its++;
      if (ksp->normtype != KSP_NORM_NONE) ksp->rnorm = res;
      else ksp->rnorm = 0;

      PetscCall((*ksp->converged)(ksp,ksp->its,ksp->rnorm,&ksp->reason,ksp->cnvP));
      if (it < pgmres->max_k+1 || ksp->reason || ksp->its == ksp->max_it) {  /* Monitor if we are done or still iterating, but not before a restart. */
        PetscCall(KSPLogResidualHistory(ksp,ksp->rnorm));
        PetscCall(KSPMonitor(ksp,ksp->its,ksp->rnorm));
      }
      if (ksp->reason) break;
      /* Catch error in happy breakdown and signal convergence and break from loop */
      if (hapend) {
        PetscCheck(!ksp->errorifnotconverged,PetscObjectComm((PetscObject)ksp),PETSC_ERR_NOT_CONVERGED,"You reached the happy break down, but convergence was not indicated. Residual norm = %g",(double)res);
        ksp->reason = KSP_DIVERGED_BREAKDOWN;
        break;
      }

      if (!(it < pgmres->max_k+1 && ksp->its < ksp->max_it)) break;

      /* The it-2 column of H was not scaled when we computed Zcur, apply correction */
      PetscCall(VecScale(Zcur,1./ *HH(it-1,it-2)));
      /* And Znext computed in this iteration was computed using the under-scaled Zcur */
      PetscCall(VecScale(Znext,1./ *HH(it-1,it-2)));

      /* In the previous iteration, we projected an unnormalized Zcur against the Krylov basis, so we need to fix the column of H resulting from that projection. */
      for (k=0; k<it; k++) *HH(k,it-1) /= *HH(it-1,it-2);
      /* When Zcur was projected against the Krylov basis, VV(it-1) was still not normalized, so fix that too. This
       * column is complete except for HH(it,it-1) which we won't know until the next iteration. */
      *HH(it-1,it-1) /= *HH(it-1,it-2);
    }

    if (it > 0) {
      PetscScalar *work;
      if (!pgmres->orthogwork) PetscCall(PetscMalloc1(pgmres->max_k + 2,&pgmres->orthogwork));
      work = pgmres->orthogwork;
      /* Apply correction computed by the VecMDot in the last iteration to Znext. The original form is
       *
       *   Znext -= sum_{j=0}^{i-1} Z[j+1] * H[j,i-1]
       *
       * where
       *
       *   Z[j] = sum_{k=0}^j V[k] * H[k,j-1]
       *
       * substituting
       *
       *   Znext -= sum_{j=0}^{i-1} sum_{k=0}^{j+1} V[k] * H[k,j] * H[j,i-1]
       *
       * rearranging the iteration space from row-column to column-row
       *
       *   Znext -= sum_{k=0}^i sum_{j=k-1}^{i-1} V[k] * H[k,j] * H[j,i-1]
       *
       * Note that column it-1 of HH is correct. For all previous columns, we must look at HES because HH has already
       * been transformed to upper triangular form.
       */
      for (k=0; k<it+1; k++) {
        work[k] = 0;
        for (j=PetscMax(0,k-1); j<it-1; j++) work[k] -= *HES(k,j) * *HH(j,it-1);
      }
      PetscCall(VecMAXPY(Znext,it+1,work,&VEC_VV(0)));
      PetscCall(VecAXPY(Znext,-*HH(it-1,it-1),Zcur));

      /* Orthogonalize Zcur against existing basis vectors. */
      for (k=0; k<it; k++) work[k] = -*HH(k,it-1);
      PetscCall(VecMAXPY(Zcur,it,work,&VEC_VV(0)));
      /* Zcur is now orthogonal, and will be referred to as VEC_VV(it) again, though it is still not normalized. */
      /* Begin computing the norm of the new vector, will be normalized after the MatMult in the next iteration. */
      PetscCall(VecNormBegin(VEC_VV(it),NORM_2,&newnorm));
    }

    /* Compute column of H (to the diagonal, but not the subdiagonal) to be able to orthogonalize the newest vector. */
    PetscCall(VecMDotBegin(Znext,it+1,&VEC_VV(0),HH(0,it)));

    /* Start an asynchronous split-mode reduction, the result of the MDot and Norm will be collected on the next iteration. */
    PetscCall(PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)Znext)));
  }

  if (itcount) *itcount = it-1; /* Number of iterations actually completed. */

  /*
    Down here we have to solve for the "best" coefficients of the Krylov
    columns, add the solution values together, and possibly unwind the
    preconditioning from the solution
   */
  /* Form the solution (or the solution so far) */
  PetscCall(KSPPGMRESBuildSoln(RS(0),ksp->vec_sol,ksp->vec_sol,ksp,it-2));
  PetscFunctionReturn(0);
}

/*
    KSPSolve_PGMRES - This routine applies the PGMRES method.

   Input Parameter:
.     ksp - the Krylov space object that was set to use pgmres

   Output Parameter:
.     outits - number of iterations used

*/
static PetscErrorCode KSPSolve_PGMRES(KSP ksp)
{
  PetscInt       its,itcount;
  KSP_PGMRES     *pgmres    = (KSP_PGMRES*)ksp->data;
  PetscBool      guess_zero = ksp->guess_zero;

  PetscFunctionBegin;
  PetscCheck(!ksp->calc_sings || pgmres->Rsvd,PetscObjectComm((PetscObject)ksp),PETSC_ERR_ORDER,"Must call KSPSetComputeSingularValues() before KSPSetUp() is called");
  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)ksp));
  ksp->its = 0;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)ksp));

  itcount     = 0;
  ksp->reason = KSP_CONVERGED_ITERATING;
  while (!ksp->reason) {
    PetscCall(KSPInitialResidual(ksp,ksp->vec_sol,VEC_TEMP,VEC_TEMP_MATOP,VEC_VV(0),ksp->vec_rhs));
    PetscCall(KSPPGMRESCycle(&its,ksp));
    itcount += its;
    if (itcount >= ksp->max_it) {
      if (!ksp->reason) ksp->reason = KSP_DIVERGED_ITS;
      break;
    }
    ksp->guess_zero = PETSC_FALSE; /* every future call to KSPInitialResidual() will have nonzero guess */
  }
  ksp->guess_zero = guess_zero; /* restore if user provided nonzero initial guess */
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPDestroy_PGMRES(KSP ksp)
{
  PetscFunctionBegin;
  PetscCall(KSPDestroy_GMRES(ksp));
  PetscFunctionReturn(0);
}

/*
    KSPPGMRESBuildSoln - create the solution from the starting vector and the
                      current iterates.

    Input parameters:
        nrs - work area of size it + 1.
        vguess  - index of initial guess
        vdest - index of result.  Note that vguess may == vdest (replace
                guess with the solution).
        it - HH upper triangular part is a block of size (it+1) x (it+1)

     This is an internal routine that knows about the PGMRES internals.
 */
static PetscErrorCode KSPPGMRESBuildSoln(PetscScalar *nrs,Vec vguess,Vec vdest,KSP ksp,PetscInt it)
{
  PetscScalar    tt;
  PetscInt       k,j;
  KSP_PGMRES     *pgmres = (KSP_PGMRES*)(ksp->data);

  PetscFunctionBegin;
  /* Solve for solution vector that minimizes the residual */

  if (it < 0) {                                 /* no pgmres steps have been performed */
    PetscCall(VecCopy(vguess,vdest)); /* VecCopy() is smart, exits immediately if vguess == vdest */
    PetscFunctionReturn(0);
  }

  /* solve the upper triangular system - RS is the right side and HH is
     the upper triangular matrix  - put soln in nrs */
  if (*HH(it,it) != 0.0) nrs[it] = *RS(it) / *HH(it,it);
  else nrs[it] = 0.0;

  for (k=it-1; k>=0; k--) {
    tt = *RS(k);
    for (j=k+1; j<=it; j++) tt -= *HH(k,j) * nrs[j];
    nrs[k] = tt / *HH(k,k);
  }

  /* Accumulate the correction to the solution of the preconditioned problem in TEMP */
  PetscCall(VecZeroEntries(VEC_TEMP));
  PetscCall(VecMAXPY(VEC_TEMP,it+1,nrs,&VEC_VV(0)));
  PetscCall(KSPUnwindPreconditioner(ksp,VEC_TEMP,VEC_TEMP_MATOP));
  /* add solution to previous solution */
  if (vdest == vguess) {
    PetscCall(VecAXPY(vdest,1.0,VEC_TEMP));
  } else {
    PetscCall(VecWAXPY(vdest,1.0,VEC_TEMP,vguess));
  }
  PetscFunctionReturn(0);
}

/*

    KSPPGMRESUpdateHessenberg - Do the scalar work for the orthogonalization.
                            Return new residual.

    input parameters:

.        ksp -    Krylov space object
.        it  -    plane rotations are applied to the (it+1)th column of the
                  modified hessenberg (i.e. HH(:,it))
.        hapend - PETSC_FALSE not happy breakdown ending.

    output parameters:
.        res - the new residual

 */
/*
.  it - column of the Hessenberg that is complete, PGMRES is actually computing two columns ahead of this
 */
static PetscErrorCode KSPPGMRESUpdateHessenberg(KSP ksp,PetscInt it,PetscBool *hapend,PetscReal *res)
{
  PetscScalar    *hh,*cc,*ss,*rs;
  PetscInt       j;
  PetscReal      hapbnd;
  KSP_PGMRES     *pgmres = (KSP_PGMRES*)(ksp->data);

  PetscFunctionBegin;
  hh = HH(0,it);   /* pointer to beginning of column to update */
  cc = CC(0);      /* beginning of cosine rotations */
  ss = SS(0);      /* beginning of sine rotations */
  rs = RS(0);      /* right hand side of least squares system */

  /* The Hessenberg matrix is now correct through column it, save that form for possible spectral analysis */
  for (j=0; j<=it+1; j++) *HES(j,it) = hh[j];

  /* check for the happy breakdown */
  hapbnd = PetscMin(PetscAbsScalar(hh[it+1] / rs[it]),pgmres->haptol);
  if (PetscAbsScalar(hh[it+1]) < hapbnd) {
    PetscCall(PetscInfo(ksp,"Detected happy breakdown, current hapbnd = %14.12e H(%" PetscInt_FMT ",%" PetscInt_FMT ") = %14.12e\n",(double)hapbnd,it+1,it,(double)PetscAbsScalar(*HH(it+1,it))));
    *hapend = PETSC_TRUE;
  }

  /* Apply all the previously computed plane rotations to the new column
     of the Hessenberg matrix */
  /* Note: this uses the rotation [conj(c)  s ; -s   c], c= cos(theta), s= sin(theta),
     and some refs have [c   s ; -conj(s)  c] (don't be confused!) */

  for (j=0; j<it; j++) {
    PetscScalar hhj = hh[j];
    hh[j]   = PetscConj(cc[j])*hhj + ss[j]*hh[j+1];
    hh[j+1] =          -ss[j] *hhj + cc[j]*hh[j+1];
  }

  /*
    compute the new plane rotation, and apply it to:
     1) the right-hand-side of the Hessenberg system (RS)
        note: it affects RS(it) and RS(it+1)
     2) the new column of the Hessenberg matrix
        note: it affects HH(it,it) which is currently pointed to
        by hh and HH(it+1, it) (*(hh+1))
    thus obtaining the updated value of the residual...
  */

  /* compute new plane rotation */

  if (!*hapend) {
    PetscReal delta = PetscSqrtReal(PetscSqr(PetscAbsScalar(hh[it])) + PetscSqr(PetscAbsScalar(hh[it+1])));
    if (delta == 0.0) {
      ksp->reason = KSP_DIVERGED_NULL;
      PetscFunctionReturn(0);
    }

    cc[it] = hh[it] / delta;    /* new cosine value */
    ss[it] = hh[it+1] / delta;  /* new sine value */

    hh[it]   = PetscConj(cc[it])*hh[it] + ss[it]*hh[it+1];
    rs[it+1] = -ss[it]*rs[it];
    rs[it]   = PetscConj(cc[it])*rs[it];
    *res     = PetscAbsScalar(rs[it+1]);
  } else { /* happy breakdown: HH(it+1, it) = 0, therefore we don't need to apply
            another rotation matrix (so RH doesn't change).  The new residual is
            always the new sine term times the residual from last time (RS(it)),
            but now the new sine rotation would be zero...so the residual should
            be zero...so we will multiply "zero" by the last residual.  This might
            not be exactly what we want to do here -could just return "zero". */

    *res = 0.0;
  }
  PetscFunctionReturn(0);
}

/*
   KSPBuildSolution_PGMRES

     Input Parameter:
.     ksp - the Krylov space object
.     ptr-

   Output Parameter:
.     result - the solution

   Note: this calls KSPPGMRESBuildSoln - the same function that KSPPGMRESCycle
   calls directly.

*/
PetscErrorCode KSPBuildSolution_PGMRES(KSP ksp,Vec ptr,Vec *result)
{
  KSP_PGMRES     *pgmres = (KSP_PGMRES*)ksp->data;

  PetscFunctionBegin;
  if (!ptr) {
    if (!pgmres->sol_temp) {
      PetscCall(VecDuplicate(ksp->vec_sol,&pgmres->sol_temp));
      PetscCall(PetscLogObjectParent((PetscObject)ksp,(PetscObject)pgmres->sol_temp));
    }
    ptr = pgmres->sol_temp;
  }
  if (!pgmres->nrs) {
    /* allocate the work area */
    PetscCall(PetscMalloc1(pgmres->max_k,&pgmres->nrs));
    PetscCall(PetscLogObjectMemory((PetscObject)ksp,pgmres->max_k*sizeof(PetscScalar)));
  }

  PetscCall(KSPPGMRESBuildSoln(pgmres->nrs,ksp->vec_sol,ptr,ksp,pgmres->it));
  if (result) *result = ptr;
  PetscFunctionReturn(0);
}

PetscErrorCode KSPSetFromOptions_PGMRES(PetscOptionItems *PetscOptionsObject,KSP ksp)
{
  PetscFunctionBegin;
  PetscCall(KSPSetFromOptions_GMRES(PetscOptionsObject,ksp));
  PetscOptionsHeadBegin(PetscOptionsObject,"KSP pipelined GMRES Options");
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

PetscErrorCode KSPReset_PGMRES(KSP ksp)
{
  PetscFunctionBegin;
  PetscCall(KSPReset_GMRES(ksp));
  PetscFunctionReturn(0);
}

/*MC
     KSPPGMRES - Implements the Pipelined Generalized Minimal Residual method.

   Options Database Keys:
+   -ksp_gmres_restart <restart> - the number of Krylov directions to orthogonalize against
.   -ksp_gmres_haptol <tol> - sets the tolerance for "happy ending" (exact convergence)
.   -ksp_gmres_preallocate - preallocate all the Krylov search directions initially (otherwise groups of
                             vectors are allocated as needed)
.   -ksp_gmres_classicalgramschmidt - use classical (unmodified) Gram-Schmidt to orthogonalize against the Krylov space (fast) (the default)
.   -ksp_gmres_modifiedgramschmidt - use modified Gram-Schmidt in the orthogonalization (more stable, but slower)
.   -ksp_gmres_cgs_refinement_type <refine_never,refine_ifneeded,refine_always> - determine if iterative refinement is used to increase the
                                   stability of the classical Gram-Schmidt  orthogonalization.
-   -ksp_gmres_krylov_monitor - plot the Krylov space generated

   Level: beginner

   Notes:
   MPI configuration may be necessary for reductions to make asynchronous progress, which is important for performance of pipelined methods.
   See the FAQ on the PETSc website for details.

   Reference:
   Ghysels, Ashby, Meerbergen, Vanroose, Hiding global communication latencies in the GMRES algorithm on massively parallel machines, 2012.

   Developer Notes:
    This object is subclassed off of KSPGMRES

.seealso: `KSPCreate()`, `KSPSetType()`, `KSPType`, `KSP`, `KSPGMRES`, `KSPLGMRES`, `KSPPIPECG`, `KSPPIPECR`,
          `KSPGMRESSetRestart()`, `KSPGMRESSetHapTol()`, `KSPGMRESSetPreAllocateVectors()`, `KSPGMRESSetOrthogonalization()`, `KSPGMRESGetOrthogonalization()`,
          `KSPGMRESClassicalGramSchmidtOrthogonalization()`, `KSPGMRESModifiedGramSchmidtOrthogonalization()`,
          `KSPGMRESCGSRefinementType`, `KSPGMRESSetCGSRefinementType()`, `KSPGMRESGetCGSRefinementType()`, `KSPGMRESMonitorKrylov()`
M*/

PETSC_EXTERN PetscErrorCode KSPCreate_PGMRES(KSP ksp)
{
  KSP_PGMRES     *pgmres;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(ksp,&pgmres));

  ksp->data                              = (void*)pgmres;
  ksp->ops->buildsolution                = KSPBuildSolution_PGMRES;
  ksp->ops->setup                        = KSPSetUp_PGMRES;
  ksp->ops->solve                        = KSPSolve_PGMRES;
  ksp->ops->reset                        = KSPReset_PGMRES;
  ksp->ops->destroy                      = KSPDestroy_PGMRES;
  ksp->ops->view                         = KSPView_GMRES;
  ksp->ops->setfromoptions               = KSPSetFromOptions_PGMRES;
  ksp->ops->computeextremesingularvalues = KSPComputeExtremeSingularValues_GMRES;
  ksp->ops->computeeigenvalues           = KSPComputeEigenvalues_GMRES;

  PetscCall(KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,3));
  PetscCall(KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_RIGHT,2));
  PetscCall(KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_RIGHT,1));

  PetscCall(PetscObjectComposeFunction((PetscObject)ksp,"KSPGMRESSetPreAllocateVectors_C",KSPGMRESSetPreAllocateVectors_GMRES));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp,"KSPGMRESSetOrthogonalization_C",KSPGMRESSetOrthogonalization_GMRES));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp,"KSPGMRESGetOrthogonalization_C",KSPGMRESGetOrthogonalization_GMRES));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp,"KSPGMRESSetRestart_C",KSPGMRESSetRestart_GMRES));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp,"KSPGMRESGetRestart_C",KSPGMRESGetRestart_GMRES));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp,"KSPGMRESSetCGSRefinementType_C",KSPGMRESSetCGSRefinementType_GMRES));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp,"KSPGMRESGetCGSRefinementType_C",KSPGMRESGetCGSRefinementType_GMRES));

  pgmres->nextra_vecs    = 1;
  pgmres->haptol         = 1.0e-30;
  pgmres->q_preallocate  = 0;
  pgmres->delta_allocate = PGMRES_DELTA_DIRECTIONS;
  pgmres->orthog         = KSPGMRESClassicalGramSchmidtOrthogonalization;
  pgmres->nrs            = NULL;
  pgmres->sol_temp       = NULL;
  pgmres->max_k          = PGMRES_DEFAULT_MAXK;
  pgmres->Rsvd           = NULL;
  pgmres->orthogwork     = NULL;
  pgmres->cgstype        = KSP_GMRES_CGS_REFINE_NEVER;
  PetscFunctionReturn(0);
}
