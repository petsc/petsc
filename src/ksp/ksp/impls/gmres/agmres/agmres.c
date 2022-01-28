
#define PETSCKSP_DLL

#include <../src/ksp/ksp/impls/gmres/agmres/agmresimpl.h>

#define AGMRES_DEFAULT_MAXK 30
#define AGMRES_DELTA_DIRECTIONS 10
static PetscErrorCode KSPAGMRESBuildSoln(KSP,PetscInt);
static PetscErrorCode KSPAGMRESBuildBasis(KSP);
static PetscErrorCode KSPAGMRESBuildHessenberg(KSP);

PetscLogEvent KSP_AGMRESComputeDeflationData, KSP_AGMRESBuildBasis, KSP_AGMRESComputeShifts, KSP_AGMRESRoddec;

extern PetscErrorCode KSPSetUp_DGMRES(KSP);
extern PetscErrorCode KSPBuildSolution_DGMRES(KSP,Vec,Vec*);
extern PetscErrorCode KSPSolve_DGMRES(KSP);
extern PetscErrorCode KSPDGMRESComputeDeflationData_DGMRES(KSP,PetscInt*);
extern PetscErrorCode KSPDGMRESComputeSchurForm_DGMRES(KSP,PetscInt*);
extern PetscErrorCode KSPDGMRESApplyDeflation_DGMRES(KSP,Vec,Vec);
extern PetscErrorCode KSPDestroy_DGMRES(KSP);
extern PetscErrorCode KSPSetFromOptions_DGMRES(PetscOptionItems *,KSP);
extern PetscErrorCode KSPDGMRESSetEigen_DGMRES(KSP,PetscInt);
/*
   This function allocates  data for the Newton basis GMRES implementation.
   Note that most data are allocated in KSPSetUp_DGMRES and KSPSetUp_GMRES, including the space for the basis vectors, the various Hessenberg matrices and the Givens rotations coefficients

*/
static PetscErrorCode KSPSetUp_AGMRES(KSP ksp)
{
  PetscErrorCode  ierr;
  PetscInt        hes;
  PetscInt        nloc;
  KSP_AGMRES      *agmres = (KSP_AGMRES*)ksp->data;
  PetscInt        neig    = agmres->neig;
  const PetscInt  max_k   = agmres->max_k;
  PetscInt        N       = MAXKSPSIZE;
  PetscInt        lwork   = PetscMax(8 * N + 16, 4 * neig * (N - neig));

  PetscFunctionBegin;
  PetscAssertFalse(ksp->pc_side == PC_SYMMETRIC,PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"no symmetric preconditioning for KSPAGMRES");
  N     = MAXKSPSIZE;
  /* Preallocate space during the call to KSPSetup_GMRES for the Krylov basis */
  agmres->q_preallocate = PETSC_TRUE; /* No allocation on the fly */
  /* Preallocate space to compute later the eigenvalues in GMRES */
  ksp->calc_sings = PETSC_TRUE;
  agmres->max_k   = N; /* Set the augmented size to be allocated in KSPSetup_GMRES */
  ierr            = KSPSetUp_DGMRES(ksp);CHKERRQ(ierr);
  agmres->max_k   = max_k;
  hes             = (N + 1) * (N + 1);

  /* Data for the Newton basis GMRES */
  ierr = PetscCalloc4(max_k,&agmres->Rshift,max_k,&agmres->Ishift,hes,&agmres->Rloc,(N+1)*4,&agmres->wbufptr);CHKERRQ(ierr);
  ierr = PetscMalloc3(N+1,&agmres->tau,lwork,&agmres->work,N+1,&agmres->nrs);CHKERRQ(ierr);
  ierr = PetscCalloc4(N+1,&agmres->Scale,N+1,&agmres->sgn,N+1,&agmres->tloc,N+1,&agmres->temp);CHKERRQ(ierr);

  /* Allocate space for the vectors in the orthogonalized basis*/
  ierr = VecGetLocalSize(agmres->vecs[0], &nloc);CHKERRQ(ierr);
  ierr = PetscMalloc1(nloc*(N+1), &agmres->Qloc);CHKERRQ(ierr);

  /* Init the ring of processors for the roddec orthogonalization */
  ierr = KSPAGMRESRoddecInitNeighboor(ksp);CHKERRQ(ierr);

  if (agmres->neig < 1) PetscFunctionReturn(0);

  /* Allocate space for the deflation */
  ierr = PetscMalloc1(N, &agmres->select);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(VEC_V(0), N, &agmres->TmpU);CHKERRQ(ierr);
  ierr = PetscMalloc2(N*N, &agmres->MatEigL, N*N, &agmres->MatEigR);CHKERRQ(ierr);
  /*  ierr = PetscMalloc6(N*N, &agmres->Q, N*N, &agmres->Z, N, &agmres->wr, N, &agmres->wi, N, &agmres->beta, N, &agmres->modul);CHKERRQ(ierr); */
  ierr = PetscMalloc3(N*N, &agmres->Q, N*N, &agmres->Z, N, &agmres->beta);CHKERRQ(ierr);
  ierr = PetscMalloc2((N+1),&agmres->perm,(2*neig*N),&agmres->iwork);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    Returns the current solution from the private data structure of AGMRES back to ptr.
*/
static PetscErrorCode KSPBuildSolution_AGMRES(KSP ksp,Vec ptr, Vec *result)
{
  KSP_AGMRES     *agmres = (KSP_AGMRES*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!ptr) {
    if (!agmres->sol_temp) {
      ierr = VecDuplicate(ksp->vec_sol,&agmres->sol_temp);CHKERRQ(ierr);
      ierr = VecCopy(ksp->vec_sol,agmres->sol_temp);CHKERRQ(ierr);
      ierr = PetscLogObjectParent((PetscObject)ksp,(PetscObject)agmres->sol_temp);CHKERRQ(ierr);
    }
    ptr = agmres->sol_temp;
  } else {
    ierr = VecCopy(ksp->vec_sol, ptr);CHKERRQ(ierr);
  }
  if (result) *result = ptr;
  PetscFunctionReturn(0);
}

/* Computes the shifts  needed to generate stable basis vectors (through the Newton polynomials)
   At input, the operators (matrix and preconditioners) are used to create a new GMRES KSP.
   One cycle of GMRES with the Arnoldi process is performed and the eigenvalues of the induced Hessenberg matrix (the Ritz values) are computed.
   NOTE: This function is not currently used; the next function is rather used when  the eigenvectors are needed next to augment the basis
*/
PetscErrorCode KSPComputeShifts_GMRES(KSP ksp)
{
  PetscErrorCode  ierr;
  KSP_AGMRES      *agmres = (KSP_AGMRES*)(ksp->data);
  KSP             kspgmres;
  Mat             Amat, Pmat;
  const PetscInt  max_k = agmres->max_k;
  PC              pc;
  PetscInt        m;
  PetscScalar     *Rshift, *Ishift;
  PetscBool       flg;

  PetscFunctionBegin;
  /* Perform one cycle of classical GMRES (with the Arnoldi process) to get the Hessenberg matrix
   We assume here that the ksp is AGMRES and that the operators for the
   linear system have been set in this ksp */
  ierr = KSPCreate(PetscObjectComm((PetscObject)ksp), &kspgmres);CHKERRQ(ierr);
  if (!ksp->pc) { ierr = KSPGetPC(ksp,&ksp->pc);CHKERRQ(ierr); }
  ierr = PCGetOperators(ksp->pc, &Amat, &Pmat);CHKERRQ(ierr);
  ierr = KSPSetOperators(kspgmres, Amat, Pmat);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(kspgmres);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,((PetscObject)ksp)->prefix, "-ksp_view", &flg);CHKERRQ(ierr);
  if (flg) { ierr = PetscOptionsClearValue(NULL,"-ksp_view");CHKERRQ(ierr); }
  ierr = KSPSetType(kspgmres, KSPGMRES);CHKERRQ(ierr);
  ierr = KSPGMRESSetRestart(kspgmres, max_k);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
  ierr = KSPSetPC(kspgmres, pc);CHKERRQ(ierr);
  /* Copy common options */
  kspgmres->pc_side = ksp->pc_side;
  /* Setup KSP context */
  ierr = KSPSetComputeEigenvalues(kspgmres, PETSC_TRUE);CHKERRQ(ierr);
  ierr = KSPSetUp(kspgmres);CHKERRQ(ierr);

  kspgmres->max_it = max_k; /* Restrict the maximum number of iterations to one cycle of GMRES */
  kspgmres->rtol   = ksp->rtol;

  ierr = KSPSolve(kspgmres, ksp->vec_rhs, ksp->vec_sol);CHKERRQ(ierr);

  ksp->guess_zero = PETSC_FALSE;
  ksp->rnorm      = kspgmres->rnorm;
  ksp->its        = kspgmres->its;
  if (kspgmres->reason == KSP_CONVERGED_RTOL) {
    ksp->reason = KSP_CONVERGED_RTOL;
    PetscFunctionReturn(0);
  } else ksp->reason = KSP_CONVERGED_ITERATING;
  /* Now, compute the Shifts values */
  ierr = PetscMalloc2(max_k,&Rshift,max_k,&Ishift);CHKERRQ(ierr);
  ierr = KSPComputeEigenvalues(kspgmres, max_k, Rshift, Ishift, &m);CHKERRQ(ierr);
  PetscAssertFalse(m < max_k,PetscObjectComm((PetscObject)ksp),PETSC_ERR_PLIB, "Unable to compute the Shifts for the Newton basis");
  else {
    ierr = KSPAGMRESLejaOrdering(Rshift, Ishift, agmres->Rshift, agmres->Ishift, max_k);CHKERRQ(ierr);

    agmres->HasShifts = PETSC_TRUE;
  }
  /* Restore KSP view options */
  if (flg) { ierr = PetscOptionsSetValue(NULL,"-ksp_view", "");CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

/* Computes the shift values (Ritz values) needed to generate stable basis vectors
   One cycle of DGMRES is performed to find the eigenvalues. The same data structures are used since AGMRES extends DGMRES
   Note that when the basis is  to be augmented, then this function computes the harmonic Ritz vectors from this first cycle.
   Input :
    - The operators (matrix, preconditioners and right hand side) are  normally required.
    - max_k : the size of the (non augmented) basis.
    - neig: The number of eigenvectors to augment, if deflation is needed
   Output :
    - The shifts as complex pair of arrays in wr and wi (size max_k).
    - The harmonic Ritz vectors (agmres->U) if deflation is needed.
*/
static PetscErrorCode KSPComputeShifts_DGMRES(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_AGMRES     *agmres = (KSP_AGMRES*)(ksp->data);
  PetscInt       max_k   = agmres->max_k; /* size of the (non augmented) Krylov subspace */
  PetscInt       Neig    = 0;
  const PetscInt max_it  = ksp->max_it;
  PetscBool      flg;

  /* Perform one cycle of dgmres to find the eigenvalues and compute the first approximations of the eigenvectors */

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(KSP_AGMRESComputeShifts, ksp, 0,0,0);CHKERRQ(ierr);
  /* Send the size of the augmented basis to DGMRES */
  ksp->max_it             = max_k; /* set this to have DGMRES performing only one cycle */
  ksp->ops->buildsolution = KSPBuildSolution_DGMRES;
  ierr                    = KSPSolve_DGMRES(ksp);CHKERRQ(ierr);
  ksp->guess_zero         = PETSC_FALSE;
  if (ksp->reason == KSP_CONVERGED_RTOL) {
    ierr = PetscLogEventEnd(KSP_AGMRESComputeShifts, ksp, 0,0,0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  } else ksp->reason = KSP_CONVERGED_ITERATING;

  if ((agmres->r == 0) && (agmres->neig > 0)) {  /* Compute the eigenvalues for the shifts and the eigenvectors (to augment the Newton basis) */
    agmres->HasSchur = PETSC_FALSE;
    ierr             = KSPDGMRESComputeDeflationData_DGMRES (ksp, &Neig);CHKERRQ (ierr);
    Neig             = max_k;
  } else { /* From DGMRES, compute only the eigenvalues needed as Shifts for the Newton Basis */
    ierr =  KSPDGMRESComputeSchurForm_DGMRES(ksp, &Neig);CHKERRQ(ierr);
  }

  /* It may happen that the Ritz values from one cycle of GMRES are not accurate enough to provide a good stability. In this case, another cycle of GMRES is performed.  The two sets of values thus generated are sorted and the most accurate are kept as shifts */
  ierr = PetscOptionsHasName(NULL,NULL, "-ksp_agmres_ImproveShifts", &flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = KSPAGMRESLejaOrdering(agmres->wr, agmres->wi, agmres->Rshift, agmres->Ishift, max_k);CHKERRQ(ierr);
  } else { /* Perform another cycle of DGMRES to find another set of eigenvalues */
    PetscInt    i;
    PetscScalar *wr, *wi,*Rshift, *Ishift;
    ierr = PetscMalloc4(2*max_k, &wr, 2*max_k, &wi, 2*max_k, &Rshift, 2*max_k, &Ishift);CHKERRQ(ierr);
    for (i = 0; i < max_k; i++) {
      wr[i] = agmres->wr[i];
      wi[i] = agmres->wi[i];
    }

    ierr = KSPSolve_DGMRES(ksp);CHKERRQ(ierr);

    ksp->guess_zero = PETSC_FALSE;
    if (ksp->reason == KSP_CONVERGED_RTOL) PetscFunctionReturn(0);
    else ksp->reason = KSP_CONVERGED_ITERATING;
    if (agmres->neig > 0) { /* Compute the eigenvalues for the shifts) and the eigenvectors (to augment the Newton basis */
      agmres->HasSchur = PETSC_FALSE;

      ierr = KSPDGMRESComputeDeflationData_DGMRES(ksp, &Neig);CHKERRQ(ierr);
      Neig = max_k;
    } else { /* From DGMRES, compute only the eigenvalues needed as Shifts for the Newton Basis */
      ierr =  KSPDGMRESComputeSchurForm_DGMRES(ksp, &Neig);CHKERRQ(ierr);
    }
    for (i = 0; i < max_k; i++) {
      wr[max_k+i] = agmres->wr[i];
      wi[max_k+i] = agmres->wi[i];
    }
    ierr = KSPAGMRESLejaOrdering(wr, wi, Rshift, Ishift, 2*max_k);CHKERRQ(ierr);
    for (i = 0; i< max_k; i++) {
      agmres->Rshift[i] = Rshift[i];
      agmres->Ishift[i] = Ishift[i];
    }
    ierr = PetscFree(Rshift);CHKERRQ(ierr);
    ierr = PetscFree(wr);CHKERRQ(ierr);
    ierr = PetscFree(Ishift);CHKERRQ(ierr);
    ierr = PetscFree(wi);CHKERRQ(ierr);
  }
  agmres->HasShifts = PETSC_TRUE;
  ksp->max_it       = max_it;
  ierr              = PetscLogEventEnd(KSP_AGMRESComputeShifts, ksp, 0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   Generate the basis vectors from the Newton polynomials with shifts and scaling factors
   The scaling factors are computed to obtain unit vectors. Note that this step can be avoided with the preprocessing option KSP_AGMRES_NONORM.
   Inputs :
    - Operators (Matrix and preconditioners and the first basis vector in VEC_V(0)
    - Shifts values in agmres->Rshift and agmres->Ishift.
   Output :
    - agmres->vecs or VEC_V : basis vectors
    - agmres->Scale : Scaling factors (equal to 1 if no scaling is done)
*/
static PetscErrorCode KSPAGMRESBuildBasis(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_AGMRES     *agmres = (KSP_AGMRES*)ksp->data;
  PetscReal      *Rshift = agmres->Rshift;
  PetscReal      *Ishift = agmres->Ishift;
  PetscReal      *Scale  = agmres->Scale;
  const PetscInt max_k   = agmres->max_k;
  PetscInt       KspSize = KSPSIZE;  /* if max_k == KspSizen then the basis should not be augmented */
  PetscInt       j       = 1;

  PetscFunctionBegin;
  ierr     = PetscLogEventBegin(KSP_AGMRESBuildBasis, ksp, 0,0,0);CHKERRQ(ierr);
  Scale[0] = 1.0;
  while (j <= max_k) {
    if (Ishift[j-1] == 0) {
      if ((ksp->pc_side == PC_LEFT) && agmres->r && agmres->DeflPrecond) {
        /* Apply the precond-matrix operators */
        ierr = KSP_PCApplyBAorAB(ksp, VEC_V(j-1), VEC_TMP, VEC_TMP_MATOP);CHKERRQ(ierr);
        /* Then apply deflation as a preconditioner */
        ierr = KSPDGMRESApplyDeflation_DGMRES(ksp, VEC_TMP, VEC_V(j));CHKERRQ(ierr);
      } else if ((ksp->pc_side == PC_RIGHT) && agmres->r && agmres->DeflPrecond) {
        ierr = KSPDGMRESApplyDeflation_DGMRES(ksp, VEC_V(j-1), VEC_TMP);CHKERRQ(ierr);
        ierr = KSP_PCApplyBAorAB(ksp, VEC_TMP, VEC_V(j), VEC_TMP_MATOP);CHKERRQ(ierr);
      } else {
        ierr = KSP_PCApplyBAorAB(ksp, VEC_V(j-1), VEC_V(j), VEC_TMP_MATOP);CHKERRQ(ierr);
      }
      ierr = VecAXPY(VEC_V(j), -Rshift[j-1], VEC_V(j-1));CHKERRQ(ierr);
#if defined(KSP_AGMRES_NONORM)
      Scale[j] = 1.0;
#else
      ierr     = VecScale(VEC_V(j), Scale[j-1]);CHKERRQ(ierr); /* This step can be postponed until all vectors are built */
      ierr     = VecNorm(VEC_V(j), NORM_2, &(Scale[j]));CHKERRQ(ierr);
      Scale[j] = 1.0/Scale[j];
#endif

      agmres->matvecs += 1;
      j++;
    } else {
      if ((ksp->pc_side == PC_LEFT) && agmres->r && agmres->DeflPrecond) {
        /* Apply the precond-matrix operators */
        ierr = KSP_PCApplyBAorAB(ksp, VEC_V(j-1), VEC_TMP, VEC_TMP_MATOP);CHKERRQ(ierr);
        /* Then apply deflation as a preconditioner */
        ierr = KSPDGMRESApplyDeflation_DGMRES(ksp, VEC_TMP, VEC_V(j));CHKERRQ(ierr);
      } else if ((ksp->pc_side == PC_RIGHT) && agmres->r && agmres->DeflPrecond) {
        ierr = KSPDGMRESApplyDeflation_DGMRES(ksp, VEC_V(j-1), VEC_TMP);CHKERRQ(ierr);
        ierr = KSP_PCApplyBAorAB(ksp, VEC_TMP, VEC_V(j), VEC_TMP_MATOP);CHKERRQ(ierr);
      } else {
        ierr = KSP_PCApplyBAorAB(ksp, VEC_V(j-1), VEC_V(j), VEC_TMP_MATOP);CHKERRQ(ierr);
      }
      ierr = VecAXPY(VEC_V(j), -Rshift[j-1], VEC_V(j-1));CHKERRQ(ierr);
#if defined(KSP_AGMRES_NONORM)
      Scale[j] = 1.0;
#else
      ierr     = VecScale(VEC_V(j), Scale[j-1]);CHKERRQ(ierr);
      ierr     = VecNorm(VEC_V(j), NORM_2, &(Scale[j]));CHKERRQ(ierr);
      Scale[j] = 1.0/Scale[j];
#endif
      agmres->matvecs += 1;
      j++;
      if ((ksp->pc_side == PC_LEFT) && agmres->r && agmres->DeflPrecond) {
        /* Apply the precond-matrix operators */
        ierr = KSP_PCApplyBAorAB(ksp, VEC_V(j-1), VEC_TMP, VEC_TMP_MATOP);CHKERRQ(ierr);
        /* Then apply deflation as a preconditioner */
        ierr = KSPDGMRESApplyDeflation_DGMRES(ksp, VEC_TMP, VEC_V(j));CHKERRQ(ierr);
      } else if ((ksp->pc_side == PC_RIGHT) && agmres->r && agmres->DeflPrecond) {
        ierr = KSPDGMRESApplyDeflation_DGMRES(ksp, VEC_V(j-1), VEC_TMP);CHKERRQ(ierr);
        ierr = KSP_PCApplyBAorAB(ksp, VEC_TMP, VEC_V(j), VEC_TMP_MATOP);CHKERRQ(ierr);
      } else {
        ierr = KSP_PCApplyBAorAB(ksp, VEC_V(j-1), VEC_V(j), VEC_TMP_MATOP);CHKERRQ(ierr);
      }
      ierr = VecAXPY(VEC_V(j), -Rshift[j-2], VEC_V(j-1));CHKERRQ(ierr);
      ierr = VecAXPY(VEC_V(j), Scale[j-2]*Ishift[j-2]*Ishift[j-2], VEC_V(j-2));CHKERRQ(ierr);
#if defined(KSP_AGMRES_NONORM)
      Scale[j] = 1.0;
#else
      ierr     = VecNorm(VEC_V(j), NORM_2, &(Scale[j]));CHKERRQ(ierr);
      Scale[j] = 1.0/Scale[j];
#endif
      agmres->matvecs += 1;
      j++;
    }
  }
  /* Augment the subspace with the eigenvectors*/
  while (j <= KspSize) {
    ierr = KSP_PCApplyBAorAB(ksp, agmres->U[j - max_k - 1], VEC_V(j), VEC_TMP_MATOP);CHKERRQ(ierr);
#if defined(KSP_AGMRES_NONORM)
    Scale[j] = 1.0;
#else
    ierr     = VecScale(VEC_V(j), Scale[j-1]);CHKERRQ(ierr);
    ierr     = VecNorm(VEC_V(j), NORM_2, &(Scale[j]));CHKERRQ(ierr);
    Scale[j] = 1.0/Scale[j];
#endif
    agmres->matvecs += 1;
    j++;
  }
  ierr = PetscLogEventEnd(KSP_AGMRESBuildBasis, ksp, 0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  Form the Hessenberg matrix for the Arnoldi-like relation.
   Inputs :
   - Shifts values in agmres->Rshift and agmres->Ishift
   - RLoc : Triangular matrix from the RODDEC orthogonalization
   Outputs :
   - H = agmres->hh_origin : The Hessenberg matrix.

   NOTE: Note that the computed Hessenberg matrix is not mathematically equivalent to that in the real Arnoldi process (in KSP GMRES). If it is needed, it can be explicitly  formed as H <-- H * RLoc^-1.
 */
static PetscErrorCode KSPAGMRESBuildHessenberg(KSP ksp)
{
  KSP_AGMRES     *agmres = (KSP_AGMRES*)ksp->data;
  PetscScalar    *Rshift = agmres->Rshift;
  PetscScalar    *Ishift = agmres->Ishift;
  PetscScalar    *Scale  = agmres->Scale;
  PetscErrorCode ierr;
  PetscInt       i       = 0, j = 0;
  const PetscInt max_k   = agmres->max_k;
  PetscInt       KspSize = KSPSIZE;
  PetscInt       N       = MAXKSPSIZE+1;

  PetscFunctionBegin;
  ierr = PetscArrayzero(agmres->hh_origin, (N+1)*N);CHKERRQ(ierr);
  while (j < max_k) {
    /* Real shifts */
    if (Ishift[j] == 0) {
      for (i = 0; i <= j; i++) {
        *H(i,j) = *RLOC(i,j+1)/Scale[j]  + (Rshift[j] * *RLOC(i,j));
      }
      *H(j+1,j) = *RLOC(j+1,j+1)/Scale[j];
      j++;
    } else if (Ishift[j] > 0) {
      for (i = 0; i <= j; i++) {
        *H(i,j) = *RLOC(i,j+1)/Scale[j] +  Rshift[j] * *RLOC(i, j);
      }
      *H(j+1,j) = *RLOC(j+1, j+1)/Scale[j];
      j++;
      for (i = 0; i <= j; i++) {
        *H(i,j) = (*RLOC(i,j+1) + Rshift[j-1] * *RLOC(i,j) - Scale[j-1] * Ishift[j-1]*Ishift[j-1]* *RLOC(i,j-1));
      }
      *H(j,j) = (*RLOC(j,j+1) + Rshift[j-1] * *RLOC(j,j));
      *H(j+1,j) = *RLOC(j+1,j+1);
      j++;
    } else SETERRQ(PetscObjectComm((PetscObject)ksp), PETSC_ERR_ORDER, "BAD ORDERING OF THE SHIFTS VALUES IN THE NEWTON BASIS");
  }
  for (j = max_k; j< KspSize; j++) { /* take into account the norm of the augmented vectors */
    for (i = 0; i <= j+1; i++) *H(i,j) = *RLOC(i, j+1)/Scale[j];
  }
  PetscFunctionReturn(0);
}

/*
  Form the new approximate solution from the least-square problem
*/
static PetscErrorCode KSPAGMRESBuildSoln(KSP ksp,PetscInt it)
{
  KSP_AGMRES     *agmres = (KSP_AGMRES*)ksp->data;
  PetscErrorCode ierr;
  const PetscInt max_k = agmres->max_k;       /* Size of the non-augmented Krylov basis */
  PetscInt       i, j;
  PetscInt       r = agmres->r;           /* current number of augmented eigenvectors */
  PetscBLASInt   KspSize;
  PetscBLASInt   lC;
  PetscBLASInt   N;
  PetscBLASInt   ldH = it + 1;
  PetscBLASInt   lwork;
  PetscBLASInt   info, nrhs = 1;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(KSPSIZE,&KspSize);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(4 * (KspSize+1),&lwork);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(KspSize+1,&lC);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(MAXKSPSIZE + 1,&N);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(N + 1,&ldH);CHKERRQ(ierr);
  /* Save a copy of the Hessenberg matrix */
  for (j = 0; j < N-1; j++) {
    for (i = 0; i < N; i++) {
      *HS(i,j) = *H(i,j);
    }
  }
  /* QR factorize the Hessenberg matrix */
  PetscStackCallBLAS("LAPACKgeqrf",LAPACKgeqrf_(&lC, &KspSize, agmres->hh_origin, &ldH, agmres->tau, agmres->work, &lwork, &info));
  PetscAssertFalse(info,PetscObjectComm((PetscObject)ksp), PETSC_ERR_LIB,"Error in LAPACK routine XGEQRF INFO=%d", info);
  /* Update the right hand side of the least square problem */
  ierr = PetscArrayzero(agmres->nrs, N);CHKERRQ(ierr);

  agmres->nrs[0] = ksp->rnorm;
  PetscStackCallBLAS("LAPACKormqr",LAPACKormqr_("L", "T", &lC, &nrhs, &KspSize, agmres->hh_origin, &ldH, agmres->tau, agmres->nrs, &N, agmres->work, &lwork, &info));
  PetscAssertFalse(info,PetscObjectComm((PetscObject)ksp), PETSC_ERR_LIB,"Error in LAPACK routine XORMQR INFO=%d",info);
  ksp->rnorm = PetscAbsScalar(agmres->nrs[KspSize]);
  /* solve the least-square problem */
  PetscStackCallBLAS("LAPACKtrtrs",LAPACKtrtrs_("U", "N", "N", &KspSize, &nrhs, agmres->hh_origin, &ldH, agmres->nrs, &N, &info));
  PetscAssertFalse(info,PetscObjectComm((PetscObject)ksp), PETSC_ERR_LIB,"Error in LAPACK routine XTRTRS INFO=%d",info);
  /* Accumulate the correction to the solution of the preconditioned problem in VEC_TMP */
  ierr = VecZeroEntries(VEC_TMP);CHKERRQ(ierr);
  ierr = VecMAXPY(VEC_TMP, max_k, agmres->nrs, &VEC_V(0));CHKERRQ(ierr);
  if (!agmres->DeflPrecond) { ierr = VecMAXPY(VEC_TMP, r, &agmres->nrs[max_k], agmres->U);CHKERRQ(ierr); }

  if ((ksp->pc_side == PC_RIGHT) && agmres->r && agmres->DeflPrecond) {
    ierr = KSPDGMRESApplyDeflation_DGMRES(ksp, VEC_TMP, VEC_TMP_MATOP);CHKERRQ(ierr);
    ierr = VecCopy(VEC_TMP_MATOP, VEC_TMP);CHKERRQ(ierr);
  }
  ierr = KSPUnwindPreconditioner(ksp, VEC_TMP, VEC_TMP_MATOP);CHKERRQ(ierr);
  /* add the solution to the previous one */
  ierr = VecAXPY(ksp->vec_sol, 1.0, VEC_TMP);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   Run  one cycle of the Newton-basis gmres, possibly augmented with eigenvectors.

   Return residual history if requested.
   Input :
   - The vector VEC_V(0) is the initia residual
   Output :
    - the solution vector is in agmres->vec_sol
   - itcount : number of inner iterations
    - res : the new residual norm
 .
 NOTE: Unlike GMRES where the residual norm is available at each (inner) iteration,  here it is available at the end of the cycle.
*/
static PetscErrorCode KSPAGMRESCycle(PetscInt *itcount,KSP ksp)
{
  KSP_AGMRES     *agmres = (KSP_AGMRES*)(ksp->data);
  PetscReal      res;
  PetscErrorCode ierr;
  PetscInt       KspSize = KSPSIZE;

  PetscFunctionBegin;
  /* check for the convergence */
  res = ksp->rnorm; /* Norm of the initial residual vector */
  if (!res) {
    if (itcount) *itcount = 0;
    ksp->reason = KSP_CONVERGED_ATOL;
    ierr        = PetscInfo(ksp,"Converged due to zero residual norm on entry\n");CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = (*ksp->converged)(ksp,ksp->its,res,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  /* Build the Krylov basis with Newton polynomials */
  ierr = KSPAGMRESBuildBasis(ksp);CHKERRQ(ierr);
  /* QR Factorize the basis with RODDEC */
  ierr = KSPAGMRESRoddec(ksp, KspSize+1);CHKERRQ(ierr);

  /* Recover a (partial) Hessenberg matrix for the Arnoldi-like relation */
  ierr = KSPAGMRESBuildHessenberg(ksp);CHKERRQ(ierr);
  /* Solve the least square problem and unwind the preconditioner */
  ierr = KSPAGMRESBuildSoln(ksp, KspSize);CHKERRQ(ierr);

  res        = ksp->rnorm;
  ksp->its  += KspSize;
  agmres->it = KspSize-1;
  /*  Test for the convergence */
  ierr = (*ksp->converged)(ksp,ksp->its,res,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  ierr = KSPLogResidualHistory(ksp,res);CHKERRQ(ierr);
  ierr = KSPMonitor(ksp,ksp->its,res);CHKERRQ(ierr);

  *itcount = KspSize;
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSolve_AGMRES(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       its;
  KSP_AGMRES     *agmres    = (KSP_AGMRES*)ksp->data;
  PetscBool      guess_zero = ksp->guess_zero;
  PetscReal      res_old, res;
  PetscInt       test;

  PetscFunctionBegin;
  ierr     = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
  ksp->its = 0;
  ierr     = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);

  ksp->reason = KSP_CONVERGED_ITERATING;
  if (!agmres->HasShifts) { /* Compute Shifts for the Newton basis */
    ierr = KSPComputeShifts_DGMRES(ksp);CHKERRQ(ierr);
  }
  /* NOTE: At this step, the initial guess is not equal to zero since one cycle of the classical GMRES is performed to compute the shifts */
  ierr = (*ksp->converged)(ksp,0,ksp->rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  while (!ksp->reason) {
    ierr = KSPInitialResidual(ksp,ksp->vec_sol,VEC_TMP,VEC_TMP_MATOP,VEC_V(0),ksp->vec_rhs);CHKERRQ(ierr);
    if ((ksp->pc_side == PC_LEFT) && agmres->r && agmres->DeflPrecond) {
      ierr = KSPDGMRESApplyDeflation_DGMRES(ksp, VEC_V(0), VEC_TMP);CHKERRQ(ierr);
      ierr = VecCopy(VEC_TMP, VEC_V(0));CHKERRQ(ierr);

      agmres->matvecs += 1;
    }
    ierr    = VecNormalize(VEC_V(0),&(ksp->rnorm));CHKERRQ(ierr);
    KSPCheckNorm(ksp,ksp->rnorm);
    res_old = ksp->rnorm; /* Record the residual norm to test if deflation is needed */

    ksp->ops->buildsolution = KSPBuildSolution_AGMRES;

    ierr     = KSPAGMRESCycle(&its,ksp);CHKERRQ(ierr);
    if (ksp->its >= ksp->max_it) {
      if (!ksp->reason) ksp->reason = KSP_DIVERGED_ITS;
      break;
    }
    /* compute the eigenvectors to augment the subspace : use an adaptive strategy */
    res = ksp->rnorm;
    if (!ksp->reason && agmres->neig > 0) {
      test = agmres->max_k * PetscLogReal(ksp->rtol/res) / PetscLogReal(res/res_old); /* estimate the remaining number of steps */
      if ((test > agmres->smv*(ksp->max_it-ksp->its)) || agmres->force) {
        if (!agmres->force && ((test > agmres->bgv*(ksp->max_it-ksp->its)) && ((agmres->r + 1) < agmres->max_neig))) {
          agmres->neig += 1; /* Augment the number of eigenvalues to deflate if the convergence is too slow */
        }
        ierr = KSPDGMRESComputeDeflationData_DGMRES(ksp,&agmres->neig);CHKERRQ(ierr);
      }
    }
    ksp->guess_zero = PETSC_FALSE; /* every future call to KSPInitialResidual() will have nonzero guess */
  }
  ksp->guess_zero = guess_zero; /* restore if user has provided nonzero initial guess */
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPDestroy_AGMRES(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_AGMRES     *agmres = (KSP_AGMRES*)ksp->data;

  PetscFunctionBegin;
  ierr = PetscFree(agmres->hh_origin);CHKERRQ(ierr);

  ierr = PetscFree(agmres->Qloc);CHKERRQ(ierr);
  ierr = PetscFree4(agmres->Rshift,agmres->Ishift,agmres->Rloc,agmres->wbufptr);CHKERRQ(ierr);
  ierr = PetscFree3(agmres->tau,agmres->work,agmres->nrs);CHKERRQ(ierr);
  ierr = PetscFree4(agmres->Scale,agmres->sgn,agmres->tloc,agmres->temp);CHKERRQ(ierr);

  ierr = PetscFree(agmres->select);CHKERRQ(ierr);
  ierr = PetscFree(agmres->wr);CHKERRQ(ierr);
  ierr = PetscFree(agmres->wi);CHKERRQ(ierr);
  if (agmres->neig) {
    ierr = VecDestroyVecs(MAXKSPSIZE,&agmres->TmpU);CHKERRQ(ierr);
    ierr = PetscFree(agmres->perm);CHKERRQ(ierr);
    ierr = PetscFree(agmres->MatEigL);CHKERRQ(ierr);
    ierr = PetscFree(agmres->MatEigR);CHKERRQ(ierr);
    ierr = PetscFree(agmres->Q);CHKERRQ(ierr);
    ierr = PetscFree(agmres->Z);CHKERRQ(ierr);
    ierr = PetscFree(agmres->beta);CHKERRQ(ierr);
  }
  ierr = KSPDestroy_DGMRES(ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPView_AGMRES(KSP ksp,PetscViewer viewer)
{
  KSP_AGMRES     *agmres = (KSP_AGMRES*)ksp->data;
  const char     *cstr   = "RODDEC ORTHOGONOLIZATION";
  char           ritzvec[25];
  PetscErrorCode ierr;
  PetscBool      iascii,isstring;
#if defined(KSP_AGMRES_NONORM)
  const char *Nstr = "SCALING FACTORS : NO";
#else
  const char *Nstr = "SCALING FACTORS : YES";
#endif

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring);CHKERRQ(ierr);

  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer, " restart=%d using %s\n", agmres->max_k, cstr);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, " %s\n", Nstr);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, " Number of matvecs : %D\n", agmres->matvecs);CHKERRQ(ierr);
    if (agmres->force) {ierr = PetscViewerASCIIPrintf (viewer, " Adaptive strategy is used: FALSE\n");CHKERRQ(ierr);}
    else PetscViewerASCIIPrintf(viewer, " Adaptive strategy is used: TRUE\n");
    if (agmres->DeflPrecond) {
      ierr = PetscViewerASCIIPrintf(viewer, " STRATEGY OF DEFLATION: PRECONDITIONER \n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, "  Frequency of extracted eigenvalues = %D\n", agmres->neig);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, "  Total number of extracted eigenvalues = %D\n", agmres->r);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, "  Maximum number of eigenvalues set to be extracted = %D\n", agmres->max_neig);CHKERRQ(ierr);
    } else {
      if (agmres->ritz) sprintf(ritzvec, "Ritz vectors");
      else sprintf(ritzvec, "Harmonic Ritz vectors");
      ierr = PetscViewerASCIIPrintf(viewer, " STRATEGY OF DEFLATION: AUGMENT\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer," augmented vectors  %d at frequency %d with %s\n", agmres->r, agmres->neig, ritzvec);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer, " Minimum relaxation parameter for the adaptive strategy(smv)  = %g\n", agmres->smv);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, " Maximum relaxation parameter for the adaptive strategy(bgv)  = %g\n", agmres->bgv);CHKERRQ(ierr);
  } else if (isstring) {
    ierr = PetscViewerStringSPrintf(viewer,"%s restart %D",cstr,agmres->max_k);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSetFromOptions_AGMRES(PetscOptionItems *PetscOptionsObject,KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       neig;
  KSP_AGMRES     *agmres = (KSP_AGMRES*)ksp->data;
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = KSPSetFromOptions_DGMRES(PetscOptionsObject,ksp);CHKERRQ(ierr);  /* Set common options from DGMRES and GMRES */
  ierr = PetscOptionsHead(PetscOptionsObject,"KSP AGMRES Options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-ksp_agmres_eigen", "Number of eigenvalues to deflate", "KSPDGMRESSetEigen", agmres->neig, &neig, &flg);CHKERRQ(ierr);
  if (flg) {
    ierr      = KSPDGMRESSetEigen_DGMRES(ksp, neig);CHKERRQ(ierr);
    agmres->r = 0;
  } else agmres->neig = 0;
  ierr = PetscOptionsInt("-ksp_agmres_maxeigen", "Maximum number of eigenvalues to deflate", "KSPDGMRESSetMaxEigen", agmres->max_neig, &neig, &flg);CHKERRQ(ierr);
  if (flg) agmres->max_neig = neig+EIG_OFFSET;
  else agmres->max_neig = agmres->neig+EIG_OFFSET;
  ierr                = PetscOptionsBool("-ksp_agmres_DeflPrecond", "Determine if the deflation should be applied as a preconditioner -- similar to KSP DGMRES", "KSPGMRESDeflPrecond",agmres->DeflPrecond,&agmres->DeflPrecond,NULL);CHKERRQ(ierr);
  ierr                = PetscOptionsBool("-ksp_agmres_ritz", "Compute the Ritz vectors instead of the Harmonic Ritz vectors ", "KSPGMRESHarmonic",agmres->ritz,&agmres->ritz ,&flg);CHKERRQ(ierr);
  ierr                = PetscOptionsReal("-ksp_agmres_MinRatio", "Relaxation parameter in the adaptive strategy; smallest multiple of the remaining number of steps allowed", "KSPGMRESSetMinRatio", agmres->smv, &agmres->smv, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ksp_agmres_MaxRatio", "Relaxation parameter in the adaptive strategy; Largest multiple of the remaining number of steps allowed", "KSPGMRESSetMaxRatio",agmres->bgv,&agmres->bgv, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
 KSPAGMRES - Newton basis GMRES implementation with adaptive augmented eigenvectors

The techniques used are best described in [1]. The contribution of this work is that it combines many of the previous work to reduce the amount of MPI messages and improve the robustness of the global approach by using deflation techniques. It has been successfully applied to a class of real and industrial problems. Please see [1] for numerical experiments and [2] for a description of these problems.
There are  many ongoing work that aim at avoiding (or minimizing) the communication in Krylov subspace methods. This code can be used as an experimental framework to combine several techniques in the particular case of GMRES. For instance, the computation of the shifts can be improved with techniques described in [3]. The orthogonalization technique can be replaced by TSQR [4]. The generation of the basis can be done using s-steps approaches[5].

 Options Database Keys:
 +   -ksp_gmres_restart <restart> -  the number of Krylov directions
 .   -ksp_gmres_krylov_monitor - plot the Krylov space generated
 .   -ksp_agmres_eigen <neig> - Number of eigenvalues to deflate (Number of vectors to augment)
 .   -ksp_agmres_maxeigen <max_neig> - Maximum number of eigenvalues to deflate
 .   -ksp_agmres_MinRatio <1> - Relaxation parameter in the adaptive strategy; smallest multiple of the remaining number of steps allowed
 .   -ksp_agmres_MaxRatio <1> - Relaxation parameter in the adaptive strategy; Largest multiple of the remaining number of steps allowed
 .   -ksp_agmres_DeflPrecond - Apply deflation as a preconditioner, this is similar to DGMRES but it rather builds a Newton basis.  This is an experimental option.
 -   -ksp_dgmres_force <0, 1> - Force the deflation at each restart.

 Level: beginner

 Notes:
    Left and right preconditioning are supported, but not symmetric preconditioning. Complex arithmetic is not supported

 Developer Notes:
    This object is subclassed off of KSPDGMRES

 Contributed by Desire NUENTSA WAKAM, INRIA <desire.nuentsa_wakam@inria.fr>
 Inputs from Guy Atenekeng <atenekeng@yahoo.com> and R.B. Sidje <roger.b.sidje@ua.edu>

 References :
 +   [1] D. Nuentsa Wakam and J. Erhel, Parallelism and robustness in GMRES with the Newton basis and the deflated restarting. Research report INRIA RR-7787, November 2011,https://hal.inria.fr/inria-00638247/en,  in revision for ETNA.
 .  [2] D. NUENTSA WAKAM and F. PACULL, Memory Efficient Hybrid Algebraic Solvers for Linear Systems Arising from Compressible Flows, Computers and Fluids, In Press, http://dx.doi.org/10.1016/j.compfluid.2012.03.023
 .  [3] B. Philippe and L. Reichel, On the generation of Krylov subspace bases, Applied Numerical
Mathematics, 62(9), pp. 1171-1186, 2012
 .  [4] J. Demmel, L. Grigori, M. F. Hoemmen, and J. Langou, Communication-optimal parallel and sequential QR and LU factorizations, SIAM journal on Scientific Computing, 34(1), A206-A239, 2012
 .  [5] M. Mohiyuddin, M. Hoemmen, J. Demmel, and K. Yelick, Minimizing communication in sparse matrix solvers, in SC '09: Proceedings of the Conference on High Performance Computing Networking, Storage and Analysis, New York, NY, USA, 2009, ACM, pp. 1154-1171.
 .    Sidje, Roger B. Alternatives for parallel Krylov subspace basis computation. Numer. Linear Algebra Appl. 4 (1997), no. 4, 305-331

 .seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPDGMRES, KSPPGMRES,
 KSPGMRESSetRestart(), KSPGMRESSetHapTol(), KSPGMRESSetPreAllocateVectors(), KSPGMRESSetOrthogonalization(), KSPGMRESGetOrthogonalization(),
 KSPGMRESClassicalGramSchmidtOrthogonalization(), KSPGMRESModifiedGramSchmidtOrthogonalization(),
 KSPGMRESCGSRefinementType, KSPGMRESSetCGSRefinementType(), KSPGMRESGetCGSRefinementType(), KSPGMRESMonitorKrylov(), KSPSetPCSide()
 M*/

PETSC_EXTERN PetscErrorCode KSPCreate_AGMRES(KSP ksp)
{
  KSP_AGMRES     *agmres;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr      = PetscNewLog(ksp,&agmres);CHKERRQ(ierr);
  ksp->data = (void*)agmres;

  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,3);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_RIGHT,2);CHKERRQ(ierr);
  ksp->ops->buildsolution                = KSPBuildSolution_AGMRES;
  ksp->ops->setup                        = KSPSetUp_AGMRES;
  ksp->ops->solve                        = KSPSolve_AGMRES;
  ksp->ops->destroy                      = KSPDestroy_AGMRES;
  ksp->ops->view                         = KSPView_AGMRES;
  ksp->ops->setfromoptions               = KSPSetFromOptions_AGMRES;
  ksp->guess_zero                        = PETSC_TRUE;
  ksp->ops->computeextremesingularvalues = KSPComputeExtremeSingularValues_GMRES;
  ksp->ops->computeeigenvalues           = KSPComputeEigenvalues_GMRES;

  ierr = PetscObjectComposeFunction((PetscObject) ksp,"KSPGMRESSetPreAllocateVectors_C",KSPGMRESSetPreAllocateVectors_GMRES);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) ksp,"KSPGMRESSetOrthogonalization_C",KSPGMRESSetOrthogonalization_GMRES);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) ksp,"KSPGMRESSetRestart_C",KSPGMRESSetRestart_GMRES);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) ksp,"KSPGMRESSetHapTol_C",KSPGMRESSetHapTol_GMRES);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) ksp,"KSPGMRESSetCGSRefinementType_C",KSPGMRESSetCGSRefinementType_GMRES);CHKERRQ(ierr);
  /* -- New functions defined in DGMRES -- */
  ierr = PetscObjectComposeFunction((PetscObject) ksp, "KSPDGMRESSetEigen_C",KSPDGMRESSetEigen_DGMRES);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) ksp, "KSPDGMRESComputeSchurForm_C",KSPDGMRESComputeSchurForm_DGMRES);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) ksp, "KSPDGMRESComputeDeflationData_C",KSPDGMRESComputeDeflationData_DGMRES);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) ksp, "KSPDGMRESApplyDeflation_C",KSPDGMRESApplyDeflation_DGMRES);CHKERRQ(ierr);

  ierr = PetscLogEventRegister("AGMRESCompDefl",   KSP_CLASSID, &KSP_AGMRESComputeDeflationData);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("AGMRESBuildBasis", KSP_CLASSID, &KSP_AGMRESBuildBasis);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("AGMRESCompShifts", KSP_CLASSID, &KSP_AGMRESComputeShifts);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("AGMRESOrthog",     KSP_CLASSID, &KSP_AGMRESRoddec);CHKERRQ(ierr);

  agmres->haptol         = 1.0e-30;
  agmres->q_preallocate  = 0;
  agmres->delta_allocate = AGMRES_DELTA_DIRECTIONS;
  agmres->orthog         = KSPGMRESClassicalGramSchmidtOrthogonalization;
  agmres->nrs            = NULL;
  agmres->sol_temp       = NULL;
  agmres->max_k          = AGMRES_DEFAULT_MAXK;
  agmres->Rsvd           = NULL;
  agmres->cgstype        = KSP_GMRES_CGS_REFINE_NEVER;
  agmres->orthogwork     = NULL;

  /* Default values for the deflation */
  agmres->r           = 0;
  agmres->neig        = 0;
  agmres->max_neig    = 0;
  agmres->lambdaN     = 0.0;
  agmres->smv         = SMV;
  agmres->bgv         = 1;
  agmres->force       = PETSC_FALSE;
  agmres->matvecs     = 0;
  agmres->improve     = PETSC_FALSE;
  agmres->HasShifts   = PETSC_FALSE;
  agmres->r           = 0;
  agmres->HasSchur    = PETSC_FALSE;
  agmres->DeflPrecond = PETSC_FALSE;
  ierr = PetscObjectGetNewTag((PetscObject)ksp,&agmres->tag);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
