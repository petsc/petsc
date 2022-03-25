
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
  PetscInt        hes;
  PetscInt        nloc;
  KSP_AGMRES      *agmres = (KSP_AGMRES*)ksp->data;
  PetscInt        neig    = agmres->neig;
  const PetscInt  max_k   = agmres->max_k;
  PetscInt        N       = MAXKSPSIZE;
  PetscInt        lwork   = PetscMax(8 * N + 16, 4 * neig * (N - neig));

  PetscFunctionBegin;
  PetscCheckFalse(ksp->pc_side == PC_SYMMETRIC,PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"no symmetric preconditioning for KSPAGMRES");
  N     = MAXKSPSIZE;
  /* Preallocate space during the call to KSPSetup_GMRES for the Krylov basis */
  agmres->q_preallocate = PETSC_TRUE; /* No allocation on the fly */
  /* Preallocate space to compute later the eigenvalues in GMRES */
  ksp->calc_sings = PETSC_TRUE;
  agmres->max_k   = N; /* Set the augmented size to be allocated in KSPSetup_GMRES */
  PetscCall(KSPSetUp_DGMRES(ksp));
  agmres->max_k   = max_k;
  hes             = (N + 1) * (N + 1);

  /* Data for the Newton basis GMRES */
  PetscCall(PetscCalloc4(max_k,&agmres->Rshift,max_k,&agmres->Ishift,hes,&agmres->Rloc,(N+1)*4,&agmres->wbufptr));
  PetscCall(PetscMalloc3(N+1,&agmres->tau,lwork,&agmres->work,N+1,&agmres->nrs));
  PetscCall(PetscCalloc4(N+1,&agmres->Scale,N+1,&agmres->sgn,N+1,&agmres->tloc,N+1,&agmres->temp));

  /* Allocate space for the vectors in the orthogonalized basis*/
  PetscCall(VecGetLocalSize(agmres->vecs[0], &nloc));
  PetscCall(PetscMalloc1(nloc*(N+1), &agmres->Qloc));

  /* Init the ring of processors for the roddec orthogonalization */
  PetscCall(KSPAGMRESRoddecInitNeighboor(ksp));

  if (agmres->neig < 1) PetscFunctionReturn(0);

  /* Allocate space for the deflation */
  PetscCall(PetscMalloc1(N, &agmres->select));
  PetscCall(VecDuplicateVecs(VEC_V(0), N, &agmres->TmpU));
  PetscCall(PetscMalloc2(N*N, &agmres->MatEigL, N*N, &agmres->MatEigR));
  /*  PetscCall(PetscMalloc6(N*N, &agmres->Q, N*N, &agmres->Z, N, &agmres->wr, N, &agmres->wi, N, &agmres->beta, N, &agmres->modul)); */
  PetscCall(PetscMalloc3(N*N, &agmres->Q, N*N, &agmres->Z, N, &agmres->beta));
  PetscCall(PetscMalloc2((N+1),&agmres->perm,(2*neig*N),&agmres->iwork));
  PetscFunctionReturn(0);
}

/*
    Returns the current solution from the private data structure of AGMRES back to ptr.
*/
static PetscErrorCode KSPBuildSolution_AGMRES(KSP ksp,Vec ptr, Vec *result)
{
  KSP_AGMRES     *agmres = (KSP_AGMRES*)ksp->data;

  PetscFunctionBegin;
  if (!ptr) {
    if (!agmres->sol_temp) {
      PetscCall(VecDuplicate(ksp->vec_sol,&agmres->sol_temp));
      PetscCall(VecCopy(ksp->vec_sol,agmres->sol_temp));
      PetscCall(PetscLogObjectParent((PetscObject)ksp,(PetscObject)agmres->sol_temp));
    }
    ptr = agmres->sol_temp;
  } else {
    PetscCall(VecCopy(ksp->vec_sol, ptr));
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
  PetscCall(KSPCreate(PetscObjectComm((PetscObject)ksp), &kspgmres));
  if (!ksp->pc) PetscCall(KSPGetPC(ksp,&ksp->pc));
  PetscCall(PCGetOperators(ksp->pc, &Amat, &Pmat));
  PetscCall(KSPSetOperators(kspgmres, Amat, Pmat));
  PetscCall(KSPSetFromOptions(kspgmres));
  PetscCall(PetscOptionsHasName(NULL,((PetscObject)ksp)->prefix, "-ksp_view", &flg));
  if (flg) PetscCall(PetscOptionsClearValue(NULL,"-ksp_view"));
  PetscCall(KSPSetType(kspgmres, KSPGMRES));
  PetscCall(KSPGMRESSetRestart(kspgmres, max_k));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(KSPSetPC(kspgmres, pc));
  /* Copy common options */
  kspgmres->pc_side = ksp->pc_side;
  /* Setup KSP context */
  PetscCall(KSPSetComputeEigenvalues(kspgmres, PETSC_TRUE));
  PetscCall(KSPSetUp(kspgmres));

  kspgmres->max_it = max_k; /* Restrict the maximum number of iterations to one cycle of GMRES */
  kspgmres->rtol   = ksp->rtol;

  PetscCall(KSPSolve(kspgmres, ksp->vec_rhs, ksp->vec_sol));

  ksp->guess_zero = PETSC_FALSE;
  ksp->rnorm      = kspgmres->rnorm;
  ksp->its        = kspgmres->its;
  if (kspgmres->reason == KSP_CONVERGED_RTOL) {
    ksp->reason = KSP_CONVERGED_RTOL;
    PetscFunctionReturn(0);
  } else ksp->reason = KSP_CONVERGED_ITERATING;
  /* Now, compute the Shifts values */
  PetscCall(PetscMalloc2(max_k,&Rshift,max_k,&Ishift));
  PetscCall(KSPComputeEigenvalues(kspgmres, max_k, Rshift, Ishift, &m));
  PetscCheckFalse(m < max_k,PetscObjectComm((PetscObject)ksp),PETSC_ERR_PLIB, "Unable to compute the Shifts for the Newton basis");
  else {
    PetscCall(KSPAGMRESLejaOrdering(Rshift, Ishift, agmres->Rshift, agmres->Ishift, max_k));

    agmres->HasShifts = PETSC_TRUE;
  }
  /* Restore KSP view options */
  if (flg) PetscCall(PetscOptionsSetValue(NULL,"-ksp_view", ""));
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
  KSP_AGMRES     *agmres = (KSP_AGMRES*)(ksp->data);
  PetscInt       max_k   = agmres->max_k; /* size of the (non augmented) Krylov subspace */
  PetscInt       Neig    = 0;
  const PetscInt max_it  = ksp->max_it;
  PetscBool      flg;

  /* Perform one cycle of dgmres to find the eigenvalues and compute the first approximations of the eigenvectors */

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(KSP_AGMRESComputeShifts, ksp, 0,0,0));
  /* Send the size of the augmented basis to DGMRES */
  ksp->max_it             = max_k; /* set this to have DGMRES performing only one cycle */
  ksp->ops->buildsolution = KSPBuildSolution_DGMRES;
  PetscCall(KSPSolve_DGMRES(ksp));
  ksp->guess_zero         = PETSC_FALSE;
  if (ksp->reason == KSP_CONVERGED_RTOL) {
    PetscCall(PetscLogEventEnd(KSP_AGMRESComputeShifts, ksp, 0,0,0));
    PetscFunctionReturn(0);
  } else ksp->reason = KSP_CONVERGED_ITERATING;

  if ((agmres->r == 0) && (agmres->neig > 0)) {  /* Compute the eigenvalues for the shifts and the eigenvectors (to augment the Newton basis) */
    agmres->HasSchur = PETSC_FALSE;
    PetscCall(KSPDGMRESComputeDeflationData_DGMRES(ksp, &Neig));
    Neig             = max_k;
  } else { /* From DGMRES, compute only the eigenvalues needed as Shifts for the Newton Basis */
    PetscCall(KSPDGMRESComputeSchurForm_DGMRES(ksp, &Neig));
  }

  /* It may happen that the Ritz values from one cycle of GMRES are not accurate enough to provide a good stability. In this case, another cycle of GMRES is performed.  The two sets of values thus generated are sorted and the most accurate are kept as shifts */
  PetscCall(PetscOptionsHasName(NULL,NULL, "-ksp_agmres_ImproveShifts", &flg));
  if (!flg) {
    PetscCall(KSPAGMRESLejaOrdering(agmres->wr, agmres->wi, agmres->Rshift, agmres->Ishift, max_k));
  } else { /* Perform another cycle of DGMRES to find another set of eigenvalues */
    PetscInt    i;
    PetscScalar *wr, *wi,*Rshift, *Ishift;
    PetscCall(PetscMalloc4(2*max_k, &wr, 2*max_k, &wi, 2*max_k, &Rshift, 2*max_k, &Ishift));
    for (i = 0; i < max_k; i++) {
      wr[i] = agmres->wr[i];
      wi[i] = agmres->wi[i];
    }

    PetscCall(KSPSolve_DGMRES(ksp));

    ksp->guess_zero = PETSC_FALSE;
    if (ksp->reason == KSP_CONVERGED_RTOL) PetscFunctionReturn(0);
    else ksp->reason = KSP_CONVERGED_ITERATING;
    if (agmres->neig > 0) { /* Compute the eigenvalues for the shifts) and the eigenvectors (to augment the Newton basis */
      agmres->HasSchur = PETSC_FALSE;

      PetscCall(KSPDGMRESComputeDeflationData_DGMRES(ksp, &Neig));
      Neig = max_k;
    } else { /* From DGMRES, compute only the eigenvalues needed as Shifts for the Newton Basis */
      PetscCall(KSPDGMRESComputeSchurForm_DGMRES(ksp, &Neig));
    }
    for (i = 0; i < max_k; i++) {
      wr[max_k+i] = agmres->wr[i];
      wi[max_k+i] = agmres->wi[i];
    }
    PetscCall(KSPAGMRESLejaOrdering(wr, wi, Rshift, Ishift, 2*max_k));
    for (i = 0; i< max_k; i++) {
      agmres->Rshift[i] = Rshift[i];
      agmres->Ishift[i] = Ishift[i];
    }
    PetscCall(PetscFree(Rshift));
    PetscCall(PetscFree(wr));
    PetscCall(PetscFree(Ishift));
    PetscCall(PetscFree(wi));
  }
  agmres->HasShifts = PETSC_TRUE;
  ksp->max_it       = max_it;
  PetscCall(PetscLogEventEnd(KSP_AGMRESComputeShifts, ksp, 0,0,0));
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
  KSP_AGMRES     *agmres = (KSP_AGMRES*)ksp->data;
  PetscReal      *Rshift = agmres->Rshift;
  PetscReal      *Ishift = agmres->Ishift;
  PetscReal      *Scale  = agmres->Scale;
  const PetscInt max_k   = agmres->max_k;
  PetscInt       KspSize = KSPSIZE;  /* if max_k == KspSizen then the basis should not be augmented */
  PetscInt       j       = 1;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(KSP_AGMRESBuildBasis, ksp, 0,0,0));
  Scale[0] = 1.0;
  while (j <= max_k) {
    if (Ishift[j-1] == 0) {
      if ((ksp->pc_side == PC_LEFT) && agmres->r && agmres->DeflPrecond) {
        /* Apply the precond-matrix operators */
        PetscCall(KSP_PCApplyBAorAB(ksp, VEC_V(j-1), VEC_TMP, VEC_TMP_MATOP));
        /* Then apply deflation as a preconditioner */
        PetscCall(KSPDGMRESApplyDeflation_DGMRES(ksp, VEC_TMP, VEC_V(j)));
      } else if ((ksp->pc_side == PC_RIGHT) && agmres->r && agmres->DeflPrecond) {
        PetscCall(KSPDGMRESApplyDeflation_DGMRES(ksp, VEC_V(j-1), VEC_TMP));
        PetscCall(KSP_PCApplyBAorAB(ksp, VEC_TMP, VEC_V(j), VEC_TMP_MATOP));
      } else {
        PetscCall(KSP_PCApplyBAorAB(ksp, VEC_V(j-1), VEC_V(j), VEC_TMP_MATOP));
      }
      PetscCall(VecAXPY(VEC_V(j), -Rshift[j-1], VEC_V(j-1)));
#if defined(KSP_AGMRES_NONORM)
      Scale[j] = 1.0;
#else
      PetscCall(VecScale(VEC_V(j), Scale[j-1])); /* This step can be postponed until all vectors are built */
      PetscCall(VecNorm(VEC_V(j), NORM_2, &(Scale[j])));
      Scale[j] = 1.0/Scale[j];
#endif

      agmres->matvecs += 1;
      j++;
    } else {
      if ((ksp->pc_side == PC_LEFT) && agmres->r && agmres->DeflPrecond) {
        /* Apply the precond-matrix operators */
        PetscCall(KSP_PCApplyBAorAB(ksp, VEC_V(j-1), VEC_TMP, VEC_TMP_MATOP));
        /* Then apply deflation as a preconditioner */
        PetscCall(KSPDGMRESApplyDeflation_DGMRES(ksp, VEC_TMP, VEC_V(j)));
      } else if ((ksp->pc_side == PC_RIGHT) && agmres->r && agmres->DeflPrecond) {
        PetscCall(KSPDGMRESApplyDeflation_DGMRES(ksp, VEC_V(j-1), VEC_TMP));
        PetscCall(KSP_PCApplyBAorAB(ksp, VEC_TMP, VEC_V(j), VEC_TMP_MATOP));
      } else {
        PetscCall(KSP_PCApplyBAorAB(ksp, VEC_V(j-1), VEC_V(j), VEC_TMP_MATOP));
      }
      PetscCall(VecAXPY(VEC_V(j), -Rshift[j-1], VEC_V(j-1)));
#if defined(KSP_AGMRES_NONORM)
      Scale[j] = 1.0;
#else
      PetscCall(VecScale(VEC_V(j), Scale[j-1]));
      PetscCall(VecNorm(VEC_V(j), NORM_2, &(Scale[j])));
      Scale[j] = 1.0/Scale[j];
#endif
      agmres->matvecs += 1;
      j++;
      if ((ksp->pc_side == PC_LEFT) && agmres->r && agmres->DeflPrecond) {
        /* Apply the precond-matrix operators */
        PetscCall(KSP_PCApplyBAorAB(ksp, VEC_V(j-1), VEC_TMP, VEC_TMP_MATOP));
        /* Then apply deflation as a preconditioner */
        PetscCall(KSPDGMRESApplyDeflation_DGMRES(ksp, VEC_TMP, VEC_V(j)));
      } else if ((ksp->pc_side == PC_RIGHT) && agmres->r && agmres->DeflPrecond) {
        PetscCall(KSPDGMRESApplyDeflation_DGMRES(ksp, VEC_V(j-1), VEC_TMP));
        PetscCall(KSP_PCApplyBAorAB(ksp, VEC_TMP, VEC_V(j), VEC_TMP_MATOP));
      } else {
        PetscCall(KSP_PCApplyBAorAB(ksp, VEC_V(j-1), VEC_V(j), VEC_TMP_MATOP));
      }
      PetscCall(VecAXPY(VEC_V(j), -Rshift[j-2], VEC_V(j-1)));
      PetscCall(VecAXPY(VEC_V(j), Scale[j-2]*Ishift[j-2]*Ishift[j-2], VEC_V(j-2)));
#if defined(KSP_AGMRES_NONORM)
      Scale[j] = 1.0;
#else
      PetscCall(VecNorm(VEC_V(j), NORM_2, &(Scale[j])));
      Scale[j] = 1.0/Scale[j];
#endif
      agmres->matvecs += 1;
      j++;
    }
  }
  /* Augment the subspace with the eigenvectors*/
  while (j <= KspSize) {
    PetscCall(KSP_PCApplyBAorAB(ksp, agmres->U[j - max_k - 1], VEC_V(j), VEC_TMP_MATOP));
#if defined(KSP_AGMRES_NONORM)
    Scale[j] = 1.0;
#else
    PetscCall(VecScale(VEC_V(j), Scale[j-1]));
    PetscCall(VecNorm(VEC_V(j), NORM_2, &(Scale[j])));
    Scale[j] = 1.0/Scale[j];
#endif
    agmres->matvecs += 1;
    j++;
  }
  PetscCall(PetscLogEventEnd(KSP_AGMRESBuildBasis, ksp, 0,0,0));
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
  PetscInt       i       = 0, j = 0;
  const PetscInt max_k   = agmres->max_k;
  PetscInt       KspSize = KSPSIZE;
  PetscInt       N       = MAXKSPSIZE+1;

  PetscFunctionBegin;
  PetscCall(PetscArrayzero(agmres->hh_origin, (N+1)*N));
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
  PetscCall(PetscBLASIntCast(KSPSIZE,&KspSize));
  PetscCall(PetscBLASIntCast(4 * (KspSize+1),&lwork));
  PetscCall(PetscBLASIntCast(KspSize+1,&lC));
  PetscCall(PetscBLASIntCast(MAXKSPSIZE + 1,&N));
  PetscCall(PetscBLASIntCast(N + 1,&ldH));
  /* Save a copy of the Hessenberg matrix */
  for (j = 0; j < N-1; j++) {
    for (i = 0; i < N; i++) {
      *HS(i,j) = *H(i,j);
    }
  }
  /* QR factorize the Hessenberg matrix */
  PetscStackCallBLAS("LAPACKgeqrf",LAPACKgeqrf_(&lC, &KspSize, agmres->hh_origin, &ldH, agmres->tau, agmres->work, &lwork, &info));
  PetscCheck(!info,PetscObjectComm((PetscObject)ksp), PETSC_ERR_LIB,"Error in LAPACK routine XGEQRF INFO=%d", info);
  /* Update the right hand side of the least square problem */
  PetscCall(PetscArrayzero(agmres->nrs, N));

  agmres->nrs[0] = ksp->rnorm;
  PetscStackCallBLAS("LAPACKormqr",LAPACKormqr_("L", "T", &lC, &nrhs, &KspSize, agmres->hh_origin, &ldH, agmres->tau, agmres->nrs, &N, agmres->work, &lwork, &info));
  PetscCheck(!info,PetscObjectComm((PetscObject)ksp), PETSC_ERR_LIB,"Error in LAPACK routine XORMQR INFO=%d",info);
  ksp->rnorm = PetscAbsScalar(agmres->nrs[KspSize]);
  /* solve the least-square problem */
  PetscStackCallBLAS("LAPACKtrtrs",LAPACKtrtrs_("U", "N", "N", &KspSize, &nrhs, agmres->hh_origin, &ldH, agmres->nrs, &N, &info));
  PetscCheck(!info,PetscObjectComm((PetscObject)ksp), PETSC_ERR_LIB,"Error in LAPACK routine XTRTRS INFO=%d",info);
  /* Accumulate the correction to the solution of the preconditioned problem in VEC_TMP */
  PetscCall(VecZeroEntries(VEC_TMP));
  PetscCall(VecMAXPY(VEC_TMP, max_k, agmres->nrs, &VEC_V(0)));
  if (!agmres->DeflPrecond) PetscCall(VecMAXPY(VEC_TMP, r, &agmres->nrs[max_k], agmres->U));

  if ((ksp->pc_side == PC_RIGHT) && agmres->r && agmres->DeflPrecond) {
    PetscCall(KSPDGMRESApplyDeflation_DGMRES(ksp, VEC_TMP, VEC_TMP_MATOP));
    PetscCall(VecCopy(VEC_TMP_MATOP, VEC_TMP));
  }
  PetscCall(KSPUnwindPreconditioner(ksp, VEC_TMP, VEC_TMP_MATOP));
  /* add the solution to the previous one */
  PetscCall(VecAXPY(ksp->vec_sol, 1.0, VEC_TMP));
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
  PetscInt       KspSize = KSPSIZE;

  PetscFunctionBegin;
  /* check for the convergence */
  res = ksp->rnorm; /* Norm of the initial residual vector */
  if (!res) {
    if (itcount) *itcount = 0;
    ksp->reason = KSP_CONVERGED_ATOL;
    PetscCall(PetscInfo(ksp,"Converged due to zero residual norm on entry\n"));
    PetscFunctionReturn(0);
  }
  PetscCall((*ksp->converged)(ksp,ksp->its,res,&ksp->reason,ksp->cnvP));
  /* Build the Krylov basis with Newton polynomials */
  PetscCall(KSPAGMRESBuildBasis(ksp));
  /* QR Factorize the basis with RODDEC */
  PetscCall(KSPAGMRESRoddec(ksp, KspSize+1));

  /* Recover a (partial) Hessenberg matrix for the Arnoldi-like relation */
  PetscCall(KSPAGMRESBuildHessenberg(ksp));
  /* Solve the least square problem and unwind the preconditioner */
  PetscCall(KSPAGMRESBuildSoln(ksp, KspSize));

  res        = ksp->rnorm;
  ksp->its  += KspSize;
  agmres->it = KspSize-1;
  /*  Test for the convergence */
  PetscCall((*ksp->converged)(ksp,ksp->its,res,&ksp->reason,ksp->cnvP));
  PetscCall(KSPLogResidualHistory(ksp,res));
  PetscCall(KSPMonitor(ksp,ksp->its,res));

  *itcount = KspSize;
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSolve_AGMRES(KSP ksp)
{
  PetscInt       its;
  KSP_AGMRES     *agmres    = (KSP_AGMRES*)ksp->data;
  PetscBool      guess_zero = ksp->guess_zero;
  PetscReal      res_old, res;
  PetscInt       test;

  PetscFunctionBegin;
  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)ksp));
  ksp->its = 0;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)ksp));

  if (!agmres->HasShifts) { /* Compute Shifts for the Newton basis */
    PetscCall(KSPComputeShifts_DGMRES(ksp));
  }
  /* NOTE: At this step, the initial guess is not equal to zero since one cycle of the classical GMRES is performed to compute the shifts */
  PetscCall((*ksp->converged)(ksp,0,ksp->rnorm,&ksp->reason,ksp->cnvP));
  while (!ksp->reason) {
    PetscCall(KSPInitialResidual(ksp,ksp->vec_sol,VEC_TMP,VEC_TMP_MATOP,VEC_V(0),ksp->vec_rhs));
    if ((ksp->pc_side == PC_LEFT) && agmres->r && agmres->DeflPrecond) {
      PetscCall(KSPDGMRESApplyDeflation_DGMRES(ksp, VEC_V(0), VEC_TMP));
      PetscCall(VecCopy(VEC_TMP, VEC_V(0)));

      agmres->matvecs += 1;
    }
    PetscCall(VecNormalize(VEC_V(0),&(ksp->rnorm)));
    KSPCheckNorm(ksp,ksp->rnorm);
    res_old = ksp->rnorm; /* Record the residual norm to test if deflation is needed */

    ksp->ops->buildsolution = KSPBuildSolution_AGMRES;

    PetscCall(KSPAGMRESCycle(&its,ksp));
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
        PetscCall(KSPDGMRESComputeDeflationData_DGMRES(ksp,&agmres->neig));
      }
    }
    ksp->guess_zero = PETSC_FALSE; /* every future call to KSPInitialResidual() will have nonzero guess */
  }
  ksp->guess_zero = guess_zero; /* restore if user has provided nonzero initial guess */
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPDestroy_AGMRES(KSP ksp)
{
  KSP_AGMRES     *agmres = (KSP_AGMRES*)ksp->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(agmres->hh_origin));

  PetscCall(PetscFree(agmres->Qloc));
  PetscCall(PetscFree4(agmres->Rshift,agmres->Ishift,agmres->Rloc,agmres->wbufptr));
  PetscCall(PetscFree3(agmres->tau,agmres->work,agmres->nrs));
  PetscCall(PetscFree4(agmres->Scale,agmres->sgn,agmres->tloc,agmres->temp));

  PetscCall(PetscFree(agmres->select));
  PetscCall(PetscFree(agmres->wr));
  PetscCall(PetscFree(agmres->wi));
  if (agmres->neig) {
    PetscCall(VecDestroyVecs(MAXKSPSIZE,&agmres->TmpU));
    PetscCall(PetscFree(agmres->perm));
    PetscCall(PetscFree(agmres->MatEigL));
    PetscCall(PetscFree(agmres->MatEigR));
    PetscCall(PetscFree(agmres->Q));
    PetscCall(PetscFree(agmres->Z));
    PetscCall(PetscFree(agmres->beta));
  }
  PetscCall(KSPDestroy_DGMRES(ksp));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPView_AGMRES(KSP ksp,PetscViewer viewer)
{
  KSP_AGMRES     *agmres = (KSP_AGMRES*)ksp->data;
  const char     *cstr   = "RODDEC ORTHOGONOLIZATION";
  char           ritzvec[25];
  PetscBool      iascii,isstring;
#if defined(KSP_AGMRES_NONORM)
  const char *Nstr = "SCALING FACTORS : NO";
#else
  const char *Nstr = "SCALING FACTORS : YES";
#endif

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring));

  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, " restart=%d using %s\n", agmres->max_k, cstr));
    PetscCall(PetscViewerASCIIPrintf(viewer, " %s\n", Nstr));
    PetscCall(PetscViewerASCIIPrintf(viewer, " Number of matvecs : %D\n", agmres->matvecs));
    if (agmres->force) PetscCall(PetscViewerASCIIPrintf (viewer, " Adaptive strategy is used: FALSE\n"));
    else PetscViewerASCIIPrintf(viewer, " Adaptive strategy is used: TRUE\n");
    if (agmres->DeflPrecond) {
      PetscCall(PetscViewerASCIIPrintf(viewer, " STRATEGY OF DEFLATION: PRECONDITIONER \n"));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Frequency of extracted eigenvalues = %D\n", agmres->neig));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Total number of extracted eigenvalues = %D\n", agmres->r));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Maximum number of eigenvalues set to be extracted = %D\n", agmres->max_neig));
    } else {
      if (agmres->ritz) sprintf(ritzvec, "Ritz vectors");
      else sprintf(ritzvec, "Harmonic Ritz vectors");
      PetscCall(PetscViewerASCIIPrintf(viewer, " STRATEGY OF DEFLATION: AUGMENT\n"));
      PetscCall(PetscViewerASCIIPrintf(viewer," augmented vectors  %d at frequency %d with %s\n", agmres->r, agmres->neig, ritzvec));
    }
    PetscCall(PetscViewerASCIIPrintf(viewer, " Minimum relaxation parameter for the adaptive strategy(smv)  = %g\n", agmres->smv));
    PetscCall(PetscViewerASCIIPrintf(viewer, " Maximum relaxation parameter for the adaptive strategy(bgv)  = %g\n", agmres->bgv));
  } else if (isstring) {
    PetscCall(PetscViewerStringSPrintf(viewer,"%s restart %D",cstr,agmres->max_k));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSetFromOptions_AGMRES(PetscOptionItems *PetscOptionsObject,KSP ksp)
{
  PetscInt       neig;
  KSP_AGMRES     *agmres = (KSP_AGMRES*)ksp->data;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscCall(KSPSetFromOptions_DGMRES(PetscOptionsObject,ksp));  /* Set common options from DGMRES and GMRES */
  PetscCall(PetscOptionsHead(PetscOptionsObject,"KSP AGMRES Options"));
  PetscCall(PetscOptionsInt("-ksp_agmres_eigen", "Number of eigenvalues to deflate", "KSPDGMRESSetEigen", agmres->neig, &neig, &flg));
  if (flg) {
    PetscCall(KSPDGMRESSetEigen_DGMRES(ksp, neig));
    agmres->r = 0;
  } else agmres->neig = 0;
  PetscCall(PetscOptionsInt("-ksp_agmres_maxeigen", "Maximum number of eigenvalues to deflate", "KSPDGMRESSetMaxEigen", agmres->max_neig, &neig, &flg));
  if (flg) agmres->max_neig = neig+EIG_OFFSET;
  else agmres->max_neig = agmres->neig+EIG_OFFSET;
  PetscCall(PetscOptionsBool("-ksp_agmres_DeflPrecond", "Determine if the deflation should be applied as a preconditioner -- similar to KSP DGMRES", "KSPGMRESDeflPrecond",agmres->DeflPrecond,&agmres->DeflPrecond,NULL));
  PetscCall(PetscOptionsBool("-ksp_agmres_ritz", "Compute the Ritz vectors instead of the Harmonic Ritz vectors ", "KSPGMRESHarmonic",agmres->ritz,&agmres->ritz ,&flg));
  PetscCall(PetscOptionsReal("-ksp_agmres_MinRatio", "Relaxation parameter in the adaptive strategy; smallest multiple of the remaining number of steps allowed", "KSPGMRESSetMinRatio", agmres->smv, &agmres->smv, NULL));
  PetscCall(PetscOptionsReal("-ksp_agmres_MaxRatio", "Relaxation parameter in the adaptive strategy; Largest multiple of the remaining number of steps allowed", "KSPGMRESSetMaxRatio",agmres->bgv,&agmres->bgv, &flg));
  PetscCall(PetscOptionsTail());
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

  PetscFunctionBegin;
  PetscCall(PetscNewLog(ksp,&agmres));
  ksp->data = (void*)agmres;

  PetscCall(KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,3));
  PetscCall(KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_RIGHT,2));
  ksp->ops->buildsolution                = KSPBuildSolution_AGMRES;
  ksp->ops->setup                        = KSPSetUp_AGMRES;
  ksp->ops->solve                        = KSPSolve_AGMRES;
  ksp->ops->destroy                      = KSPDestroy_AGMRES;
  ksp->ops->view                         = KSPView_AGMRES;
  ksp->ops->setfromoptions               = KSPSetFromOptions_AGMRES;
  ksp->guess_zero                        = PETSC_TRUE;
  ksp->ops->computeextremesingularvalues = KSPComputeExtremeSingularValues_GMRES;
  ksp->ops->computeeigenvalues           = KSPComputeEigenvalues_GMRES;

  PetscCall(PetscObjectComposeFunction((PetscObject) ksp,"KSPGMRESSetPreAllocateVectors_C",KSPGMRESSetPreAllocateVectors_GMRES));
  PetscCall(PetscObjectComposeFunction((PetscObject) ksp,"KSPGMRESSetOrthogonalization_C",KSPGMRESSetOrthogonalization_GMRES));
  PetscCall(PetscObjectComposeFunction((PetscObject) ksp,"KSPGMRESSetRestart_C",KSPGMRESSetRestart_GMRES));
  PetscCall(PetscObjectComposeFunction((PetscObject) ksp,"KSPGMRESSetHapTol_C",KSPGMRESSetHapTol_GMRES));
  PetscCall(PetscObjectComposeFunction((PetscObject) ksp,"KSPGMRESSetCGSRefinementType_C",KSPGMRESSetCGSRefinementType_GMRES));
  /* -- New functions defined in DGMRES -- */
  PetscCall(PetscObjectComposeFunction((PetscObject) ksp, "KSPDGMRESSetEigen_C",KSPDGMRESSetEigen_DGMRES));
  PetscCall(PetscObjectComposeFunction((PetscObject) ksp, "KSPDGMRESComputeSchurForm_C",KSPDGMRESComputeSchurForm_DGMRES));
  PetscCall(PetscObjectComposeFunction((PetscObject) ksp, "KSPDGMRESComputeDeflationData_C",KSPDGMRESComputeDeflationData_DGMRES));
  PetscCall(PetscObjectComposeFunction((PetscObject) ksp, "KSPDGMRESApplyDeflation_C",KSPDGMRESApplyDeflation_DGMRES));

  PetscCall(PetscLogEventRegister("AGMRESCompDefl",   KSP_CLASSID, &KSP_AGMRESComputeDeflationData));
  PetscCall(PetscLogEventRegister("AGMRESBuildBasis", KSP_CLASSID, &KSP_AGMRESBuildBasis));
  PetscCall(PetscLogEventRegister("AGMRESCompShifts", KSP_CLASSID, &KSP_AGMRESComputeShifts));
  PetscCall(PetscLogEventRegister("AGMRESOrthog",     KSP_CLASSID, &KSP_AGMRESRoddec));

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
  PetscCall(PetscObjectGetNewTag((PetscObject)ksp,&agmres->tag));
  PetscFunctionReturn(0);
}
