/*
  * Private data structure used for the KSP AGMRES.
  * It extends the definition of KSP_GMRES and KSP_DGMRES data structures. If you modify something there (located in gmresimpl.h and in dgmresimpl.h), you should  modify it here as well.
  * In this KSP, KSPSIZE denotes the size of the basis (possibly augmented with Schur vectors) and MAXKSPSIZE denotes the maximum size of the augmented basis (with respect to the input
*/
#if !defined(__AGMRES)
#define __AGMRES

#include <../src/ksp/ksp/impls/gmres/dgmres/dgmresimpl.h>
typedef struct {
  KSPGMRESHEADER
  KSPDGMRESHEADER

  /* Data specific to AGMRES */
  PetscReal    bgv;             /* large multiple of the remaining allowed number of steps -- used for the adaptive strategy */
  PetscBool    ritz;            /* Compute the Harmonic Ritz vectors instead of the Ritz vectors */
  PetscBool    DeflPrecond;     /* Apply deflation by building adaptively a preconditioner, otherwise augment the basis */
  PetscScalar  *Qloc;           /* Orthogonal reflectors from the QR of the basis */
  PetscScalar  *Rloc;           /* triangular matrix obtained from the QR of the basis */
  PetscScalar  *Rshift, *Ishift; /* Real and Imaginary parts of the shifts in the Newton basis */
  PetscScalar  *Scale;          /* Norm of the vectors in the Newton basis */
  PetscBool    HasShifts;       /* Estimation of shifts exists */
  PetscMPIInt  rank,size;       /* Rank and size of the current process; to be used in RODDEC*/
  PetscMPIInt  First, Last, Ileft, Iright;  /* Create a ring of processors for RODDEC */
  PetscScalar  *MatEigL, *MatEigR; /* matrices for the eigenvalue problem */
  PetscScalar  *sgn;            /* Sign of the rotation in the QR factorization of the basis */
  PetscScalar  *tloc;           /* */
  Vec          *TmpU;           /* Temporary vectors */
  PetscScalar  *beta;           /* needed for the eigenvalues */
  PetscBLASInt *select;         /* array used to select the Schur vectors to order */
  PetscScalar  *temp,*wbufptr;
  PetscScalar  *tau;            /* Scalar factors of the elementary reflectors in xgeqrf */
  PetscMPIInt  tag;
} KSP_AGMRES;

PETSC_EXTERN PetscLogEvent KSP_AGMRESComputeDeflationData;
PETSC_EXTERN PetscLogEvent KSP_AGMRESBuildBasis;
PETSC_EXTERN PetscLogEvent KSP_AGMRESComputeShifts;
PETSC_EXTERN PetscLogEvent KSP_AGMRESRoddec;

/* vector names */
#define VEC_TMP        agmres->vecs[0]
#define VEC_TMP_MATOP  agmres->vecs[1]
#define VEC_V(i)       agmres->vecs[VEC_OFFSET+i]

#define MAXKSPSIZE  ((agmres->DeflPrecond) ? (agmres->max_k) : (agmres->max_k + agmres->max_neig))
#define KSPSIZE ((agmres->DeflPrecond) ? (agmres->max_k) : (agmres->max_k + agmres->r))
#define H(a,b) (agmres->hh_origin + (b)*(MAXKSPSIZE + 2)+(a))
#define HS(a,b) (agmres->hes_origin + (b)*(MAXKSPSIZE + 1)+(a))
#define RLOC(a,b) (agmres->Rloc + (b)*(MAXKSPSIZE + 1)+(a))

PetscErrorCode KSPAGMRESRoddec(KSP, PetscInt);
PetscErrorCode KSPAGMRESRodvec(KSP, PetscInt, PetscScalar*, Vec);
PetscErrorCode KSPAGMRESLejaOrdering(PetscScalar*, PetscScalar*, PetscScalar*, PetscScalar*, PetscInt);
PetscErrorCode KSPAGMRESRoddecInitNeighboor(KSP);
PetscErrorCode KSPAGMRESComputeDeflationData (KSP);
#endif
