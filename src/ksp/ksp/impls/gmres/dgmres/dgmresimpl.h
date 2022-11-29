#ifndef PETSC_DGMRESIMPL_H
#define PETSC_DGMRESIMPL_H

#define KSPGMRES_NO_MACROS
#include <../src/ksp/ksp/impls/gmres/gmresimpl.h>
#include <petscblaslapack.h>

#define KSPDGMRESHEADER \
  /* Data specific to DGMRES */ \
  Vec          *U;               /* Vectors that form the basis of the invariant subspace */ \
  PetscScalar  *T;               /* T=U^T*M^{-1}*A*U */ \
  PetscScalar  *TF;              /* The factors L and U from T = P*L*U */ \
  PetscBLASInt *InvP;            /* Permutation Vector from the LU factorization of T */ \
  PetscInt      neig;            /* number of eigenvalues to extract at each restart */ \
  PetscInt      r;               /* current number of deflated eigenvalues */ \
  PetscInt      max_neig;        /* Maximum number of eigenvalues to deflate */ \
  PetscReal     lambdaN;         /* modulus of the largest eigenvalue of A */ \
  PetscReal     smv;             /* smaller multiple of the remaining allowed number of steps -- used for the adaptive strategy */ \
  PetscBool     force;           /* Force the use of the deflation at the restart */ \
  PetscInt      matvecs;         /* Total number of matrix-vectors */ \
  PetscInt      GreatestEig;     /* Extract the greatest eigenvalues instead */ \
  PetscReal    *wr, *wi, *modul; /* Real and complex part and modulus of eigenvalues */ \
  PetscScalar  *Q, *Z;           /* Left and right schur/eigenvectors from the QZ algorithm */ \
  PetscInt     *perm;            /* temporary permutation vector */ \
  /* Work spaces */ \
  Vec          *mu;       /* Save the product M^{-1}AU */ \
  PetscScalar  *Sr;       /* Schur vectors to extract */ \
  Vec          *X;        /* Schurs Vectors Sr projected to the entire space */ \
  Vec          *mx;       /* store the product M^{-1}*A*X */ \
  PetscScalar  *umx;      /* store the product U^T*M^{-1}*A*X */ \
  PetscScalar  *xmu;      /* store the product X^T*M^{-1}*A*U */ \
  PetscScalar  *xmx;      /* store the product X^T*M^{-1}*A*X */ \
  PetscScalar  *x1;       /* store the product U^T*x */ \
  PetscScalar  *x2;       /* store the product U^T*x */ \
  PetscScalar  *Sr2;      /* Schur vectors at the improvement step */ \
  PetscScalar  *auau;     /* product of (M*A*U)^T*M*A*U */ \
  PetscScalar  *auu;      /* product of (M*A*U)^T*U */ \
  PetscScalar  *work;     /* work space for LAPACK functions */ \
  PetscBLASInt *iwork;    /* work space for LAPACK functions */ \
  PetscReal    *orth;     /* Coefficients for the orthogonalization */ \
  PetscBool     HasSchur; /* Indicate if the Schur form had already been computed in this cycle */ \
  PetscBool     improve;  /* 0 = do not improve the eigenvalues; This is an experimental option */

typedef struct {
  KSPGMRESHEADER
  KSPDGMRESHEADER
} KSP_DGMRES;

PETSC_INTERN PetscErrorCode KSPDGMRESComputeDeflationData(KSP, PetscInt *);

PETSC_EXTERN PetscLogEvent KSP_DGMRESComputeDeflationData;
PETSC_EXTERN PetscLogEvent KSP_DGMRESApplyDeflation;

#define HH(a, b)  (dgmres->hh_origin + (b) * (dgmres->max_k + 2) + (a))
#define HES(a, b) (dgmres->hes_origin + (b) * (dgmres->max_k + 1) + (a))
#define CC(a)     (dgmres->cc_origin + (a))
#define SS(a)     (dgmres->ss_origin + (a))
#define GRS(a)    (dgmres->rs_origin + (a))

/* vector names */
#define VEC_OFFSET     2
#define VEC_TEMP       dgmres->vecs[0]
#define VEC_TEMP_MATOP dgmres->vecs[1]
#define VEC_VV(i)      dgmres->vecs[VEC_OFFSET + i]

#define EIG_OFFSET            1
#define DGMRES_DEFAULT_EIG    1
#define DGMRES_DEFAULT_MAXEIG 10

#define UU       dgmres->U
#define TT       dgmres->T
#define TTF      dgmres->TF
#define XX       dgmres->X
#define INVP     dgmres->InvP
#define MU       dgmres->mu
#define MX       dgmres->mx
#define UMX      dgmres->umx
#define XMU      dgmres->xmu
#define XMX      dgmres->xmx
#define X1       dgmres->x1
#define X2       dgmres->x2
#define SR       dgmres->Sr
#define SR2      dgmres->Sr2
#define AUAU     dgmres->auau
#define AUU      dgmres->auu
#define MAX_K    dgmres->max_k
#define MAX_NEIG dgmres->max_neig
#define WORK     dgmres->work
#define IWORK    dgmres->iwork
#define ORTH     dgmres->orth
#define SMV      1
#endif // PETSC_DGMRESIMPL_H
