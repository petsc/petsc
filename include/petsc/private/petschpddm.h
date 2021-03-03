#if !defined(PETSCHPDDM_H)
#define PETSCHPDDM_H

#include <petsc/private/kspimpl.h>

#define PETSC_HPDDM_MAXLEVELS 9
PETSC_EXTERN PetscLogEvent PC_HPDDM_PtAP;
PETSC_EXTERN PetscLogEvent PC_HPDDM_PtBP;
PETSC_EXTERN PetscLogEvent PC_HPDDM_Next;
PETSC_INTERN PetscErrorCode HPDDMLoadDL_Private(PetscBool*);

namespace HPDDM {
  template<class K> class Schwarz;       /* forward definitions of two needed HPDDM classes */
  class PETScOperator;
}

struct PC_HPDDM_Level {
  VecScatter                  scatter;   /* scattering from PETSc nonoverlapping numbering to HPDDM overlapping */
  Vec                         *v[2];     /* working vectors */
  Mat                         V;         /* working matrix */
  KSP                         ksp;       /* KSP coupling the action of pc and P */
  PC                          pc;        /* inner fine-level PC, acting like a multigrid smoother */
  HPDDM::Schwarz<PetscScalar> *P;        /* coarse-level HPDDM solver */
  Vec                         D;         /* partition of unity */
  PetscReal                   threshold; /* threshold for selecting local deflation vectors */
  PetscInt                    nu;        /* number of local deflation vectors */
  const struct PC_HPDDM*      parent;    /* parent PC */
};

struct PC_HPDDM {
  PC_HPDDM_Level              **levels;   /* array of shells */
  Mat                         aux;        /* local auxiliary matrix defined at the finest level on PETSC_COMM_SELF */
  Mat                         B;          /* right-hand side matrix defined at the finest level on PETSC_COMM_SELF */
  Vec                         normal;     /* temporary Vec when preconditioning the normal equations with KSPLSQR */
  IS                          is;         /* global numbering of the auxiliary matrix */
  PetscInt                    N;          /* number of levels */
  PCHPDDMCoarseCorrectionType correction; /* type of coarse correction */
  PetscBool                   Neumann;    /* aux is the local Neumann matrix? */
  PetscBool                   log_separate; /* separate events for each level? */
  PetscBool                   share;      /* shared KSP between SLEPc ST and the fine-level subdomain solver? */
  PetscErrorCode              (*setup)(Mat, PetscReal, Vec, Vec, PetscReal, IS, void*); /* setup function for the auxiliary matrix */
  void*                       setup_ctx;  /* context for setup */
};

struct KSP_HPDDM {
  HPDDM::PETScOperator *op;
  PetscReal            rcntl[1];
  int                  icntl[2];
  unsigned short       scntl[2];
  char                 cntl [5];
};

#include <HPDDM.hpp>

#endif /* PETSCHPDDM_H */
