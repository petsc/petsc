#ifndef PETSCHPDDM_H
#define PETSCHPDDM_H

#include <petsc/private/kspimpl.h>

#define PETSC_KSPHPDDM_DEFAULT_PRECISION \
  (PetscDefined(USE_REAL_SINGLE) ? KSP_HPDDM_PRECISION_SINGLE : (PetscDefined(USE_REAL_DOUBLE) ? KSP_HPDDM_PRECISION_DOUBLE : (PetscDefined(USE_REAL___FLOAT128) ? KSP_HPDDM_PRECISION_QUADRUPLE : KSP_HPDDM_PRECISION_HALF)))
#define PETSC_PCHPDDM_MAXLEVELS 9

namespace HPDDM
{
template <class>
class Schwarz; /* forward definitions of two needed HPDDM classes */
class PETScOperator;
} // namespace HPDDM

struct PC_HPDDM_Level {
  VecScatter                   scatter;   /* scattering from PETSc nonoverlapping numbering to HPDDM overlapping */
  Vec                         *v[2];      /* working vectors */
  Mat                          V[3];      /* working matrices */
  KSP                          ksp;       /* KSP coupling the action of pc and P */
  PC                           pc;        /* inner fine-level PC, acting like a multigrid smoother */
  HPDDM::Schwarz<PetscScalar> *P;         /* coarse-level HPDDM solver */
  Vec                          D;         /* partition of unity */
  PetscReal                    threshold; /* threshold for selecting local deflation vectors */
  PetscInt                     nu;        /* number of local deflation vectors */
  const struct PC_HPDDM       *parent;    /* parent PC */
};

struct PC_HPDDM {
  PC_HPDDM_Level            **levels;                                       /* array of shells */
  Mat                         aux;                                          /* local auxiliary matrix defined at the finest level on PETSC_COMM_SELF */
  Mat                         B;                                            /* right-hand side matrix defined at the finest level on PETSC_COMM_SELF */
  Vec                         normal;                                       /* temporary Vec when preconditioning the normal equations with KSPLSQR */
  IS                          is;                                           /* global numbering of the auxiliary matrix */
  PetscInt                    N;                                            /* number of levels */
  PCHPDDMCoarseCorrectionType correction;                                   /* type of coarse correction */
  PetscBool3                  Neumann;                                      /* aux is the local Neumann matrix? */
  PetscBool                   log_separate;                                 /* separate events for each level? */
  PetscBool                   share;                                        /* shared subdomain KSP between SLEPc and PETSc? */
  PetscBool                   deflation;                                    /* aux is the local deflation space? */
  PetscErrorCode (*setup)(Mat, PetscReal, Vec, Vec, PetscReal, IS, void *); /* setup function for the auxiliary matrix */
  void *setup_ctx;                                                          /* context for setup */
};

struct KSP_HPDDM {
  HPDDM::PETScOperator *op;
  PetscReal             rcntl[1];
  int                   icntl[2];
  unsigned short        scntl[2];
  char                  cntl[5];
  KSPHPDDMPrecision     precision;
};

PETSC_EXTERN PetscLogEvent  PC_HPDDM_PtAP;
PETSC_EXTERN PetscLogEvent  PC_HPDDM_PtBP;
PETSC_EXTERN PetscLogEvent  PC_HPDDM_Next;
PETSC_INTERN PetscErrorCode HPDDMLoadDL_Private(PetscBool *);
PETSC_INTERN const char     HPDDMCitation[];
PETSC_INTERN PetscBool      HPDDMCite;
#if PetscDefined(HAVE_CUDA) && PetscDefined(HAVE_HPDDM)
PETSC_INTERN PetscErrorCode KSPSolve_HPDDM_CUDA_Private(KSP_HPDDM *, const PetscScalar *, PetscScalar *, PetscInt, MPI_Comm);
#endif

#include <HPDDM.hpp>

#endif /* PETSCHPDDM_H */
