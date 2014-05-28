#if !defined(_PETSCPROBLEMIMPL_H)
#define _PETSCPROBLEMIMPL_H

#include <petscproblem.h>
#include <petsc-private/petscimpl.h>

typedef struct _PetscProblemOps *PetscProblemOps;
struct _PetscProblemOps {
  PetscErrorCode (*setfromoptions)(PetscProblem);
  PetscErrorCode (*setup)(PetscProblem);
  PetscErrorCode (*view)(PetscProblem,PetscViewer);
  PetscErrorCode (*destroy)(PetscProblem);
};

typedef void (*PointFunc)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]);
typedef void (*BdPointFunc)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]);

struct _p_PetscProblem {
  PETSCHEADER(struct _PetscProblemOps);
  void        *data;      /* Implementation object */
  PetscBool    setup;     /* Flag for setup */
  PetscInt     Nf;        /* The number of solution fields */
  PetscObject *disc;      /* The discretization for each solution field (PetscFE, PetscFV, etc.) */
  PetscObject *discBd;    /* The boundary discretization for each solution field (PetscFE, PetscFV, etc.) */
  PointFunc   *f,   *g;   /* Weak form integrands f_0, f_1, g_0, g_1, g_2, g_3 */
  BdPointFunc *fBd, *gBd; /* Weak form boundary integrands f_0, f_1, g_0, g_1, g_2, g_3 */
  PetscInt     dim;       /* The spatial dimension */
  /* Computed sizes */
  PetscInt     totDim, totDimBd;       /* Total problem dimension */
  PetscInt     totComp;                /* Total field components */
  /* Work space */
  PetscReal  **basis,    **basisBd;    /* Default basis tabulation for each field */
  PetscReal  **basisDer, **basisDerBd; /* Default basis derivative tabulation for each field */
  PetscScalar *u;                      /* Field evaluation */
  PetscScalar *u_t;                    /* Field time derivative evaluation */
  PetscScalar *u_x;                    /* Field gradient evaluation */
  PetscScalar *refSpaceDer;            /* Workspace for computing derivative in the reference coordinates */
  PetscReal   *x;                      /* Workspace for computing real coordinates */
  PetscScalar *f0, *f1;                /* Point evaluations of weak form residual integrands */
  PetscScalar *g0, *g1, *g2, *g3;      /* Point evaluations of weak form Jacobian integrands */
};

typedef struct {
  PetscInt dummy; /* */
} PetscProblem_Basic;

#endif
