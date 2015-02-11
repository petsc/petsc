#if !defined(_PETSCDSIMPL_H)
#define _PETSCDSIMPL_H

#include <petscds.h>
#include <petsc-private/petscimpl.h>

PETSC_EXTERN PetscBool      PetscDSRegisterAllCalled;
PETSC_EXTERN PetscErrorCode PetscDSRegisterAll(void);

typedef struct _PetscDSOps *PetscDSOps;
struct _PetscDSOps {
  PetscErrorCode (*setfromoptions)(PetscDS);
  PetscErrorCode (*setup)(PetscDS);
  PetscErrorCode (*view)(PetscDS,PetscViewer);
  PetscErrorCode (*destroy)(PetscDS);
};

typedef void (*PointFunc)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]);
typedef void (*BdPointFunc)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]);
typedef void (*RiemannFunc)(const PetscReal[], const PetscReal[], const PetscScalar[], const PetscScalar[], PetscScalar[], void *);

struct _p_PetscDS {
  PETSCHEADER(struct _PetscDSOps);
  void        *data;      /* Implementation object */
  PetscBool    setup;     /* Flag for setup */
  PetscInt     Nf;        /* The number of solution fields */
  PetscObject *disc;      /* The discretization for each solution field (PetscFE, PetscFV, etc.) */
  PetscObject *discBd;    /* The boundary discretization for each solution field (PetscFE, PetscFV, etc.) */
  PointFunc   *obj;       /* Scalar integral (like an objective function) */
  PointFunc   *f,   *g;   /* Weak form integrands f_0, f_1, g_0, g_1, g_2, g_3 */
  BdPointFunc *fBd, *gBd; /* Weak form boundary integrands f_0, f_1, g_0, g_1, g_2, g_3 */
  RiemannFunc *r;         /* Riemann solvers */
  void       **ctx;       /* User contexts for each field */
  PetscInt     dim;       /* The spatial dimension */
  /* Computed sizes */
  PetscInt     totDim, totDimBd;       /* Total system dimension */
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
} PetscDS_Basic;

#endif
