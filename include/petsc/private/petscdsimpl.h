#if !defined(_PETSCDSIMPL_H)
#define _PETSCDSIMPL_H

#include <petscds.h>
#include <petsc/private/petscimpl.h>

PETSC_EXTERN PetscBool      PetscDSRegisterAllCalled;
PETSC_EXTERN PetscErrorCode PetscDSRegisterAll(void);

typedef struct _PetscDSOps *PetscDSOps;
struct _PetscDSOps {
  PetscErrorCode (*setfromoptions)(PetscDS);
  PetscErrorCode (*setup)(PetscDS);
  PetscErrorCode (*view)(PetscDS,PetscViewer);
  PetscErrorCode (*destroy)(PetscDS);
};

struct _p_PetscDS {
  PETSCHEADER(struct _PetscDSOps);
  void        *data;      /* Implementation object */
  PetscBool    setup;     /* Flag for setup */
  PetscInt     Nf;        /* The number of solution fields */
  PetscBool   *implicit;  /* Flag for implicit or explicit solve */
  PetscBool   *adjacency; /* Flag for variable influence */
  PetscObject *disc;      /* The discretization for each solution field (PetscFE, PetscFV, etc.) */
  PetscObject *discBd;    /* The boundary discretization for each solution field (PetscFE, PetscFV, etc.) */
  PetscPointFunc   *obj;  /* Scalar integral (like an objective function) */
  PetscPointFunc   *f;    /* Weak form integrands for F, f_0, f_1 */
  PetscPointJac    *g;    /* Weak form integrands for J = dF/du, g_0, g_1, g_2, g_3 */
  PetscPointJac    *gp;   /* Weak form integrands for preconditioner for J, g_0, g_1, g_2, g_3 */
  PetscPointJac    *gt;   /* Weak form integrands for dF/du_t, g_0, g_1, g_2, g_3 */
  PetscBdPointFunc *fBd;  /* Weak form boundary integrands F_bd, f_0, f_1 */
  PetscBdPointJac  *gBd;  /* Weak form boundary integrands J_bd = dF_bd/du, g_0, g_1, g_2, g_3 */
  PetscRiemannFunc *r;    /* Riemann solvers */
  void       **ctx;       /* User contexts for each field */
  PetscInt     dim;       /* The spatial dimension */
  /* Computed sizes */
  PetscInt     totDim, totDimBd;       /* Total system dimension */
  PetscInt     totComp;                /* Total field components */
  /* Work space */
  PetscInt    *off,       *offBd;      /* Offsets for each field */
  PetscInt    *offDer,    *offDerBd;   /* Derivative offsets for each field */
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
