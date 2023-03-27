#ifndef PETSCDSIMPL_H
#define PETSCDSIMPL_H

#include <petscds.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/hashmap.h>

PETSC_EXTERN PetscBool      PetscDSRegisterAllCalled;
PETSC_EXTERN PetscErrorCode PetscDSRegisterAll(void);

typedef struct _n_DSBoundary *DSBoundary;

struct _n_DSBoundary {
  const char             *name;   /* A unique identifier for the condition */
  DMLabel                 label;  /* The DMLabel indicating the mesh region over which the condition holds */
  const char             *lname;  /* The label name if the label is missing from the DM */
  PetscInt                Nv;     /* The number of label values defining the region */
  PetscInt               *values; /* The labels values defining the region */
  PetscWeakForm           wf;     /* Holds the pointwise functions defining the form (only for NATURAL conditions) */
  DMBoundaryConditionType type;   /* The type of condition, usually either ESSENTIAL or NATURAL */
  PetscInt                field;  /* The field constrained by the condition */
  PetscInt                Nc;     /* The number of constrained field components */
  PetscInt               *comps;  /* The constrained field components */
  void (*func)(void);             /* Function that provides the boundary values (only for ESSENTIAL conditions) */
  void (*func_t)(void);           /* Function that provides the time derivative of the boundary values (only for ESSENTIAL conditions) */
  void      *ctx;                 /* The user context for func and func_t */
  DSBoundary next;
};

typedef struct {
  PetscInt start;    /* Starting entry of the chunk in an array (in bytes) */
  PetscInt size;     /* Current number of entries of the chunk */
  PetscInt reserved; /* Number of reserved entries in the chunk */
} PetscChunk;

typedef struct {
  size_t size;      /* Current number of entries used in array */
  size_t alloc;     /* Number of bytes allocated for array */
  size_t unitbytes; /* Number of bytes per entry */
  char  *array;
} PetscChunkBuffer;

#define PetscFormKeyHash(key) PetscHashCombine(PetscHashCombine(PetscHashCombine(PetscHashPointer((key).label), PetscHashInt((key).value)), PetscHashInt((key).field)), PetscHashInt((key).part))

#define PetscFormKeyEqual(k1, k2) (((k1).label == (k2).label) ? ((k1).value == (k2).value) ? ((k1).field == (k2).field) ? ((k1).part == (k2).part) : 0 : 0 : 0)

static PetscChunk _PetscInvalidChunk = {-1, -1, -1};

PETSC_HASH_MAP(HMapForm, PetscFormKey, PetscChunk, PetscFormKeyHash, PetscFormKeyEqual, _PetscInvalidChunk)

/*
  We sort lexicographically on the structure.
  Returns
  -1: left < right
   0: left = right
   1: left > right
*/
static inline int Compare_PetscFormKey_Private(const void *left, const void *right, PETSC_UNUSED void *ctx)
{
  PetscFormKey l = *(const PetscFormKey *)left;
  PetscFormKey r = *(const PetscFormKey *)right;
  return (l.label < r.label) ? -1 : ((l.label > r.label) ? 1 : ((l.value < r.value) ? -1 : (l.value > r.value) ? 1 : ((l.field < r.field) ? -1 : (l.field > r.field) ? 1 : ((l.part < r.part) ? -1 : (l.part > r.part)))));
}

typedef struct _PetscWeakFormOps *PetscWeakFormOps;
struct _PetscWeakFormOps {
  PetscErrorCode (*setfromoptions)(PetscWeakForm);
  PetscErrorCode (*setup)(PetscWeakForm);
  PetscErrorCode (*view)(PetscWeakForm, PetscViewer);
  PetscErrorCode (*destroy)(PetscWeakForm);
};

struct _p_PetscWeakForm {
  PETSCHEADER(struct _PetscWeakFormOps);
  void *data; /* Implementation object */

  PetscInt          Nf;    /* The number of fields in the system */
  PetscChunkBuffer *funcs; /* Buffer holding all function pointers */
  PetscHMapForm    *form;  /* Stores function pointers for forms */
};

typedef struct _PetscDSOps *PetscDSOps;
struct _PetscDSOps {
  PetscErrorCode (*setfromoptions)(PetscDS);
  PetscErrorCode (*setup)(PetscDS);
  PetscErrorCode (*view)(PetscDS, PetscViewer);
  PetscErrorCode (*destroy)(PetscDS);
};

struct _p_PetscDS {
  PETSCHEADER(struct _PetscDSOps);
  void        *data;       /* Implementation object */
  PetscDS     *subprobs;   /* The subspaces for each dimension */
  PetscBool    setup;      /* Flag for setup */
  PetscInt     dimEmbed;   /* The real space coordinate dimension */
  PetscInt     Nf;         /* The number of solution fields */
  PetscObject *disc;       /* The discretization for each solution field (PetscFE, PetscFV, etc.) */
  PetscBool   *cohesive;   /* Flag for cohesive discretization */
  PetscBool    isCohesive; /* We are on a cohesive cell, meaning lower dimensional FE used on a 0-volume cell. Normal fields appear on both endcaps, whereas cohesive field only appear once in the middle */
  /* Quadrature */
  PetscBool forceQuad;                  /* Flag to force matching quadratures in discretizations */
  IS       *quadPerm[DM_NUM_POLYTOPES]; /* qP[ct][o]: q point permutation for orientation o of integ domain */
  /* Equations */
  DSBoundary            boundary;     /* Linked list of boundary conditions */
  PetscBool             useJacPre;    /* Flag for using the Jacobian preconditioner */
  PetscBool            *implicit;     /* Flag for implicit or explicit solve for each field */
  PetscInt             *jetDegree;    /* The highest derivative for each field equation, or the k-jet that each discretization needs to tabulate */
  PetscWeakForm         wf;           /* The PetscWeakForm holding all pointwise functions */
  PetscPointFunc       *update;       /* Direct update of field coefficients */
  PetscSimplePointFunc *exactSol;     /* Exact solutions for each field */
  void                **exactCtx;     /* Contexts for the exact solution functions */
  PetscSimplePointFunc *exactSol_t;   /* Time derivative of the exact solutions for each field */
  void                **exactCtx_t;   /* Contexts for the time derivative of the exact solution functions */
  PetscInt              numConstants; /* Number of constants passed to point functions */
  PetscScalar          *constants;    /* Array of constants passed to point functions */
  void                **ctx;          /* User contexts for each field */
  /* Computed sizes */
  PetscInt         totDim;            /* Total system dimension */
  PetscInt         totComp;           /* Total field components */
  PetscInt        *Nc;                /* Number of components for each field */
  PetscInt        *Nb;                /* Number of basis functions for each field */
  PetscInt        *off;               /* Offsets for each field */
  PetscInt        *offDer;            /* Derivative offsets for each field */
  PetscInt        *offCohesive[3];    /* Offsets for each field on side s of a cohesive cell */
  PetscInt        *offDerCohesive[3]; /* Derivative offsets for each field on side s of a cohesive cell */
  PetscTabulation *T;                 /* Basis function and derivative tabulation for each field */
  PetscTabulation *Tf;                /* Basis function and derivative tabulation for each local face and field */
  /* Work space */
  PetscScalar *u;                 /* Field evaluation */
  PetscScalar *u_t;               /* Field time derivative evaluation */
  PetscScalar *u_x;               /* Field gradient evaluation */
  PetscScalar *basisReal;         /* Workspace for pushforward */
  PetscScalar *basisDerReal;      /* Workspace for derivative pushforward */
  PetscScalar *testReal;          /* Workspace for pushforward */
  PetscScalar *testDerReal;       /* Workspace for derivative pushforward */
  PetscReal   *x;                 /* Workspace for computing real coordinates */
  PetscScalar *f0, *f1;           /* Point evaluations of weak form residual integrands */
  PetscScalar *g0, *g1, *g2, *g3; /* Point evaluations of weak form Jacobian integrands */
};

typedef struct {
  PetscInt dummy; /* */
} PetscDS_Basic;

PETSC_INTERN PetscErrorCode PetscDSGetDiscType_Internal(PetscDS, PetscInt, PetscDiscType *);

#endif
