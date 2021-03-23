#if !defined(PETSCDSIMPL_H)
#define PETSCDSIMPL_H

#include <petscds.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/hashmap.h>

PETSC_EXTERN PetscBool      PetscDSRegisterAllCalled;
PETSC_EXTERN PetscErrorCode PetscDSRegisterAll(void);

typedef struct _n_DSBoundary *DSBoundary;

struct _n_DSBoundary {
  const char *name;
  const char *labelname;
  DMBoundaryConditionType type;
  PetscInt    field;
  PetscInt    numcomps;
  PetscInt   *comps;
  void      (*func)(void);
  void      (*func_t)(void);
  PetscInt    numids;
  PetscInt   *ids;
  void       *ctx;
  DSBoundary  next;
};

typedef struct {
  PetscInt start;    /* Starting entry of the chunk in an array (in bytes) */
  PetscInt size;     /* Current number of entries of the chunk */
  PetscInt reserved; /* Number of reserved entries in the chunk */
} PetscChunk;

typedef struct {
  size_t  size;      /* Current number of entries used in array */
  size_t  alloc;     /* Number of bytes allocated for array */
  size_t  unitbytes; /* Number of bytes per entry */
  char   *array;
} PetscChunkBuffer;

#define PetscHashFormKeyHash(key) \
  PetscHashCombine(PetscHashCombine(PetscHashPointer((key).label),PetscHashInt((key).value)),PetscHashInt((key).field))

#define PetscHashFormKeyEqual(k1,k2) \
  (((k1).label == (k2).label) ? \
   ((k1).value == (k2).value) ? \
   ((k1).field == (k2).field) : 0 : 0)

static PetscChunk _PetscInvalidChunk = {-1, -1, -1};

PETSC_HASH_MAP(HMapForm, PetscHashFormKey, PetscChunk, PetscHashFormKeyHash, PetscHashFormKeyEqual, _PetscInvalidChunk)

/*
  We sort lexicographically on the structure.
  Returns
  -1: left < right
   0: left = right
   1: left > right
*/
PETSC_STATIC_INLINE int Compare_PetscHashFormKey_Private(const void *left, const void *right, PETSC_UNUSED void *ctx)
{
  PetscHashFormKey l = *(PetscHashFormKey *) left;
  PetscHashFormKey r = *(PetscHashFormKey *) right;
  return (l.label < r.label) ? -1 : ((l.label > r.label) ? 1 :
           ((l.value < r.value) ? -1 : (l.value > r.value) ? 1 :
             ((l.field < r.field) ? -1 : (l.field > r.field))));
}

typedef struct _PetscWeakFormOps *PetscWeakFormOps;
struct _PetscWeakFormOps {
  PetscErrorCode (*setfromoptions)(PetscWeakForm);
  PetscErrorCode (*setup)(PetscWeakForm);
  PetscErrorCode (*view)(PetscWeakForm,PetscViewer);
  PetscErrorCode (*destroy)(PetscWeakForm);
};

struct _p_PetscWeakForm {
  PETSCHEADER(struct _PetscWeakFormOps);
  void *data; /* Implementation object */

  PetscInt          Nf;    /* The number of fields in the system */
  PetscChunkBuffer *funcs; /* Buffer holding all function pointers */
  PetscHMapForm     obj;   /* Scalar integral (like an objective function) */
  PetscHMapForm     f0;    /* Weak form integrands against test function for F */
  PetscHMapForm     f1;    /* Weak form integrands against test function derivative for F */
  PetscHMapForm     g0;    /* Weak form integrands for J = dF/du */
  PetscHMapForm     g1;    /* Weak form integrands for J = dF/du */
  PetscHMapForm     g2;    /* Weak form integrands for J = dF/du */
  PetscHMapForm     g3;    /* Weak form integrands for J = dF/du */
  PetscHMapForm     gp0;   /* Weak form integrands for preconditioner for J */
  PetscHMapForm     gp1;   /* Weak form integrands for preconditioner for J */
  PetscHMapForm     gp2;   /* Weak form integrands for preconditioner for J */
  PetscHMapForm     gp3;   /* Weak form integrands for preconditioner for J */
  PetscHMapForm     gt0;   /* Weak form integrands for dF/du_t */
  PetscHMapForm     gt1;   /* Weak form integrands for dF/du_t */
  PetscHMapForm     gt2;   /* Weak form integrands for dF/du_t */
  PetscHMapForm     gt3;   /* Weak form integrands for dF/du_t */
  PetscHMapForm     bdf0;  /* Weak form boundary integrands F_bd */
  PetscHMapForm     bdf1;  /* Weak form boundary integrands F_bd */
  PetscHMapForm     bdg0;  /* Weak form boundary integrands J_bd = dF_bd/du */
  PetscHMapForm     bdg1;  /* Weak form boundary integrands J_bd = dF_bd/du */
  PetscHMapForm     bdg2;  /* Weak form boundary integrands J_bd = dF_bd/du */
  PetscHMapForm     bdg3;  /* Weak form boundary integrands J_bd = dF_bd/du */
  PetscHMapForm     bdgp0; /* Weak form integrands for preconditioner for J_bd */
  PetscHMapForm     bdgp1; /* Weak form integrands for preconditioner for J_bd */
  PetscHMapForm     bdgp2; /* Weak form integrands for preconditioner for J_bd */
  PetscHMapForm     bdgp3; /* Weak form integrands for preconditioner for J_bd */
  PetscHMapForm     r;     /* Riemann solvers */
};

typedef struct _PetscDSOps *PetscDSOps;
struct _PetscDSOps {
  PetscErrorCode (*setfromoptions)(PetscDS);
  PetscErrorCode (*setup)(PetscDS);
  PetscErrorCode (*view)(PetscDS,PetscViewer);
  PetscErrorCode (*destroy)(PetscDS);
};

struct _p_PetscDS {
  PETSCHEADER(struct _PetscDSOps);
  void        *data;              /* Implementation object */
  PetscDS     *subprobs;          /* The subspaces for each dimension */
  PetscBool    setup;             /* Flag for setup */
  PetscBool    isHybrid;          /* Flag for hybrid cell (this is crappy, but the only thing I can see to do now) */
  PetscInt     dimEmbed;          /* The real space coordinate dimension */
  PetscInt     Nf;                /* The number of solution fields */
  PetscObject *disc;              /* The discretization for each solution field (PetscFE, PetscFV, etc.) */
  /* Equations */
  DSBoundary            boundary;      /* Linked list of boundary conditions */
  PetscBool             useJacPre;     /* Flag for using the Jacobian preconditioner */
  PetscBool            *implicit;      /* Flag for implicit or explicit solve for each field */
  PetscInt             *jetDegree;     /* The highest derivative for each field equation, or the k-jet that each discretization needs to tabulate */
  PetscWeakForm         wf;            /* The PetscWeakForm holding all pointwise functions */
  PetscPointFunc       *update;        /* Direct update of field coefficients */
  PetscSimplePointFunc *exactSol;      /* Exact solutions for each field */
  void                **exactCtx;      /* Contexts for the exact solution functions */
  PetscSimplePointFunc *exactSol_t;    /* Time derivative of the exact solutions for each field */
  void                **exactCtx_t;    /* Contexts for the time derivative of the exact solution functions */
  PetscInt              numConstants;  /* Number of constants passed to point functions */
  PetscScalar          *constants;     /* Array of constants passed to point functions */
  void                 **ctx;          /* User contexts for each field */
  /* Computed sizes */
  PetscInt         totDim;             /* Total system dimension */
  PetscInt         totComp;            /* Total field components */
  PetscInt        *Nc;                 /* Number of components for each field */
  PetscInt        *Nb;                 /* Number of basis functions for each field */
  PetscInt        *off;                /* Offsets for each field */
  PetscInt        *offDer;             /* Derivative offsets for each field */
  PetscTabulation *T;                  /* Basis function and derivative tabulation for each field */
  PetscTabulation *Tf;                 /* Basis function and derivative tabulation for each local face and field */
  /* Work space */
  PetscScalar *u;                      /* Field evaluation */
  PetscScalar *u_t;                    /* Field time derivative evaluation */
  PetscScalar *u_x;                    /* Field gradient evaluation */
  PetscScalar *basisReal;              /* Workspace for pushforward */
  PetscScalar *basisDerReal;           /* Workspace for derivative pushforward */
  PetscScalar *testReal;               /* Workspace for pushforward */
  PetscScalar *testDerReal;            /* Workspace for derivative pushforward */
  PetscReal   *x;                      /* Workspace for computing real coordinates */
  PetscScalar *f0, *f1;                /* Point evaluations of weak form residual integrands */
  PetscScalar *g0, *g1, *g2, *g3;      /* Point evaluations of weak form Jacobian integrands */
};

typedef struct {
  PetscInt dummy; /* */
} PetscDS_Basic;

PETSC_INTERN PetscErrorCode PetscDSGetDiscType_Internal(PetscDS, PetscInt, PetscDiscType *);

#endif
