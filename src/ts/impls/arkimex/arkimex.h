typedef struct _ARKTableau *ARKTableau;
struct _ARKTableau {
  char      *name;
  PetscBool  additive;             /* If False, it is a DIRK method */
  PetscInt   order;                /* Classical approximation order of the method */
  PetscInt   s;                    /* Number of stages */
  PetscBool  stiffly_accurate;     /* The implicit part is stiffly accurate */
  PetscBool  FSAL_implicit;        /* The implicit part is FSAL */
  PetscBool  explicit_first_stage; /* The implicit part has an explicit first stage */
  PetscInt   pinterp;              /* Interpolation order */
  PetscReal *At, *bt, *ct;         /* Stiff tableau */
  PetscReal *A, *b, *c;            /* Non-stiff tableau */
  PetscReal *bembedt, *bembed;     /* Embedded formula of order one less (order-1) */
  PetscReal *binterpt, *binterp;   /* Dense output formula */
  PetscReal  ccfl;                 /* Placeholder for CFL coefficient relative to forward Euler */
};
typedef struct _ARKTableauLink *ARKTableauLink;
struct _ARKTableauLink {
  struct _ARKTableau tab;
  ARKTableauLink     next;
};

typedef struct {
  ARKTableau   tableau;
  Vec         *Y;            /* States computed during the step */
  Vec         *YdotI;        /* Time derivatives for the stiff part */
  Vec         *YdotRHS;      /* Function evaluations for the non-stiff part */
  Vec         *Y_prev;       /* States computed during the previous time step */
  Vec         *YdotI_prev;   /* Time derivatives for the stiff part for the previous time step*/
  Vec         *YdotRHS_prev; /* Function evaluations for the non-stiff part for the previous time step*/
  Vec          Ydot0;        /* Holds the slope from the previous step in FSAL case */
  Vec          Ydot;         /* Work vector holding Ydot during residual evaluation */
  Vec          Z;            /* Ydot = shift(Y-Z) */
  IS           alg_is;       /* Index set for algebraic variables, needed when restarting with DIRK */
  PetscScalar *work;         /* Scalar work */
  PetscReal    scoeff;       /* shift = scoeff/dt */
  PetscReal    stage_time;
  PetscBool    imex;
  PetscBool    extrapolate; /* Extrapolate initial guess from previous time-step stage values */
  TSStepStatus status;

  /* context for fast-slow split */
  Vec       Y_snes;       /* Work vector for SNES */
  Vec      *YdotI_fast;   /* Function evaluations for the fast components in YdotI */
  Vec      *YdotRHS_fast; /* Function evaluations for the fast components in YdotRHS */
  Vec      *YdotRHS_slow; /* Function evaluations for the slow components in YdotRHS */
  IS        is_slow, is_fast;
  TS        subts_slow, subts_fast;
  PetscBool fastslowsplit;

  /* context for sensitivity analysis */
  Vec *VecsDeltaLam;   /* Increment of the adjoint sensitivity w.r.t IC at stage */
  Vec *VecsSensiTemp;  /* Vectors to be multiplied with Jacobian transpose */
  Vec *VecsSensiPTemp; /* Temporary Vectors to store JacobianP-transpose-vector product */
} TS_ARKIMEX;
