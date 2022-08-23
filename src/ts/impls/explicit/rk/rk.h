typedef struct _RKTableau *RKTableau;
struct _RKTableau {
  char      *name;
  PetscInt   order;     /* Classical approximation order of the method i              */
  PetscInt   s;         /* Number of stages                                           */
  PetscInt   p;         /* Interpolation order                                        */
  PetscBool  FSAL;      /* flag to indicate if tableau is FSAL                        */
  PetscReal *A, *b, *c; /* Tableau                                                    */
  PetscReal *bembed;    /* Embedded formula of order one less (order-1)               */
  PetscReal *binterp;   /* Dense output formula                                       */
  PetscReal  ccfl;      /* Placeholder for CFL coefficient relative to forward Euler  */
};
typedef struct _RKTableauLink *RKTableauLink;
struct _RKTableauLink {
  struct _RKTableau tab;
  RKTableauLink     next;
};

typedef struct {
  RKTableau    tableau;
  Vec          X0;
  Vec         *Y;            /* States computed during the step                                              */
  Vec         *YdotRHS;      /* Function evaluations for the non-stiff part and contains all components      */
  Vec         *YdotRHS_fast; /* Function evaluations for the non-stiff part and contains fast components     */
  Vec         *YdotRHS_slow; /* Function evaluations for the non-stiff part and contains slow components     */
  Vec         *VecsDeltaLam; /* Increment of the adjoint sensitivity w.r.t IC at stage                       */
  Vec         *VecsSensiTemp;
  Vec          VecDeltaMu;    /* Increment of the adjoint sensitivity w.r.t P at stage                        */
  Vec         *VecsDeltaLam2; /* Increment of the 2nd-order adjoint sensitivity w.r.t IC at stage */
  Vec          VecDeltaMu2;   /* Increment of the 2nd-order adjoint sensitivity w.r.t P at stage */
  Vec         *VecsSensi2Temp;
  PetscScalar *work; /* Scalar work                                                                  */
  PetscInt     slow; /* flag indicates call slow components solver (0) or fast components solver (1) */
  PetscReal    stage_time;
  TSStepStatus status;
  PetscReal    ptime;
  PetscReal    time_step;
  PetscInt     dtratio; /* ratio between slow time step size and fast step size                         */
  IS           is_fast, is_slow;
  TS           subts_fast, subts_slow, subts_current, ts_root;
  PetscBool    use_multirate;
  Mat          MatFwdSensip0;
  Mat         *MatsFwdStageSensip;
  Mat         *MatsFwdSensipTemp;
  Vec          VecDeltaFwdSensipCol; /* Working vector for holding one column of the sensitivity matrix */
} TS_RK;
