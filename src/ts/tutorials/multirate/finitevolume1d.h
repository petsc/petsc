#include <petscts.h>
#include <petscdm.h>

typedef struct _LimitInfo {
  PetscReal hx;
  PetscInt  m;

  /* context for partitioned system */
  PetscReal hxs, hxm, hxf;
}   *LimitInfo;
void Limit_Upwind(LimitInfo, const PetscScalar *, const PetscScalar *, PetscScalar *);
void Limit_LaxWendroff(LimitInfo, const PetscScalar *, const PetscScalar *, PetscScalar *);
void Limit_BeamWarming(LimitInfo, const PetscScalar *, const PetscScalar *, PetscScalar *);
void Limit_Fromm(LimitInfo, const PetscScalar *, const PetscScalar *, PetscScalar *);
void Limit_Minmod(LimitInfo, const PetscScalar *, const PetscScalar *, PetscScalar *);
void Limit_Superbee(LimitInfo, const PetscScalar *, const PetscScalar *, PetscScalar *);
void Limit_MC(LimitInfo, const PetscScalar *, const PetscScalar *, PetscScalar *);
void Limit_VanLeer(LimitInfo, const PetscScalar *, const PetscScalar *, PetscScalar *);
void Limit_VanAlbada(LimitInfo, const PetscScalar *, const PetscScalar *, PetscScalar *); /* differentiable */
void Limit_VanAlbadaTVD(LimitInfo, const PetscScalar *, const PetscScalar *, PetscScalar *);
void Limit_Koren(LimitInfo, const PetscScalar *, const PetscScalar *, PetscScalar *);    /* differentiable */
void Limit_KorenSym(LimitInfo, const PetscScalar *, const PetscScalar *, PetscScalar *); /* differentiable */
void Limit_Koren3(LimitInfo, const PetscScalar *, const PetscScalar *, PetscScalar *);
void Limit_CadaTorrilhon2(LimitInfo, const PetscScalar *, const PetscScalar *, PetscScalar *);
void Limit_CadaTorrilhon3R(PetscReal r, LimitInfo, const PetscScalar *, const PetscScalar *, PetscScalar *);
void Limit_CadaTorrilhon3R0p1(LimitInfo, const PetscScalar *, const PetscScalar *, PetscScalar *);
void Limit_CadaTorrilhon3R1(LimitInfo, const PetscScalar *, const PetscScalar *, PetscScalar *);
void Limit_CadaTorrilhon3R10(LimitInfo, const PetscScalar *, const PetscScalar *, PetscScalar *);
void Limit_CadaTorrilhon3R100(LimitInfo, const PetscScalar *, const PetscScalar *, PetscScalar *);

/* --------------------------------- Finite Volume data structures ----------------------------------- */

typedef enum {
  FVBC_PERIODIC,
  FVBC_OUTFLOW,
  FVBC_INFLOW
} FVBCType;
extern const char *FVBCTypes[];
/* we add three new variables at the end of input parameters of function to be position of cell center, left boundary of domain, right boundary of domain */
typedef PetscErrorCode (*RiemannFunction)(void *, PetscInt, const PetscScalar *, const PetscScalar *, PetscScalar *, PetscReal *, PetscReal, PetscReal, PetscReal);
typedef PetscErrorCode (*ReconstructFunction)(void *, PetscInt, const PetscScalar *, PetscScalar *, PetscScalar *, PetscReal *, PetscReal);

PetscErrorCode RiemannListAdd(PetscFunctionList *, const char *, RiemannFunction);
PetscErrorCode RiemannListFind(PetscFunctionList, const char *, RiemannFunction *);
PetscErrorCode ReconstructListAdd(PetscFunctionList *, const char *, ReconstructFunction);
PetscErrorCode ReconstructListFind(PetscFunctionList, const char *, ReconstructFunction *);
PetscErrorCode PhysicsDestroy_SimpleFree(void *);

typedef struct {
  PetscErrorCode (*sample)(void *, PetscInt, FVBCType, PetscReal, PetscReal, PetscReal, PetscReal, PetscReal *);
  RiemannFunction     riemann;
  ReconstructFunction characteristic;
  PetscErrorCode (*destroy)(void *);
  void    *user;
  PetscInt dof;
  char    *fieldname[16];
} PhysicsCtx;

void Limit2_Upwind(LimitInfo info, const PetscScalar *jL, const PetscScalar *jR, const PetscInt len_slow, const PetscInt len_fast, PetscInt n, PetscScalar *lmt);
void Limit2_LaxWendroff(LimitInfo info, const PetscScalar *jL, const PetscScalar *jR, const PetscInt len_slow, const PetscInt len_fast, PetscInt n, PetscScalar *lmt);
void Limit2_BeamWarming(LimitInfo info, const PetscScalar *jL, const PetscScalar *jR, const PetscInt len_slow, const PetscInt len_fast, PetscInt n, PetscScalar *lmt);
void Limit2_Fromm(LimitInfo info, const PetscScalar *jL, const PetscScalar *jR, const PetscInt len_slow, const PetscInt len_fast, PetscInt n, PetscScalar *lmt);
void Limit2_Minmod(LimitInfo info, const PetscScalar *jL, const PetscScalar *jR, const PetscInt len_slow, const PetscInt len_fast, PetscInt n, PetscScalar *lmt);
void Limit2_Superbee(LimitInfo info, const PetscScalar *jL, const PetscScalar *jR, const PetscInt len_slow, const PetscInt len_fast, PetscInt n, PetscScalar *lmt);
void Limit2_MC(LimitInfo info, const PetscScalar *jL, const PetscScalar *jR, const PetscInt len_slow, const PetscInt len_fast, PetscInt n, PetscScalar *lmt);
void Limit2_Koren3(LimitInfo info, const PetscScalar *jL, const PetscScalar *jR, const PetscInt len_slow, const PetscInt len_fast, PetscInt n, PetscScalar *lmt);

void Limit3_Upwind(LimitInfo info, const PetscScalar *jL, const PetscScalar *jR, const PetscInt sm, const PetscInt mf, const PetscInt fm, const PetscInt ms, PetscInt n, PetscScalar *lmt);
void Limit3_LaxWendroff(LimitInfo info, const PetscScalar *jL, const PetscScalar *jR, const PetscInt sm, const PetscInt mf, const PetscInt fm, const PetscInt ms, PetscInt n, PetscScalar *lmt);
void Limit3_BeamWarming(LimitInfo info, const PetscScalar *jL, const PetscScalar *jR, const PetscInt sm, const PetscInt mf, const PetscInt fm, const PetscInt ms, PetscInt n, PetscScalar *lmt);
void Limit3_Fromm(LimitInfo info, const PetscScalar *jL, const PetscScalar *jR, const PetscInt sm, const PetscInt mf, const PetscInt fm, const PetscInt ms, PetscInt n, PetscScalar *lmt);
void Limit3_Minmod(LimitInfo info, const PetscScalar *jL, const PetscScalar *jR, const PetscInt sm, const PetscInt mf, const PetscInt fm, const PetscInt ms, PetscInt n, PetscScalar *lmt);
void Limit3_Superbee(LimitInfo info, const PetscScalar *jL, const PetscScalar *jR, const PetscInt sm, const PetscInt mf, const PetscInt fm, const PetscInt ms, PetscInt n, PetscScalar *lmt);
void Limit3_MC(LimitInfo info, const PetscScalar *jL, const PetscScalar *jR, const PetscInt sm, const PetscInt mf, const PetscInt fm, const PetscInt ms, PetscInt n, PetscScalar *lmt);
void Limit3_Koren3(LimitInfo info, const PetscScalar *jL, const PetscScalar *jR, const PetscInt sm, const PetscInt mf, const PetscInt fm, const PetscInt ms, PetscInt n, PetscScalar *lmt);

typedef PetscErrorCode (*RiemannFunction_2WaySplit)(void *, PetscInt, const PetscScalar *, const PetscScalar *, PetscScalar *, PetscReal *);
typedef PetscErrorCode (*ReconstructFunction_2WaySplit)(void *, PetscInt, const PetscScalar *, PetscScalar *, PetscScalar *, PetscReal *);

PetscErrorCode RiemannListAdd_2WaySplit(PetscFunctionList *, const char *, RiemannFunction_2WaySplit);
PetscErrorCode RiemannListFind_2WaySplit(PetscFunctionList, const char *, RiemannFunction_2WaySplit *);
PetscErrorCode ReconstructListAdd_2WaySplit(PetscFunctionList *, const char *, ReconstructFunction_2WaySplit);
PetscErrorCode ReconstructListFind_2WaySplit(PetscFunctionList, const char *, ReconstructFunction_2WaySplit *);

typedef struct {
  PetscErrorCode (*sample2)(void *, PetscInt, FVBCType, PetscReal, PetscReal, PetscReal, PetscReal, PetscReal *);
  PetscErrorCode (*inflow)(void *, PetscReal, PetscReal, PetscReal *);
  RiemannFunction_2WaySplit     riemann2;
  ReconstructFunction_2WaySplit characteristic2;
  PetscErrorCode (*destroy)(void *);
  void      *user;
  PetscInt   dof;
  char      *fieldname[16];
  PetscBool *bcinflowindex; /* Boolean array where bcinflowindex[dof*i+j] = TRUE indicates that the jth component of the solution
                                                     is an inflow boundary condition and i = 0 is left bc, i = 1 is right bc. FALSE implies outflow
                                                     outflow boundary condition. */
} PhysicsCtx2;

typedef struct {
  void (*limit)(LimitInfo, const PetscScalar *, const PetscScalar *, PetscScalar *);
  PhysicsCtx physics;
  MPI_Comm   comm;
  char       prefix[256];

  /* Local work arrays */
  PetscScalar *R, *Rinv; /* Characteristic basis, and it's inverse.  COLUMN-MAJOR */
  PetscScalar *cjmpLR;   /* Jumps at left and right edge of cell, in characteristic basis, len=2*dof */
  PetscScalar *cslope;   /* Limited slope, written in characteristic basis */
  PetscScalar *uLR;      /* Solution at left and right of interface, conservative variables, len=2*dof */
  PetscScalar *flux;     /* Flux across interface */
  PetscReal   *speeds;   /* Speeds of each wave */
  PetscReal   *ub;       /* Boundary data for inflow boundary conditions */

  PetscReal cfl_idt; /* Max allowable value of 1/Delta t */
  PetscReal cfl;
  PetscReal xmin, xmax;
  PetscInt  initial;
  PetscBool simulation;
  FVBCType  bctype;
  PetscBool exact;

  /* context for partitioned system */
  void (*limit3)(LimitInfo, const PetscScalar *, const PetscScalar *, const PetscInt, const PetscInt, const PetscInt, const PetscInt, PetscInt, PetscScalar *);
  void (*limit2)(LimitInfo, const PetscScalar *, const PetscScalar *, PetscInt, PetscInt, PetscInt, PetscScalar *);
  PhysicsCtx2 physics2;
  PetscInt    hratio; /* hratio = hslow/hfast */
  IS          isf, iss, isf2, iss2, ism, issb, ismb;
  PetscBool   recursive;
  PetscInt    sm, mf, fm, ms;     /* positions (array index) for slow-medium, medium-fast, fast-medium, medium-slow interfaces */
  PetscInt    sf, fs;             /* slow-fast and fast-slow interfaces */
  PetscInt    lsbwidth, rsbwidth; /* left slow buffer width and right slow buffer width */
  PetscInt    lmbwidth, rmbwidth; /* left medium buffer width and right medium buffer width */
} FVCtx;

/* --------------------------------- Finite Volume Solver ----------------------------------- */
PetscErrorCode FVRHSFunction(TS, PetscReal, Vec, Vec, void *);
PetscErrorCode FVSample(FVCtx *, DM, PetscReal, Vec);
PetscErrorCode SolutionStatsView(DM, Vec, PetscViewer);
