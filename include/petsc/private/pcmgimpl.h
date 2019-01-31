/*
      Data structure used for Multigrid preconditioner.
*/
#if !defined(__MG_IMPL)
#define __MG_IMPL
#include <petsc/private/pcimpl.h>
#include <petscksp.h>
#define PETSC_MG_MAXLEVELS 10
/*
     Each level has its own copy of this data.
     Level (0) is always the coarsest level and Level (levels-1) is the finest.
*/
typedef struct {
  PetscInt cycles;                             /* Type of cycle to run: 1 V 2 W */
  PetscInt level;                              /* level = 0 coarsest level */
  PetscInt levels;                             /* number of active levels used */
  Vec      b;                                  /* Right hand side */
  Vec      x;                                  /* Solution */
  Vec      r;                                  /* Residual */
  Vec     *coarseSpace;                        /* A vector space which should be accurately captured by the next coarser mesh,
                                                  and thus accurately interpolated. This array should have the same size on each
                                                  level, and the vectors should correspond to the same function discretized in
                                                  the sequence of spaces. */

  PetscErrorCode (*residual)(Mat,Vec,Vec,Vec);

  Mat           A;                             /* matrix used in forming residual*/
  KSP           smoothd;                       /* pre smoother */
  KSP           smoothu;                       /* post smoother */
  Mat           interpolate;
  Mat           restrct;                       /* restrict is a reserved word in C99 and on Cray */
  Mat           inject;                        /* Used for moving state if provided. */
  Vec           rscale;                        /* scaling of restriction matrix */
  PetscLogEvent eventsmoothsetup;              /* if logging times for each level */
  PetscLogEvent eventsmoothsolve;
  PetscLogEvent eventresidual;
  PetscLogEvent eventinterprestrict;
} PC_MG_Levels;

/*
    This data structure is shared by all the levels.
*/
typedef struct {
  PCMGType            am;                     /* Multiplicative, additive or full */
  PetscInt            cyclesperpcapply;       /* Number of cycles to use in each PCApply(), multiplicative only*/
  PetscInt            maxlevels;              /* total number of levels allocated */
  PCMGGalerkinType    galerkin;               /* use Galerkin process to compute coarser matrices */
  PetscBool           usedmfornumberoflevels; /* sets the number of levels by getting this information out of the DM */

  PetscBool           adaptInterpolation;     /* flag to adapt the interpolator based upon the coarseSpace */
  PCMGCoarseSpaceType coarseSpaceType;        /* Type of coarse space: polynomials, harmonics, eigenvectors, ... */
  PetscInt            Nc;                     /* The number of vectors in coarseSpace */
  PetscInt            eigenvalue;             /* Key for storing the eigenvalue as a scalar in the eigenvector Vec */
  PetscBool           mespMonitor;            /* flag to monitor the multilevel eigensolver */

  PetscInt     nlevels;
  PC_MG_Levels **levels;
  PetscInt     default_smoothu;               /* number of smooths per level if not over-ridden */
  PetscInt     default_smoothd;               /*  with calls to KSPSetTolerances() */
  PetscReal    rtol,abstol,dtol,ttol;         /* tolerances for when running with PCApplyRichardson_MG */

  void          *innerctx;                    /* optional data for preconditioner, like PCEXOTIC that inherits off of PCMG */
  PetscLogStage stageApply;
  PetscErrorCode (*view)(PC,PetscViewer);     /* GAMG and other objects that use PCMG can set their own viewer here */
  PetscReal      min_eigen_DinvA[PETSC_MG_MAXLEVELS];
  PetscReal      max_eigen_DinvA[PETSC_MG_MAXLEVELS];
} PC_MG;

PETSC_INTERN PetscErrorCode PCSetUp_MG(PC);
PETSC_INTERN PetscErrorCode PCDestroy_MG(PC);
PETSC_INTERN PetscErrorCode PCSetFromOptions_MG(PetscOptionItems *PetscOptionsObject,PC);
PETSC_INTERN PetscErrorCode PCView_MG(PC,PetscViewer);
PETSC_INTERN PetscErrorCode PCMGGetLevels_MG(PC,PetscInt *);
PETSC_INTERN PetscErrorCode PCMGSetLevels_MG(PC,PetscInt,MPI_Comm *);
PETSC_DEPRECATED_FUNCTION("Use PCMGResidualDefault() (since version 3.5)") PETSC_STATIC_INLINE PetscErrorCode PCMGResidual_Default(Mat A,Vec b,Vec x,Vec r) {
  return PCMGResidualDefault(A,b,x,r);
}

PETSC_INTERN PetscErrorCode DMSetBasisFunction_Internal(PetscInt, PetscBool, PetscInt, PetscErrorCode (**)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *));
PETSC_INTERN PetscErrorCode PCMGComputeCoarseSpace_Internal(PC, PetscInt, PCMGCoarseSpaceType, PetscInt, const Vec[], Vec *[]);
PETSC_INTERN PetscErrorCode PCMGAdaptInterpolator_Internal(PC, PetscInt, KSP, KSP, PetscInt, Vec[], Vec[]);
PETSC_INTERN PetscErrorCode PCMGRecomputeLevelOperators_Internal(PC, PetscInt);


#endif
