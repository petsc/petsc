/*
      Preconditioner module.
*/
#if !defined(__PETSCPC_H)
#define __PETSCPC_H
#include <petscmat.h>
#include <petscdmtypes.h>

PETSC_EXTERN PetscErrorCode PCInitializePackage(void);

/*
    PCList contains the list of preconditioners currently registered
   These are added with PCRegister()
*/
PETSC_EXTERN PetscFunctionList PCList;

/*S
     PC - Abstract PETSc object that manages all preconditioners including direct solvers such as PCLU

   Level: beginner

  Concepts: preconditioners

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types)
S*/
typedef struct _p_PC* PC;

/*J
    PCType - String with the name of a PETSc preconditioner method.

   Level: beginner

   Notes: Click on the links below to see details on a particular solver

          PCRegister() is used to register preconditioners that are then accessible via PCSetType()

.seealso: PCSetType(), PC, PCCreate(), PCRegister(), PCSetFromOptions()
J*/
typedef const char* PCType;
#define PCNONE            "none"
#define PCJACOBI          "jacobi"
#define PCSOR             "sor"
#define PCLU              "lu"
#define PCSHELL           "shell"
#define PCBJACOBI         "bjacobi"
#define PCMG              "mg"
#define PCEISENSTAT       "eisenstat"
#define PCILU             "ilu"
#define PCICC             "icc"
#define PCASM             "asm"
#define PCGASM            "gasm"
#define PCKSP             "ksp"
#define PCCOMPOSITE       "composite"
#define PCREDUNDANT       "redundant"
#define PCSPAI            "spai"
#define PCNN              "nn"
#define PCCHOLESKY        "cholesky"
#define PCPBJACOBI        "pbjacobi"
#define PCMAT             "mat"
#define PCHYPRE           "hypre"
#define PCPARMS           "parms"
#define PCFIELDSPLIT      "fieldsplit"
#define PCTFS             "tfs"
#define PCML              "ml"
#define PCGALERKIN        "galerkin"
#define PCEXOTIC          "exotic"
#define PCHMPI            "hmpi"
#define PCASA             "asa"
#define PCCP              "cp"
#define PCBFBT            "bfbt"
#define PCLSC             "lsc"
#define PCPYTHON          "python"
#define PCPFMG            "pfmg"
#define PCSYSPFMG         "syspfmg"
#define PCREDISTRIBUTE    "redistribute"
#define PCSVD             "svd"
#define PCGAMG            "gamg"
#define PCSACUSP          "sacusp"        /* these four run on NVIDIA GPUs using CUSP */
#define PCSACUSPPOLY      "sacusppoly"
#define PCBICGSTABCUSP    "bicgstabcusp"
#define PCAINVCUSP        "ainvcusp"
#define PCBDDC            "bddc"

/* Logging support */
PETSC_EXTERN PetscClassId PC_CLASSID;

/*E
    PCSide - If the preconditioner is to be applied to the left, right
     or symmetrically around the operator.

   Level: beginner

.seealso:
E*/
typedef enum { PC_SIDE_DEFAULT=-1,PC_LEFT,PC_RIGHT,PC_SYMMETRIC} PCSide;
#define PC_SIDE_MAX (PC_SYMMETRIC + 1)
PETSC_EXTERN const char *const *const PCSides;

PETSC_EXTERN PetscErrorCode PCCreate(MPI_Comm,PC*);
PETSC_EXTERN PetscErrorCode PCSetType(PC,PCType);
PETSC_EXTERN PetscErrorCode PCSetUp(PC);
PETSC_EXTERN PetscErrorCode PCSetUpOnBlocks(PC);
PETSC_EXTERN PetscErrorCode PCApply(PC,Vec,Vec);
PETSC_EXTERN PetscErrorCode PCApplySymmetricLeft(PC,Vec,Vec);
PETSC_EXTERN PetscErrorCode PCApplySymmetricRight(PC,Vec,Vec);
PETSC_EXTERN PetscErrorCode PCApplyBAorAB(PC,PCSide,Vec,Vec,Vec);
PETSC_EXTERN PetscErrorCode PCApplyTranspose(PC,Vec,Vec);
PETSC_EXTERN PetscErrorCode PCApplyTransposeExists(PC,PetscBool *);
PETSC_EXTERN PetscErrorCode PCApplyBAorABTranspose(PC,PCSide,Vec,Vec,Vec);

#define PC_FILE_CLASSID 1211222

/*E
    PCRichardsonConvergedReason - reason a PCApplyRichardson method terminates

   Level: advanced

   Notes: this must match finclude/petscpc.h and the KSPConvergedReason values in petscksp.h

.seealso: PCApplyRichardson()
E*/
typedef enum {
              PCRICHARDSON_CONVERGED_RTOL               =  2,
              PCRICHARDSON_CONVERGED_ATOL               =  3,
              PCRICHARDSON_CONVERGED_ITS                =  4,
              PCRICHARDSON_DIVERGED_DTOL                = -4} PCRichardsonConvergedReason;

PETSC_EXTERN PetscErrorCode PCApplyRichardson(PC,Vec,Vec,Vec,PetscReal,PetscReal,PetscReal,PetscInt,PetscBool ,PetscInt*,PCRichardsonConvergedReason*);
PETSC_EXTERN PetscErrorCode PCApplyRichardsonExists(PC,PetscBool *);
PETSC_EXTERN PetscErrorCode PCSetInitialGuessNonzero(PC,PetscBool );
PETSC_EXTERN PetscErrorCode PCSetUseAmat(PC,PetscBool);
PETSC_EXTERN PetscErrorCode PCGetUseAmat(PC,PetscBool*);

PETSC_EXTERN PetscErrorCode PCRegisterAll(void);
PETSC_EXTERN PetscBool PCRegisterAllCalled;

PETSC_EXTERN PetscErrorCode PCRegister(const char[],PetscErrorCode(*)(PC));

PETSC_EXTERN PetscErrorCode PCReset(PC);
PETSC_EXTERN PetscErrorCode PCDestroy(PC*);
PETSC_EXTERN PetscErrorCode PCSetFromOptions(PC);
PETSC_EXTERN PetscErrorCode PCGetType(PC,PCType*);

PETSC_EXTERN PetscErrorCode PCFactorGetMatrix(PC,Mat*);
PETSC_EXTERN PetscErrorCode PCSetModifySubMatrices(PC,PetscErrorCode(*)(PC,PetscInt,const IS[],const IS[],Mat[],void*),void*);
PETSC_EXTERN PetscErrorCode PCModifySubMatrices(PC,PetscInt,const IS[],const IS[],Mat[],void*);

PETSC_EXTERN PetscErrorCode PCSetOperators(PC,Mat,Mat,MatStructure);
PETSC_EXTERN PetscErrorCode PCGetOperators(PC,Mat*,Mat*,MatStructure*);
PETSC_EXTERN PetscErrorCode PCGetOperatorsSet(PC,PetscBool *,PetscBool *);

PETSC_EXTERN PetscErrorCode PCView(PC,PetscViewer);
PETSC_EXTERN PetscErrorCode PCLoad(PC,PetscViewer);

PETSC_EXTERN PetscErrorCode PCSetOptionsPrefix(PC,const char[]);
PETSC_EXTERN PetscErrorCode PCAppendOptionsPrefix(PC,const char[]);
PETSC_EXTERN PetscErrorCode PCGetOptionsPrefix(PC,const char*[]);

PETSC_EXTERN PetscErrorCode PCComputeExplicitOperator(PC,Mat*);

/*
      These are used to provide extra scaling of preconditioned
   operator for time-stepping schemes like in SUNDIALS
*/
PETSC_EXTERN PetscErrorCode PCGetDiagonalScale(PC,PetscBool *);
PETSC_EXTERN PetscErrorCode PCDiagonalScaleLeft(PC,Vec,Vec);
PETSC_EXTERN PetscErrorCode PCDiagonalScaleRight(PC,Vec,Vec);
PETSC_EXTERN PetscErrorCode PCSetDiagonalScale(PC,Vec);

/* ------------- options specific to particular preconditioners --------- */

PETSC_EXTERN PetscErrorCode PCJacobiSetUseRowMax(PC);
PETSC_EXTERN PetscErrorCode PCJacobiSetUseRowSum(PC);
PETSC_EXTERN PetscErrorCode PCJacobiSetUseAbs(PC);
PETSC_EXTERN PetscErrorCode PCSORSetSymmetric(PC,MatSORType);
PETSC_EXTERN PetscErrorCode PCSORSetOmega(PC,PetscReal);
PETSC_EXTERN PetscErrorCode PCSORSetIterations(PC,PetscInt,PetscInt);

PETSC_EXTERN PetscErrorCode PCEisenstatSetOmega(PC,PetscReal);
PETSC_EXTERN PetscErrorCode PCEisenstatNoDiagonalScaling(PC);

PETSC_EXTERN PetscErrorCode PCBJacobiSetTotalBlocks(PC,PetscInt,const PetscInt[]);
PETSC_EXTERN PetscErrorCode PCBJacobiSetLocalBlocks(PC,PetscInt,const PetscInt[]);

PETSC_EXTERN PetscErrorCode PCShellSetApply(PC,PetscErrorCode (*)(PC,Vec,Vec));
PETSC_EXTERN PetscErrorCode PCShellSetApplyBA(PC,PetscErrorCode (*)(PC,PCSide,Vec,Vec,Vec));
PETSC_EXTERN PetscErrorCode PCShellSetApplyTranspose(PC,PetscErrorCode (*)(PC,Vec,Vec));
PETSC_EXTERN PetscErrorCode PCShellSetSetUp(PC,PetscErrorCode (*)(PC));
PETSC_EXTERN PetscErrorCode PCShellSetApplyRichardson(PC,PetscErrorCode (*)(PC,Vec,Vec,Vec,PetscReal,PetscReal,PetscReal,PetscInt,PetscBool ,PetscInt*,PCRichardsonConvergedReason*));
PETSC_EXTERN PetscErrorCode PCShellSetView(PC,PetscErrorCode (*)(PC,PetscViewer));
PETSC_EXTERN PetscErrorCode PCShellSetDestroy(PC,PetscErrorCode (*)(PC));
PETSC_EXTERN PetscErrorCode PCShellGetContext(PC,void**);
PETSC_EXTERN PetscErrorCode PCShellSetContext(PC,void*);
PETSC_EXTERN PetscErrorCode PCShellSetName(PC,const char[]);
PETSC_EXTERN PetscErrorCode PCShellGetName(PC,const char*[]);

PETSC_EXTERN PetscErrorCode PCFactorSetZeroPivot(PC,PetscReal);

PETSC_EXTERN PetscErrorCode PCFactorSetShiftType(PC,MatFactorShiftType);
PETSC_EXTERN PetscErrorCode PCFactorSetShiftAmount(PC,PetscReal);

PETSC_EXTERN PetscErrorCode PCFactorSetMatSolverPackage(PC,const MatSolverPackage);
PETSC_EXTERN PetscErrorCode PCFactorGetMatSolverPackage(PC,const MatSolverPackage*);
PETSC_EXTERN PetscErrorCode PCFactorSetUpMatSolverPackage(PC);

PETSC_EXTERN PetscErrorCode PCFactorSetFill(PC,PetscReal);
PETSC_EXTERN PetscErrorCode PCFactorSetColumnPivot(PC,PetscReal);
PETSC_EXTERN PetscErrorCode PCFactorReorderForNonzeroDiagonal(PC,PetscReal);

PETSC_EXTERN PetscErrorCode PCFactorSetMatOrderingType(PC,MatOrderingType);
PETSC_EXTERN PetscErrorCode PCFactorSetReuseOrdering(PC,PetscBool );
PETSC_EXTERN PetscErrorCode PCFactorSetReuseFill(PC,PetscBool );
PETSC_EXTERN PetscErrorCode PCFactorSetUseInPlace(PC);
PETSC_EXTERN PetscErrorCode PCFactorSetAllowDiagonalFill(PC);
PETSC_EXTERN PetscErrorCode PCFactorSetPivotInBlocks(PC,PetscBool );

PETSC_EXTERN PetscErrorCode PCFactorGetLevels(PC,PetscInt*);
PETSC_EXTERN PetscErrorCode PCFactorSetLevels(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCFactorSetDropTolerance(PC,PetscReal,PetscReal,PetscInt);

PETSC_EXTERN PetscErrorCode PCASMSetLocalSubdomains(PC,PetscInt,IS[],IS[]);
PETSC_EXTERN PetscErrorCode PCASMSetTotalSubdomains(PC,PetscInt,IS[],IS[]);
PETSC_EXTERN PetscErrorCode PCASMSetOverlap(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCASMSetDMSubdomains(PC,PetscBool);
PETSC_EXTERN PetscErrorCode PCASMGetDMSubdomains(PC,PetscBool*);
PETSC_EXTERN PetscErrorCode PCASMSetSortIndices(PC,PetscBool );

/*E
    PCASMType - Type of additive Schwarz method to use

$  PC_ASM_BASIC        - Symmetric version where residuals from the ghost points are used
$                        and computed values in ghost regions are added together.
$                        Classical standard additive Schwarz.
$  PC_ASM_RESTRICT     - Residuals from ghost points are used but computed values in ghost
$                        region are discarded.
$                        Default.
$  PC_ASM_INTERPOLATE  - Residuals from ghost points are not used, computed values in ghost
$                        region are added back in.
$  PC_ASM_NONE         - Residuals from ghost points are not used, computed ghost values are
$                        discarded.
$                        Not very good.

   Level: beginner

.seealso: PCASMSetType()
E*/
typedef enum {PC_ASM_BASIC = 3,PC_ASM_RESTRICT = 1,PC_ASM_INTERPOLATE = 2,PC_ASM_NONE = 0} PCASMType;
PETSC_EXTERN const char *const PCASMTypes[];

PETSC_EXTERN PetscErrorCode PCASMSetType(PC,PCASMType);
PETSC_EXTERN PetscErrorCode PCASMCreateSubdomains(Mat,PetscInt,IS*[]);
PETSC_EXTERN PetscErrorCode PCASMDestroySubdomains(PetscInt,IS[],IS[]);
PETSC_EXTERN PetscErrorCode PCASMCreateSubdomains2D(PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt*,IS**,IS**);
PETSC_EXTERN PetscErrorCode PCASMGetLocalSubdomains(PC,PetscInt*,IS*[],IS*[]);
PETSC_EXTERN PetscErrorCode PCASMGetLocalSubmatrices(PC,PetscInt*,Mat*[]);

/*E
    PCGASMType - Type of generalized additive Schwarz method to use (differs from ASM in allowing multiple processors per subdomain).

   Each subdomain has nested inner and outer parts.  The inner subdomains are assumed to form a non-overlapping covering of the computational
   domain, while the outer subdomains contain the inner subdomains and overlap with each other.  This preconditioner will compute
   a subdomain correction over each *outer* subdomain from a residual computed there, but its different variants will differ in
   (a) how the outer subdomain residual is computed, and (b) how the outer subdomain correction is computed.

$  PC_GASM_BASIC       - Symmetric version where the full from the outer subdomain is used, and the resulting correction is applied
$                        over the outer subdomains.  As a result, points in the overlap will receive the sum of the corrections
$                        from neighboring subdomains.
$                        Classical standard additive Schwarz.
$  PC_GASM_RESTRICT    - Residual from the outer subdomain is used but the correction is restricted to the inner subdomain only
$                        (i.e., zeroed out over the overlap portion of the outer subdomain before being applied).  As a result,
$                        each point will receive a correction only from the unique inner subdomain containing it (nonoverlapping covering
$                        assumption).
$                        Default.
$  PC_GASM_INTERPOLATE - Residual is zeroed out over the overlap portion of the outer subdomain, but the resulting correction is
$                        applied over the outer subdomain. As a result, points in the overlap will receive the sum of the corrections
$                        from neighboring subdomains.
$
$  PC_GASM_NONE        - Residuals and corrections are zeroed out outside the local subdomains.
$                        Not very good.

   Level: beginner

.seealso: PCGASMSetType()
E*/
typedef enum {PC_GASM_BASIC = 3,PC_GASM_RESTRICT = 1,PC_GASM_INTERPOLATE = 2,PC_GASM_NONE = 0} PCGASMType;
PETSC_EXTERN const char *const PCGASMTypes[];

PETSC_EXTERN PetscErrorCode PCGASMSetSubdomains(PC,PetscInt,IS[],IS[]);
PETSC_EXTERN PetscErrorCode PCGASMSetTotalSubdomains(PC,PetscInt,PetscBool);
PETSC_EXTERN PetscErrorCode PCGASMSetOverlap(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCGASMSetDMSubdomains(PC,PetscBool);
PETSC_EXTERN PetscErrorCode PCGASMGetDMSubdomains(PC,PetscBool*);
PETSC_EXTERN PetscErrorCode PCGASMSetSortIndices(PC,PetscBool );

PETSC_EXTERN PetscErrorCode PCGASMSetType(PC,PCGASMType);
PETSC_EXTERN PetscErrorCode PCGASMCreateLocalSubdomains(Mat,PetscInt,PetscInt,IS*[],IS*[]);
PETSC_EXTERN PetscErrorCode PCGASMDestroySubdomains(PetscInt,IS[],IS[]);
PETSC_EXTERN PetscErrorCode PCGASMCreateSubdomains2D(PC,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt*,IS**,IS**);
PETSC_EXTERN PetscErrorCode PCGASMGetSubdomains(PC,PetscInt*,IS*[],IS*[]);
PETSC_EXTERN PetscErrorCode PCGASMGetSubmatrices(PC,PetscInt*,Mat*[]);

/*E
    PCCompositeType - Determines how two or more preconditioner are composed

$  PC_COMPOSITE_ADDITIVE - results from application of all preconditioners are added together
$  PC_COMPOSITE_MULTIPLICATIVE - preconditioners are applied sequentially to the residual freshly
$                                computed after the previous preconditioner application
$  PC_COMPOSITE_SYMMETRIC_MULTIPLICATIVE - preconditioners are applied sequentially to the residual freshly
$                                computed from first preconditioner to last and then back (Use only for symmetric matrices and preconditions)
$  PC_COMPOSITE_SPECIAL - This is very special for a matrix of the form alpha I + R + S
$                         where first preconditioner is built from alpha I + S and second from
$                         alpha I + R

   Level: beginner

.seealso: PCCompositeSetType()
E*/
typedef enum {PC_COMPOSITE_ADDITIVE,PC_COMPOSITE_MULTIPLICATIVE,PC_COMPOSITE_SYMMETRIC_MULTIPLICATIVE,PC_COMPOSITE_SPECIAL,PC_COMPOSITE_SCHUR} PCCompositeType;
PETSC_EXTERN const char *const PCCompositeTypes[];

PETSC_EXTERN PetscErrorCode PCCompositeSetType(PC,PCCompositeType);
PETSC_EXTERN PetscErrorCode PCCompositeAddPC(PC,PCType);
PETSC_EXTERN PetscErrorCode PCCompositeGetPC(PC,PetscInt,PC *);
PETSC_EXTERN PetscErrorCode PCCompositeSpecialSetAlpha(PC,PetscScalar);

PETSC_EXTERN PetscErrorCode PCRedundantSetNumber(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCRedundantSetScatter(PC,VecScatter,VecScatter);
PETSC_EXTERN PetscErrorCode PCRedundantGetOperators(PC,Mat*,Mat*);

PETSC_EXTERN PetscErrorCode PCSPAISetEpsilon(PC,double);
PETSC_EXTERN PetscErrorCode PCSPAISetNBSteps(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCSPAISetMax(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCSPAISetMaxNew(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCSPAISetBlockSize(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCSPAISetCacheSize(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCSPAISetVerbose(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCSPAISetSp(PC,PetscInt);

PETSC_EXTERN PetscErrorCode PCHYPRESetType(PC,const char[]);
PETSC_EXTERN PetscErrorCode PCHYPREGetType(PC,const char*[]);
PETSC_EXTERN PetscErrorCode PCBJacobiGetLocalBlocks(PC,PetscInt*,const PetscInt*[]);
PETSC_EXTERN PetscErrorCode PCBJacobiGetTotalBlocks(PC,PetscInt*,const PetscInt*[]);

PETSC_EXTERN PetscErrorCode PCFieldSplitSetFields(PC,const char[],PetscInt,const PetscInt*,const PetscInt*);
PETSC_EXTERN PetscErrorCode PCFieldSplitGetType(PC,PCCompositeType*);
PETSC_EXTERN PetscErrorCode PCFieldSplitSetType(PC,PCCompositeType);
PETSC_EXTERN PetscErrorCode PCFieldSplitSetBlockSize(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCFieldSplitSetIS(PC,const char[],IS);
PETSC_EXTERN PetscErrorCode PCFieldSplitGetIS(PC,const char[],IS*);
PETSC_EXTERN PetscErrorCode PCFieldSplitSetDMSplits(PC,PetscBool);
PETSC_EXTERN PetscErrorCode PCFieldSplitGetDMSplits(PC,PetscBool*);

/*E
    PCFieldSplitSchurPreType - Determines how to precondition Schur complement

    Level: intermediate

.seealso: PCFieldSplitSchurPrecondition()
E*/
typedef enum {PC_FIELDSPLIT_SCHUR_PRE_SELF,PC_FIELDSPLIT_SCHUR_PRE_A11,PC_FIELDSPLIT_SCHUR_PRE_USER} PCFieldSplitSchurPreType;
PETSC_EXTERN const char *const PCFieldSplitSchurPreTypes[];

/*E
    PCFieldSplitSchurFactType - determines which off-diagonal parts of the approximate block factorization to use

    Level: intermediate

.seealso: PCFieldSplitSetSchurFactType()
E*/
typedef enum {
  PC_FIELDSPLIT_SCHUR_FACT_DIAG,
  PC_FIELDSPLIT_SCHUR_FACT_LOWER,
  PC_FIELDSPLIT_SCHUR_FACT_UPPER,
  PC_FIELDSPLIT_SCHUR_FACT_FULL
} PCFieldSplitSchurFactType;
PETSC_EXTERN const char *const PCFieldSplitSchurFactTypes[];

PETSC_EXTERN PetscErrorCode PCFieldSplitSchurPrecondition(PC,PCFieldSplitSchurPreType,Mat);
PETSC_EXTERN PetscErrorCode PCFieldSplitSetSchurFactType(PC,PCFieldSplitSchurFactType);
PETSC_EXTERN PetscErrorCode PCFieldSplitGetSchurBlocks(PC,Mat*,Mat*,Mat*,Mat*);

PETSC_EXTERN PetscErrorCode PCGalerkinSetRestriction(PC,Mat);
PETSC_EXTERN PetscErrorCode PCGalerkinSetInterpolation(PC,Mat);

PETSC_EXTERN PetscErrorCode PCSetCoordinates(PC,PetscInt,PetscInt,PetscReal*);
PETSC_EXTERN PetscErrorCode PCSASetVectors(PC,PetscInt,PetscReal *);

PETSC_EXTERN PetscErrorCode PCPythonSetType(PC,const char[]);

PETSC_EXTERN PetscErrorCode PCSetDM(PC,DM);
PETSC_EXTERN PetscErrorCode PCGetDM(PC,DM*);

PETSC_EXTERN PetscErrorCode PCSetApplicationContext(PC,void*);
PETSC_EXTERN PetscErrorCode PCGetApplicationContext(PC,void*);

PETSC_EXTERN PetscErrorCode PCBiCGStabCUSPSetTolerance(PC,PetscReal);
PETSC_EXTERN PetscErrorCode PCBiCGStabCUSPSetIterations(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCBiCGStabCUSPSetUseVerboseMonitor(PC,PetscBool);

PETSC_EXTERN PetscErrorCode PCAINVCUSPSetDropTolerance(PC,PetscReal);
PETSC_EXTERN PetscErrorCode PCAINVCUSPUseScaling(PC,PetscBool);
PETSC_EXTERN PetscErrorCode PCAINVCUSPSetNonzeros(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCAINVCUSPSetLinParameter(PC,PetscInt);
/*E
    PCPARMSGlobalType - Determines the global preconditioner method in PARMS

    Level: intermediate

.seealso: PCPARMSSetGlobal()
E*/
typedef enum {PC_PARMS_GLOBAL_RAS,PC_PARMS_GLOBAL_SCHUR,PC_PARMS_GLOBAL_BJ} PCPARMSGlobalType;
PETSC_EXTERN const char *const PCPARMSGlobalTypes[];
/*E
    PCPARMSLocalType - Determines the local preconditioner method in PARMS

    Level: intermediate

.seealso: PCPARMSSetLocal()
E*/
typedef enum {PC_PARMS_LOCAL_ILU0,PC_PARMS_LOCAL_ILUK,PC_PARMS_LOCAL_ILUT,PC_PARMS_LOCAL_ARMS} PCPARMSLocalType;
PETSC_EXTERN const char *const PCPARMSLocalTypes[];

PETSC_EXTERN PetscErrorCode PCPARMSSetGlobal(PC pc,PCPARMSGlobalType type);
PETSC_EXTERN PetscErrorCode PCPARMSSetLocal(PC pc,PCPARMSLocalType type);
PETSC_EXTERN PetscErrorCode PCPARMSSetSolveTolerances(PC pc,PetscReal tol,PetscInt maxits);
PETSC_EXTERN PetscErrorCode PCPARMSSetSolveRestart(PC pc,PetscInt restart);
PETSC_EXTERN PetscErrorCode PCPARMSSetNonsymPerm(PC pc,PetscBool nonsym);
PETSC_EXTERN PetscErrorCode PCPARMSSetFill(PC pc,PetscInt lfil0,PetscInt lfil1,PetscInt lfil2);

/*E
    PCGAMGType - type of generalized algebraic multigrid (PCGAMG) method

    Level: intermediate

.seealso: PCMG, PCSetType(), PCGAMGSetThreshold(), PCGAMGSetThreshold(), PCGAMGSetReuseProl()
E*/
typedef const char *PCGAMGType;
#define PCGAMGAGG         "agg"
#define PCGAMGGEO         "geo"
#define PCGAMGCLASSICAL   "classical"
PETSC_EXTERN PetscErrorCode PCGAMGSetProcEqLim(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCGAMGSetRepartitioning(PC,PetscBool);
PETSC_EXTERN PetscErrorCode PCGAMGSetUseASMAggs(PC,PetscBool);
PETSC_EXTERN PetscErrorCode PCGAMGSetSolverType(PC,char[],PetscInt);
PETSC_EXTERN PetscErrorCode PCGAMGSetThreshold(PC,PetscReal);
PETSC_EXTERN PetscErrorCode PCGAMGSetCoarseEqLim(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCGAMGSetNlevels(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCGAMGSetType( PC,PCGAMGType );
PETSC_EXTERN PetscErrorCode PCGAMGSetNSmooths(PC pc, PetscInt n);
PETSC_EXTERN PetscErrorCode PCGAMGSetSymGraph(PC pc, PetscBool n);
PETSC_EXTERN PetscErrorCode PCGAMGSetSquareGraph(PC,PetscBool);
PETSC_EXTERN PetscErrorCode PCGAMGSetReuseProl(PC,PetscBool);
PETSC_EXTERN PetscErrorCode PCGAMGFinalizePackage(void);
PETSC_EXTERN PetscErrorCode PCGAMGInitializePackage(void);

#if defined(PETSC_HAVE_PCBDDC)
PETSC_EXTERN PetscErrorCode PCBDDCSetPrimalVerticesLocalIS(PC,IS);
PETSC_EXTERN PetscErrorCode PCBDDCSetCoarseningRatio(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCBDDCSetLevels(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCBDDCSetNullSpace(PC,MatNullSpace);
PETSC_EXTERN PetscErrorCode PCBDDCSetDirichletBoundaries(PC,IS);
PETSC_EXTERN PetscErrorCode PCBDDCGetDirichletBoundaries(PC,IS*);
PETSC_EXTERN PetscErrorCode PCBDDCSetNeumannBoundaries(PC,IS);
PETSC_EXTERN PetscErrorCode PCBDDCGetNeumannBoundaries(PC,IS*);
PETSC_EXTERN PetscErrorCode PCBDDCSetDofsSplitting(PC,PetscInt,IS[]);
PETSC_EXTERN PetscErrorCode PCBDDCSetLocalAdjacencyGraph(PC,PetscInt,const PetscInt[],const PetscInt[],PetscCopyMode);
PETSC_EXTERN PetscErrorCode PCBDDCCreateFETIDPOperators(PC,Mat*,PC*);
PETSC_EXTERN PetscErrorCode PCBDDCMatFETIDPGetRHS(Mat,Vec,Vec);
PETSC_EXTERN PetscErrorCode PCBDDCMatFETIDPGetSolution(Mat,Vec,Vec);
#endif

PETSC_EXTERN PetscErrorCode PCISSetUseStiffnessScaling(PC,PetscBool);
PETSC_EXTERN PetscErrorCode PCISSetSubdomainScalingFactor(PC,PetscScalar);
PETSC_EXTERN PetscErrorCode PCISSetSubdomainDiagonalScaling(PC,Vec);

/*E
    PCMGType - Determines the type of multigrid method that is run.

   Level: beginner

   Values:
+  PC_MG_MULTIPLICATIVE (default) - traditional V or W cycle as determined by PCMGSetCycles()
.  PC_MG_ADDITIVE - the additive multigrid preconditioner where all levels are
                smoothed before updating the residual. This only uses the
                down smoother, in the preconditioner the upper smoother is ignored
.  PC_MG_FULL - same as multiplicative except one also performs grid sequencing,
            that is starts on the coarsest grid, performs a cycle, interpolates
            to the next, performs a cycle etc. This is much like the F-cycle presented in "Multigrid" by Trottenberg, Oosterlee, Schuller page 49, but that
            algorithm supports smoothing on before the restriction on each level in the initial restriction to the coarsest stage. In addition that algorithm
            calls the V-cycle only on the coarser level and has a post-smoother instead.
-  PC_MG_KASKADE - like full multigrid except one never goes back to a coarser level
               from a finer

.seealso: PCMGSetType()

E*/
typedef enum { PC_MG_MULTIPLICATIVE,PC_MG_ADDITIVE,PC_MG_FULL,PC_MG_KASKADE } PCMGType;
PETSC_EXTERN const char *const PCMGTypes[];
#define PC_MG_CASCADE PC_MG_KASKADE;

/*E
    PCMGCycleType - Use V-cycle or W-cycle

   Level: beginner

   Values:
+  PC_MG_V_CYCLE
-  PC_MG_W_CYCLE

.seealso: PCMGSetCycleType()

E*/
typedef enum { PC_MG_CYCLE_V = 1,PC_MG_CYCLE_W = 2 } PCMGCycleType;
PETSC_EXTERN const char *const PCMGCycleTypes[];

PETSC_EXTERN PetscErrorCode PCMGSetType(PC,PCMGType);
PETSC_EXTERN PetscErrorCode PCMGSetLevels(PC,PetscInt,MPI_Comm*);
PETSC_EXTERN PetscErrorCode PCMGGetLevels(PC,PetscInt*);

PETSC_EXTERN PetscErrorCode PCMGSetNumberSmoothUp(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCMGSetNumberSmoothDown(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCMGSetCycleType(PC,PCMGCycleType);
PETSC_EXTERN PetscErrorCode PCMGSetCycleTypeOnLevel(PC,PetscInt,PCMGCycleType);
PETSC_EXTERN PetscErrorCode PCMGSetCyclesOnLevel(PC,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode PCMGMultiplicativeSetCycles(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCMGSetGalerkin(PC,PetscBool);
PETSC_EXTERN PetscErrorCode PCMGGetGalerkin(PC,PetscBool*);


PETSC_EXTERN PetscErrorCode PCMGSetRhs(PC,PetscInt,Vec);
PETSC_EXTERN PetscErrorCode PCMGSetX(PC,PetscInt,Vec);
PETSC_EXTERN PetscErrorCode PCMGSetR(PC,PetscInt,Vec);

PETSC_EXTERN PetscErrorCode PCMGSetRestriction(PC,PetscInt,Mat);
PETSC_EXTERN PetscErrorCode PCMGGetRestriction(PC,PetscInt,Mat*);
PETSC_EXTERN PetscErrorCode PCMGSetInterpolation(PC,PetscInt,Mat);
PETSC_EXTERN PetscErrorCode PCMGGetInterpolation(PC,PetscInt,Mat*);
PETSC_EXTERN PetscErrorCode PCMGSetRScale(PC,PetscInt,Vec);
PETSC_EXTERN PetscErrorCode PCMGGetRScale(PC,PetscInt,Vec*);
PETSC_EXTERN PetscErrorCode PCMGSetResidual(PC,PetscInt,PetscErrorCode (*)(Mat,Vec,Vec,Vec),Mat);
PETSC_EXTERN PetscErrorCode PCMGResidualDefault(Mat,Vec,Vec,Vec);

/*E
    PCExoticType - Face based or wirebasket based coarse grid space

   Level: beginner

.seealso: PCExoticSetType(), PCEXOTIC
E*/
typedef enum { PC_EXOTIC_FACE,PC_EXOTIC_WIREBASKET } PCExoticType;
PETSC_EXTERN const char *const PCExoticTypes[];
PETSC_EXTERN PetscErrorCode PCExoticSetType(PC,PCExoticType);

#endif /* __PETSCPC_H */
