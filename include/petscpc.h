/*
      Preconditioner module. 
*/
#if !defined(__PETSCPC_H)
#define __PETSCPC_H
#include "petscdm.h"
PETSC_EXTERN_CXX_BEGIN

extern PetscErrorCode   PCInitializePackage(const char[]);

/*
    PCList contains the list of preconditioners currently registered
   These are added with the PCRegisterDynamic() macro
*/
extern PetscFList PCList;

/*S
     PC - Abstract PETSc object that manages all preconditioners

   Level: beginner

  Concepts: preconditioners

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types)
S*/
typedef struct _p_PC* PC;

/*E
    PCType - String with the name of a PETSc preconditioner method or the creation function
       with an optional dynamic library name, for example
       http://www.mcs.anl.gov/petsc/lib.a:mypccreate()

   Level: beginner

   Notes: Click on the links below to see details on a particular solver

.seealso: PCSetType(), PC, PCCreate()
E*/
#define PCType char*
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
#define PCFIELDSPLIT      "fieldsplit"
#define PCTFS             "tfs"
#define PCML              "ml"
#define PCPROMETHEUS      "prometheus"
#define PCGALERKIN        "galerkin"
#define PCEXOTIC          "exotic"
#define PCOPENMP          "openmp"
#define PCSUPPORTGRAPH    "supportgraph"
#define PCASA             "asa"
#define PCCP              "cp"
#define PCBFBT            "bfbt"
#define PCLSC             "lsc"
#define PCPYTHON          "python"
#define PCPFMG            "pfmg"
#define PCSYSPFMG         "syspfmg"
#define PCREDISTRIBUTE    "redistribute"
#define PCSACUSP          "sacusp"
#define PCSACUSPPOLY      "sacusppoly"
#define PCBICGSTABCUSP    "bicgstabcusp"
#define PCSVD             "svd"
#define PCAINVCUSP        "ainvcusp"

/* Logging support */
extern PetscClassId  PC_CLASSID;

/*E
    PCSide - If the preconditioner is to be applied to the left, right
     or symmetrically around the operator.

   Level: beginner

.seealso: 
E*/
typedef enum { PC_LEFT,PC_RIGHT,PC_SYMMETRIC } PCSide;
extern const char *PCSides[];

extern PetscErrorCode  PCCreate(MPI_Comm,PC*);
extern PetscErrorCode  PCSetType(PC,const PCType);
extern PetscErrorCode  PCSetUp(PC);
extern PetscErrorCode  PCSetUpOnBlocks(PC);
extern PetscErrorCode  PCApply(PC,Vec,Vec);
extern PetscErrorCode  PCApplySymmetricLeft(PC,Vec,Vec);
extern PetscErrorCode  PCApplySymmetricRight(PC,Vec,Vec);
extern PetscErrorCode  PCApplyBAorAB(PC,PCSide,Vec,Vec,Vec);
extern PetscErrorCode  PCApplyTranspose(PC,Vec,Vec);
extern PetscErrorCode  PCApplyTransposeExists(PC,PetscBool *);
extern PetscErrorCode  PCApplyBAorABTranspose(PC,PCSide,Vec,Vec,Vec);

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

extern PetscErrorCode  PCApplyRichardson(PC,Vec,Vec,Vec,PetscReal,PetscReal,PetscReal,PetscInt,PetscBool ,PetscInt*,PCRichardsonConvergedReason*);
extern PetscErrorCode  PCApplyRichardsonExists(PC,PetscBool *);
extern PetscErrorCode  PCSetInitialGuessNonzero(PC,PetscBool );

extern PetscErrorCode  PCRegisterDestroy(void);
extern PetscErrorCode  PCRegisterAll(const char[]);
extern PetscBool  PCRegisterAllCalled;

extern PetscErrorCode  PCRegister(const char[],const char[],const char[],PetscErrorCode(*)(PC));

/*MC
   PCRegisterDynamic - Adds a method to the preconditioner package.

   Synopsis:
   PetscErrorCode PCRegisterDynamic(const char *name_solver,const char *path,const char *name_create,PetscErrorCode (*routine_create)(PC))

   Not collective

   Input Parameters:
+  name_solver - name of a new user-defined solver
.  path - path (either absolute or relative) the library containing this solver
.  name_create - name of routine to create method context
-  routine_create - routine to create method context

   Notes:
   PCRegisterDynamic() may be called multiple times to add several user-defined preconditioners.

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Sample usage:
.vb
   PCRegisterDynamic("my_solver","/home/username/my_lib/lib/libO/solaris/mylib",
              "MySolverCreate",MySolverCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     PCSetType(pc,"my_solver")
   or at runtime via the option
$     -pc_type my_solver

   Level: advanced

   Notes: ${PETSC_ARCH}, ${PETSC_DIR}, ${PETSC_LIB_DIR},  or ${any environmental variable}
           occuring in pathname will be replaced with appropriate values.
         If your function is not being put into a shared library then use PCRegister() instead

.keywords: PC, register

.seealso: PCRegisterAll(), PCRegisterDestroy()
M*/
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define PCRegisterDynamic(a,b,c,d) PCRegister(a,b,c,0)
#else
#define PCRegisterDynamic(a,b,c,d) PCRegister(a,b,c,d)
#endif

extern PetscErrorCode  PCReset(PC);
extern PetscErrorCode  PCDestroy(PC);
extern PetscErrorCode  PCSetFromOptions(PC);
extern PetscErrorCode  PCGetType(PC,const PCType*);

extern PetscErrorCode  PCFactorGetMatrix(PC,Mat*);
extern PetscErrorCode  PCSetModifySubMatrices(PC,PetscErrorCode(*)(PC,PetscInt,const IS[],const IS[],Mat[],void*),void*);
extern PetscErrorCode  PCModifySubMatrices(PC,PetscInt,const IS[],const IS[],Mat[],void*);

extern PetscErrorCode  PCSetOperators(PC,Mat,Mat,MatStructure);
extern PetscErrorCode  PCGetOperators(PC,Mat*,Mat*,MatStructure*);
extern PetscErrorCode  PCGetOperatorsSet(PC,PetscBool *,PetscBool *);

extern PetscErrorCode  PCView(PC,PetscViewer);

extern PetscErrorCode  PCSetOptionsPrefix(PC,const char[]);
extern PetscErrorCode  PCAppendOptionsPrefix(PC,const char[]);
extern PetscErrorCode  PCGetOptionsPrefix(PC,const char*[]);

extern PetscErrorCode  PCComputeExplicitOperator(PC,Mat*);

/*
      These are used to provide extra scaling of preconditioned 
   operator for time-stepping schemes like in SUNDIALS 
*/
extern PetscErrorCode  PCGetDiagonalScale(PC,PetscBool *);
extern PetscErrorCode  PCDiagonalScaleLeft(PC,Vec,Vec);
extern PetscErrorCode  PCDiagonalScaleRight(PC,Vec,Vec);
extern PetscErrorCode  PCSetDiagonalScale(PC,Vec);

/* ------------- options specific to particular preconditioners --------- */

extern PetscErrorCode  PCJacobiSetUseRowMax(PC);
extern PetscErrorCode  PCJacobiSetUseRowSum(PC);
extern PetscErrorCode  PCJacobiSetUseAbs(PC);
extern PetscErrorCode  PCSORSetSymmetric(PC,MatSORType);
extern PetscErrorCode  PCSORSetOmega(PC,PetscReal);
extern PetscErrorCode  PCSORSetIterations(PC,PetscInt,PetscInt);

extern PetscErrorCode  PCEisenstatSetOmega(PC,PetscReal);
extern PetscErrorCode  PCEisenstatNoDiagonalScaling(PC);

#define USE_PRECONDITIONER_MATRIX 0
#define USE_TRUE_MATRIX           1
extern PetscErrorCode  PCBJacobiSetUseTrueLocal(PC);
extern PetscErrorCode  PCBJacobiSetTotalBlocks(PC,PetscInt,const PetscInt[]);
extern PetscErrorCode  PCBJacobiSetLocalBlocks(PC,PetscInt,const PetscInt[]);

extern PetscErrorCode  PCKSPSetUseTrue(PC);

extern PetscErrorCode  PCShellSetApply(PC,PetscErrorCode (*)(PC,Vec,Vec)); 
extern PetscErrorCode  PCShellSetApplyBA(PC,PetscErrorCode (*)(PC,PCSide,Vec,Vec,Vec)); 
extern PetscErrorCode  PCShellSetApplyTranspose(PC,PetscErrorCode (*)(PC,Vec,Vec));
extern PetscErrorCode  PCShellSetSetUp(PC,PetscErrorCode (*)(PC));
extern PetscErrorCode  PCShellSetApplyRichardson(PC,PetscErrorCode (*)(PC,Vec,Vec,Vec,PetscReal,PetscReal,PetscReal,PetscInt,PetscBool ,PetscInt*,PCRichardsonConvergedReason*));
extern PetscErrorCode  PCShellSetView(PC,PetscErrorCode (*)(PC,PetscViewer));
extern PetscErrorCode  PCShellSetDestroy(PC,PetscErrorCode (*)(PC));
extern PetscErrorCode  PCShellGetContext(PC,void**);
extern PetscErrorCode  PCShellSetContext(PC,void*);
extern PetscErrorCode  PCShellSetName(PC,const char[]);
extern PetscErrorCode  PCShellGetName(PC,char*[]);

extern PetscErrorCode  PCFactorSetZeroPivot(PC,PetscReal);

extern PetscErrorCode  PCFactorSetShiftType(PC,MatFactorShiftType); 
extern PetscErrorCode  PCFactorSetShiftAmount(PC,PetscReal); 

extern PetscErrorCode  PCFactorSetMatSolverPackage(PC,const MatSolverPackage);
extern PetscErrorCode  PCFactorGetMatSolverPackage(PC,const MatSolverPackage*);
extern PetscErrorCode  PCFactorSetUpMatSolverPackage(PC);

extern PetscErrorCode  PCFactorSetFill(PC,PetscReal);
extern PetscErrorCode  PCFactorSetColumnPivot(PC,PetscReal);
extern PetscErrorCode  PCFactorReorderForNonzeroDiagonal(PC,PetscReal);

extern PetscErrorCode  PCFactorSetMatOrderingType(PC,const MatOrderingType);
extern PetscErrorCode  PCFactorSetReuseOrdering(PC,PetscBool );
extern PetscErrorCode  PCFactorSetReuseFill(PC,PetscBool );
extern PetscErrorCode  PCFactorSetUseInPlace(PC);
extern PetscErrorCode  PCFactorSetAllowDiagonalFill(PC);
extern PetscErrorCode  PCFactorSetPivotInBlocks(PC,PetscBool );

extern PetscErrorCode  PCFactorSetLevels(PC,PetscInt);
extern PetscErrorCode  PCFactorSetDropTolerance(PC,PetscReal,PetscReal,PetscInt);

extern PetscErrorCode  PCASMSetLocalSubdomains(PC,PetscInt,IS[],IS[]);
extern PetscErrorCode  PCASMSetTotalSubdomains(PC,PetscInt,IS[],IS[]);
extern PetscErrorCode  PCASMSetOverlap(PC,PetscInt);
extern PetscErrorCode  PCASMSetSortIndices(PC,PetscBool );

/*E
    PCASMType - Type of additive Schwarz method to use

$  PC_ASM_BASIC - symmetric version where residuals from the ghost points are used
$                 and computed values in ghost regions are added together. Classical
$                 standard additive Schwarz
$  PC_ASM_RESTRICT - residuals from ghost points are used but computed values in ghost
$                    region are discarded. Default
$  PC_ASM_INTERPOLATE - residuals from ghost points are not used, computed values in ghost
$                       region are added back in
$  PC_ASM_NONE - ghost point residuals are not used, computed ghost values are discarded
$                not very good.                

   Level: beginner

.seealso: PCASMSetType()
E*/
typedef enum {PC_ASM_BASIC = 3,PC_ASM_RESTRICT = 1,PC_ASM_INTERPOLATE = 2,PC_ASM_NONE = 0} PCASMType;
extern const char *PCASMTypes[];

extern PetscErrorCode  PCASMSetType(PC,PCASMType);
extern PetscErrorCode  PCASMCreateSubdomains(Mat,PetscInt,IS*[]);
extern PetscErrorCode  PCASMDestroySubdomains(PetscInt,IS[],IS[]);
extern PetscErrorCode  PCASMCreateSubdomains2D(PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt*,IS**,IS**);
extern PetscErrorCode  PCASMGetLocalSubdomains(PC,PetscInt*,IS*[],IS*[]);
extern PetscErrorCode  PCASMGetLocalSubmatrices(PC,PetscInt*,Mat*[]);

/*E
    PCGASMType - Type of generalized additive Schwarz method to use (differs from ASM in allowing multiple processors per domain)

$  PC_GASM_BASIC - symmetric version where residuals from the ghost points are used
$                 and computed values in ghost regions are added together. Classical
$                 standard additive Schwarz
$  PC_GASM_RESTRICT - residuals from ghost points are used but computed values in ghost
$                    region are discarded. Default
$  PC_GASM_INTERPOLATE - residuals from ghost points are not used, computed values in ghost
$                       region are added back in
$  PC_GASM_NONE - ghost point residuals are not used, computed ghost values are discarded
$                not very good.                

   Level: beginner

.seealso: PCGASMSetType()
E*/
typedef enum {PC_GASM_BASIC = 3,PC_GASM_RESTRICT = 1,PC_GASM_INTERPOLATE = 2,PC_GASM_NONE = 0} PCGASMType;
extern const char *PCGASMTypes[];

extern PetscErrorCode  PCGASMSetLocalSubdomains(PC,PetscInt,IS[],IS[]);
extern PetscErrorCode  PCGASMSetTotalSubdomains(PC,PetscInt);
extern PetscErrorCode  PCGASMSetOverlap(PC,PetscInt);
extern PetscErrorCode  PCGASMSetSortIndices(PC,PetscBool );

extern PetscErrorCode  PCGASMSetType(PC,PCGASMType);
extern PetscErrorCode  PCGASMCreateSubdomains(Mat,PetscInt,IS*[]);
extern PetscErrorCode  PCGASMDestroySubdomains(PetscInt,IS[],IS[]);
extern PetscErrorCode  PCGASMCreateSubdomains2D(PC,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt*,IS**,IS**);
extern PetscErrorCode  PCGASMGetLocalSubdomains(PC,PetscInt*,IS*[],IS*[]);
extern PetscErrorCode  PCGASMGetLocalSubmatrices(PC,PetscInt*,Mat*[]);

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
extern const char *PCCompositeTypes[];

extern PetscErrorCode  PCCompositeSetUseTrue(PC);
extern PetscErrorCode  PCCompositeSetType(PC,PCCompositeType);
extern PetscErrorCode  PCCompositeAddPC(PC,PCType);
extern PetscErrorCode  PCCompositeGetPC(PC,PetscInt,PC *);
extern PetscErrorCode  PCCompositeSpecialSetAlpha(PC,PetscScalar);

extern PetscErrorCode  PCRedundantSetNumber(PC,PetscInt);
extern PetscErrorCode  PCRedundantSetScatter(PC,VecScatter,VecScatter);
extern PetscErrorCode  PCRedundantGetOperators(PC,Mat*,Mat*);

extern PetscErrorCode  PCSPAISetEpsilon(PC,double);
extern PetscErrorCode  PCSPAISetNBSteps(PC,PetscInt);
extern PetscErrorCode  PCSPAISetMax(PC,PetscInt);
extern PetscErrorCode  PCSPAISetMaxNew(PC,PetscInt);
extern PetscErrorCode  PCSPAISetBlockSize(PC,PetscInt);
extern PetscErrorCode  PCSPAISetCacheSize(PC,PetscInt);
extern PetscErrorCode  PCSPAISetVerbose(PC,PetscInt);
extern PetscErrorCode  PCSPAISetSp(PC,PetscInt);

extern PetscErrorCode  PCHYPRESetType(PC,const char[]);
extern PetscErrorCode  PCHYPREGetType(PC,const char*[]);
extern PetscErrorCode  PCBJacobiGetLocalBlocks(PC,PetscInt*,const PetscInt*[]);
extern PetscErrorCode  PCBJacobiGetTotalBlocks(PC,PetscInt*,const PetscInt*[]);

extern PetscErrorCode  PCFieldSplitSetFields(PC,const char[],PetscInt,const PetscInt*);
extern PetscErrorCode  PCFieldSplitSetType(PC,PCCompositeType);
extern PetscErrorCode  PCFieldSplitSetBlockSize(PC,PetscInt);
extern PetscErrorCode  PCFieldSplitSetIS(PC,const char[],IS);
extern PetscErrorCode  PCFieldSplitGetIS(PC,const char[],IS*);

/*E
    PCFieldSplitSchurPreType - Determines how to precondition Schur complement

    Level: intermediate

.seealso: PCFieldSplitSchurPrecondition()
E*/
typedef enum {PC_FIELDSPLIT_SCHUR_PRE_SELF,PC_FIELDSPLIT_SCHUR_PRE_DIAG,PC_FIELDSPLIT_SCHUR_PRE_USER} PCFieldSplitSchurPreType;
extern const char *const PCFieldSplitSchurPreTypes[];

extern PetscErrorCode  PCFieldSplitSchurPrecondition(PC,PCFieldSplitSchurPreType,Mat);
extern PetscErrorCode  PCFieldSplitGetSchurBlocks(PC,Mat*,Mat*,Mat*,Mat*);

extern PetscErrorCode  PCGalerkinSetRestriction(PC,Mat);
extern PetscErrorCode  PCGalerkinSetInterpolation(PC,Mat);

extern PetscErrorCode  PCSetCoordinates(PC,PetscInt,PetscReal*);
extern PetscErrorCode  PCSASetVectors(PC,PetscInt,PetscReal *);

extern PetscErrorCode  PCPythonSetType(PC,const char[]);

extern PetscErrorCode  PCSetDM(PC,DM);
extern PetscErrorCode  PCGetDM(PC,DM*);

extern PetscErrorCode  PCBiCGStabCUSPSetTolerance(PC,PetscReal);
extern PetscErrorCode  PCBiCGStabCUSPSetIterations(PC,PetscInt);
extern PetscErrorCode  PCBiCGStabCUSPSetUseVerboseMonitor(PC,PetscBool);

extern PetscErrorCode  PCAINVCUSPSetDropTolerance(PC,PetscReal);
extern PetscErrorCode  PCAINVCUSPUseScaling(PC,PetscBool);
extern PetscErrorCode  PCAINVCUSPSetNonzeros(PC,PetscInt);
extern PetscErrorCode  PCAINVCUSPSetLinParameter(PC,PetscInt);
PETSC_EXTERN_CXX_END

#endif /* __PETSCPC_H */
