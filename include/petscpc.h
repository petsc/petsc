/*
      Preconditioner module. 
*/
#if !defined(__PETSCPC_H)
#define __PETSCPC_H
#include "petscmat.h"
PETSC_EXTERN_CXX_BEGIN

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT  PCInitializePackage(const char[]);

/*
    PCList contains the list of preconditioners currently registered
   These are added with the PCRegisterDynamic() macro
*/
extern PetscFList PCList;
#define PCType const char*

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
#define PCKSP             "ksp"
#define PCCOMPOSITE       "composite"
#define PCREDUNDANT       "redundant"
#define PCSPAI            "spai"
#define PCNN              "nn"
#define PCCHOLESKY        "cholesky"
#define PCSAMG            "samg"
#define PCPBJACOBI        "pbjacobi"
#define PCMAT             "mat"
#define PCHYPRE           "hypre"
#define PCFIELDSPLIT      "fieldsplit"
#define PCTFS             "tfs"
#define PCML              "ml"
#define PCPROMETHEUS      "prometheus"

/* Logging support */
extern PetscCookie PETSCKSP_DLLEXPORT PC_COOKIE;
extern PetscEvent  PC_SetUp, PC_SetUpOnBlocks, PC_Apply, PC_ApplyCoarse, PC_ApplyMultiple, PC_ApplySymmetricLeft;
extern PetscEvent  PC_ApplySymmetricRight, PC_ModifySubMatrices;

/*E
    PCSide - If the preconditioner is to be applied to the left, right
     or symmetrically around the operator.

   Level: beginner

.seealso: 
E*/
typedef enum { PC_LEFT,PC_RIGHT,PC_SYMMETRIC } PCSide;
extern const char *PCSides[];

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCCreate(MPI_Comm,PC*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCSetType(PC,PCType);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCSetUp(PC);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCSetUpOnBlocks(PC);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCApply(PC,Vec,Vec);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCApplySymmetricLeft(PC,Vec,Vec);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCApplySymmetricRight(PC,Vec,Vec);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCApplyBAorAB(PC,PCSide,Vec,Vec,Vec);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCApplyTranspose(PC,Vec,Vec);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCHasApplyTranspose(PC,PetscTruth*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCApplyBAorABTranspose(PC,PCSide,Vec,Vec,Vec);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCApplyRichardson(PC,Vec,Vec,Vec,PetscReal,PetscReal,PetscReal,PetscInt);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCApplyRichardsonExists(PC,PetscTruth*);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCRegisterDestroy(void);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCRegisterAll(const char[]);
extern PetscTruth PCRegisterAllCalled;

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCRegister(const char[],const char[],const char[],PetscErrorCode(*)(PC));

/*MC
   PCRegisterDynamic - Adds a method to the preconditioner package.

   Synopsis:
   PetscErrorCode PCRegisterDynamic(char *name_solver,char *path,char *name_create,PetscErrorCode (*routine_create)(PC))

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

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCDestroy(PC);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCSetFromOptions(PC);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCGetType(PC,PCType*);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCGetFactoredMatrix(PC,Mat*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCSetModifySubMatrices(PC,PetscErrorCode(*)(PC,PetscInt,const IS[],const IS[],Mat[],void*),void*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCModifySubMatrices(PC,PetscInt,const IS[],const IS[],Mat[],void*);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCSetOperators(PC,Mat,Mat,MatStructure);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCGetOperators(PC,Mat*,Mat*,MatStructure*);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCView(PC,PetscViewer);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCSetOptionsPrefix(PC,const char[]);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCAppendOptionsPrefix(PC,const char[]);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCGetOptionsPrefix(PC,const char*[]);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCComputeExplicitOperator(PC,Mat*);

/*
      These are used to provide extra scaling of preconditioned 
   operator for time-stepping schemes like in PVODE 
*/
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCDiagonalScale(PC,PetscTruth*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCDiagonalScaleLeft(PC,Vec,Vec);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCDiagonalScaleRight(PC,Vec,Vec);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCDiagonalScaleSet(PC,Vec);

/* ------------- options specific to particular preconditioners --------- */

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCJacobiSetUseRowMax(PC);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCSORSetSymmetric(PC,MatSORType);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCSORSetOmega(PC,PetscReal);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCSORSetIterations(PC,PetscInt,PetscInt);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCEisenstatSetOmega(PC,PetscReal);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCEisenstatNoDiagonalScaling(PC);

#define USE_PRECONDITIONER_MATRIX 0
#define USE_TRUE_MATRIX           1
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCBJacobiSetUseTrueLocal(PC);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCBJacobiSetTotalBlocks(PC,PetscInt,const PetscInt[]);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCBJacobiSetLocalBlocks(PC,PetscInt,const PetscInt[]);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCKSPSetUseTrue(PC);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCShellSetApply(PC,PetscErrorCode (*)(void*,Vec,Vec)); 
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCShellSetApplyTranspose(PC,PetscErrorCode (*)(void*,Vec,Vec));
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCShellSetSetUp(PC,PetscErrorCode (*)(void*));
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCShellSetApplyRichardson(PC,PetscErrorCode (*)(void*,Vec,Vec,Vec,PetscReal,PetscReal,PetscReal,PetscInt));
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCShellSetPreSolve(PC,PetscErrorCode (*)(void*,KSP,Vec,Vec));
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCShellSetPostSolve(PC,PetscErrorCode (*)(void*,KSP,Vec,Vec));
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCShellSetView(PC,PetscErrorCode (*)(void*,PetscViewer));
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCShellSetDestroy(PC,PetscErrorCode (*)(void*));
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCShellGetContext(PC,void**);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCShellSetContext(PC,void*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCShellSetName(PC,const char[]);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCShellGetName(PC,char*[]);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetZeroPivot(PC,PetscReal);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetShiftNonzero(PC,PetscReal); 
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetShiftPd(PC,PetscTruth); 

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCLUSetMatOrdering(PC,MatOrderingType);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCLUSetReuseOrdering(PC,PetscTruth);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCLUSetReuseFill(PC,PetscTruth);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCLUSetUseInPlace(PC);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCLUSetFill(PC,PetscReal);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCLUSetPivoting(PC,PetscReal);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCLUSetPivotInBlocks(PC,PetscTruth);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCLUReorderForNonzeroDiagonal(PC,PetscReal);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCCholeskySetMatOrdering(PC,MatOrderingType);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCCholeskySetReuseOrdering(PC,PetscTruth);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCCholeskySetReuseFill(PC,PetscTruth);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCCholeskySetUseInPlace(PC);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCCholeskySetFill(PC,PetscReal);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCCholeskySetPivotInBlocks(PC,PetscTruth);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCILUSetMatOrdering(PC,MatOrderingType);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCILUSetUseInPlace(PC);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCILUSetFill(PC,PetscReal);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCILUSetLevels(PC,PetscInt);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCILUSetReuseOrdering(PC,PetscTruth);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCILUSetUseDropTolerance(PC,PetscReal,PetscReal,PetscInt);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCILUDTSetReuseFill(PC,PetscTruth);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCILUSetAllowDiagonalFill(PC);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCILUSetPivotInBlocks(PC,PetscTruth);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCILUReorderForNonzeroDiagonal(PC,PetscReal);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCICCSetMatOrdering(PC,MatOrderingType);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCICCSetFill(PC,PetscReal);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCICCSetLevels(PC,PetscInt);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCICCSetPivotInBlocks(PC,PetscTruth);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCASMSetLocalSubdomains(PC,PetscInt,IS[]);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCASMSetTotalSubdomains(PC,PetscInt,IS[]);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCASMSetOverlap(PC,PetscInt);
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

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCASMSetType(PC,PCASMType);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCASMCreateSubdomains2D(PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt *,IS **);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCASMSetUseInPlace(PC);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCASMGetLocalSubdomains(PC,PetscInt*,IS*[]);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCASMGetLocalSubmatrices(PC,PetscInt*,Mat*[]);

/*E
    PCCompositeType - Determines how two or more preconditioner are composed

$  PC_COMPOSITE_ADDITIVE - results from application of all preconditioners are added together
$  PC_COMPOSITE_MULTIPLICATIVE - preconditioners are applied sequentially to the residual freshly
$                                computed after the previous preconditioner application
$  PC_COMPOSITE_SPECIAL - This is very special for a matrix of the form alpha I + R + S
$                         where first preconditioner is built from alpha I + S and second from
$                         alpha I + R

   Level: beginner

.seealso: PCCompositeSetType()
E*/
typedef enum {PC_COMPOSITE_ADDITIVE,PC_COMPOSITE_MULTIPLICATIVE,PC_COMPOSITE_SPECIAL} PCCompositeType;
extern const char *PCCompositeTypes[];

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCCompositeSetUseTrue(PC);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCCompositeSetType(PC,PCCompositeType);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCCompositeAddPC(PC,PCType);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCCompositeGetPC(PC pc,PetscInt n,PC *);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCCompositeSpecialSetAlpha(PC,PetscScalar);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCRedundantSetScatter(PC,VecScatter,VecScatter);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCRedundantGetOperators(PC,Mat*,Mat*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCRedundantGetPC(PC,PC*);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCSPAISetEpsilon(PC,double);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCSPAISetNBSteps(PC,PetscInt);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCSPAISetMax(PC,PetscInt);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCSPAISetMaxNew(PC,PetscInt);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCSPAISetBlockSize(PC,PetscInt);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCSPAISetCacheSize(PC,PetscInt);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCSPAISetVerbose(PC,PetscInt);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCSPAISetSp(PC,PetscInt);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCHYPRESetType(PC,const char[]);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCBJacobiGetLocalBlocks(PC,PetscInt*,const PetscInt*[]);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCBJacobiGetTotalBlocks(PC,PetscInt*,const PetscInt*[]);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCFieldSplitSetFields(PC,PetscInt,PetscInt*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCFieldSplitSetType(PC,PCCompositeType);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCPrometheusSetCoordinates(PC,PetscReal*);

PETSC_EXTERN_CXX_END
#endif /* __PETSCPC_H */
