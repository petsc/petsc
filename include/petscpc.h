/*
      Preconditioner module. 
*/
#if !defined(__PETSCPC_H)
#define __PETSCPC_H
#include "petscmat.h"
PETSC_EXTERN_CXX_BEGIN

EXTERN PetscErrorCode PCInitializePackage(const char[]);

/*
    PCList contains the list of preconditioners currently registered
   These are added with the PCRegisterDynamic() macro
*/
extern PetscFList PCList;
#define PCType char*

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
#define PCNONE       "none"
#define PCJACOBI     "jacobi"
#define PCSOR        "sor"
#define PCLU         "lu"
#define PCSHELL      "shell"
#define PCBJACOBI    "bjacobi"
#define PCMG         "mg"
#define PCEISENSTAT  "eisenstat"
#define PCILU        "ilu"
#define PCICC        "icc"
#define PCASM        "asm"
#define PCKSP        "ksp"
#define PCCOMPOSITE  "composite"
#define PCREDUNDANT  "redundant"
#define PCSPAI       "spai"
#define PCNN         "nn"
#define PCCHOLESKY   "cholesky"
#define PCRAMG       "ramg"
#define PCSAMG       "samg"
#define PCPBJACOBI   "pbjacobi"
#define PCESI        "esi"
#define PCPETSCESI   "petscesi"
#define PCMAT        "mat"
#define PCHYPRE      "hypre"

/* Logging support */
extern int PC_COOKIE;
extern int PC_SetUp, PC_SetUpOnBlocks, PC_Apply, PC_ApplyCoarse, PC_ApplyMultiple, PC_ApplySymmetricLeft;
extern int PC_ApplySymmetricRight, PC_ModifySubMatrices;

/*E
    PCSide - If the preconditioner is to be applied to the left, right
     or symmetrically around the operator.

   Level: beginner

.seealso: 
E*/
typedef enum { PC_LEFT,PC_RIGHT,PC_SYMMETRIC } PCSide;

EXTERN PetscErrorCode PCCreate(MPI_Comm,PC*);
EXTERN PetscErrorCode PCSetType(PC,const PCType);
EXTERN PetscErrorCode PCSetUp(PC);
EXTERN PetscErrorCode PCSetUpOnBlocks(PC);
EXTERN PetscErrorCode PCApply(PC,Vec,Vec);
EXTERN PetscErrorCode PCApplySymmetricLeft(PC,Vec,Vec);
EXTERN PetscErrorCode PCApplySymmetricRight(PC,Vec,Vec);
EXTERN PetscErrorCode PCApplyBAorAB(PC,PCSide,Vec,Vec,Vec);
EXTERN PetscErrorCode PCApplyTranspose(PC,Vec,Vec);
EXTERN PetscErrorCode PCHasApplyTranspose(PC,PetscTruth*);
EXTERN PetscErrorCode PCApplyBAorABTranspose(PC,PCSide,Vec,Vec,Vec);
EXTERN PetscErrorCode PCApplyRichardson(PC,Vec,Vec,Vec,PetscReal,PetscReal,PetscReal,int);
EXTERN PetscErrorCode PCApplyRichardsonExists(PC,PetscTruth*);

EXTERN PetscErrorCode        PCRegisterDestroy(void);
EXTERN PetscErrorCode        PCRegisterAll(const char[]);
extern PetscTruth PCRegisterAllCalled;

EXTERN PetscErrorCode PCRegister(const char[],const char[],const char[],int(*)(PC));

/*MC
   PCRegisterDynamic - Adds a method to the preconditioner package.

   Synopsis:
   int PCRegisterDynamic(char *name_solver,char *path,char *name_create,int (*routine_create)(PC))

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

   Notes: ${PETSC_ARCH}, ${PETSC_DIR}, ${PETSC_LIB_DIR}, ${BOPT}, or ${any environmental variable}
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

EXTERN PetscErrorCode PCDestroy(PC);
EXTERN PetscErrorCode PCSetFromOptions(PC);
EXTERN PetscErrorCode PCESISetFromOptions(PC);
EXTERN PetscErrorCode PCGetType(PC,PCType*);

EXTERN PetscErrorCode PCGetFactoredMatrix(PC,Mat*);
EXTERN PetscErrorCode PCSetModifySubMatrices(PC,int(*)(PC,int,const IS[],const IS[],Mat[],void*),void*);
EXTERN PetscErrorCode PCModifySubMatrices(PC,int,const IS[],const IS[],Mat[],void*);

EXTERN PetscErrorCode PCSetOperators(PC,Mat,Mat,MatStructure);
EXTERN PetscErrorCode PCGetOperators(PC,Mat*,Mat*,MatStructure*);

EXTERN PetscErrorCode PCView(PC,PetscViewer);

EXTERN PetscErrorCode PCSetOptionsPrefix(PC,const char[]);
EXTERN PetscErrorCode PCAppendOptionsPrefix(PC,const char[]);
EXTERN PetscErrorCode PCGetOptionsPrefix(PC,char*[]);

EXTERN PetscErrorCode PCComputeExplicitOperator(PC,Mat*);

/*
      These are used to provide extra scaling of preconditioned 
   operator for time-stepping schemes like in PVODE 
*/
EXTERN PetscErrorCode PCDiagonalScale(PC,PetscTruth*);
EXTERN PetscErrorCode PCDiagonalScaleLeft(PC,Vec,Vec);
EXTERN PetscErrorCode PCDiagonalScaleRight(PC,Vec,Vec);
EXTERN PetscErrorCode PCDiagonalScaleSet(PC,Vec);

/* ------------- options specific to particular preconditioners --------- */

EXTERN PetscErrorCode PCJacobiSetUseRowMax(PC);
EXTERN PetscErrorCode PCSORSetSymmetric(PC,MatSORType);
EXTERN PetscErrorCode PCSORSetOmega(PC,PetscReal);
EXTERN PetscErrorCode PCSORSetIterations(PC,int,int);

EXTERN PetscErrorCode PCEisenstatSetOmega(PC,PetscReal);
EXTERN PetscErrorCode PCEisenstatNoDiagonalScaling(PC);

#define USE_PRECONDITIONER_MATRIX 0
#define USE_TRUE_MATRIX           1
EXTERN PetscErrorCode PCBJacobiSetUseTrueLocal(PC);
EXTERN PetscErrorCode PCBJacobiSetTotalBlocks(PC,int,const int[]);
EXTERN PetscErrorCode PCBJacobiSetLocalBlocks(PC,int,const int[]);

EXTERN PetscErrorCode PCKSPSetUseTrue(PC);

EXTERN PetscErrorCode PCShellSetApply(PC,int (*)(void*,Vec,Vec),void*); 
EXTERN PetscErrorCode PCShellSetApplyTranspose(PC,int (*)(void*,Vec,Vec));
EXTERN PetscErrorCode PCShellSetSetUp(PC,int (*)(void*));
EXTERN PetscErrorCode PCShellSetApplyRichardson(PC,int (*)(void*,Vec,Vec,Vec,PetscReal,PetscReal,PetscReal,int),void*);
EXTERN PetscErrorCode PCShellSetView(PC,int (*)(void*,PetscViewer));
EXTERN PetscErrorCode PCShellSetName(PC,const char[]);
EXTERN PetscErrorCode PCShellGetName(PC,char*[]);

EXTERN PetscErrorCode PCLUSetMatOrdering(PC,MatOrderingType);
EXTERN PetscErrorCode PCLUSetReuseOrdering(PC,PetscTruth);
EXTERN PetscErrorCode PCLUSetReuseFill(PC,PetscTruth);
EXTERN PetscErrorCode PCLUSetUseInPlace(PC);
EXTERN PetscErrorCode PCLUSetFill(PC,PetscReal);
EXTERN PetscErrorCode PCLUSetDamping(PC,PetscReal);
EXTERN PetscErrorCode PCLUSetShift(PC,PetscTruth);
EXTERN PetscErrorCode PCLUSetPivoting(PC,PetscReal);
EXTERN PetscErrorCode PCLUSetPivotInBlocks(PC,PetscTruth);
EXTERN PetscErrorCode PCLUSetZeroPivot(PC,PetscReal);

EXTERN PetscErrorCode PCCholeskySetMatOrdering(PC,MatOrderingType);
EXTERN PetscErrorCode PCCholeskySetReuseOrdering(PC,PetscTruth);
EXTERN PetscErrorCode PCCholeskySetReuseFill(PC,PetscTruth);
EXTERN PetscErrorCode PCCholeskySetUseInPlace(PC);
EXTERN PetscErrorCode PCCholeskySetFill(PC,PetscReal);
EXTERN PetscErrorCode PCCholeskySetDamping(PC,PetscReal);
EXTERN PetscErrorCode PCCholeskySetShift(PC,PetscTruth);
EXTERN PetscErrorCode PCCholeskySetPivotInBlocks(PC,PetscTruth);

EXTERN PetscErrorCode PCILUSetMatOrdering(PC,MatOrderingType);
EXTERN PetscErrorCode PCILUSetUseInPlace(PC);
EXTERN PetscErrorCode PCILUSetFill(PC,PetscReal);
EXTERN PetscErrorCode PCILUSetLevels(PC,int);
EXTERN PetscErrorCode PCILUSetReuseOrdering(PC,PetscTruth);
EXTERN PetscErrorCode PCILUSetUseDropTolerance(PC,PetscReal,PetscReal,int);
EXTERN PetscErrorCode PCILUDTSetReuseFill(PC,PetscTruth);
EXTERN PetscErrorCode PCILUSetAllowDiagonalFill(PC);
EXTERN PetscErrorCode PCILUSetDamping(PC,PetscReal);
EXTERN PetscErrorCode PCILUSetShift(PC,PetscTruth);
EXTERN PetscErrorCode PCILUSetPivotInBlocks(PC,PetscTruth);
EXTERN PetscErrorCode PCILUSetZeroPivot(PC,PetscReal);

EXTERN PetscErrorCode PCICCSetMatOrdering(PC,MatOrderingType);
EXTERN PetscErrorCode PCICCSetFill(PC,PetscReal);
EXTERN PetscErrorCode PCICCSetLevels(PC,int);
EXTERN PetscErrorCode PCICCSetDamping(PC,PetscReal);
EXTERN PetscErrorCode PCICCSetShift(PC,PetscTruth);
EXTERN PetscErrorCode PCICCSetPivotInBlocks(PC,PetscTruth);
EXTERN PetscErrorCode PCICCSetZeroPivot(PC,PetscReal);

EXTERN PetscErrorCode PCASMSetLocalSubdomains(PC,int,IS[]);
EXTERN PetscErrorCode PCASMSetTotalSubdomains(PC,int,IS[]);
EXTERN PetscErrorCode PCASMSetOverlap(PC,int);
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
EXTERN PetscErrorCode PCASMSetType(PC,PCASMType);
EXTERN PetscErrorCode PCASMCreateSubdomains2D(int,int,int,int,int,int,int *,IS **);
EXTERN PetscErrorCode PCASMSetUseInPlace(PC);
EXTERN PetscErrorCode PCASMGetLocalSubdomains(PC,int*,IS*[]);
EXTERN PetscErrorCode PCASMGetLocalSubmatrices(PC,int*,Mat*[]);

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
EXTERN PetscErrorCode PCCompositeSetUseTrue(PC);
EXTERN PetscErrorCode PCCompositeSetType(PC,PCCompositeType);
EXTERN PetscErrorCode PCCompositeAddPC(PC,PCType);
EXTERN PetscErrorCode PCCompositeGetPC(PC pc,int n,PC *);
EXTERN PetscErrorCode PCCompositeSpecialSetAlpha(PC,PetscScalar);

EXTERN PetscErrorCode PCRedundantSetScatter(PC,VecScatter,VecScatter);
EXTERN PetscErrorCode PCRedundantGetOperators(PC,Mat*,Mat*);
EXTERN PetscErrorCode PCRedundantGetPC(PC,PC*);
EXTERN PetscErrorCode MatGetOrderingList(PetscFList *list);

EXTERN PetscErrorCode PCSPAISetEpsilon(PC,double);
EXTERN PetscErrorCode PCSPAISetNBSteps(PC,int);
EXTERN PetscErrorCode PCSPAISetMax(PC,int);
EXTERN PetscErrorCode PCSPAISetMaxNew(PC,int);
EXTERN PetscErrorCode PCSPAISetBlockSize(PC,int);
EXTERN PetscErrorCode PCSPAISetCacheSize(PC,int);
EXTERN PetscErrorCode PCSPAISetVerbose(PC,int);
EXTERN PetscErrorCode PCSPAISetSp(PC,int);

EXTERN PetscErrorCode PCHYPRESetType(PC,const char[]);
EXTERN PetscErrorCode PCBJacobiGetLocalBlocks(PC,int*,const int*[]);
EXTERN PetscErrorCode PCBJacobiGetTotalBlocks(PC,int*,const int*[]);

PETSC_EXTERN_CXX_END
#endif /* __PETSCPC_H */
