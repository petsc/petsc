/* $Id: petscpc.h,v 1.122 2001/08/21 21:03:12 bsmith Exp $ */

/*
      Preconditioner module. 
*/
#if !defined(__PETSCPC_H)
#define __PETSCPC_H
#include "petscmat.h"
PETSC_EXTERN_CXX_BEGIN

EXTERN int PCInitializePackage(const char[]);

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

EXTERN int PCCreate(MPI_Comm,PC*);
EXTERN int PCSetType(PC,const PCType);
EXTERN int PCSetUp(PC);
EXTERN int PCSetUpOnBlocks(PC);
EXTERN int PCApply(PC,Vec,Vec,PCSide);
EXTERN int PCApplySymmetricLeft(PC,Vec,Vec);
EXTERN int PCApplySymmetricRight(PC,Vec,Vec);
EXTERN int PCApplyBAorAB(PC,PCSide,Vec,Vec,Vec);
EXTERN int PCApplyTranspose(PC,Vec,Vec);
EXTERN int PCHasApplyTranspose(PC,PetscTruth*);
EXTERN int PCApplyBAorABTranspose(PC,PCSide,Vec,Vec,Vec);
EXTERN int PCApplyRichardson(PC,Vec,Vec,Vec,PetscReal,PetscReal,PetscReal,int);
EXTERN int PCApplyRichardsonExists(PC,PetscTruth*);

EXTERN int        PCRegisterDestroy(void);
EXTERN int        PCRegisterAll(const char[]);
extern PetscTruth PCRegisterAllCalled;

EXTERN int PCRegister(const char[],const char[],const char[],int(*)(PC));

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

EXTERN int PCDestroy(PC);
EXTERN int PCSetFromOptions(PC);
EXTERN int PCESISetFromOptions(PC);
EXTERN int PCGetType(PC,PCType*);

EXTERN int PCGetFactoredMatrix(PC,Mat*);
EXTERN int PCSetModifySubMatrices(PC,int(*)(PC,int,const IS[],const IS[],Mat[],void*),void*);
EXTERN int PCModifySubMatrices(PC,int,const IS[],const IS[],Mat[],void*);

EXTERN int PCSetOperators(PC,Mat,Mat,MatStructure);
EXTERN int PCGetOperators(PC,Mat*,Mat*,MatStructure*);

EXTERN int PCView(PC,PetscViewer);

EXTERN int PCSetOptionsPrefix(PC,const char[]);
EXTERN int PCAppendOptionsPrefix(PC,const char[]);
EXTERN int PCGetOptionsPrefix(PC,char*[]);

EXTERN int PCNullSpaceAttach(PC,MatNullSpace);

EXTERN int PCComputeExplicitOperator(PC,Mat*);

/*
      These are used to provide extra scaling of preconditioned 
   operator for time-stepping schemes like in PVODE 
*/
EXTERN int PCDiagonalScale(PC,PetscTruth*);
EXTERN int PCDiagonalScaleLeft(PC,Vec,Vec);
EXTERN int PCDiagonalScaleRight(PC,Vec,Vec);
EXTERN int PCDiagonalScaleSet(PC,Vec);

/* ------------- options specific to particular preconditioners --------- */

EXTERN int PCJacobiSetUseRowMax(PC);
EXTERN int PCSORSetSymmetric(PC,MatSORType);
EXTERN int PCSORSetOmega(PC,PetscReal);
EXTERN int PCSORSetIterations(PC,int,int);

EXTERN int PCEisenstatSetOmega(PC,PetscReal);
EXTERN int PCEisenstatNoDiagonalScaling(PC);

#define USE_PRECONDITIONER_MATRIX 0
#define USE_TRUE_MATRIX           1
EXTERN int PCBJacobiSetUseTrueLocal(PC);
EXTERN int PCBJacobiSetTotalBlocks(PC,int,const int[]);
EXTERN int PCBJacobiSetLocalBlocks(PC,int,const int[]);

EXTERN int PCKSPSetUseTrue(PC);

EXTERN int PCShellSetApply(PC,int (*)(void*,Vec,Vec),void*); 
EXTERN int PCShellSetApplyTranspose(PC,int (*)(void*,Vec,Vec));
EXTERN int PCShellSetSetUp(PC,int (*)(void*));
EXTERN int PCShellSetApplyRichardson(PC,int (*)(void*,Vec,Vec,Vec,PetscReal,PetscReal,PetscReal,int),void*);
EXTERN int PCShellSetView(PC,int (*)(void*,PetscViewer));
EXTERN int PCShellSetName(PC,const char[]);
EXTERN int PCShellGetName(PC,char*[]);

EXTERN int PCLUSetMatOrdering(PC,MatOrderingType);
EXTERN int PCLUSetReuseOrdering(PC,PetscTruth);
EXTERN int PCLUSetReuseFill(PC,PetscTruth);
EXTERN int PCLUSetUseInPlace(PC);
EXTERN int PCLUSetFill(PC,PetscReal);
EXTERN int PCLUSetDamping(PC,PetscReal);
EXTERN int PCLUSetShift(PC,PetscTruth);
EXTERN int PCLUSetPivoting(PC,PetscReal);
EXTERN int PCLUSetPivotInBlocks(PC,PetscTruth);
EXTERN int PCLUSetZeroPivot(PC,PetscReal);

EXTERN int PCCholeskySetMatOrdering(PC,MatOrderingType);
EXTERN int PCCholeskySetReuseOrdering(PC,PetscTruth);
EXTERN int PCCholeskySetReuseFill(PC,PetscTruth);
EXTERN int PCCholeskySetUseInPlace(PC);
EXTERN int PCCholeskySetFill(PC,PetscReal);
EXTERN int PCCholeskySetDamping(PC,PetscReal);
EXTERN int PCCholeskySetShift(PC,PetscTruth);
EXTERN int PCCholeskySetPivotInBlocks(PC,PetscTruth);

EXTERN int PCILUSetMatOrdering(PC,MatOrderingType);
EXTERN int PCILUSetUseInPlace(PC);
EXTERN int PCILUSetFill(PC,PetscReal);
EXTERN int PCILUSetLevels(PC,int);
EXTERN int PCILUSetReuseOrdering(PC,PetscTruth);
EXTERN int PCILUSetUseDropTolerance(PC,PetscReal,PetscReal,int);
EXTERN int PCILUDTSetReuseFill(PC,PetscTruth);
EXTERN int PCILUSetAllowDiagonalFill(PC);
EXTERN int PCILUSetDamping(PC,PetscReal);
EXTERN int PCILUSetShift(PC,PetscTruth);
EXTERN int PCILUSetPivotInBlocks(PC,PetscTruth);
EXTERN int PCILUSetZeroPivot(PC,PetscReal);

EXTERN int PCICCSetMatOrdering(PC,MatOrderingType);
EXTERN int PCICCSetFill(PC,PetscReal);
EXTERN int PCICCSetLevels(PC,int);
EXTERN int PCICCSetDamping(PC,PetscReal);
EXTERN int PCICCSetShift(PC,PetscTruth);
EXTERN int PCICCSetPivotInBlocks(PC,PetscTruth);
EXTERN int PCICCSetZeroPivot(PC,PetscReal);

EXTERN int PCASMSetLocalSubdomains(PC,int,IS[]);
EXTERN int PCASMSetTotalSubdomains(PC,int,IS[]);
EXTERN int PCASMSetOverlap(PC,int);
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
EXTERN int PCASMSetType(PC,PCASMType);
EXTERN int PCASMCreateSubdomains2D(int,int,int,int,int,int,int *,IS **);
EXTERN int PCASMSetUseInPlace(PC);
EXTERN int PCASMGetLocalSubdomains(PC,int*,IS*[]);
EXTERN int PCASMGetLocalSubmatrices(PC,int*,Mat*[]);

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
EXTERN int PCCompositeSetUseTrue(PC);
EXTERN int PCCompositeSetType(PC,PCCompositeType);
EXTERN int PCCompositeAddPC(PC,PCType);
EXTERN int PCCompositeGetPC(PC pc,int n,PC *);
EXTERN int PCCompositeSpecialSetAlpha(PC,PetscScalar);

EXTERN int PCRedundantSetScatter(PC,VecScatter,VecScatter);
EXTERN int PCRedundantGetOperators(PC,Mat*,Mat*);
EXTERN int PCRedundantGetPC(PC,PC*);
EXTERN int MatGetOrderingList(PetscFList *list);

EXTERN int PCSPAISetEpsilon(PC,double);
EXTERN int PCSPAISetNBSteps(PC,int);
EXTERN int PCSPAISetMax(PC,int);
EXTERN int PCSPAISetMaxNew(PC,int);
EXTERN int PCSPAISetBlockSize(PC,int);
EXTERN int PCSPAISetCacheSize(PC,int);
EXTERN int PCSPAISetVerbose(PC,int);
EXTERN int PCSPAISetSp(PC,int);

EXTERN int PCHYPRESetType(PC,const char[]);
EXTERN int PCBJacobiGetLocalBlocks(PC,int*,const int*[]);
EXTERN int PCBJacobiGetTotalBlocks(PC,int*,const int*[]);

PETSC_EXTERN_CXX_END
#endif /* __PETSCPC_H */
