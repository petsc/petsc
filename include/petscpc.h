/* $Id: petscpc.h,v 1.111 2000/09/25 18:50:22 curfman Exp bsmith $ */

/*
      Preconditioner module. 
*/
#if !defined(__PETSCPC_H)
#define __PETSCPC_H
#include "petscmat.h"

/*
    PCList contains the list of preconditioners currently registered
   These are added with the PCRegisterDynamic() macro
*/
extern FList PCList;
typedef char *PCType;

/*
    Standard PETSc preconditioners
*/
#define PCNONE      "none"
#define PCJACOBI    "jacobi"
#define PCSOR       "sor"
#define PCLU        "lu"
#define PCSHELL     "shell"
#define PCBJACOBI   "bjacobi"
#define PCMG        "mg"
#define PCEISENSTAT "eisenstat"
#define PCILU       "ilu"
#define PCICC       "icc"
#define PCASM       "asm"
#define PCSLES      "sles"
#define PCCOMPOSITE "composite"
#define PCREDUNDANT "redundant"
#define PCSPAI      "spai"
#define PCMILU      "milu"
#define PCNN        "nn"
#define PCCHOLESKY  "cholesky"
#define PCRAMG      "ramg"

typedef struct _p_PC* PC;
#define PC_COOKIE     PETSC_COOKIE+9


typedef enum { PC_LEFT,PC_RIGHT,PC_SYMMETRIC } PCSide;

EXTERN int PCCreate(MPI_Comm,PC*);
EXTERN int PCSetType(PC,PCType);
EXTERN int PCSetUp(PC);
EXTERN int PCSetUpOnBlocks(PC);
EXTERN int PCApply(PC,Vec,Vec);
EXTERN int PCApplySymmetricLeft(PC,Vec,Vec);
EXTERN int PCApplySymmetricRight(PC,Vec,Vec);
EXTERN int PCApplyBAorAB(PC,PCSide,Vec,Vec,Vec);
EXTERN int PCApplyTranspose(PC,Vec,Vec);
EXTERN int PCApplyBAorABTranspose(PC,PCSide,Vec,Vec,Vec);
EXTERN int PCApplyRichardson(PC,Vec,Vec,Vec,int);
EXTERN int PCApplyRichardsonExists(PC,PetscTruth*);

EXTERN int        PCRegisterDestroy(void);
EXTERN int        PCRegisterAll(char*);
extern PetscTruth PCRegisterAllCalled;

EXTERN int PCRegister(char*,char*,char*,int(*)(PC));
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define PCRegisterDynamic(a,b,c,d) PCRegister(a,b,c,0)
#else
#define PCRegisterDynamic(a,b,c,d) PCRegister(a,b,c,d)
#endif

EXTERN int PCDestroy(PC);
EXTERN int PCSetFromOptions(PC);
EXTERN int PCSetTypeFromOptions(PC);
EXTERN int PCGetType(PC,PCType*);

EXTERN int PCGetFactoredMatrix(PC,Mat*);
EXTERN int PCSetModifySubMatrices(PC,int(*)(PC,int,IS*,IS*,Mat*,void*),void*);
EXTERN int PCModifySubMatrices(PC,int,IS*,IS*,Mat*,void*);

EXTERN int PCSetOperators(PC,Mat,Mat,MatStructure);
EXTERN int PCGetOperators(PC,Mat*,Mat*,MatStructure*);

EXTERN int PCSetVector(PC,Vec);
EXTERN int PCGetVector(PC,Vec*);
EXTERN int PCView(PC,Viewer);

EXTERN int PCSetOptionsPrefix(PC,char*);
EXTERN int PCAppendOptionsPrefix(PC,char*);
EXTERN int PCGetOptionsPrefix(PC,char**);

EXTERN int PCNullSpaceAttach(PC,MatNullSpace);

EXTERN int PCComputeExplicitOperator(PC,Mat*);

/* ------------- options specific to particular preconditioners --------- */

EXTERN int PCJacobiSetUseRowMax(PC);
EXTERN int PCSORSetSymmetric(PC,MatSORType);
EXTERN int PCSORSetOmega(PC,double);
EXTERN int PCSORSetIterations(PC,int);

EXTERN int PCEisenstatSetOmega(PC,double);
EXTERN int PCEisenstatNoDiagonalScaling(PC);

#define USE_PRECONDITIONER_MATRIX 0
#define USE_TRUE_MATRIX           1
EXTERN int PCBJacobiSetUseTrueLocal(PC);
EXTERN int PCBJacobiSetTotalBlocks(PC,int,int*);
EXTERN int PCBJacobiSetLocalBlocks(PC,int,int*);

EXTERN int PCSLESSetUseTrue(PC);
EXTERN int PCCompositeSetUseTrue(PC);

EXTERN int PCShellSetApply(PC,int (*)(void*,Vec,Vec),void*);
EXTERN int PCShellSetSetUp(PC,int (*)(void*));
EXTERN int PCShellSetApplyRichardson(PC,int (*)(void*,Vec,Vec,Vec,int),void*);
EXTERN int PCShellSetName(PC,char*);
EXTERN int PCShellGetName(PC,char**);

EXTERN int PCLUSetMatOrdering(PC,MatOrderingType);
EXTERN int PCLUSetReuseOrdering(PC,PetscTruth);
EXTERN int PCLUSetReuseFill(PC,PetscTruth);
EXTERN int PCLUSetUseInPlace(PC);
EXTERN int PCLUSetFill(PC,double);
EXTERN int PCLUSetDamping(PC,double);
EXTERN int PCLUSetColumnPivoting(PC,PetscReal);

EXTERN int PCCholeskySetMatOrdering(PC,MatOrderingType);
EXTERN int PCCholeskySetReuseOrdering(PC,PetscTruth);
EXTERN int PCCholeskySetReuseFill(PC,PetscTruth);
EXTERN int PCCholeskySetUseInPlace(PC);
EXTERN int PCCholeskySetFill(PC,double);
EXTERN int PCCholeskySetDamping(PC,double);

EXTERN int PCILUSetMatOrdering(PC,MatOrderingType);
EXTERN int PCILUSetUseInPlace(PC);
EXTERN int PCILUSetFill(PC,double);
EXTERN int PCILUSetLevels(PC,int);
EXTERN int PCILUSetReuseOrdering(PC,PetscTruth);
EXTERN int PCILUSetUseDropTolerance(PC,PetscReal,PetscReal,int);
EXTERN int PCILUDTSetReuseFill(PC,PetscTruth);
EXTERN int PCILUSetAllowDiagonalFill(PC);
EXTERN int PCILUSetDamping(PC,double);

EXTERN int PCICCSetMatOrdering(PC,MatOrderingType);
EXTERN int PCICCSetFill(PC,double);
EXTERN int PCICCSetLevels(PC,int);

EXTERN int PCASMSetLocalSubdomains(PC,int,IS *);
EXTERN int PCASMSetTotalSubdomains(PC,int,IS *);
EXTERN int PCASMSetOverlap(PC,int);
typedef enum {PC_ASM_BASIC = 3,PC_ASM_RESTRICT = 1,PC_ASM_INTERPOLATE = 2,PC_ASM_NONE = 0} PCASMType;
EXTERN int PCASMSetType(PC,PCASMType);
EXTERN int PCASMCreateSubdomains2D(int,int,int,int,int,int,int *,IS **);
EXTERN int PCASMSetUseInPlace(PC);
EXTERN int PCASMGetLocalSubdomains(PC,int*,IS**);

typedef enum {PC_COMPOSITE_ADDITIVE,PC_COMPOSITE_MULTIPLICATIVE,PC_COMPOSITE_SPECIAL} PCCompositeType;
EXTERN int PCCompositeSetType(PC,PCCompositeType);
EXTERN int PCCompositeAddPC(PC,PCType);
EXTERN int PCCompositeGetPC(PC pc,int n,PC *);

EXTERN int PCRedundantSetScatter(PC,VecScatter,VecScatter);
EXTERN int PCRedundantGetOperators(PC,Mat*,Mat*);
EXTERN int PCRedundantGetPC(PC,PC*);
EXTERN int MatGetOrderingList(FList *list);
#endif




