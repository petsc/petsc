/* $Id: pc.h,v 1.87 1999/02/08 16:58:36 balay Exp bsmith $ */

/*
      Preconditioner module. 
*/
#if !defined(__PC_H)
#define __PC_H
#include "petsc.h"
#include "mat.h"

/*
    PCList contains the list of preconditioners currently registered
   These are added with the PCRegister() macro
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

typedef struct _p_PC* PC;
#define PC_COOKIE     PETSC_COOKIE+9

/*
      Null space code is not yet developed 
*/
typedef struct _p_PCNullSpace* PCNullSpace;
#define PCNULLSPACE_COOKIE    PETSC_COOKIE+17

typedef enum { PC_LEFT, PC_RIGHT, PC_SYMMETRIC } PCSide;

extern int PCCreate(MPI_Comm,PC*);
extern int PCSetType(PC,PCType);
extern int PCSetUp(PC);
extern int PCSetUpOnBlocks(PC);
extern int PCApply(PC,Vec,Vec);
extern int PCApplySymmetricLeft(PC,Vec,Vec);
extern int PCApplySymmetricRight(PC,Vec,Vec);
extern int PCApplyBAorAB(PC,PCSide,Vec,Vec,Vec);
extern int PCApplyTrans(PC,Vec,Vec);
extern int PCApplyBAorABTrans(PC,PCSide,Vec,Vec,Vec);
extern int PCApplyRichardson(PC,Vec,Vec,Vec,int);
extern int PCApplyRichardsonExists(PC,PetscTruth*);

extern int PCRegisterDestroy(void);
extern int PCRegisterAll(char*);
extern int PCRegisterAllCalled;

extern int PCRegister_Private(char*,char*,char*,int(*)(PC));
#if defined(USE_DYNAMIC_LIBRARIES)
#define PCRegister(a,b,c,d) PCRegister_Private(a,b,c,0)
#else
#define PCRegister(a,b,c,d) PCRegister_Private(a,b,c,d)
#endif

extern int PCDestroy(PC);
extern int PCSetFromOptions(PC);
extern int PCGetType(PC,PCType*);

extern int PCGetFactoredMatrix(PC,Mat*);
extern int PCSetModifySubMatrices(PC,int(*)(PC,int,IS*,IS*,Mat*,void*),void*);
extern int PCModifySubMatrices(PC,int,IS*,IS*,Mat*,void*);

extern int PCSetOperators(PC,Mat,Mat,MatStructure);
extern int PCGetOperators(PC,Mat*,Mat*,MatStructure*);

extern int PCSetVector(PC,Vec);
extern int PCGetVector(PC,Vec*);
extern int PCPrintHelp(PC);
extern int PCView(PC,Viewer);

extern int PCSetOptionsPrefix(PC,char*);
extern int PCAppendOptionsPrefix(PC,char*);
extern int PCGetOptionsPrefix(PC,char**);

extern int PCNullSpaceCreate(MPI_Comm,int,int,Vec *,PCNullSpace*);
extern int PCNullSpaceDestroy(PCNullSpace);
extern int PCNullSpaceRemove(PCNullSpace,Vec);

/* ------------- options specific to particular preconditioners --------- */
extern int PCSORSetSymmetric(PC, MatSORType);
extern int PCSORSetOmega(PC, double);
extern int PCEisenstatSetOmega(PC, double);
extern int PCSORSetIterations(PC, int);

#define USE_PRECONDITIONER_MATRIX 0
#define USE_TRUE_MATRIX           1
extern int PCBJacobiSetUseTrueLocal(PC);
extern int PCBJacobiSetTotalBlocks(PC, int, int*);
extern int PCBJacobiSetLocalBlocks(PC, int, int*);

extern int PCSLESSetUseTrue(PC);
extern int PCCompositeSetUseTrue(PC);

extern int PCShellSetApply(PC, int (*)(void*,Vec,Vec), void*);
extern int PCShellSetSetUp(PC, int (*)(void*));
extern int PCShellSetApplyRichardson(PC,int (*)(void*,Vec,Vec,Vec,int),void*);
extern int PCShellSetName(PC,char*);
extern int PCShellGetName(PC,char**);

extern int PCLUSetMatReordering(PC,MatReorderingType);
extern int PCLUSetReuseReordering(PC,PetscTruth);
extern int PCLUSetReuseFill(PC,PetscTruth);

extern int PCILUSetMatReordering(PC,MatReorderingType);
extern int PCLUSetUseInPlace(PC);
extern int PCLUSetFill(PC,double);
extern int PCILUSetUseInPlace(PC);
extern int PCILUSetFill(PC,double);
extern int PCILUSetLevels(PC,int);
extern int PCILUSetReuseReordering(PC,PetscTruth);
extern int PCILUSetUseDropTolerance(PC,double,int);
extern int PCILUSetReuseFill(PC,PetscTruth);
extern int PCILUSetAllowDiagonalFill(PC);

extern int PCEisenstatNoDiagonalScaling(PC);

extern int PCASMSetLocalSubdomains(PC, int, IS *);
extern int PCASMSetTotalSubdomains(PC, int, IS *);
extern int PCASMSetOverlap(PC, int);
typedef enum {PC_ASM_BASIC = 3,PC_ASM_RESTRICT = 1,PC_ASM_INTERPOLATE = 2,PC_ASM_NONE = 0} PCASMType;
extern int PCASMSetType(PC,PCASMType);
extern int PCASMCreateSubdomains2D(int,int,int,int,int,int,int *,IS **);
extern int PCASMSetUseInPlace(PC);

typedef enum {PC_COMPOSITE_ADDITIVE, PC_COMPOSITE_MULTIPLICATIVE} PCCompositeType;
extern int PCCompositeSetType(PC,PCCompositeType);
extern int PCCompositeAddPC(PC,PCType);
extern int PCCompositeGetPC(PC pc,int n,PC *);

#endif




