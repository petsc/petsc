/* $Id: pc.h,v 1.54 1996/02/08 19:37:51 curfman Exp bsmith $ */

/*
      Preconditioner module. Defines the preconditioner routines.
*/
#if !defined(__PC_PACKAGE)
#define __PC_PACKAGE
#include "petsc.h"
#include "mat.h"

typedef enum { PCNONE, PCJACOBI, PCSOR, PCLU, PCSHELL, PCBJACOBI, PCMG,
               PCEISENSTAT, PCILU, PCICC, PCASM, PCBGS } PCType;

typedef struct _PC* PC;
#define PC_COOKIE    PETSC_COOKIE+9

typedef struct _PCNullSpace* PCNullSpace;
#define PCNULLSPACE_COOKIE    PETSC_COOKIE+17

typedef enum { PC_LEFT, PC_RIGHT, PC_SYMMETRIC } PCSide;

extern int    PCCreate(MPI_Comm,PC*);
extern int    PCSetType(PC,PCType);
extern int    PCSetUp(PC);
extern int    PCSetUpOnBlocks(PC);
extern int    PCApply(PC,Vec,Vec);
extern int    PCApplySymmLeft(PC,Vec,Vec);
extern int    PCApplySymmRight(PC,Vec,Vec);
extern int    PCApplyBAorAB(PC,PCSide,Vec,Vec,Vec);
extern int    PCApplyTrans(PC,Vec,Vec);
extern int    PCApplyBAorABTrans(PC,PCSide,Vec,Vec,Vec);
extern int    PCApplyRichardson(PC,Vec,Vec,Vec,int);
extern int    PCApplyRichardsonExists(PC);
extern int    PCRegisterAll();
extern int    PCRegisterDestroy();
extern int    PCRegister(PCType,char *,int (*)(PC));
extern int    PCDestroy(PC);
extern int    PCSetFromOptions(PC);
extern int    PCGetType(PC,PCType*,char**);
extern int    PCGetFactoredMatrix(PC,Mat*);

typedef enum {SAME_NONZERO_PATTERN=0,DIFFERENT_NONZERO_PATTERN=1,SAME_PRECONDITIONER=2} 
              MatStructure;
extern int PCSetOperators(PC,Mat,Mat,MatStructure);
extern int PCGetOperators(PC,Mat*,Mat*,MatStructure*);

extern int PCSetVector(PC,Vec);
extern int PCPrintHelp(PC);
extern int PCView(PC,Viewer);

extern int PCSetOptionsPrefix(PC,char*);
extern int PCAppendOptionsPrefix(PC,char*);
extern int PCGetOptionsPrefix(PC,char**);

extern int PCNullSpaceCreate(MPI_Comm,int,int,Vec *,PCNullSpace*);
extern int PCNullSpaceDestroy(PCNullSpace);
extern int PCNullSpaceRemove(PCNullSpace,Vec);

/* options specific to particular preconditioners */
extern int PCSORSetSymmetric(PC, MatSORType);
extern int PCSORSetOmega(PC, double);
extern int PCEisenstatSetOmega(PC, double);
extern int PCSORSetIterations(PC, int);

typedef enum {BGS_FORWARD_SWEEP=1,BGS_SYMMETRIC_SWEEP=2} PCBGSType;
extern int PCBGSSetSymmetric(PC, PCBGSType);

#define USE_PRECONDITIONER_MATRIX 0
#define USE_TRUE_MATRIX           1

extern int PCBJacobiSetUseTrueLocal(PC);
extern int PCBJacobiSetTotalBlocks(PC, int, int*,int*);
extern int PCBJacobiSetLocalBlocks(PC, int, int*,int*);

extern int PCBGSSetUseTrueLocal(PC);
extern int PCBGSSetTotalBlocks(PC, int, int*,int*);
extern int PCBGSSetLocalBlocks(PC, int, int*,int*);
extern int PCBGSSetSymmetric(PC, PCBGSType);

extern int PCBSIterSetBlockSolve(PC);
extern int PCBSIterSetFromOptions(PC);
extern int PCBSIterSolve(PC,Vec,Vec,int*);

extern int PCShellSetApply(PC, int (*)(void*,Vec,Vec), void*);
extern int PCShellSetApplyRichardson(PC,int (*)(void*,Vec,Vec,Vec,int),void*);
extern int PCShellSetName(PC,char*);
extern int PCShellGetName(PC,char**);

extern int PCLUSetUseInPlace(PC);
extern int PCILUSetUseInPlace(PC);
extern int PCILUSetLevels(PC,int);
extern int PCEisenstatUseDiagonalScaling(PC);

extern int PCASMCreateSubdomains2D(int,int,int,int,int,int,int *,IS **);
extern int PCASMSetLocalSubdomains(PC, int, IS *);
extern int PCASMSetTotalSubdomains(PC, int, IS *);
extern int PCASMSetOverlap(PC, int);
#endif

