/* $Id: pc.h,v 1.60 1996/07/08 22:24:30 bsmith Exp bsmith $ */

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
extern int    PCApplySymmetricLeft(PC,Vec,Vec);
extern int    PCApplySymmetricRight(PC,Vec,Vec);
extern int    PCApplyBAorAB(PC,PCSide,Vec,Vec,Vec);
extern int    PCApplyTrans(PC,Vec,Vec);
extern int    PCApplyBAorABTrans(PC,PCSide,Vec,Vec,Vec);
extern int    PCApplyRichardson(PC,Vec,Vec,Vec,int);
extern int    PCApplyRichardsonExists(PC,PetscTruth*);
extern int    PCRegisterAll();
extern int    PCRegisterDestroy();
extern int    PCRegister(PCType,char *,int (*)(PC));
extern int    PCDestroy(PC);
extern int    PCSetFromOptions(PC);
extern int    PCGetType(PC,PCType*,char**);
extern int    PCGetFactoredMatrix(PC,Mat*);

typedef enum {SAME_NONZERO_PATTERN,DIFFERENT_NONZERO_PATTERN,SAME_PRECONDITIONER} 
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

typedef enum {PCBGS_FORWARD_SWEEP=1,PCBGS_SYMMETRIC_SWEEP=2} PCBGSType;
extern int PCBGSSetSymmetric(PC, PCBGSType);

#define USE_PRECONDITIONER_MATRIX 0
#define USE_TRUE_MATRIX           1
extern int PCBJacobiSetUseTrueLocal(PC);
extern int PCBJacobiSetTotalBlocks(PC, int, int*);
extern int PCBJacobiSetLocalBlocks(PC, int, int*);

extern int PCBGSSetUseTrueLocal(PC);
extern int PCBGSSetTotalBlocks(PC, int, int*);
extern int PCBGSSetLocalBlocks(PC, int, int*);
extern int PCBGSSetSymmetric(PC, PCBGSType);

extern int PCShellSetApply(PC, int (*)(void*,Vec,Vec), void*);
extern int PCShellSetApplyRichardson(PC,int (*)(void*,Vec,Vec,Vec,int),void*);
extern int PCShellSetName(PC,char*);
extern int PCShellGetName(PC,char**);

extern int PCLUSetUseInPlace(PC);
extern int PCILUSetUseInPlace(PC);
extern int PCILUSetLevels(PC,int);
extern int PCILUSetReuseReordering(PC,PetscTruth);
extern int PCILUSetUseDropTolerance(PC,double,int);
extern int PCILUSetReuseFill(PC,PetscTruth);

extern int PCEisenstatUseDiagonalScaling(PC);

extern int PCASMSetLocalSubdomains(PC, int, IS *);
extern int PCASMSetTotalSubdomains(PC, int, IS *);
extern int PCASMSetOverlap(PC, int);

extern int PCASMCreateSubdomains2D(int,int,int,int,int,int,int *,IS **);
#endif

