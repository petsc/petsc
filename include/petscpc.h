
/*
      Preconditioner module.
*/
#if !defined(__PC_PACKAGE)
#define __PC_PACKAGE
#include "petsc.h"
#include "mat.h"

typedef enum { PCNONE, PCJACOBI, PCSOR, PCDIRECT, PCSHELL, PCBJACOBI, PCMG,
               PCESOR } PCMETHOD;

typedef struct _PC* PC;
#define PC_COOKIE         0x505050

extern int    PCCreate(PC*);
extern int    PCSetMethod(PC,PCMETHOD);
extern int    PCSetUp(PC);
extern int    PCApply(PC,Vec,Vec);
extern int    PCApplyBAorAB(PC,int,Vec,Vec,Vec);
extern int    PCApplyTrans(PC,Vec,Vec);
extern int    PCApplyBAorABTrans(PC,int,Vec,Vec,Vec);
extern int    PCApplyRichardson(PC,Vec,Vec,Vec,int);
extern int    PCApplyRichardsonExists(PC);
extern int    PCRegisterAll();
extern int    PCRegisterDestroy();
extern int    PCRegister(PCMETHOD,char *,int (*)(PC));
extern int    PCDestroy(PC);
extern int    PCSetFromOptions(PC);
extern int    PCGetMethodFromOptions(PC pc,PCMETHOD *);
extern int    PCPrintMethods(char*,char *);
extern int    PCGetMethodFromContext(PC,PCMETHOD*);
extern int    PCGetMethodName(PCMETHOD,char **);
extern int    PCSetMat(PC,Mat);
extern int    PCSetVector(PC,Vec);
extern int    PCPrintHelp(PC);
extern int    PCSetOptionsPrefix(PC,char*);

extern int PCSORSetSymmetric(PC, int);
extern int PCSORSetOmega(PC, double);
extern int PCESORSetOmega(PC, double);
extern int PCSORSetIterations(PC, int);

extern int PCBJacobiSetBlocks(PC, int);

extern int PCShellSetApply(PC, int (*)(void*,Vec,Vec), void*);
extern int PCShellSetApplyRichardson(PC,int (*)(void*,Vec,Vec,Vec,int),void*);

extern Mat PCGetMat(PC);

#endif

