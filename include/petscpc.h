
/*
      Preconditioner module.
*/
#if !defined(__PC_PACKAGE)
#define __PC_PACKAGE
#include "petsc.h"
#include "mat.h"

typedef enum { PCNONE, PCJACOBI, PCSOR, PCDIRECT } PCMETHOD;

typedef struct _PC* PC;

extern int    PCCreate(PC*);
extern int    PCSetMethod(PC,PCMETHOD);
extern int    PCApply(void*,Vec,Vec);
extern int    PCSetUp(PC);
extern int    PCApplyRichardson(void *,Vec,Vec,Vec,int);
extern int    PCApplyRichardsonExists(PC);
extern int    PCRegisterAll();
extern int    PCRegister(PCMETHOD,char *,int (*)(PC));
extern int    PCDestroy(PC);
extern int    PCSetFromOptions(PC);
extern int    PCGetMethodFromOptions(int,char *,PCMETHOD *);
extern int    PCPrintMethods(char *);
extern int    PCGetMethodFromContext(PC,PCMETHOD*);
extern int    PCGetMethodName(PCMETHOD,char **);
extern int    PCSetMatrix(PC,Mat);
extern int    PCGetMatrix(PC,Mat*);
extern int    PCSetVector(PC,Vec);
extern int    PCPrintHelp(PC);

extern int PCSORSetSymmetric(PC, int);
extern int PCSORSetOmega(PC, double);

#endif
