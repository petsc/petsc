
/*
      Preconditioner module.
*/
#if !defined(__PC_PACKAGE)
#define __PC_PACKAGE
#include "petsc.h"
#include "mat.h"

typedef enum { PCNONE, PCJACOBI, PCSOR } PCMETHOD;

typedef struct _PC* PC;

int    PCCreate(PC*);
int    PCSetMethod(PC,PCMETHOD);
int    PCApply(void*,Vec,Vec);
int    PCSetUp(PC);
int    PCApplyRichardson(void *,Vec,Vec,Vec,int);
int    PCApplyRichardsonExists(PC);
int    PCRegisterAll();
int    PCRegister(PCMETHOD,char *,int (*)(PC));
int    PCDestroy(PC);

#endif
