
#if !defined(__SLES_PACKAGE)
#define __SLES_PACKAGE
#include "pc.h"
#include "ksp.h"

#define SLES_COOKIE 0x70707070

typedef struct _SLES* SLES;

extern int SLESCreate(SLES*);
extern int SLESGetPC(SLES,PC*);
extern int SLESGetKSP(SLES,KSP*);
extern int SLESSetOperators(SLES,Mat,Mat,int);
extern int SLESSolve(SLES,Vec,Vec,int*);
extern int SLESSetFromOptions(SLES);
extern int SLESDestroy(SLES);
extern int SLESPrintHelp(SLES);
extern int SLESSetOptionsPrefix(SLES,char*);

#endif
