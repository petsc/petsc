
#if !defined(_SLES_H)
#define _SLES_H
#include "pc.h"
#include "ksp.h"

typedef struct _SLES* SLES;

extern int SLESCreate(SLES*);
extern int SLESGetPC(SLES,PC*);
extern int SLESGetKSP(SLES,KSP*);
extern int SLESSetMat(SLES,Mat);
extern int SLESSolve(SLES,Vec,Vec);
extern int SLESSetFromOptions(SLES);
extern int SLESDestroy(SLES);
extern int SLESPrintHelp(SLES);

#endif
