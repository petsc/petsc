
#if !defined(_SLES_H)
#define _SLES_H
#include "pc.h"
#include "ksp.h"

typedef struct _SLES* SLES;

extern int SLESCreate(SLES*);
extern int SLESGetPC(SLES,PC*);
extern int SLESGetKSP(SLES,KSP*);
extern int SLESSetMat(SLES,Mat);
extern int SLESSetVec(SLES,Vec);
extern int SLESSolve(SLES,Vec,Vec);
extern int SLESSetFromOptions(SLES);

extern int SLESPrintHelp(SLES);

#endif
