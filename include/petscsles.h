
#if !defined(_SLES_H)
#define _SLES_H
#include "pc.h"
#include "ksp.h"

typedef struct _SLES* SLES;

int SLESCreate(SLES*);
int SLESGetPC(SLES,PC*);
int SLESGetKSP(SLES,KSP*);
int SLESSetMat(SLES,Mat);
int SLESSetVec(SLES,Vec);

#define SLES_DIRECT    1
#define SLES_ITERATIVE 2

int SLESSetSolverType(SLES,int);

#endif
