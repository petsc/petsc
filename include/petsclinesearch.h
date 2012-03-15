#ifndef __PETSCLINESEARCH_H
#define __PETSCLINESEARCH_H

#include <petscsnes.h>

PETSC_EXTERN_CXX_BEGIN

typedef struct _p_LineSearch* PetscLineSearch;

/*
 User interface for Line Searches
*/

#define PetscLineSearchType char*
#define PETSCLINESEARCHBT                 "bt"
#define PETSCLINESEARCHBASIC              "basic"
#define PETSCLINESEARCHL2                 "l2"
#define PETSCLINESEARCHCP                 "cp"
#define PETSCLINESEARCHSHELL              "shell"

extern PetscClassId  PETSCLINESEARCH_CLASSID;
extern PetscBool     PetscLineSearchRegisterAllCalled;
extern PetscFList    PetscLineSearchList;
extern PetscLogEvent PetscLineSearch_Apply;

typedef PetscErrorCode (*PetscLineSearchPreCheckFunc)(PetscLineSearch,Vec,Vec,PetscBool *);
typedef PetscErrorCode (*PetscLineSearchMidCheckFunc)(PetscLineSearch,Vec,Vec,PetscScalar *);
typedef PetscErrorCode (*PetscLineSearchPostCheckFunc)(PetscLineSearch,Vec,Vec,Vec,PetscBool *,PetscBool *);
typedef PetscErrorCode (*PetscLineSearchApplyFunc)(PetscLineSearch);
typedef PetscErrorCode (*PetscLineSearchUserFunc)(PetscLineSearch, void *);

extern PetscErrorCode PetscLineSearchCreate(MPI_Comm, PetscLineSearch*);
extern PetscErrorCode PetscLineSearchReset(PetscLineSearch);
extern PetscErrorCode PetscLineSearchView(PetscLineSearch);
extern PetscErrorCode PetscLineSearchDestroy(PetscLineSearch *);
extern PetscErrorCode PetscLineSearchSetType(PetscLineSearch, const PetscLineSearchType);
extern PetscErrorCode PetscLineSearchSetFromOptions(PetscLineSearch);
extern PetscErrorCode PetscLineSearchSetUp(PetscLineSearch);
extern PetscErrorCode PetscLineSearchApply(PetscLineSearch, Vec, Vec, PetscReal *, Vec);
extern PetscErrorCode PetscLineSearchPreCheck(PetscLineSearch, PetscBool *);
extern PetscErrorCode PetscLineSearchMidCheck(PetscLineSearch, Vec, Vec, PetscReal *);
extern PetscErrorCode PetscLineSearchPostCheck(PetscLineSearch, PetscBool *, PetscBool *);
extern PetscErrorCode PetscLineSearchGetWork(PetscLineSearch, PetscInt);

/* INELEGANT HACK pointers to the associated SNES in order to be able to get the function evaluation out */
extern PetscErrorCode  PetscLineSearchSetSNES(PetscLineSearch,SNES);
extern PetscErrorCode  PetscLineSearchGetSNES(PetscLineSearch,SNES*);

/* set and get the parameters and vectors */
extern PetscErrorCode  PetscLineSearchGetTolerances(PetscLineSearch,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscInt*);
extern PetscErrorCode  PetscLineSearchSetTolerances(PetscLineSearch,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal,PetscInt);

extern PetscErrorCode  PetscLineSearchGetLambda(PetscLineSearch,PetscReal*);
extern PetscErrorCode  PetscLineSearchSetLambda(PetscLineSearch,PetscReal);

extern PetscErrorCode  PetscLineSearchGetDamping(PetscLineSearch,PetscReal*);
extern PetscErrorCode  PetscLineSearchSetDamping(PetscLineSearch,PetscReal);

extern PetscErrorCode  PetscLineSearchGetSuccess(PetscLineSearch, PetscBool*);
extern PetscErrorCode  PetscLineSearchSetSuccess(PetscLineSearch, PetscBool);

extern PetscErrorCode PetscLineSearchGetVecs(PetscLineSearch,Vec*,Vec*,Vec*,Vec*,Vec*);
extern PetscErrorCode PetscLineSearchSetVecs(PetscLineSearch,Vec,Vec,Vec,Vec,Vec);

extern PetscErrorCode PetscLineSearchGetNorms(PetscLineSearch, PetscReal *, PetscReal *, PetscReal *);
extern PetscErrorCode PetscLineSearchSetNorms(PetscLineSearch, PetscReal, PetscReal, PetscReal);
extern PetscErrorCode PetscLineSearchComputeNorms(PetscLineSearch);

extern PetscErrorCode  PetscLineSearchSetMonitor(PetscLineSearch, PetscBool);
extern PetscErrorCode  PetscLineSearchGetMonitor(PetscLineSearch, PetscViewer*);

extern PetscErrorCode  PetscLineSearchAppendOptionsPrefix(PetscLineSearch, const char prefix[]);
extern PetscErrorCode  PetscLineSearchGetOptionsPrefix(PetscLineSearch, const char *prefix[]);


/* Shell interface functions */
extern PetscErrorCode PetscLineSearchShellSetUserFunc(PetscLineSearch,PetscLineSearchUserFunc,void*);
extern PetscErrorCode PetscLineSearchShellGetUserFunc(PetscLineSearch,PetscLineSearchUserFunc*,void**);

/*register line search types */
extern PetscErrorCode PetscLineSearchRegister(const char[],const char[],const char[],PetscErrorCode(*)(PetscLineSearch));
extern PetscErrorCode PetscLineSearchRegisterAll(const char path[]);
extern PetscErrorCode PetscLineSearchRegisterDestroy(void);

#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define PetscLineSearchRegisterDynamic(a,b,c,d) PetscLineSearchRegister(a,b,c,0)
#else
#define PetscLineSearchRegisterDynamic(a,b,c,d) PetscLineSearchRegister(a,b,c,d)
#endif

PETSC_EXTERN_CXX_END
#endif
