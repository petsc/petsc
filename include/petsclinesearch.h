#ifndef __PETSCLINESEARCH_H
#define __PETSCLINESEARCH_H

#include <petscsnes.h>

PETSC_EXTERN_CXX_BEGIN

typedef struct _p_LineSearch* LineSearch;

/*
 User interface for Line Searches
*/

#define LineSearchType char*
#define LINESEARCHCUBIC              "cubic"
#define LINESEARCHQUADRATIC          "quadratic"
#define LINESEARCHBASIC              "basic"
#define LINESEARCHL2                 "l2"
#define LINESEARCHCP                 "cp"
#define LINESEARCHSHELL              "shell"

extern PetscClassId  LineSearch_CLASSID;
extern PetscBool     LineSearchRegisterAllCalled;
extern PetscFList    LineSearchList;
extern PetscLogEvent LineSearch_Apply;

typedef PetscErrorCode (*LineSearchPreCheckFunc)(LineSearch,Vec,Vec,PetscBool *);
typedef PetscErrorCode (*LineSearchPostCheckFunc)(LineSearch,Vec,Vec,Vec,PetscBool *,PetscBool *);
typedef PetscErrorCode (*LineSearchApplyFunc)(LineSearch);
typedef PetscErrorCode (*LineSearchUserFunc)(LineSearch,Vec,Vec,Vec,PetscReal,PetscReal,Vec,Vec,PetscReal*,PetscReal*,PetscBool*);

extern PetscErrorCode LineSearchCreate(MPI_Comm, LineSearch*);
extern PetscErrorCode LineSearchReset(LineSearch);
extern PetscErrorCode LineSearchView(LineSearch);
extern PetscErrorCode LineSearchDestroy(LineSearch *);
extern PetscErrorCode LineSearchSetType(LineSearch, const LineSearchType);
extern PetscErrorCode LineSearchSetFromOptions(LineSearch);
extern PetscErrorCode LineSearchSetUp(LineSearch);
extern PetscErrorCode LineSearchApply(LineSearch, Vec, Vec, PetscReal *, Vec);
extern PetscErrorCode LineSearchPreCheck(LineSearch, PetscBool *);
extern PetscErrorCode LineSearchPostCheck(LineSearch, PetscBool *, PetscBool *);
extern PetscErrorCode LineSearchGetWork(LineSearch, PetscInt);
extern PetscErrorCode LineSearchGetNorms(LineSearch, PetscReal *, PetscReal *, PetscReal *);

/*
extern PetscErrorCode  LineSearchSetFunction(LineSearch,Vec,SNESFunction,void*);
extern PetscErrorCode  LineSearchGetFunction(LineSearch,Vec*,SNESFunction*,void**);
extern PetscErrorCode  LineSearchComputeFunction(LineSearch,Vec,Vec);
extern PetscErrorCode  LineSearchSetJacobian(LineSearch,Mat,Mat,SNESJacobian,void*);
extern PetscErrorCode  LineSearchGetJacobian(LineSearch,Mat*,Mat*,SNESJacobian*,void**);
 */

/* TEMPORARY pointers to the associated SNES in order to be able to get the function evaluation out */
extern PetscErrorCode  LineSearchSetSNES(LineSearch,SNES);
extern PetscErrorCode  LineSearchGetSNES(LineSearch,SNES*);

/* unlike before; set and get the parameters */
extern PetscErrorCode  LineSearchGetLambda(LineSearch,PetscScalar*);
extern PetscErrorCode  LineSearchGetDamping(LineSearch,PetscScalar*);
extern PetscErrorCode  LineSearchGetMaxStep(LineSearch,PetscScalar*);
extern PetscErrorCode  LineSearchGetStepTolerance(LineSearch,PetscScalar*);

extern PetscErrorCode  LineSearchSetLambda(LineSearch,PetscScalar*);
extern PetscErrorCode  LineSearchSetDamping(LineSearch,PetscScalar*);
extern PetscErrorCode  LineSearchSetMaxStep(LineSearch,PetscScalar*);
extern PetscErrorCode  LineSearchSetStepTolerance(LineSearch,PetscScalar*);

extern PetscErrorCode  LineSearchGetSuccess(LineSearch, PetscBool*);

extern PetscErrorCode  LineSearchSetMonitor(LineSearch, PetscBool);

extern PetscErrorCode  LineSearchAppendOptionsPrefix(LineSearch, const char prefix[]);
extern PetscErrorCode  LineSearchGetOptionsPrefix(LineSearch, const char *prefix[]);

/*register line search types */
extern PetscErrorCode LineSearchRegister(const char[],const char[],const char[],PetscErrorCode(*)(LineSearch));
extern PetscErrorCode LineSearchRegisterAll(const char path[]);
extern PetscErrorCode LineSearchRegisterDestroy(void);

#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define LineSearchRegisterDynamic(a,b,c,d) LineSearchRegister(a,b,c,0)
#else
#define LineSearchRegisterDynamic(a,b,c,d) LineSearchRegister(a,b,c,d)
#endif

PETSC_EXTERN_CXX_END
#endif
