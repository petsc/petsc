/*
      mathematical function module. 
*/
#if !defined(__PETSCPF_H)
#define __PETSCPF_H
#include "petscvec.h"
PETSC_EXTERN_CXX_BEGIN

/*
    PPetscFList contains the list of preconditioners currently registered
   These are added with the PFRegisterDynamic() macro
*/
extern PetscFList PPetscFList;

/*E
    PFType - Type of PETSc mathematical function, a string name

   Level: beginner

.seealso: PFSetType(), PF
E*/
#define PFType char*
#define PFCONSTANT      "constant"
#define PFMAT           "mat"
#define PFSTRING        "string"
#define PFQUICK         "quick"
#define PFIDENTITY      "identity"
#define PFMATLAB        "matlab"

/*S
     PF - Abstract PETSc mathematical function

   Level: beginner

  Concepts: functions

.seealso:  PFCreate(), PFDestroy(), PFSetType(), PFApply(), PFApplyVec(), PFSet(), PFType
S*/
typedef struct _p_PF* PF;

extern PetscCookie PF_COOKIE;

EXTERN PetscErrorCode PFCreate(MPI_Comm,PetscInt,PetscInt,PF*);
EXTERN PetscErrorCode PFSetType(PF,const PFType,void*);
EXTERN PetscErrorCode PFSet(PF,PetscErrorCode(*)(void*,PetscInt,PetscScalar*,PetscScalar*),PetscErrorCode(*)(void*,Vec,Vec),PetscErrorCode(*)(void*,PetscViewer),PetscErrorCode(*)(void*),void*);
EXTERN PetscErrorCode PFApply(PF,PetscInt,PetscScalar*,PetscScalar*);
EXTERN PetscErrorCode PFApplyVec(PF,Vec,Vec);

EXTERN PetscErrorCode        PFRegisterDestroy(void);
EXTERN PetscErrorCode        PFRegisterAll(const char[]);
extern PetscTruth PFRegisterAllCalled;

EXTERN PetscErrorCode PFRegister(const char[],const char[],const char[],PetscErrorCode (*)(PF,void*));
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define PFRegisterDynamic(a,b,c,d) PFRegister(a,b,c,0)
#else
#define PFRegisterDynamic(a,b,c,d) PFRegister(a,b,c,d)
#endif

EXTERN PetscErrorCode PFDestroy(PF);
EXTERN PetscErrorCode PFSetFromOptions(PF);
EXTERN PetscErrorCode PFGetType(PF,PFType*);

EXTERN PetscErrorCode PFView(PF,PetscViewer);

#define PFSetOptionsPrefix(a,s) PetscObjectSetOptionsPrefix((PetscObject)(a),s)

PETSC_EXTERN_CXX_END
#endif
