/*
      mathematical function module. 
*/
#if !defined(__PETSCPF_H)
#define __PETSCPF_H
#include "petscvec.h"
PETSC_EXTERN_CXX_BEGIN

/*
    PFList contains the list of preconditioners currently registered
   These are added with the PFRegisterDynamic() macro
*/
extern PetscFList PFList;

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

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT PFCreate(MPI_Comm,PetscInt,PetscInt,PF*);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT PFSetType(PF,const PFType,void*);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT PFSet(PF,PetscErrorCode(*)(void*,PetscInt,PetscScalar*,PetscScalar*),PetscErrorCode(*)(void*,Vec,Vec),PetscErrorCode(*)(void*,PetscViewer),PetscErrorCode(*)(void*),void*);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT PFApply(PF,PetscInt,PetscScalar*,PetscScalar*);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT PFApplyVec(PF,Vec,Vec);

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT PFRegisterDestroy(void);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT PFRegisterAll(const char[]);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT PFInitializePackage(const char[]);
extern PetscTruth PFRegisterAllCalled;

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT PFRegister(const char[],const char[],const char[],PetscErrorCode (*)(PF,void*));
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define PFRegisterDynamic(a,b,c,d) PFRegister(a,b,c,0)
#else
#define PFRegisterDynamic(a,b,c,d) PFRegister(a,b,c,d)
#endif

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT PFDestroy(PF);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT PFSetFromOptions(PF);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT PFGetType(PF,const PFType*);

EXTERN PetscErrorCode PETSCVEC_DLLEXPORT PFView(PF,PetscViewer);

#define PFSetOptionsPrefix(a,s) PetscObjectSetOptionsPrefix((PetscObject)(a),s)

PETSC_EXTERN_CXX_END
#endif
