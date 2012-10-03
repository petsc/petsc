/*
      mathematical function module.
*/
#if !defined(__PETSCPF_H)
#define __PETSCPF_H
#include <petscvec.h>

/*
    PFList contains the list of preconditioners currently registered
   These are added with the PFRegisterDynamic() macro
*/
PETSC_EXTERN PetscFList PFList;

/*J
    PFType - Type of PETSc mathematical function, a string name

   Level: beginner

.seealso: PFSetType(), PF
J*/
typedef const char* PFType;
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

PETSC_EXTERN PetscClassId PF_CLASSID;

PETSC_EXTERN PetscErrorCode PFCreate(MPI_Comm,PetscInt,PetscInt,PF*);
PETSC_EXTERN PetscErrorCode PFSetType(PF,PFType,void*);
PETSC_EXTERN PetscErrorCode PFSet(PF,PetscErrorCode(*)(void*,PetscInt,const PetscScalar*,PetscScalar*),PetscErrorCode(*)(void*,Vec,Vec),PetscErrorCode(*)(void*,PetscViewer),PetscErrorCode(*)(void*),void*);
PETSC_EXTERN PetscErrorCode PFApply(PF,PetscInt,const PetscScalar*,PetscScalar*);
PETSC_EXTERN PetscErrorCode PFApplyVec(PF,Vec,Vec);

PETSC_EXTERN PetscErrorCode PFRegisterDestroy(void);
PETSC_EXTERN PetscErrorCode PFRegisterAll(const char[]);
PETSC_EXTERN PetscErrorCode PFInitializePackage(const char[]);
PETSC_EXTERN PetscBool PFRegisterAllCalled;

PETSC_EXTERN PetscErrorCode PFRegister(const char[],const char[],const char[],PetscErrorCode (*)(PF,void*));
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define PFRegisterDynamic(a,b,c,d) PFRegister(a,b,c,0)
#else
#define PFRegisterDynamic(a,b,c,d) PFRegister(a,b,c,d)
#endif

PETSC_EXTERN PetscErrorCode PFDestroy(PF*);
PETSC_EXTERN PetscErrorCode PFSetFromOptions(PF);
PETSC_EXTERN PetscErrorCode PFGetType(PF,PFType*);

PETSC_EXTERN PetscErrorCode PFView(PF,PetscViewer);

#define PFSetOptionsPrefix(a,s) PetscObjectSetOptionsPrefix((PetscObject)(a),s)

#endif
