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

extern PetscClassId PF_CLASSID;

extern PetscErrorCode  PFCreate(MPI_Comm,PetscInt,PetscInt,PF*);
extern PetscErrorCode  PFSetType(PF,const PFType,void*);
extern PetscErrorCode  PFSet(PF,PetscErrorCode(*)(void*,PetscInt,const PetscScalar*,PetscScalar*),PetscErrorCode(*)(void*,Vec,Vec),PetscErrorCode(*)(void*,PetscViewer),PetscErrorCode(*)(void*),void*);
extern PetscErrorCode  PFApply(PF,PetscInt,const PetscScalar*,PetscScalar*);
extern PetscErrorCode  PFApplyVec(PF,Vec,Vec);

extern PetscErrorCode  PFRegisterDestroy(void);
extern PetscErrorCode  PFRegisterAll(const char[]);
extern PetscErrorCode  PFInitializePackage(const char[]);
extern PetscBool  PFRegisterAllCalled;

extern PetscErrorCode  PFRegister(const char[],const char[],const char[],PetscErrorCode (*)(PF,void*));
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define PFRegisterDynamic(a,b,c,d) PFRegister(a,b,c,0)
#else
#define PFRegisterDynamic(a,b,c,d) PFRegister(a,b,c,d)
#endif

extern PetscErrorCode  PFDestroy(PF*);
extern PetscErrorCode  PFSetFromOptions(PF);
extern PetscErrorCode  PFGetType(PF,const PFType*);

extern PetscErrorCode  PFView(PF,PetscViewer);

#define PFSetOptionsPrefix(a,s) PetscObjectSetOptionsPrefix((PetscObject)(a),s)

PETSC_EXTERN_CXX_END
#endif
