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

extern int PF_COOKIE;

EXTERN int PFCreate(MPI_Comm,int,int,PF*);
EXTERN int PFSetType(PF,const PFType,void*);
EXTERN int PFSet(PF,int(*)(void*,int,PetscScalar*,PetscScalar*),int(*)(void*,Vec,Vec),int(*)(void*,PetscViewer),int(*)(void*),void*);
EXTERN int PFApply(PF,int,PetscScalar*,PetscScalar*);
EXTERN int PFApplyVec(PF,Vec,Vec);

EXTERN int        PFRegisterDestroy(void);
EXTERN int        PFRegisterAll(const char[]);
extern PetscTruth PFRegisterAllCalled;

EXTERN int PFRegister(const char[],const char[],const char[],int(*)(PF,void*));
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define PFRegisterDynamic(a,b,c,d) PFRegister(a,b,c,0)
#else
#define PFRegisterDynamic(a,b,c,d) PFRegister(a,b,c,d)
#endif

EXTERN int PFDestroy(PF);
EXTERN int PFSetFromOptions(PF);
EXTERN int PFGetType(PF,PFType*);

EXTERN int PFView(PF,PetscViewer);

#define PFSetOptionsPrefix(a,s) PetscObjectSetOptionsPrefix((PetscObject)(a),s)

PETSC_EXTERN_CXX_END
#endif
