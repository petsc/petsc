/* $Id: petscpf.h,v 1.6 2000/05/10 16:44:25 bsmith Exp bsmith $ */

/*
      mathematical function module. 
*/
#if !defined(__PETSCPF_H)
#define __PETSCPF_H
#include "petscmat.h"

/*
    PFList contains the list of preconditioners currently registered
   These are added with the PFRegisterDynamic() macro
*/
extern FList PFList;
typedef char *PFType;

/*
    Standard PETSc functions
*/
#define PFCONSTANT      "constant"
#define PFMAT           "mat"
#define PFSTRING        "string"
#define PFQUICK         "quick"
#define PFIDENTITY      "identity"
#define PFMATLAB        "matlab"

typedef struct _p_PF* PF;
#define PF_COOKIE     PETSC_COOKIE+9


EXTERN int PFCreate(MPI_Comm,int,int,PF*);
EXTERN int PFSetType(PF,PFType,void*);
EXTERN int PFSet(PF,int(*)(void*,int,Scalar*,Scalar*),int(*)(void*,Vec,Vec),int(*)(void*,Viewer),int(*)(void*),void*);
EXTERN int PFApply(PF,int,Scalar*,Scalar*);
EXTERN int PFApplyVec(PF,Vec,Vec);

EXTERN int        PFRegisterDestroy(void);
EXTERN int        PFRegisterAll(char*);
extern PetscTruth PFRegisterAllCalled;

EXTERN int PFRegister(char*,char*,char*,int(*)(PF,void*));
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define PFRegisterDynamic(a,b,c,d) PFRegister(a,b,c,0)
#else
#define PFRegisterDynamic(a,b,c,d) PFRegister(a,b,c,d)
#endif

EXTERN int PFDestroy(PF);
EXTERN int PFSetFromOptions(PF);
EXTERN int PFSetTypeFromOptions(PF);
EXTERN int PFGetType(PF,PFType*);

EXTERN int PFView(PF,Viewer);
EXTERN int PFPrintHelp(PF);

#define PFSetOptionsPrefix(a,s) PetscObjectSetOptionsPrefix((PetscObject)(a),s)
#endif




