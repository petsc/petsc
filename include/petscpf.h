/* $Id: petscpf.h,v 1.4 2000/04/09 03:11:53 bsmith Exp balay $ */

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

typedef struct _p_PF* PF;
#define PF_COOKIE     PETSC_COOKIE+9


extern int PFCreate(MPI_Comm,int,int,PF*);
extern int PFSetType(PF,PFType,void*);
extern int PFSet(PF,int(*)(void*,int,Scalar*,Scalar*),int(*)(void*,Vec,Vec),int(*)(void*,Viewer),int(*)(void*),void*);
extern int PFApply(PF,int,Scalar*,Scalar*);
extern int PFApplyVec(PF,Vec,Vec);

extern int        PFRegisterDestroy(void);
extern int        PFRegisterAll(char*);
extern PetscTruth PFRegisterAllCalled;

extern int PFRegister(char*,char*,char*,int(*)(PF,void*));
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define PFRegisterDynamic(a,b,c,d) PFRegister(a,b,c,0)
#else
#define PFRegisterDynamic(a,b,c,d) PFRegister(a,b,c,d)
#endif

extern int PFDestroy(PF);
extern int PFSetFromOptions(PF);
extern int PFSetTypeFromOptions(PF);
extern int PFGetType(PF,PFType*);

extern int PFView(PF,Viewer);
extern int PFPrintHelp(PF);

#define PFSetOptionsPrefix(a,s) PetscObjectSetOptionsPrefix((PetscObject)(a),s)
#endif




