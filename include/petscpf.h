/* $Id: pf.h,v 1.2 2000/01/24 04:05:37 bsmith Exp bsmith $ */

/*
      mathematical function module. 
*/
#if !defined(__PF_H)
#define __PF_H
#include "mat.h"

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

typedef struct _p_PF* PF;
#define PF_COOKIE     PETSC_COOKIE+9


extern int PFCreate(MPI_Comm,int,int,PF*);
extern int PFSetType(PF,PFType,void*);
extern int PFSet(PF,int(*)(void*,int,Scalar*,Scalar*),int(*)(void*,Vec,Vec),int(*)(void*,Viewer),int(*)(void*),void*);
extern int PFApply(PF,int,Scalar*,Scalar*);
extern int PFApplyVec(PF,Vec,Vec);

extern int PFRegisterDestroy(void);
extern int PFRegisterAll(char*);
extern int PFRegisterAllCalled;

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

#endif




