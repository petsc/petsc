#define TAOLINESEARCH_DLL
#include "private/taolinesearch_impl.h"

EXTERN_C_BEGIN
extern PetscErrorCode TaoLineSearchCreate_Unit(TaoLineSearch);
extern PetscErrorCode TaoLineSearchCreate_MT(TaoLineSearch);
extern PetscErrorCode TaoLineSearchCreate_GPCG(TaoLineSearch);
EXTERN_C_END



//PetscClassId TAOLINESEARCH_CLASSID=0;

#ifdef PETSC_USE_DYNAMIC_LIBRARIES
#define TaoLineSearchRegisterDynamic(a,b,c,d) TaoLineSearchRegister(a,b,c,0)
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PetscDLLibraryRegister_taolinesearch"
PetscErrorCode PetscDLLibraryRegister_taolinesearch(const char path[]) 
{
    PetscErrorCode info;
    info = PetscInitializeNoArguments(); if (info) return 1;
    PetscFunctionBegin;
    info = TaoLineSearchInitializePackage(path); CHKERRQ(info);
    PetscFunctionReturn(0);
}
EXTERN_C_END
#else
#define TaoLineSearchRegisterDynamic(a,b,c,d) TaoLineSearchRegister(a,b,c,d)
#endif

EXTERN_C_BEGIN
extern PetscErrorCode TaoLineSearchCreate_Unit(TaoLineSearch);
extern PetscErrorCode TaoLineSearchCreate_MT(TaoLineSearch);
extern PetscErrorCode TaoLineSearchCreate_GPCG(TaoLineSearch);
EXTERN_C_END
    
extern PetscBool TaoLineSearchRegisterAllCalled;

#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchRegisterAll"
PetscErrorCode TaoLineSearchRegisterAll(const char path[])
{
    PetscErrorCode info;
    PetscFunctionBegin;
    TaoLineSearchRegisterAllCalled=PETSC_TRUE;
    info = TaoLineSearchRegisterDynamic("unit",path,"TaoLineSearchCreate_Unit",TaoLineSearchCreate_Unit); CHKERRQ(info);
    info = TaoLineSearchRegisterDynamic("more-thuente",path,"TaoLineSearchCreate_MT",TaoLineSearchCreate_MT); CHKERRQ(info);
    info = TaoLineSearchRegisterDynamic("gpcg",path,"TaoLineSearchCreate_GPCG",TaoLineSearchCreate_GPCG); CHKERRQ(info);
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchInitializePackage"
PetscErrorCode TaoLineSearchInitializePackage(const char path[])
{
    static PetscBool initialized = PETSC_FALSE;
    PetscErrorCode info;

    PetscFunctionBegin;
    if (initialized) PetscFunctionReturn(0);
    initialized = PETSC_TRUE;
    info = PetscClassIdRegister("TaoLineSearch",&TAOLINESEARCH_CLASSID); CHKERRQ(info);
    info = TaoLineSearchRegisterAll(path);
    info = PetscLogEventRegister(  "TaoLineSearchApply",TAOLINESEARCH_CLASSID,&TaoLineSearch_ApplyEvent); CHKERRQ(info);
    info = PetscLogEventRegister("TaoLineSearchComputeObjective[Gradient]",TAOLINESEARCH_CLASSID,&TaoLineSearch_EvalEvent); CHKERRQ(info);
    PetscFunctionReturn(0);
}



    

