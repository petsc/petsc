#define TAOLINESEARCH_DLL
#include "private/taolinesearch_impl.h"

EXTERN_C_BEGIN
EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchCreate_Unit(TaoLineSearch);
EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchCreate_MT(TaoLineSearch);
EXTERN_C_END



//PetscCookie TAOLINESEARCH_COOKIE=0;

#ifdef PETSC_USE_DYNAMIC_LIBRARIES
#define TaoLineSearchRegisterDynamic(a,b,c,d) TaoLineSearchRegister(a,b,c,0)
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PetscDLLibraryRegister_taolinesearch"
PetscErrorCode TAOLINESEARCH_DLLEXPORT PetscDLLibraryRegister_taolinesearch(const char path[]) 
{
    PetscErrorCode info;
    info = PetscInitializeNoArguments(); if (info) return 1;
    PetscFunctionBegin;
    info = TaoLineSearchInitializePackage(path); CHKERRQ(info);
    PetscFunctionReturn(0);

EXTERN_C_END
#else
#define TaoLineSearchRegisterDynamic(a,b,c,d) TaoLineSearchRegister(a,b,c,d)
#endif

EXTERN_C_BEGIN
EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchCreate_Unit(TaoLineSearch);
EXTERN_C_END

extern PetscTruth TaoLineSearchRegisterAllCalled;

#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchRegisterAll"
PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchRegisterAll(const char path[])
{
    PetscErrorCode info;
    PetscFunctionBegin;
    TaoLineSearchRegisterAllCalled=PETSC_TRUE;
    info = TaoLineSearchRegisterDynamic("unit",path,"TaoLineSearchCreate_Unit",TaoLineSearchCreate_Unit); CHKERRQ(info);
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchInitializePackage"
PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchInitializePackage(const char path[])
{
    static PetscTruth initialized = PETSC_FALSE;
    PetscErrorCode info;

    PetscFunctionBegin;
    if (initialized) PetscFunctionReturn(0);
    initialized = PETSC_TRUE;
    info = PetscCookieRegister("TaoLineSearch",&TAOLINESEARCH_COOKIE); CHKERRQ(info);
    info = TaoLineSearchRegisterAll(path);
    info = PetscLogEventRegister(  "TaoLineSearchApply",TAOLINESEARCH_COOKIE,&TaoLineSearch_ApplyEvent); CHKERRQ(info);
    info = PetscLogEventRegister("TaoLineSearchComputeObjective[Gradient]",TAOLINESEARCH_COOKIE,&TaoLineSearch_EvalEvent); CHKERRQ(info);
    PetscFunctionReturn(0);
}



    

