#define TAOLINESEARCH_DLL
#include "private/taolinesearch_impl.h"


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
    info = PetscLogClassRegister(&TAOLINESEARCH_COOKIE,"TaoLineSearch"); CHKERRQ(info);
    info = TaoLineSearchRegisterAll(path);
    info = PetscLogEventRegister(&TaoLineSearch_ApplyEvent, "TaoLineSearchApply", TAOLINESEARCH_COOKIE); CHKERRQ(info);
    info = PetscLogEventRegister(&TaoLineSearch_EvalEvent, "TaoLineSearchComputeObjective[Gradient]",TAOLINESEARCH_COOKIE); CHKERRQ(info);
    PetscFunctionReturn(0);
}



    

