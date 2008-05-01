#include "taosolver.h"
#include "private/taolinesearch_impl.h"

PetscTruth TaoLineSearchRegisterAllCalled = PETSC_FALSE;
PetscFList TaoLineSearchList              = PETSC_NULL;

PetscCookie TAOLINESEARCH_COOKIE=0;
PetscEvent TaoLineSearch_ApplyEvent = 0, TaoLineSearch_EvalEvent=0;

#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchView"
PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchView(TaoLineSearch ls, PetscViewer viewer)
{
    PetscErrorCode info;
    PetscTruth isascii, isstring;
    TaoLineSearchType type;
    PetscFunctionBegin;
    PetscValidHeaderSpecific(ls,TAOLINESEARCH_COOKIE,1);
    if (!viewer) {
	info = PetscViewerASCIIGetStdout(((PetscObject)ls)->comm, &viewer); CHKERRQ(info);
    }
    PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE,2);
    PetscCheckSameComm(ls,1,viewer,2);

    info = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(info);
    info = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_STRING,&isstring);CHKERRQ(info);
    if (isascii) {
	if (((PetscObject)ls)->prefix) {
	    info = PetscViewerASCIIPrintf(viewer,"TaoLineSearch Object:(%s)\n",((PetscObject)ls)->prefix);CHKERRQ(info);
	} else {
	    info = PetscViewerASCIIPrintf(viewer,"TaoLineSearch Object:\n");CHKERRQ(info);
	}
	info = TaoLineSearchGetType(ls,&type);CHKERRQ(info);
	if (type) {
	    info = PetscViewerASCIIPrintf(viewer,"  type: %s\n",type);CHKERRQ(info);
	} else {
	    info = PetscViewerASCIIPrintf(viewer,"  type: not set yet\n");CHKERRQ(info);
	}
	if (ls->ops->view) {
	    info = PetscViewerASCIIPushTab(viewer);CHKERRQ(info);
	    info = (*ls->ops->view)(ls,viewer);CHKERRQ(info);
	    info = PetscViewerASCIIPopTab(viewer);CHKERRQ(info);
	}
	info = PetscViewerASCIIPrintf(viewer,"  maximum function evaluations=%D\n",ls->max_funcs);CHKERRQ(info);
/*	info = PetscViewerASCIIPrintf(viewer,"  tolerances: relative=%G, absolute=%G, solution=%G\n",
	ls->rtol,ls->abstol,ls->xtol);CHKERRQ(info); */
	info = PetscViewerASCIIPrintf(viewer,"  total number of function evaluations=%D\n",ls->nfuncs);CHKERRQ(info);
    } else if (isstring) {
	info = TaoLineSearchGetType(ls,&type);CHKERRQ(info);
	info = PetscViewerStringSPrintf(viewer," %-3.3s",type);CHKERRQ(info);
    }
    PetscFunctionReturn(0);
    
}

#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchGetNumberFunctionEvals"
PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchGetNumberFunctionEvals(TaoLineSearch ls, PetscInt *nfuncs) 
{
    PetscFunctionBegin;
    PetscValidHeaderSpecific(ls,TAOLINESEARCH_COOKIE,1);
    PetscValidIntPointer(nfuncs,2);
    *nfuncs = ls->nfuncs;
    PetscFunctionReturn(0);
}
    


#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchCreate"
PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchCreate(MPI_Comm comm, TaoLineSearch *newls)
{
    PetscErrorCode info;
    TaoLineSearch ls;
    
    PetscFunctionBegin;
    PetscValidPointer(newls,2);
    *newls = PETSC_NULL;

#ifndef PETSC_USE_DYNAMIC_LIBRARIES
    info = TaoLineSearchInitializePackage(PETSC_NULL); CHKERRQ(info);
#endif 

    info = PetscHeaderCreate(ls,_p_TaoLineSearch,struct _TaoLineSearchOps,
			     TAOLINESEARCH_COOKIE, 0, "TaoLineSearch",
			     comm,TaoLineSearchDestroy, TaoLineSearchView);
    CHKERRQ(info);
    
    ls->max_fev=30;
    ls->ftol = 0.0001;
    ls->gtol = 0.9;
    ls->rtol = 1.0e-10;

    ls->nfuncs=0;
    
    ls->ops->computeobjective=0;
    ls->ops->computegradient=0;
    ls->ops->setup=0;
    ls->ops->apply=0;
    ls->ops->view=0;
    ls->ops->setfromoptions=0;
    ls->ops->destroy=0;
    *newls = ls;
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchSetUp"
PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchSetUp(TaoLineSearch ls)
{
    PetscErrorCode info;
    PetscFunctionBegin;
    PetscValidHeaderSpecific(ls,TAOLINESEARCH_COOKIE,1);
    if (ls->setupcalled) PetscFunctionReturn(0);
    if (!((PetscObject)ls)->type_name) {
	info = TaoLineSearchSetType(ls,TAOLINESEARCH_UNIT); CHKERRQ(info);
    }
    if (!ls->start_x) {
	SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Starting vector not valid.");
    }

    info = VecDuplicate(ls->start_x, &ls->new_x); CHKERRQ(info);
    info = VecDuplicate(ls->start_x, &ls->new_g); CHKERRQ(info);
    if (ls->ops->setup) {
	info = (*ls->ops->setup)(ls); CHKERRQ(info);
    }
    ls->setupcalled = PETSC_TRUE;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchDestroy"
PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchDestroy(TaoLineSearch ls)
{
    PetscErrorCode info;
    PetscFunctionBegin;
    PetscValidHeaderSpecific(ls,TAOLINESEARCH_COOKIE,1);
    if (--((PetscObject)ls)->refct > 0) PetscFunctionReturn(0);
    if (ls->ops->destroy) {
	info = (*ls->ops->destroy)(ls); CHKERRQ(info);
    }
    if (ls->start_x) {
	info = VecDestroy(ls->start_x); CHKERRQ(info);
    }
    if (ls->new_x) {
	info = VecDestroy(ls->new_x); CHKERRQ(info);
    }
    if (ls->new_g) {
	info = VecDestroy(ls->new_g); CHKERRQ(info);
    }

    info = PetscHeaderDestroy(ls); CHKERRQ(info);
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchApply"
PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchApply(TaoLineSearch ls, Vec x, PetscReal f, Vec g, Vec s)
{
    PetscErrorCode info;
    PetscTruth flg;
    PetscViewer viewer;
    char filename[PETSC_MAX_PATH_LEN];

    PetscFunctionBegin;
    PetscValidHeaderSpecific(ls,TAOLINESEARCH_COOKIE,1);
    PetscValidHeaderSpecific(x,VEC_COOKIE,2);

    PetscValidHeaderSpecific(g,VEC_COOKIE,4);
    PetscValidHeaderSpecific(s,VEC_COOKIE,5);
    PetscCheckSameComm(ls,1,x,2);
    PetscCheckSameComm(ls,1,g,4);
    PetscCheckSameComm(ls,1,s,5);
    
    if (x != ls->start_x) {
	info = PetscObjectReference((PetscObject)x);
	if (ls->start_x) {
	    info = VecDestroy(ls->start_x); CHKERRQ(info);
	}
	ls->start_x = x;
    }
/*    if (g != ls->gradient) {
	info = PetscObjectReference((PetscObject)g);
	if (ls->gradient) {
	    info = VecDestroy(ls->gradient); CHKERRQ(info);
	}
	ls->gradient = g;
    if (s != ls->step_direction) {
	info = PetscObjectReference((PetscObject)s);
	if (ls->step_direction) {
	    info = VecDestroy(ls->step_direction); CHKERRQ(info);
	}
	ls->step_direction = s;
    }
    ls->fval = *f;
*/


    info = TaoLineSearchSetUp(ls); CHKERRQ(info);
    ls->nfuncs=0;
    info = PetscLogEventBegin(TaoLineSearch_ApplyEvent,ls,0,0,0); CHKERRQ(info);
    info = (*ls->ops->apply)(ls,x,f,g,s); CHKERRQ(info);
    info = PetscLogEventEnd(TaoLineSearch_ApplyEvent, ls, 0,0,0); CHKERRQ(info);
    
    info = PetscOptionsGetString(((PetscObject)ls)->prefix,"-taolinesearch_view",filename,PETSC_MAX_PATH_LEN,&flg); CHKERRQ(info);
    if (flg && !PetscPreLoadingOn) {
	info = PetscViewerASCIIOpen(((PetscObject)ls)->comm,filename,&viewer); CHKERRQ(info);
	info = TaoLineSearchView(ls,viewer); CHKERRQ(info);
	info = PetscViewerDestroy(viewer); CHKERRQ(info);
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchSetType"
PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchSetType(TaoLineSearch ls, TaoLineSearchType type)
{
    PetscErrorCode info;
    PetscErrorCode (*r)(TaoLineSearch);
    PetscTruth flg;

    PetscFunctionBegin;
    PetscValidHeaderSpecific(ls,TAOLINESEARCH_COOKIE,1);
    PetscValidCharPointer(type,2);
    info = PetscTypeCompare((PetscObject)ls, type, &flg); CHKERRQ(info);
    if (flg) PetscFunctionReturn(0);
    
    info = PetscFListFind(TaoLineSearchList, ((PetscObject)ls)->comm,type, (void (**)(void)) &r); CHKERRQ(info);
    if (!r) SETERRQ1(PETSC_ERR_ARG_UNKNOWN_TYPE,"Unyoable to find requested TaoLineSearch type %s",type);
    if (ls->ops->destroy) {
	info = (*(ls)->ops->destroy)(ls); CHKERRQ(info);
    }
    ls->ops->setup=0;
    ls->ops->apply=0;
    ls->ops->view=0;
    ls->ops->setfromoptions=0;
    ls->ops->destroy=0;
    
    ls->setupcalled = PETSC_FALSE;
    info = (*r)(ls); CHKERRQ(info);
    info = PetscObjectChangeTypeName((PetscObject)ls, type); CHKERRQ(info);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchSetFromOptions"
PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchSetFromOptions(TaoLineSearch ls)
{

    PetscFunctionBegin;
    PetscValidHeaderSpecific(ls,TAOLINESEARCH_COOKIE,1);
    /* Not implemented yet */
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchGetType"
PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchGetType(TaoLineSearch ls, TaoLineSearchType *type)
{
    PetscFunctionBegin;
    PetscValidHeaderSpecific(ls,TAOLINESEARCH_COOKIE,1);
    PetscValidPointer(type,2);
    *type = ((PetscObject)ls)->type_name;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchSetObjective"
PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchSetObjective(TaoLineSearch ls, PetscErrorCode(*func)(TaoLineSearch ls, Vec x, PetscReal*, void*), void *ctx)
{
    PetscErrorCode info;
    PetscFunctionBegin;
    PetscValidHeaderSpecific(ls,TAOLINESEARCH_COOKIE,1);
    
    ls->ops->computeobjective=func;
    if (ctx) ls->userctx_func=ctx;
    PetscFunctionReturn(0);


}


#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchSetGradient"
PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchSetGradient(TaoLineSearch ls, PetscErrorCode(*func)(TaoLineSearch ls, Vec x, Vec g, void*), void *ctx)
{
    PetscErrorCode info;
    PetscFunctionBegin;
    PetscValidHeaderSpecific(ls,TAOLINESEARCH_COOKIE,1);
    
    ls->ops->computegradient=func;
    if (ctx) ls->userctx_grad=ctx;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchSetObjectiveGradient"
PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchSetObjectiveGradient(TaoLineSearch ls, PetscErrorCode(*func)(TaoLineSearch ls, Vec x, PetscReal *, Vec g, void*), void *ctx)
{
    PetscErrorCode info;
    PetscFunctionBegin;
    PetscValidHeaderSpecific(ls,TAOLINESEARCH_COOKIE,1);
    
    ls->ops->computeobjectiveandgradient=func;
    if (ctx) ls->userctx_funcgrad=ctx;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchObjective_Default"
PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchObjectiveGradient_Default(TaoLineSearch ls, Vec x, PetscReal *f, Vec g, void *ctx) 
{ 
    PetscErrorCode info;
    PetscFunctionBegin;
    PetscValidHeaderSpecific(ls,TAOLINESEARCH_COOKIE,1);
    PetscValidHeaderSpecific(x,VEC_COOKIE,2);
    PetscValidPointer(f,3);
    PetscValidHeaderSpecific(g,VEC_COOKIE,4);
    if (!ls->taosolver) {
	SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Cannot use TaoSolver object's Objective/Gradient routine: TaoSolver object not declared with TaoLineSearchUseTaoSolverRoutines().");
    } else {
	//TaoSolverComputeObjectiveGradient(ls->taosolver,x,f,g); CHKERRQ(info);
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchUseTaoSolverRoutines"
PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchUseTaoSolverRoutines(TaoLineSearch ls, TaoSolver ts) 
{
    PetscFunctionBegin;
    PetscValidHeaderSpecific(ls,TAOLINESEARCH_COOKIE,1);
    PetscValidHeaderSpecific(ts,TAOSOLVER_COOKIE,1);
    ls->taosolver = ts;
    ls->ops->computeobjective=0;
    ls->ops->computegradient=0;
    ls->ops->computeobjectiveandgradient = TaoLineSearchObjectiveGradient_Default;
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchComputeObjective"
PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchComputeObjective(TaoLineSearch ls, Vec x, PetscReal *f) 
{
    PetscErrorCode info;
    Vec gdummy;
    PetscFunctionBegin;
    PetscValidHeaderSpecific(ls,TAOLINESEARCH_COOKIE,1);
    PetscValidHeaderSpecific(x,VEC_COOKIE,2);
    PetscValidPointer(f,3);
    PetscCheckSameComm(ls,1,x,2);
    info = PetscLogEventBegin(TaoLineSearch_EvalEvent,ls,0,0,0); CHKERRQ(info);
    if (!ls->ops->computeobjective && !ls->ops->computeobjectiveandgradient) {
	SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Line Search does not have objective function set");
    }
    PetscStackPush("TaoLineSearch user objective routine"); 
    CHKMEMQ;
    if (ls->ops->computeobjective) {
	info = (*ls->ops->computeobjective)(ls,x,f,ls->userctx_func); CHKERRQ(info);
    } else {
	info = VecDuplicate(x,&gdummy); CHKERRQ(info);
	info = (*ls->ops->computeobjectiveandgradient)(ls,x,f,gdummy,ls->userctx_funcgrad); CHKERRQ(info);
	info = VecDestroy(gdummy); CHKERRQ(info);
    }
    CHKMEMQ;
    PetscStackPop;
    info = PetscLogEventEnd(TaoLineSearch_EvalEvent,ls,0,0,0); CHKERRQ(info);
    PetscFunctionReturn(0);
    
}


#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchComputeObjectiveGradient"
PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchComputeObjectiveGradient(TaoLineSearch ls, Vec x, PetscReal *f, Vec g) 
{
    PetscErrorCode info;
    PetscFunctionBegin;
    PetscValidHeaderSpecific(ls,TAOLINESEARCH_COOKIE,1);
    PetscValidHeaderSpecific(x,VEC_COOKIE,2);
    PetscValidPointer(f,3);
    PetscValidHeaderSpecific(g,VEC_COOKIE,4);
    PetscCheckSameComm(ls,1,x,2);
    PetscCheckSameComm(ls,1,g,4);
    info = PetscLogEventBegin(TaoLineSearch_EvalEvent,ls,0,0,0); CHKERRQ(info);
    if (!ls->ops->computeobjective && !ls->ops->computeobjectiveandgradient) {
	SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Line Search does not have objective function set");
    }
    if (!ls->ops->computegradient && !ls->ops->computeobjectiveandgradient) {
	SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Line Search does not have gradient function set");
    }

    PetscStackPush("TaoLineSearch user objective/gradient routine"); 
    CHKMEMQ;
    if (ls->ops->computeobjectiveandgradient) {
	info = (*ls->ops->computeobjectiveandgradient)(ls,x,f,g,ls->userctx_funcgrad); CHKERRQ(info);
    } else {
	info = (*ls->ops->computeobjective)(ls,x,f,ls->userctx_func); CHKERRQ(info);
	info = (*ls->ops->computegradient)(ls,x,g,ls->userctx_grad); CHKERRQ(info);
    }
    CHKMEMQ;
    PetscStackPop;
    info = PetscLogEventEnd(TaoLineSearch_EvalEvent,ls,0,0,0); CHKERRQ(info);
}

#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchComputeGradient"
PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchComputeGradient(TaoLineSearch ls, Vec x, Vec g) 
{
    PetscErrorCode info;
    PetscReal fdummy;
    PetscFunctionBegin;
    PetscValidHeaderSpecific(ls,TAOLINESEARCH_COOKIE,1);
    PetscValidHeaderSpecific(x,VEC_COOKIE,2);
    PetscValidHeaderSpecific(g,VEC_COOKIE,3);
    PetscCheckSameComm(ls,1,x,2);
    PetscCheckSameComm(ls,1,g,3);
    info = PetscLogEventBegin(TaoLineSearch_EvalEvent,ls,0,0,0); CHKERRQ(info);
    if (!ls->ops->computegradient && !ls->ops->computeobjectiveandgradient) {
	SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Line Search does not have gradient functions set");
    }
    PetscStackPush("TaoLineSearch user gradient routine"); 
    CHKMEMQ;
    if (ls->ops->computegradient) { 
	info = (*ls->ops->computegradient)(ls,x,g,ls->userctx_grad); CHKERRQ(info);
    } else {
	info = (*ls->ops->computeobjectiveandgradient)(ls,x,&fdummy,g,ls->userctx_funcgrad); CHKERRQ(info);
    }
    CHKMEMQ;
    PetscStackPop;
    info = PetscLogEventEnd(TaoLineSearch_EvalEvent,ls,0,0,0); CHKERRQ(info);
}

#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchGetSolution"
PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchGetSolution(TaoLineSearch ls, Vec x, PetscReal *f, Vec g, TaoLineSearchTerminationReason *reason) 
{
    PetscErrorCode info;
    PetscFunctionBegin;
    PetscValidHeaderSpecific(ls,TAOLINESEARCH_COOKIE,1);
    PetscValidHeaderSpecific(x,VEC_COOKIE,2);
    PetscValidPointer(f,3);
    PetscValidHeaderSpecific(g,VEC_COOKIE,4);
    PetscValidIntPointer(reason,5);

    if (ls->new_x) {
	info = VecCopy(ls->new_x,x); CHKERRQ(info);
    }
    *f = ls->new_f;
    if (ls->new_g) {
	info = VecCopy(ls->new_g,g); CHKERRQ(info);
    }
    *reason = ls->reason;
    PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchRegister"
PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchRegister(const char s[], const char path[], const char name[], PetscErrorCode (*func)(TaoLineSearch))
{
    char full[PETSC_MAX_PATH_LEN];
    PetscErrorCode info;
    PetscFunctionBegin;
    info = PetscFListConcat(path,name,full); CHKERRQ(info);
    info = PetscFListAdd(&TaoLineSearchList, s, full, (void (*)(void))func); CHKERRQ(info);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchRegisterDestroy"
PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchRegisterDestroy(void)
{
    PetscErrorCode info;
    PetscFunctionBegin;
    info = PetscFListDestroy(&TaoLineSearchList); CHKERRQ(info);
    TaoLineSearchRegisterAllCalled = PETSC_FALSE;
    PetscFunctionReturn(0);
}
