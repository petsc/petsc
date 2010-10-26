#include "taosolver.h"
#include "taolinesearch.h"
#include "private/taolinesearch_impl.h"

PetscBool TaoLineSearchRegisterAllCalled = PETSC_FALSE;
PetscFList TaoLineSearchList              = PETSC_NULL;

PetscClassId TAOLINESEARCH_CLASSID=0;
PetscLogEvent TaoLineSearch_ApplyEvent = 0, TaoLineSearch_EvalEvent=0;

#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchView"
PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchView(TaoLineSearch ls, PetscViewer viewer)
 {
     PetscErrorCode info;
     PetscBool isascii, isstring;
     const TaoLineSearchType type;
     PetscFunctionBegin;
     PetscValidHeaderSpecific(ls,TAOLINESEARCH_CLASSID,1);
     if (!viewer) {
	 info = PetscViewerASCIIGetStdout(((PetscObject)ls)->comm, &viewer); CHKERRQ(info);
     }
     PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
     PetscCheckSameComm(ls,1,viewer,2);

     info = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(info);
     info = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring);CHKERRQ(info);
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
	 info = PetscViewerASCIIPrintf(viewer,"  maximum function evaluations=%D\n",ls->maxfev);CHKERRQ(info);
	 info = PetscViewerASCIIPrintf(viewer,"  tolerances: ftol=%G, rtol=%G, gtol=%G\n", ls->ftol, ls->rtol,ls->gtol);CHKERRQ(info); 
	 info = PetscViewerASCIIPrintf(viewer,"  total number of function evaluations=%D\n",ls->nfev);CHKERRQ(info);
	 if (ls->bounded) {
	     info = PetscViewerASCIIPrintf(viewer,"  using variable bounds\n");CHKERRQ(info);
	 }
     } else if (isstring) {
	 info = TaoLineSearchGetType(ls,&type);CHKERRQ(info);
	 info = PetscViewerStringSPrintf(viewer," %-3.3s",type);CHKERRQ(info);
     }
     PetscFunctionReturn(0);

 }

 #undef __FUNCT__
 #define __FUNCT__ "TaoLineSearchGetNumberFunctionEvals"
 PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchGetNumberFunctionEvals(TaoLineSearch ls, PetscInt *nfev) 
 {
     PetscFunctionBegin;
     PetscValidHeaderSpecific(ls,TAOLINESEARCH_CLASSID,1);
     PetscValidIntPointer(nfev,2);
     *nfev = ls->nfev;
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
			      TAOLINESEARCH_CLASSID, 0, "TaoLineSearch",
			      comm,TaoLineSearchDestroy, TaoLineSearchView);
     CHKERRQ(info);
     ls->bounded = 0;
     ls->maxfev=30;
     ls->ftol = 0.0001;
     ls->gtol = 0.9;
     ls->rtol = 1.0e-10;
     ls->stepmin=1.0e-20;
     ls->stepmax=1.0e+20;
     ls->step=1.0;
     ls->nfev=0;

     ls->ops->computeobjective=0;
     ls->ops->computegradient=0;
     ls->ops->setup=0;
     ls->ops->apply=0;
     ls->ops->view=0;
     ls->ops->setfromoptions=0;
     ls->ops->destroy=0;
     ls->setupcalled=PETSC_FALSE;
     *newls = ls;
     PetscFunctionReturn(0);
 }


 #undef __FUNCT__
 #define __FUNCT__ "TaoLineSearchSetUp"
 PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchSetUp(TaoLineSearch ls)
 {
     PetscErrorCode info;
     const char *default_type=TAOLINESEARCH_MT;
     PetscFunctionBegin;
     PetscValidHeaderSpecific(ls,TAOLINESEARCH_CLASSID,1);
     if (ls->setupcalled) PetscFunctionReturn(0);
     if (!((PetscObject)ls)->type_name) {
	 info = TaoLineSearchSetType(ls,default_type); CHKERRQ(info);
     }
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
     PetscValidHeaderSpecific(ls,TAOLINESEARCH_CLASSID,1);
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
 PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchApply(TaoLineSearch ls, Vec x, PetscReal *f, Vec g, Vec s, PetscReal *steplength, TaoLineSearchTerminationReason *reason)
 {
     PetscErrorCode ierr;
     PetscBool flg;
     PetscViewer viewer;
     PetscInt low1,low2,low3,high1,high2,high3;
     char filename[PETSC_MAX_PATH_LEN];

     PetscFunctionBegin;
     *reason = TAOLINESEARCH_CONTINUE_ITERATING;
     PetscValidHeaderSpecific(ls,TAOLINESEARCH_CLASSID,1);
     PetscValidHeaderSpecific(x,VEC_CLASSID,2);
     PetscValidScalarPointer(f,3);
     PetscValidHeaderSpecific(g,VEC_CLASSID,4);
     PetscValidHeaderSpecific(s,VEC_CLASSID,5);
     PetscValidPointer(reason,7);
     PetscCheckSameComm(ls,1,x,2);
     PetscCheckSameTypeAndComm(x,2,g,4);
     PetscCheckSameTypeAndComm(x,2,s,5);
     ierr = VecGetOwnershipRange(x, &low1, &high1); CHKERRQ(ierr);
     ierr = VecGetOwnershipRange(g, &low2, &high2); CHKERRQ(ierr);
     ierr = VecGetOwnershipRange(s, &low3, &high3); CHKERRQ(ierr);
     if ( low1!= low2 || low1!= low3 || high1!= high2 || high1!= high3) {
       SETERRQ(PETSC_COMM_SELF,1,"InCompatible vector local lengths");
     }



     ierr = TaoLineSearchSetUp(ls); CHKERRQ(ierr);
     if (!ls->ops->apply) {
	 SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Line Search Object does not have 'apply' routine");
     }
     ls->nfev=0;

     /* Check parameter values */
     if (ls->ftol < 0.0) {
       ierr = PetscInfo1(ls,"Bad Line Search Parameter: ftol (%g) < 0\n",ls->ftol); CHKERRQ(ierr);
       *reason=TAOLINESEARCH_FAILED_BADPARAMETER;
     }
     if (ls->rtol < 0.0) {
       ierr = PetscInfo1(ls,"Bad Line Search Parameter: rtol (%g) < 0\n",ls->rtol); CHKERRQ(ierr);
       *reason=TAOLINESEARCH_FAILED_BADPARAMETER;
     }      

     if (ls->gtol < 0.0) {
       ierr = PetscInfo1(ls,"Bad Line Search Parameter: gtol (%g) < 0\n",ls->gtol); CHKERRQ(ierr);
       *reason=TAOLINESEARCH_FAILED_BADPARAMETER;
     }      
     if (ls->stepmin < 0.0) {
       ierr = PetscInfo1(ls,"Bad Line Search Parameter: stepmin (%g) < 0\n",ls->stepmin); CHKERRQ(ierr);
       *reason=TAOLINESEARCH_FAILED_BADPARAMETER;
     }      
     if (ls->stepmax < ls->stepmin) {
       ierr = PetscInfo2(ls,"Bad Line Search Parameter: stepmin (%g) > stepmax (%g)\n",ls->stepmin,ls->stepmax); CHKERRQ(ierr);
       *reason=TAOLINESEARCH_FAILED_BADPARAMETER;
     }      
     if (ls->maxfev < 0) {
       ierr = PetscInfo1(ls,"Bad Line Search Parameter: maxfev (%d) < 0\n",ls->maxfev); CHKERRQ(ierr);
       *reason=TAOLINESEARCH_FAILED_BADPARAMETER;
     }      
     if (PetscIsInfOrNanReal(*f)) {
       ierr = PetscInfo1(ls,"Initial Line Search Function Value is Inf or Nan (%g)\n",*f); CHKERRQ(ierr);
       *reason=TAOLINESEARCH_FAILED_INFORNAN;
     }

 /*    
     if (x != ls->start_x) {
	 ierr = PetscObjectReference((PetscObject)x);
	 if (ls->start_x) {
	     ierr = VecDestroy(ls->start_x); CHKERRQ(ierr);
	 }
	 if (ls->new_x) {
	   ierr = VecDestroy(ls->new_x); CHKERRQ(ierr);
	 }

	 ls->start_x = x;

	 ierr = VecDuplicate(ls->start_x, &ls->new_x); CHKERRQ(ierr);
	 ierr = VecDuplicate(ls->start_x, &ls->new_g); CHKERRQ(ierr);
     }
     if (g != ls->gradient) {
	 ierr = PetscObjectReference((PetscObject)g);
	 if (ls->gradient) {
	     ierr = VecDestroy(ls->gradient); CHKERRQ(ierr);
	 }
	 ls->gradient = g;
     if (s != ls->step_direction) {
	 ierr = PetscObjectReference((PetscObject)s);
	 if (ls->step_direction) {
	     ierr = VecDestroy(ls->step_direction); CHKERRQ(ierr);
	 }
	 ls->step_direction = s;
     }

     ls->fval = *f;
 */


     ierr = PetscLogEventBegin(TaoLineSearch_ApplyEvent,ls,0,0,0); CHKERRQ(ierr);
     ierr = (*ls->ops->apply)(ls,x,f,g,s); CHKERRQ(ierr);
     ierr = PetscLogEventEnd(TaoLineSearch_ApplyEvent, ls, 0,0,0); CHKERRQ(ierr);
     *reason=ls->reason;

     if (steplength) { 
       *steplength=ls->step;
     }


     ierr = PetscOptionsGetString(((PetscObject)ls)->prefix,"-tao_ls_view",filename,PETSC_MAX_PATH_LEN,&flg); CHKERRQ(ierr);
     if (flg && !PetscPreLoadingOn) {
	 ierr = PetscViewerASCIIOpen(((PetscObject)ls)->comm,filename,&viewer); CHKERRQ(ierr);
	 ierr = TaoLineSearchView(ls,viewer); CHKERRQ(ierr);
	 ierr = PetscViewerDestroy(viewer); CHKERRQ(ierr);
     }
     PetscFunctionReturn(0);
 }

 #undef __FUNCT__
 #define __FUNCT__ "TaoLineSearchSetType"
 PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchSetType(TaoLineSearch ls, const TaoLineSearchType type)
 {
     PetscErrorCode ierr;
     PetscErrorCode (*r)(TaoLineSearch);
     PetscBool flg;

     PetscFunctionBegin;
     PetscValidHeaderSpecific(ls,TAOLINESEARCH_CLASSID,1);
     PetscValidCharPointer(type,2);
     ierr = PetscTypeCompare((PetscObject)ls, type, &flg); CHKERRQ(ierr);
     if (flg) PetscFunctionReturn(0);

     ierr = PetscFListFind(TaoLineSearchList, ((PetscObject)ls)->comm,type, (void (**)(void)) &r); CHKERRQ(ierr);
     if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested TaoLineSearch type %s",type);
     if (ls->ops->destroy) {
	 ierr = (*(ls)->ops->destroy)(ls); CHKERRQ(ierr);
     }
     ls->maxfev=30;
     ls->ftol = 0.0001;
     ls->gtol = 0.9;
     ls->rtol = 1.0e-10;
     ls->stepmin=1.0e-20;
     ls->stepmax=1.0e+20;

     ls->nfev=0;
     ls->ops->setup=0;
     ls->ops->apply=0;
     ls->ops->view=0;
     ls->ops->setfromoptions=0;
     ls->ops->destroy=0;
     ls->setupcalled = PETSC_FALSE;
     ierr = (*r)(ls); CHKERRQ(ierr);
     ierr = PetscObjectChangeTypeName((PetscObject)ls, type); CHKERRQ(ierr);
     PetscFunctionReturn(0);
 }

 #undef __FUNCT__
 #define __FUNCT__ "TaoLineSearchSetFromOptions"
 PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchSetFromOptions(TaoLineSearch ls)
 {
   PetscErrorCode ierr;

   const char *default_type=TAOLINESEARCH_MT;
   char type[256];
   PetscBool flg;
   PetscFunctionBegin;
   PetscValidHeaderSpecific(ls,TAOLINESEARCH_CLASSID,1);

   ierr = PetscOptionsBegin(((PetscObject)ls)->comm, ((PetscObject)ls)->prefix,"Tao line search options","TaoLineSearch"); CHKERRQ(ierr);
   {
     if (!TaoLineSearchRegisterAllCalled) {
       ierr = TaoLineSearchRegisterAll(PETSC_NULL); CHKERRQ(ierr);
     }
     if (((PetscObject)ls)->type_name) {
       default_type = ((PetscObject)ls)->type_name;
     }
     /* Check for type from options */
     ierr = PetscOptionsList("-tao_ls_type","Tao Line Search type","TaoLineSearchSetType",TaoLineSearchList,default_type,type,256,&flg); CHKERRQ(ierr);
     if (flg) {
       ierr = TaoLineSearchSetType(ls,type); CHKERRQ(ierr);
     } else if (!((PetscObject)ls)->type_name) {
       ierr = TaoLineSearchSetType(ls,default_type);
     }

     ierr = PetscOptionsInt("-tao_ls_maxfev","max function evals in line search",
			  "",ls->maxfev,&ls->maxfev,0);CHKERRQ(ierr);
     ierr = PetscOptionsReal("-tao_ls_ftol","tol for sufficient decrease","",
			   ls->ftol,&ls->ftol,0);CHKERRQ(ierr);
     ierr = PetscOptionsReal("-tao_ls_gtol","tol for curvature condition","",
			   ls->gtol,&ls->gtol,0);CHKERRQ(ierr);
     ierr = PetscOptionsReal("-tao_ls_rtol","relative tol for acceptable step","",
			   ls->rtol,&ls->rtol,0);CHKERRQ(ierr);
     ierr = PetscOptionsReal("-tao_ls_stepmin","lower bound for step","",
			   ls->stepmin,&ls->stepmin,0);CHKERRQ(ierr);
     ierr = PetscOptionsReal("-tao_ls_stepmax","upper bound for step","",
			   ls->stepmax,&ls->stepmax,0);CHKERRQ(ierr);


     if (ls->ops->setfromoptions) {
       ierr = (*ls->ops->setfromoptions)(ls); CHKERRQ(ierr);
     }
   }
   ierr = PetscOptionsEnd(); CHKERRQ(ierr);


   PetscFunctionReturn(0);
 }

 #undef __FUNCT__
 #define __FUNCT__ "TaoLineSearchGetType"
 PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchGetType(TaoLineSearch ls, const TaoLineSearchType *type)
 {
     PetscFunctionBegin;
     PetscValidHeaderSpecific(ls,TAOLINESEARCH_CLASSID,1);
     PetscValidPointer(type,2);
     *type = ((PetscObject)ls)->type_name;
     PetscFunctionReturn(0);
 }

 #undef __FUNCT__
 #define __FUNCT__ "TaoLineSearchSetObjective"
 PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchSetObjective(TaoLineSearch ls, PetscErrorCode(*func)(TaoLineSearch ls, Vec x, PetscReal*, void*), void *ctx)
 {
     PetscFunctionBegin;
     PetscValidHeaderSpecific(ls,TAOLINESEARCH_CLASSID,1);

     ls->ops->computeobjective=func;
     if (ctx) ls->userctx_func=ctx;
     PetscFunctionReturn(0);


 }


 #undef __FUNCT__
 #define __FUNCT__ "TaoLineSearchSetGradient"
 PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchSetGradient(TaoLineSearch ls, PetscErrorCode(*func)(TaoLineSearch ls, Vec x, Vec g, void*), void *ctx)
 {
     PetscFunctionBegin;
     PetscValidHeaderSpecific(ls,TAOLINESEARCH_CLASSID,1);

     ls->ops->computegradient=func;
     if (ctx) ls->userctx_grad=ctx;
     PetscFunctionReturn(0);
 }

 #undef __FUNCT__
 #define __FUNCT__ "TaoLineSearchSetObjectiveAndGradient"
PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchSetObjectiveAndGradient(TaoLineSearch ls, PetscErrorCode(*func)(TaoLineSearch ls, Vec x, PetscReal *, Vec g, void*), void *ctx)
{
    PetscFunctionBegin;
    PetscValidHeaderSpecific(ls,TAOLINESEARCH_CLASSID,1);
    
    ls->ops->computeobjectiveandgradient=func;
    if (ctx) ls->userctx_funcgrad=ctx;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchObjectiveGradient_Default"
PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchObjectiveGradient_Default(TaoLineSearch ls, Vec x, PetscReal *f, Vec g, void *ctx) 
{ 
    PetscErrorCode ierr;
    PetscFunctionBegin;
    PetscValidHeaderSpecific(ls,TAOLINESEARCH_CLASSID,1);
    PetscValidHeaderSpecific(x,VEC_CLASSID,2);
    PetscValidPointer(f,3);
    PetscValidHeaderSpecific(g,VEC_CLASSID,4);
    if (!ls->taosolver) {
	SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Cannot use TaoSolver object's Objective/Gradient routine: TaoSolver object not declared with TaoLineSearchUseTaoSolverRoutines().");
    } else {
	ierr = TaoSolverComputeObjectiveAndGradient(ls->taosolver,x,f,g); CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchUseTaoSolverRoutines"
PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchUseTaoSolverRoutines(TaoLineSearch ls, TaoSolver ts) 
{
    PetscFunctionBegin;
    PetscValidHeaderSpecific(ls,TAOLINESEARCH_CLASSID,1);
    PetscValidHeaderSpecific(ts,TAOSOLVER_CLASSID,1);
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
    PetscErrorCode ierr;
    Vec gdummy;
    PetscFunctionBegin;
    PetscValidHeaderSpecific(ls,TAOLINESEARCH_CLASSID,1);
    PetscValidHeaderSpecific(x,VEC_CLASSID,2);
    PetscValidPointer(f,3);
    PetscCheckSameComm(ls,1,x,2);
    ierr = PetscLogEventBegin(TaoLineSearch_EvalEvent,ls,0,0,0); CHKERRQ(ierr);
    if (!ls->ops->computeobjective && !ls->ops->computeobjectiveandgradient) {
	SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Line Search does not have objective function set");
    }
    PetscStackPush("TaoLineSearch user objective routine"); 
    CHKMEMQ;
    if (ls->ops->computeobjective) {
	ierr = (*ls->ops->computeobjective)(ls,x,f,ls->userctx_func); CHKERRQ(ierr);
    } else {
	ierr = VecDuplicate(x,&gdummy); CHKERRQ(ierr);
	ierr = (*ls->ops->computeobjectiveandgradient)(ls,x,f,gdummy,ls->userctx_funcgrad); CHKERRQ(ierr);
	ierr = VecDestroy(gdummy); CHKERRQ(ierr);
    }
    CHKMEMQ;
    PetscStackPop;
    ierr = PetscLogEventEnd(TaoLineSearch_EvalEvent,ls,0,0,0); CHKERRQ(ierr);
    ls->nfev++;
    PetscFunctionReturn(0);
    
}


#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchComputeObjectiveAndGradient"
PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchComputeObjectiveAndGradient(TaoLineSearch ls, Vec x, PetscReal *f, Vec g) 
{
    PetscErrorCode ierr;
    PetscFunctionBegin;
    PetscValidHeaderSpecific(ls,TAOLINESEARCH_CLASSID,1);
    PetscValidHeaderSpecific(x,VEC_CLASSID,2);
    PetscValidPointer(f,3);
    PetscValidHeaderSpecific(g,VEC_CLASSID,4);
    PetscCheckSameComm(ls,1,x,2);
    PetscCheckSameComm(ls,1,g,4);
    ierr = PetscLogEventBegin(TaoLineSearch_EvalEvent,ls,0,0,0); CHKERRQ(ierr);
    if (!ls->ops->computeobjective && !ls->ops->computeobjectiveandgradient) {
	SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Line Search does not have objective function set");
    }
    if (!ls->ops->computegradient && !ls->ops->computeobjectiveandgradient) {
	SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Line Search does not have gradient function set");
    }

    PetscStackPush("TaoLineSearch user objective/gradient routine"); 
    CHKMEMQ;
    if (ls->ops->computeobjectiveandgradient) {
	ierr = (*ls->ops->computeobjectiveandgradient)(ls,x,f,g,ls->userctx_funcgrad); CHKERRQ(ierr);
    } else {
	ierr = (*ls->ops->computeobjective)(ls,x,f,ls->userctx_func); CHKERRQ(ierr);
	ierr = (*ls->ops->computegradient)(ls,x,g,ls->userctx_grad); CHKERRQ(ierr);
    }
    CHKMEMQ;
    PetscStackPop;
    ierr = PetscLogEventEnd(TaoLineSearch_EvalEvent,ls,0,0,0); CHKERRQ(ierr);
    ierr = PetscInfo1(ls,"TaoLineSearch Function evaluation: %14.12e\n",*f);CHKERRQ(ierr);    ls->nfev++;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchComputeGradient"
PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchComputeGradient(TaoLineSearch ls, Vec x, Vec g) 
{
    PetscErrorCode ierr;
    PetscReal fdummy;
    PetscFunctionBegin;
    PetscValidHeaderSpecific(ls,TAOLINESEARCH_CLASSID,1);
    PetscValidHeaderSpecific(x,VEC_CLASSID,2);
    PetscValidHeaderSpecific(g,VEC_CLASSID,3);
    PetscCheckSameComm(ls,1,x,2);
    PetscCheckSameComm(ls,1,g,3);
    ierr = PetscLogEventBegin(TaoLineSearch_EvalEvent,ls,0,0,0); CHKERRQ(ierr);
    if (!ls->ops->computegradient && !ls->ops->computeobjectiveandgradient) {
	SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Line Search does not have gradient functions set");
    }
    PetscStackPush("TaoLineSearch user gradient routine"); 
    CHKMEMQ;
    if (ls->ops->computegradient) { 
	ierr = (*ls->ops->computegradient)(ls,x,g,ls->userctx_grad); CHKERRQ(ierr);
    } else {
	ierr = (*ls->ops->computeobjectiveandgradient)(ls,x,&fdummy,g,ls->userctx_funcgrad); CHKERRQ(ierr);
    }
    CHKMEMQ;
    PetscStackPop;
    ierr = PetscLogEventEnd(TaoLineSearch_EvalEvent,ls,0,0,0); CHKERRQ(ierr);
    ls->nfev++;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchGetSolution"
PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchGetSolution(TaoLineSearch ls, Vec x, PetscReal *f, Vec g, PetscReal *steplength, TaoLineSearchTerminationReason *reason) 
{
    PetscErrorCode ierr;
    PetscFunctionBegin;
    PetscValidHeaderSpecific(ls,TAOLINESEARCH_CLASSID,1);
    PetscValidHeaderSpecific(x,VEC_CLASSID,2);
    PetscValidPointer(f,3);
    PetscValidHeaderSpecific(g,VEC_CLASSID,4);
    PetscValidIntPointer(reason,6);

    if (ls->new_x) {
	ierr = VecCopy(ls->new_x,x); CHKERRQ(ierr);
    }
    *f = ls->new_f;
    if (ls->new_g) {
	ierr = VecCopy(ls->new_g,g); CHKERRQ(ierr);
    }
    if (steplength) {
      *steplength=ls->step;
    }
    *reason = ls->reason;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchGetFullStepObjective"
PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchGetFullStepObjective(TaoLineSearch ls, PetscReal *f_fullstep)
{
    PetscFunctionBegin;
    PetscValidHeaderSpecific(ls,TAOLINESEARCH_CLASSID,1);
    *f_fullstep = ls->f_fullstep;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchSetVariableBounds"
PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchSetVariableBounds(TaoLineSearch ls,Vec xl, Vec xu)
{
    PetscFunctionBegin;
    PetscValidHeaderSpecific(ls,TAOLINESEARCH_CLASSID,1);
    PetscValidHeaderSpecific(xl,VEC_CLASSID,2);
    PetscValidHeaderSpecific(xu,VEC_CLASSID,3);
    
    ls->lower = xl;
    ls->upper = xu;
    ls->bounded = 1;

    PetscFunctionReturn(0);
    
}


#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchSetInitialStepLength"
PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchSetInitialStepLength(TaoLineSearch ls,PetscReal s)
{
    PetscFunctionBegin;
    PetscValidHeaderSpecific(ls,TAOLINESEARCH_CLASSID,1);
    
    ls->initstep = s;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchGetStepLength"
PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchGetStepLength(TaoLineSearch ls,PetscReal *s)
{
    PetscFunctionBegin;
    PetscValidHeaderSpecific(ls,TAOLINESEARCH_CLASSID,1);
    *s = ls->step;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchRegister"
PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchRegister(const char s[], const char path[], const char name[], PetscErrorCode (*func)(TaoLineSearch))
{
    char full[PETSC_MAX_PATH_LEN];
    PetscErrorCode ierr;
    PetscFunctionBegin;
    ierr = PetscFListConcat(path,name,full); CHKERRQ(ierr);
    ierr = PetscFListAdd(&TaoLineSearchList, s, full, (void (*)(void))func); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchRegisterDestroy"
PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchRegisterDestroy(void)
{
    PetscErrorCode ierr;
    PetscFunctionBegin;
    ierr = PetscFListDestroy(&TaoLineSearchList); CHKERRQ(ierr);
    TaoLineSearchRegisterAllCalled = PETSC_FALSE;
    PetscFunctionReturn(0);
}
