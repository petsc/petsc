#define PETSCTS_DLL

/*
    Provides a PETSc interface to SUNDIALS. Alan Hindmarsh's parallel ODE
    solver.
*/

#include "src/ts/impls/implicit/sundials/petscsundials.h"  /*I "petscts.h" I*/    

#undef DEBUG_SUNDIAL

/*
      TSPrecond_Sundials - function that we provide to SUNDIALS to
                        evaluate the preconditioner.
    Contributed by: Liyang Xu
*/
#undef __FUNCT__
#define __FUNCT__ "TSPrecond_Sundials"
PetscErrorCode TSPrecond_Sundials(realtype tn,N_Vector y,N_Vector fy,
                    booleantype jok,booleantype *jcurPtr,
                    realtype _gamma,void *P_data,
                    N_Vector vtemp1,N_Vector vtemp2,N_Vector vtemp3)
{
  TS             ts = (TS) P_data;
  TS_Sundials    *cvode = (TS_Sundials*)ts->data;
  PC             pc = cvode->pc;
  PetscErrorCode ierr;
  Mat            Jac = ts->B;
  Vec            tmpy = cvode->w1;
  PetscScalar    one = 1.0,gm;
  MatStructure   str = DIFFERENT_NONZERO_PATTERN;
  realtype *tmp;
  PetscInt locsize,i;
  PetscScalar *parray;
  
  PetscFunctionBegin;
#ifdef DEBUG_SUNDIAL
  printf(" TSPrecond ...\n");
#endif
  /* This allows us to construct preconditioners in-place if we like */
  ierr = MatSetUnfactored(Jac);CHKERRQ(ierr);

  /* jok - TRUE means reuse current Jacobian else recompute Jacobian */
  if (jok) {
    /* ierr  = MatCopy(cvode->pmat,Jac,SAME_NONZERO_PATTERN);CHKERRQ(ierr); fails - will find out why */
    ierr     = MatCopy(cvode->pmat,Jac,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    str      = SAME_NONZERO_PATTERN;
    *jcurPtr = FALSE;
  } else {
    /* make PETSc vector tmpy point to SUNDIALS vector y */
    tmp  = N_VGetArrayPointer(y);
    ierr = VecGetArray(cvode->w1,&parray);CHKERRQ(ierr);
    ierr = VecGetLocalSize(ts->vec_sol,&locsize);CHKERRQ(ierr);
    for (i=0; i<locsize; i++){
      parray[i] = tmp[i];
    }
    ierr = VecRestoreArray(cvode->w1,&parray);CHKERRQ(ierr);
#ifdef DEBUG_SUNDIAL
    printf(" in Precond, tmpy ...\n");
    ierr = VecView(tmpy,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
#endif

    /* compute the Jacobian */
    ierr = TSComputeRHSJacobian(ts,ts->ptime,tmpy,&Jac,&Jac,&str);CHKERRQ(ierr);

    /* copy the Jacobian matrix */
    if (!cvode->pmat) {
      ierr = MatDuplicate(Jac,MAT_COPY_VALUES,&cvode->pmat);CHKERRQ(ierr);
      ierr = PetscLogObjectParent(ts,cvode->pmat);CHKERRQ(ierr);
    }
    ierr = MatCopy(Jac,cvode->pmat,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    *jcurPtr = TRUE;
  }

  /* construct I-gamma*Jac  */
  gm   = -_gamma;
  ierr = MatScale(Jac,gm);CHKERRQ(ierr);
  ierr = MatShift(Jac,one);CHKERRQ(ierr);
  
  ierr = PCSetOperators(pc,Jac,Jac,str);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     TSPSolve_Sundials -  routine that we provide to Sundials that applies the preconditioner.
      
    Contributed by: Liyang Xu
*/    
#undef __FUNCT__
#define __FUNCT__ "TSPSolve_Sundials"
PetscErrorCode TSPSolve_Sundials(realtype tn,N_Vector y,N_Vector fy,
                                 N_Vector r,N_Vector z,
                                 realtype _gamma,realtype delta,
                                int lr,void *P_data,N_Vector vtemp)
{ 
  TS          ts = (TS) P_data;
  TS_Sundials *cvode = (TS_Sundials*)ts->data;
  PC          pc = cvode->pc;
  Vec         rr = cvode->w1,xx = cvode->w2;
  PetscErrorCode ierr;
  realtype *tmp;
  PetscInt locsize,i;
  PetscScalar *parray;

  PetscFunctionBegin;
#ifdef DEBUG_SUNDIAL
  printf(" ... TSPSolve \n");
#endif
  /* Make the PETSc work vectors rr and xx point to the arrays in the SUNDIALS vectors */
  ierr = VecGetLocalSize(ts->vec_sol,&locsize);CHKERRQ(ierr);
  tmp  = N_VGetArrayPointer(r);
  ierr = VecGetArray(rr,&parray);CHKERRQ(ierr);
  for (i=0; i<locsize; i++){
    parray[i] = tmp[i];
  }
  ierr = VecRestoreArray(rr,&parray);CHKERRQ(ierr);
  /*
  printf(" rr ..\n");
  ierr = VecView(rr,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr); */

  tmp  = N_VGetArrayPointer(z);
  ierr = VecGetArray(xx,&parray);CHKERRQ(ierr);
  for (i=0; i<locsize; i++){
    parray[i] = tmp[i];
  }
  ierr = VecRestoreArray(xx,&parray);CHKERRQ(ierr);
  /*
  printf(" xx ..\n");
  ierr = VecView(xx,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr); */

  /* Solve the Px=r and put the result in xx */
  ierr = PCApply(pc,rr,xx);CHKERRQ(ierr);

  cvode->linear_solves++;
  PetscFunctionReturn(0);
}

/*
        TSFunction_Sundials - routine that we provide to Sundials that applies the right hand side.
    Contributed by: Liyang Xu
*/  
#undef __FUNCT__  
#define __FUNCT__ "TSFunction_Sundials"
void TSFunction_Sundials(realtype t,N_Vector y,N_Vector ydot,void *ctx)
{
  TS        ts = (TS) ctx;
  TS_Sundials *cvode = (TS_Sundials*)ts->data;
  Vec       tmpx = cvode->w1,tmpy = cvode->w2;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#ifdef DEBUG_SUNDIAL
  printf(" ...TSFunction \n");
#endif
  /* Make the PETSc work vectors tmpx and tmpy point to the arrays in the SUNDIALS vectors */
  ierr = VecPlaceArray(tmpx,N_VGetArrayPointer(y));
  if (ierr) {
    (*PetscErrorPrintf)("TSFunction_Sundials:Could not place array. Error code %d",(int)ierr);
  }
  ierr = VecResetArray(tmpx);

  ierr = VecPlaceArray(tmpy,N_VGetArrayPointer(ydot));
  if (ierr) {
    (*PetscErrorPrintf)("TSFunction_Sundials:Could not place array. Error code %d",(int)ierr);
  }
  ierr = VecResetArray(tmpy);

  /* now compute the right hand side function */
  ierr = TSComputeRHSFunction(ts,t,tmpx,tmpy);
  if (ierr) {
    (*PetscErrorPrintf)("TSFunction_Sundials:Could not compute RHS function. Error code %d",(int)ierr);
  }
}

/*
       TSStep_Sundials_Nonlinear - Calls Sundials to integrate the ODE.
    Contributed by: Liyang Xu
*/
#undef __FUNCT__  
#define __FUNCT__ "TSStep_Sundials_Nonlinear"
/* 
    TSStep_Sundials_Nonlinear - 
  
   steps - number of time steps
   time - time that integrater is  terminated. 
*/
PetscErrorCode TSStep_Sundials_Nonlinear(TS ts,int *steps,double *time)
{
  TS_Sundials  *cvode = (TS_Sundials*)ts->data;
  Vec          sol = ts->vec_sol;
  PetscErrorCode ierr;
  int          i,max_steps = ts->max_steps,flag;
  long int     its;
  realtype     t,tout;
  PetscScalar  *ydata,*parray;
  PetscInt     locsize;
  void         *mem;

  PetscFunctionBegin;
#ifdef DEBUG_SUNDIAL
  printf(" ... call TSStep_Sundials_Nonlinear\n");
#endif
  /* 
     Call CVodeCreate to create the solver memory:
     CV_ADAMS   specifies the Adams Method
     CV_FUNCTIONAL  specifies functional iteration  
     A pointer to the integrator memory is returned and stored in cvode_mem.
  */
  mem = CVodeCreate(cvode->cvode_type, CV_NEWTON); 
  if (!mem) SETERRQ(1,"CVodeCreate() fails");
  flag = CVodeSetFdata(mem,ts);
  if (flag) SETERRQ(1,"CVodeSetFdata() fails");

  /* 
     Call CVodeMalloc to initialize the integrator memory: 
     mem is the pointer to the integrator memory returned by CVodeCreate
     f       is the user's right hand side function in y'=f(t,y)
     T0      is the initial time
     u       is the initial dependent variable vector
     CV_SS   specifies scalar relative and absolute tolerances
     reltol  is the relative tolerance
     &abstol is a pointer to the scalar absolute tolerance
  */
  flag = CVodeMalloc(mem,TSFunction_Sundials,ts->ptime,cvode->y,CV_SS,cvode->reltol,&cvode->abstol);
  if (flag) SETERRQ(1,"CVodeMalloc() fails");

  /* initialize the number of steps */
  *steps = -ts->steps;
  ierr   = TSMonitor(ts,ts->steps,ts->ptime,sol);CHKERRQ(ierr); 

  /* call CVSpgmr to use GMRES as the linear solver. */
  /* setup the ode integrator with the given preconditioner */
  /* flag  = CVSpgmr(mem,PREC_LEFT,cvode->restart); */
  flag  = CVSpgmr(mem,PREC_LEFT,0);
  if (flag) SETERRQ(1,"CVSpgmr() fails");
  flag = CVSpgmrSetGSType(mem,MODIFIED_GS);
  if (flag) SETERRQ(1,"CVSpgmrSetGSType() fails");

  /* Set preconditioner setup and solve routines Precond and PSolve, 
     and the pointer to the user-defined block data */
  flag = CVSpgmrSetPreconditioner(mem,TSPrecond_Sundials,TSPSolve_Sundials,ts);
  if (flag) SETERRQ(1,"CVSpgmrSetPreconditioner() fails");

  tout = ts->max_time;
  ierr = VecGetLocalSize(ts->vec_sol,&locsize);CHKERRQ(ierr);
  NV_LENGTH_S(cvode->y) = locsize;
  for (i = 0; i < max_steps; i++) {
    if (ts->ptime >= ts->max_time) break;
    ierr = VecGetArray(ts->vec_sol,&ydata);CHKERRQ(ierr);
    NV_DATA_S(cvode->y) = (realtype *)ydata;
    ierr = CVode(mem,tout,cvode->y,&t,CV_ONE_STEP);CHKERRQ(ierr); 
    ierr = VecRestoreArray(ts->vec_sol,&ydata);CHKERRQ(ierr);
    
    ierr = CVodeGetNumNonlinSolvIters(mem,&its);CHKERRQ(ierr);
    cvode->nonlinear_solves += its; 

    if (t > ts->max_time && cvode->exact_final_time) { 
      /* interpolate to final requested time */
      ierr = CVodeGetDky(mem,tout,0,cvode->y);CHKERRQ(ierr);
      t = tout;
    }
    ts->time_step = t - ts->ptime;
    ts->ptime     = t; 

    /* copy the solution from cvode->y to cvode->update and sol */
    ydata = (PetscScalar *) NV_DATA_S(cvode->y);
    /*ierr = VecPlaceArray(cvode->w1,ydata);CHKERRQ(ierr);*/
    ierr = VecGetArray(cvode->w1,&parray);CHKERRQ(ierr);
    /*ierr = VecGetLocalSize(ts->vec_sol,&locsize);CHKERRQ(ierr);*/
    ierr = PetscMemcpy(parray,ydata,locsize*sizeof(PetscScalar));
    ierr = VecRestoreArray(cvode->w1,&parray);CHKERRQ(ierr);
    ierr = VecCopy(cvode->w1,cvode->update);CHKERRQ(ierr);
    ierr = VecCopy(cvode->update,sol);CHKERRQ(ierr);
#ifdef DEBUG_SUNDIAL   
    printf(" in Step, sol ...\n");
    ierr = VecView(sol,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);  
#endif    
    ts->steps++;
    ierr = TSMonitor(ts,ts->steps,t,sol);CHKERRQ(ierr); 
    ierr = CVodeGetNumNonlinSolvIters(mem,&its);CHKERRQ(ierr);
    ts->nonlinear_its = its;
    ierr = CVSpgmrGetNumLinIters(mem, &its);
    ts->linear_its = its; 
  }
  *steps += ts->steps;
  *time   = t;
  PetscFunctionReturn(0);
}

/* Contributed by: Liyang Xu */
#undef __FUNCT__  
#define __FUNCT__ "TSDestroy_Sundials"
PetscErrorCode TSDestroy_Sundials(TS ts)
{
  TS_Sundials    *cvode = (TS_Sundials*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (cvode->pmat)   {ierr = MatDestroy(cvode->pmat);CHKERRQ(ierr);} 
  if (cvode->pc)     {ierr = PCDestroy(cvode->pc);CHKERRQ(ierr);} 
  if (cvode->update) {ierr = VecDestroy(cvode->update);CHKERRQ(ierr);}
  if (cvode->func)   {ierr = VecDestroy(cvode->func);CHKERRQ(ierr);}
  if (cvode->rhs)    {ierr = VecDestroy(cvode->rhs);CHKERRQ(ierr);}
  if (cvode->w1)     {ierr = VecDestroy(cvode->w1);CHKERRQ(ierr);}
  if (cvode->w2)     {ierr = VecDestroy(cvode->w2);CHKERRQ(ierr);}
  ierr = MPI_Comm_free(&(cvode->comm_sundials));CHKERRQ(ierr);

  /*  CVodeFree(mem); */
  ierr = PetscFree(cvode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    Contributed by: Liyang Xu
*/
#undef __FUNCT__  
#define __FUNCT__ "TSSetUp_Sundials_Nonlinear"
PetscErrorCode TSSetUp_Sundials_Nonlinear(TS ts)
{
  TS_Sundials    *cvode = (TS_Sundials*)ts->data;
  PetscErrorCode ierr;
  int            M,locsize,i;
  realtype       *tmp;
  PetscScalar    *parray;

  PetscFunctionBegin;
  ierr = PCSetFromOptions(cvode->pc);CHKERRQ(ierr);
  /* get the vector size */
  ierr = VecGetSize(ts->vec_sol,&M);CHKERRQ(ierr);
  ierr = VecGetLocalSize(ts->vec_sol,&locsize);CHKERRQ(ierr);

  /* allocate the memory for N_Vec y */
  /* cvode->y = N_VNew_Parallel(ts->comm,locsize,M);*/ 
  cvode->y = N_VNew_Serial(M);
  if (!cvode->y) SETERRQ(1,"cvode->y is not allocated");

  /* initialize N_Vec y */
#ifdef DEBUG_SUNDIAL
  printf(" initial vec_sol ...\n");
  ierr = VecView(ts->vec_sol,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
#endif
  ierr = VecGetArray(ts->vec_sol,&parray);CHKERRQ(ierr);
  tmp  = NV_DATA_S(cvode->y);
  for (i=0; i<locsize; i++) tmp[i] = parray[i];
  ierr = VecRestoreArray(ts->vec_sol,PETSC_NULL);CHKERRQ(ierr);

  /* initialize vector update and func */
  ierr = VecDuplicate(ts->vec_sol,&cvode->update);CHKERRQ(ierr);  
  ierr = VecDuplicate(ts->vec_sol,&cvode->func);CHKERRQ(ierr);  
  ierr = PetscLogObjectParent(ts,cvode->update);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(ts,cvode->func);CHKERRQ(ierr);

  /* 
      Create work vectors for the TSPSolve_Sundials() routine. Note these are
    allocated with zero space arrays because the actual array space is provided 
    by Sundials and set using VecPlaceArray().
  */
  ierr = VecDuplicate(ts->vec_sol,&cvode->w1);CHKERRQ(ierr); 
  ierr = VecDuplicate(ts->vec_sol,&cvode->w2);CHKERRQ(ierr); 
  ierr = PetscLogObjectParent(ts,cvode->w1);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(ts,cvode->w2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    Contributed by: Liyang Xu
*/
#undef __FUNCT__  
#define __FUNCT__ "TSSetFromOptions_Sundials_Nonlinear"
PetscErrorCode TSSetFromOptions_Sundials_Nonlinear(TS ts)
{
  TS_Sundials    *cvode = (TS_Sundials*)ts->data;
  PetscErrorCode ierr;
  int            indx;
  const char     *btype[] = {"bdf","adams"},*otype[] = {"modified","unmodified"};
  PetscTruth     flag;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("SUNDIALS ODE solver options");CHKERRQ(ierr);
    ierr = PetscOptionsEList("-ts_sundials_type","Scheme","TSSundialsSetType",btype,2,"bdf",&indx,&flag);CHKERRQ(ierr);
    if (flag) {
      ierr = TSSundialsSetType(ts,(TSSundialsType)indx);CHKERRQ(ierr);
    }
    ierr = PetscOptionsEList("-ts_sundials_gramschmidt_type","Type of orthogonalization","TSSundialsSetGramSchmidtType",otype,2,"unmodified",&indx,&flag);CHKERRQ(ierr);
    if (flag) {
      ierr = TSSundialsSetGramSchmidtType(ts,(TSSundialsGramSchmidtType)indx);CHKERRQ(ierr);
    }
    ierr = PetscOptionsReal("-ts_sundials_atol","Absolute tolerance for convergence","TSSundialsSetTolerance",cvode->abstol,&cvode->abstol,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ts_sundials_rtol","Relative tolerance for convergence","TSSundialsSetTolerance",cvode->reltol,&cvode->reltol,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ts_sundials_linear_tolerance","Convergence tolerance for linear solve","TSSundialsSetLinearTolerance",cvode->linear_tol,&cvode->linear_tol,&flag);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-ts_sundials_gmres_restart","Number of GMRES orthogonalization directions","TSSundialsSetGMRESRestart",cvode->restart,&cvode->restart,&flag);CHKERRQ(ierr);
    ierr = PetscOptionsName("-ts_sundials_not_exact_final_time","Allow SUNDIALS to stop near the final time, not exactly on it","TSSundialsSetExactFinalTime",&cvode->exact_final_time);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    Contributed by: Liyang Xu
*/
#undef __FUNCT__  
#define __FUNCT__ "TSPrintHelp_Sundials" 
PetscErrorCode TSPrintHelp_Sundials(TS ts,char *p)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = (*PetscHelpPrintf)(ts->comm," Options for TSSUNDIALS integrater:\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ts->comm," -ts_sundials_type <bdf,adams>: integration approach\n",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ts->comm," -ts_sundials_atol aabs: absolute tolerance of ODE solution\n",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ts->comm," -ts_sundials_rtol rel: relative tolerance of ODE solution\n",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ts->comm," -ts_sundials_gramschmidt_type <unmodified,modified>\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ts->comm," -ts_sundials_gmres_restart <restart_size> (also max. GMRES its)\n");CHKERRQ(ierr); 
  ierr = (*PetscHelpPrintf)(ts->comm," -ts_sundials_linear_tolerance <tol>\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ts->comm," -ts_sundials_not_exact_final_time\n");CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

/*

    Contributed by: Liyang Xu
*/
#undef __FUNCT__  
#define __FUNCT__ "TSView_Sundials" 
PetscErrorCode TSView_Sundials(TS ts,PetscViewer viewer)
{
  TS_Sundials    *cvode = (TS_Sundials*)ts->data;
  PetscErrorCode ierr;
  char           *type;
  char            atype[] = "Adams";
  char            btype[] = "BDF: backward differentiation formula";
  PetscTruth     iascii,isstring;

  PetscFunctionBegin;
  if (cvode->cvode_type == SUNDIALS_ADAMS) {type = atype;}
  else                                     {type = btype;}

  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_STRING,&isstring);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"Sundials integrater does not use SNES!\n");CHKERRQ(ierr); 
    ierr = PetscViewerASCIIPrintf(viewer,"Sundials integrater type %s\n",type);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Sundials abs tol %g rel tol %g\n",cvode->abstol,cvode->reltol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Sundials linear solver tolerance factor %g\n",cvode->linear_tol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Sundials GMRES max iterations (same as restart in SUNDIALS) %D\n",cvode->restart);CHKERRQ(ierr);
    if (cvode->gtype == SUNDIALS_MODIFIED_GS) {
      ierr = PetscViewerASCIIPrintf(viewer,"Sundials using modified Gram-Schmidt for orthogonalization in GMRES\n");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"Sundials using unmodified (classical) Gram-Schmidt for orthogonalization in GMRES\n");CHKERRQ(ierr);
    }
  } else if (isstring) {
    ierr = PetscViewerStringSPrintf(viewer,"Sundials type %s",type);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported by TS Sundials",((PetscObject)viewer)->type_name);
  }
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = PCView(cvode->pc,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* --------------------------------------------------------------------------*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "TSSundialsSetType_Sundials"
PetscErrorCode PETSCTS_DLLEXPORT TSSundialsSetType_Sundials(TS ts,TSSundialsType type)
{
  TS_Sundials *cvode = (TS_Sundials*)ts->data;
  
  PetscFunctionBegin;
  cvode->cvode_type = type;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "TSSundialsSetGMRESRestart_Sundials"
PetscErrorCode PETSCTS_DLLEXPORT TSSundialsSetGMRESRestart_Sundials(TS ts,int restart)
{
  TS_Sundials *cvode = (TS_Sundials*)ts->data;
  
  PetscFunctionBegin;
  cvode->restart = restart;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "TSSundialsSetLinearTolerance_Sundials"
PetscErrorCode PETSCTS_DLLEXPORT TSSundialsSetLinearTolerance_Sundials(TS ts,double tol)
{
  TS_Sundials *cvode = (TS_Sundials*)ts->data;
  
  PetscFunctionBegin;
  cvode->linear_tol = tol;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "TSSundialsSetGramSchmidtType_Sundials"
PetscErrorCode PETSCTS_DLLEXPORT TSSundialsSetGramSchmidtType_Sundials(TS ts,TSSundialsGramSchmidtType type)
{
  TS_Sundials *cvode = (TS_Sundials*)ts->data;
  
  PetscFunctionBegin;
  cvode->gtype = type;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "TSSundialsSetTolerance_Sundials"
PetscErrorCode PETSCTS_DLLEXPORT TSSundialsSetTolerance_Sundials(TS ts,double aabs,double rel)
{
  TS_Sundials *cvode = (TS_Sundials*)ts->data;
  
  PetscFunctionBegin;
  if (aabs != PETSC_DECIDE) cvode->abstol = aabs;
  if (rel != PETSC_DECIDE)  cvode->reltol = rel;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "TSSundialsGetPC_Sundials"
PetscErrorCode PETSCTS_DLLEXPORT TSSundialsGetPC_Sundials(TS ts,PC *pc)
{ 
  TS_Sundials *cvode = (TS_Sundials*)ts->data;

  PetscFunctionBegin;
  *pc = cvode->pc;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "TSSundialsGetIterations_Sundials"
PetscErrorCode PETSCTS_DLLEXPORT TSSundialsGetIterations_Sundials(TS ts,int *nonlin,int *lin)
{
  TS_Sundials *cvode = (TS_Sundials*)ts->data;
  
  PetscFunctionBegin;
  if (nonlin) *nonlin = cvode->nonlinear_solves;
  if (lin)    *lin    = cvode->linear_solves;
  PetscFunctionReturn(0);
}
EXTERN_C_END
  
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "TSSundialsSetExactFinalTime_Sundials"
PetscErrorCode PETSCTS_DLLEXPORT TSSundialsSetExactFinalTime_Sundials(TS ts,PetscTruth s)
{
  TS_Sundials *cvode = (TS_Sundials*)ts->data;
  
  PetscFunctionBegin;
  cvode->exact_final_time = s;
  PetscFunctionReturn(0);
}
EXTERN_C_END
/* -------------------------------------------------------------------------------------------*/

#undef __FUNCT__
#define __FUNCT__ "TSSundialsGetIterations"
/*@C
   TSSundialsGetIterations - Gets the number of nonlinear and linear iterations used so far by Sundials.

   Not Collective

   Input parameters:
.    ts     - the time-step context

   Output Parameters:
+   nonlin - number of nonlinear iterations
-   lin    - number of linear iterations

   Level: advanced

   Notes:
    These return the number since the creation of the TS object

.keywords: non-linear iterations, linear iterations

.seealso: TSSundialsSetType(), TSSundialsSetGMRESRestart(),
          TSSundialsSetLinearTolerance(), TSSundialsSetGramSchmidtType(), TSSundialsSetTolerance(),
          TSSundialsGetIterations(), TSSundialsSetType(), TSSundialsSetGMRESRestart(),
          TSSundialsSetLinearTolerance(), TSSundialsSetTolerance(), TSSundialsGetPC(),
          TSSundialsSetExactFinalTime()

@*/
PetscErrorCode PETSCTS_DLLEXPORT TSSundialsGetIterations(TS ts,int *nonlin,int *lin)
{
  PetscErrorCode ierr,(*f)(TS,int*,int*);
  
  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)ts,"TSSundialsGetIterations_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ts,nonlin,lin);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSundialsSetType"
/*@
   TSSundialsSetType - Sets the method that Sundials will use for integration.

   Collective on TS

   Input parameters:
+    ts     - the time-step context
-    type   - one of  SUNDIALS_ADAMS or SUNDIALS_BDF

    Contributed by: Liyang Xu

   Level: intermediate

.keywords: Adams, backward differentiation formula

.seealso: TSSundialsGetIterations(),  TSSundialsSetGMRESRestart(),
          TSSundialsSetLinearTolerance(), TSSundialsSetGramSchmidtType(), TSSundialsSetTolerance(),
          TSSundialsGetIterations(), TSSundialsSetType(), TSSundialsSetGMRESRestart(),
          TSSundialsSetLinearTolerance(), TSSundialsSetTolerance(), TSSundialsGetPC(),
          TSSundialsSetExactFinalTime()
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSSundialsSetType(TS ts,TSSundialsType type)
{
  PetscErrorCode ierr,(*f)(TS,TSSundialsType);
  
  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)ts,"TSSundialsSetType_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ts,type);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSundialsSetGMRESRestart"
/*@
   TSSundialsSetGMRESRestart - Sets the dimension of the Krylov space used by 
       GMRES in the linear solver in SUNDIALS. SUNDIALS DOES NOT use restarted GMRES so
       this is ALSO the maximum number of GMRES steps that will be used.

   Collective on TS

   Input parameters:
+    ts      - the time-step context
-    restart - number of direction vectors (the restart size).

   Level: advanced

.keywords: GMRES, restart

.seealso: TSSundialsGetIterations(), TSSundialsSetType(), 
          TSSundialsSetLinearTolerance(), TSSundialsSetGramSchmidtType(), TSSundialsSetTolerance(),
          TSSundialsGetIterations(), TSSundialsSetType(), TSSundialsSetGMRESRestart(),
          TSSundialsSetLinearTolerance(), TSSundialsSetTolerance(), TSSundialsGetPC(),
          TSSundialsSetExactFinalTime()

@*/
PetscErrorCode PETSCTS_DLLEXPORT TSSundialsSetGMRESRestart(TS ts,int restart)
{
  PetscErrorCode ierr,(*f)(TS,int);  

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)ts,"TSSundialsSetGMRESRestart_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ts,restart);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSundialsSetLinearTolerance"
/*@
   TSSundialsSetLinearTolerance - Sets the tolerance used to solve the linear
       system by SUNDIALS.

   Collective on TS

   Input parameters:
+    ts     - the time-step context
-    tol    - the factor by which the tolerance on the nonlinear solver is
             multiplied to get the tolerance on the linear solver, .05 by default.

   Level: advanced

.keywords: GMRES, linear convergence tolerance, SUNDIALS

.seealso: TSSundialsGetIterations(), TSSundialsSetType(), TSSundialsSetGMRESRestart(),
          TSSundialsSetGramSchmidtType(), TSSundialsSetTolerance(),
          TSSundialsGetIterations(), TSSundialsSetType(), TSSundialsSetGMRESRestart(),
          TSSundialsSetLinearTolerance(), TSSundialsSetTolerance(), TSSundialsGetPC(),
          TSSundialsSetExactFinalTime()

@*/
PetscErrorCode PETSCTS_DLLEXPORT TSSundialsSetLinearTolerance(TS ts,double tol)
{
  PetscErrorCode ierr,(*f)(TS,double);  
  
  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)ts,"TSSundialsSetLinearTolerance_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ts,tol);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSundialsSetGramSchmidtType"
/*@
   TSSundialsSetGramSchmidtType - Sets type of orthogonalization used
        in GMRES method by SUNDIALS linear solver.

   Collective on TS

   Input parameters:
+    ts  - the time-step context
-    type - either SUNDIALS_MODIFIED_GS or SUNDIALS_CLASSICAL_GS

   Level: advanced

.keywords: Sundials, orthogonalization

.seealso: TSSundialsGetIterations(), TSSundialsSetType(), TSSundialsSetGMRESRestart(),
          TSSundialsSetLinearTolerance(),  TSSundialsSetTolerance(),
          TSSundialsGetIterations(), TSSundialsSetType(), TSSundialsSetGMRESRestart(),
          TSSundialsSetLinearTolerance(), TSSundialsSetTolerance(), TSSundialsGetPC(),
          TSSundialsSetExactFinalTime()

@*/
PetscErrorCode PETSCTS_DLLEXPORT TSSundialsSetGramSchmidtType(TS ts,TSSundialsGramSchmidtType type)
{
  PetscErrorCode ierr,(*f)(TS,TSSundialsGramSchmidtType);  
  
  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)ts,"TSSundialsSetGramSchmidtType_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ts,type);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSundialsSetTolerance"
/*@
   TSSundialsSetTolerance - Sets the absolute and relative tolerance used by 
                         Sundials for error control.

   Collective on TS

   Input parameters:
+    ts  - the time-step context
.    aabs - the absolute tolerance  
-    rel - the relative tolerance

    Contributed by: Liyang Xu

     See the Cvode/Sundials users manual for exact details on these parameters. Essentially
    these regulate the size of the error for a SINGLE timestep.

   Level: intermediate

.keywords: Sundials, tolerance

.seealso: TSSundialsGetIterations(), TSSundialsSetType(), TSSundialsSetGMRESRestart(),
          TSSundialsSetLinearTolerance(), TSSundialsSetGramSchmidtType(), 
          TSSundialsGetIterations(), TSSundialsSetType(), TSSundialsSetGMRESRestart(),
          TSSundialsSetLinearTolerance(), TSSundialsSetTolerance(), TSSundialsGetPC(),
          TSSundialsSetExactFinalTime()

@*/
PetscErrorCode PETSCTS_DLLEXPORT TSSundialsSetTolerance(TS ts,double aabs,double rel)
{
  PetscErrorCode ierr,(*f)(TS,double,double);  
  
  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)ts,"TSSundialsSetTolerance_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ts,aabs,rel);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSSundialsGetPC"
/*@
   TSSundialsGetPC - Extract the PC context from a time-step context for Sundials.

   Input Parameter:
.    ts - the time-step context

   Output Parameter:
.    pc - the preconditioner context

   Level: advanced

    Contributed by: Liyang Xu

.seealso: TSSundialsGetIterations(), TSSundialsSetType(), TSSundialsSetGMRESRestart(),
          TSSundialsSetLinearTolerance(), TSSundialsSetGramSchmidtType(), TSSundialsSetTolerance(),
          TSSundialsGetIterations(), TSSundialsSetType(), TSSundialsSetGMRESRestart(),
          TSSundialsSetLinearTolerance(), TSSundialsSetTolerance()
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSSundialsGetPC(TS ts,PC *pc)
{ 
  PetscErrorCode ierr,(*f)(TS,PC *);  

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)ts,"TSSundialsGetPC_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ts,pc);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"TS must be of Sundials type to extract the PC");
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSSundialsSetExactFinalTime"
/*@
   TSSundialsSetExactFinalTime - Determines if Sundials interpolates solution to the 
      exact final time requested by the user or just returns it at the final time
      it computed. (Defaults to true).

   Input Parameter:
+   ts - the time-step context
-   ft - PETSC_TRUE if interpolates, else PETSC_FALSE

   Level: beginner

.seealso:TSSundialsGetIterations(), TSSundialsSetType(), TSSundialsSetGMRESRestart(),
          TSSundialsSetLinearTolerance(), TSSundialsSetGramSchmidtType(), TSSundialsSetTolerance(),
          TSSundialsGetIterations(), TSSundialsSetType(), TSSundialsSetGMRESRestart(),
          TSSundialsSetLinearTolerance(), TSSundialsSetTolerance(), TSSundialsGetPC() 
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSSundialsSetExactFinalTime(TS ts,PetscTruth ft)
{ 
  PetscErrorCode ierr,(*f)(TS,PetscTruth);  

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)ts,"TSSundialsSetExactFinalTime_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ts,ft);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------------*/
/*MC
      TS_Sundials - ODE solver using the LLNL CVODE/SUNDIALS package (now called SUNDIALS)

   Options Database:
+    -ts_sundials_type <bdf,adams>
.    -ts_sundials_gramschmidt_type <modified, classical> - type of orthogonalization inside GMRES
.    -ts_sundials_atol <tol> - Absolute tolerance for convergence
.    -ts_sundials_rtol <tol> - Relative tolerance for convergence
.    -ts_sundials_linear_tolerance <tol> 
.    -ts_sundials_gmres_restart <restart> - Number of GMRES orthogonalization directions
-    -ts_sundials_not_exact_final_time -Allow SUNDIALS to stop near the final time, not exactly on it

    Notes: This uses its own nonlinear solver and Krylov method so PETSc SNES and KSP options do not apply
           only PETSc PC options

    Contributed by: Liyang Xu

    Level: beginner

.seealso:  TSCreate(), TS, TSSetType(), TSSundialsSetType(), TSSundialsSetGMRESRestart(), TSSundialsSetLinearTolerance(),
           TSSundialsSetGramSchmidtType(), TSSundialsSetTolerance(), TSSundialsGetPC(), TSSundialsGetIterations(), TSSundialsSetExactFinalTime()

M*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "TSCreate_Sundials"
PetscErrorCode PETSCTS_DLLEXPORT TSCreate_Sundials(TS ts)
{
  TS_Sundials *cvode;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ts->ops->destroy         = TSDestroy_Sundials;
  ts->ops->view            = TSView_Sundials;

  if (ts->problem_type != TS_NONLINEAR) {
    SETERRQ(PETSC_ERR_SUP,"Only support for nonlinear problems");
  }
  ts->ops->setup           = TSSetUp_Sundials_Nonlinear;  
  ts->ops->step            = TSStep_Sundials_Nonlinear;
  ts->ops->setfromoptions  = TSSetFromOptions_Sundials_Nonlinear;

  ierr = PetscNew(TS_Sundials,&cvode);CHKERRQ(ierr);
  ierr = PCCreate(ts->comm,&cvode->pc);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(ts,cvode->pc);CHKERRQ(ierr);
  ts->data          = (void*)cvode;
  cvode->cvode_type = CV_BDF;
  cvode->gtype      = SUNDIALS_UNMODIFIED_GS;
  cvode->restart    = 5;
  cvode->linear_tol = .05;

  cvode->exact_final_time = PETSC_TRUE;

  ierr = MPI_Comm_dup(ts->comm,&(cvode->comm_sundials));CHKERRQ(ierr);
  /* set tolerance for Sundials */
  cvode->abstol = 1e-6;
  cvode->reltol = 1e-6;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSundialsSetType_C","TSSundialsSetType_Sundials",
                    TSSundialsSetType_Sundials);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSundialsSetGMRESRestart_C",
                    "TSSundialsSetGMRESRestart_Sundials",
                    TSSundialsSetGMRESRestart_Sundials);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSundialsSetLinearTolerance_C",
                    "TSSundialsSetLinearTolerance_Sundials",
                     TSSundialsSetLinearTolerance_Sundials);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSundialsSetGramSchmidtType_C",
                    "TSSundialsSetGramSchmidtType_Sundials",
                     TSSundialsSetGramSchmidtType_Sundials);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSundialsSetTolerance_C",
                    "TSSundialsSetTolerance_Sundials",
                     TSSundialsSetTolerance_Sundials);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSundialsGetPC_C",
                    "TSSundialsGetPC_Sundials",
                     TSSundialsGetPC_Sundials);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSundialsGetIterations_C",
                    "TSSundialsGetIterations_Sundials",
                     TSSundialsGetIterations_Sundials);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSundialsSetExactFinalTime_C",
                    "TSSundialsSetExactFinalTime_Sundials",
                     TSSundialsSetExactFinalTime_Sundials);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END










