#define PETSCTS_DLL

/*
    Provides a PETSc interface to SUNDIALS/CVODE solver.
    The interface to PVODE (old version of CVODE) was originally contributed 
    by Liyang Xu. It has been redone by Hong Zhang and Dinesh Kaushik.

    Reference: sundials-2.4.0/examples/cvode/parallel/cvDiurnal_kry_p.c
*/
#include "sundials.h"  /*I "petscts.h" I*/

/*
      TSPrecond_Sundials - function that we provide to SUNDIALS to
                        evaluate the preconditioner.
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
  Vec            yy = cvode->w1;
  PetscScalar    one = 1.0,gm;
  MatStructure   str = DIFFERENT_NONZERO_PATTERN;
  PetscScalar    *y_data;
  
  PetscFunctionBegin;
  /* This allows us to construct preconditioners in-place if we like */
  ierr = MatSetUnfactored(Jac);CHKERRQ(ierr);
  
  /* jok - TRUE means reuse current Jacobian else recompute Jacobian */
  if (jok) {
    ierr     = MatCopy(cvode->pmat,Jac,str);CHKERRQ(ierr);
    *jcurPtr = FALSE;
  } else {
    /* make PETSc vector yy point to SUNDIALS vector y */
    y_data = (PetscScalar *) N_VGetArrayPointer(y);
    ierr   = VecPlaceArray(yy,y_data); CHKERRQ(ierr);

    /* compute the Jacobian */
    ierr = TSComputeRHSJacobian(ts,ts->ptime,yy,&Jac,&Jac,&str);CHKERRQ(ierr);
    ierr = VecResetArray(yy); CHKERRQ(ierr);

    /* copy the Jacobian matrix */
    if (!cvode->pmat) {
      ierr = MatDuplicate(Jac,MAT_COPY_VALUES,&cvode->pmat);CHKERRQ(ierr);
      ierr = PetscLogObjectParent(ts,cvode->pmat);CHKERRQ(ierr);
    } else {
      ierr = MatCopy(Jac,cvode->pmat,str);CHKERRQ(ierr);
    }
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
*/    
#undef __FUNCT__
#define __FUNCT__ "TSPSolve_Sundials"
PetscErrorCode TSPSolve_Sundials(realtype tn,N_Vector y,N_Vector fy,N_Vector r,N_Vector z,
                                 realtype _gamma,realtype delta,int lr,void *P_data,N_Vector vtemp)
{ 
  TS              ts = (TS) P_data;
  TS_Sundials     *cvode = (TS_Sundials*)ts->data;
  PC              pc = cvode->pc;
  Vec             rr = cvode->w1,zz = cvode->w2;
  PetscErrorCode  ierr;
  PetscScalar     *r_data,*z_data;

  PetscFunctionBegin;
  /* Make the PETSc work vectors rr and zz point to the arrays in the SUNDIALS vectors r and z respectively*/
  r_data  = (PetscScalar *) N_VGetArrayPointer(r);
  z_data  = (PetscScalar *) N_VGetArrayPointer(z);
  ierr = VecPlaceArray(rr,r_data); CHKERRQ(ierr);
  ierr = VecPlaceArray(zz,z_data); CHKERRQ(ierr);

  /* Solve the Px=r and put the result in zz */
  ierr = PCApply(pc,rr,zz); CHKERRQ(ierr);
  ierr = VecResetArray(rr); CHKERRQ(ierr);
  ierr = VecResetArray(zz); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
        TSFunction_Sundials - routine that we provide to Sundials that applies the right hand side.
*/  
#undef __FUNCT__  
#define __FUNCT__ "TSFunction_Sundials"
int TSFunction_Sundials(realtype t,N_Vector y,N_Vector ydot,void *ctx)
{
  TS              ts = (TS) ctx;
  MPI_Comm        comm = ((PetscObject)ts)->comm;
  TS_Sundials     *cvode = (TS_Sundials*)ts->data;
  Vec             yy = cvode->w1,yyd = cvode->w2;
  PetscScalar     *y_data,*ydot_data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /* Make the PETSc work vectors yy and yyd point to the arrays in the SUNDIALS vectors y and ydot respectively*/
  y_data     = (PetscScalar *) N_VGetArrayPointer(y);
  ydot_data  = (PetscScalar *) N_VGetArrayPointer(ydot);
  ierr = VecPlaceArray(yy,y_data);CHKERRABORT(comm,ierr);
  ierr = VecPlaceArray(yyd,ydot_data); CHKERRABORT(comm,ierr);

  /* now compute the right hand side function */
  ierr = TSComputeRHSFunction(ts,t,yy,yyd); CHKERRABORT(comm,ierr);
  ierr = VecResetArray(yy); CHKERRABORT(comm,ierr);
  ierr = VecResetArray(yyd); CHKERRABORT(comm,ierr);
  PetscFunctionReturn(0);
}

/*
       TSStep_Sundials_Nonlinear - Calls Sundials to integrate the ODE.
*/
#undef __FUNCT__  
#define __FUNCT__ "TSStep_Sundials_Nonlinear"
PetscErrorCode TSStep_Sundials_Nonlinear(TS ts,int *steps,double *time)
{
  TS_Sundials    *cvode = (TS_Sundials*)ts->data;
  Vec            sol = ts->vec_sol;
  PetscErrorCode ierr;
  PetscInt       i,max_steps = ts->max_steps,flag;
  long int       its;
  realtype       t,tout;
  PetscScalar    *y_data;
  void           *mem;
 
  PetscFunctionBegin;
  mem  = cvode->mem;
  tout = ts->max_time;
  ierr = VecGetArray(ts->vec_sol,&y_data);CHKERRQ(ierr);
  N_VSetArrayPointer((realtype *)y_data,cvode->y);
  ierr = VecRestoreArray(ts->vec_sol,PETSC_NULL);CHKERRQ(ierr);
  for (i = 0; i < max_steps; i++) {
    if (ts->ptime >= ts->max_time) break;
    ierr = TSPreStep(ts);CHKERRQ(ierr);
    if (cvode->monitorstep){
      flag = CVode(mem,tout,cvode->y,&t,CV_ONE_STEP);
    } else {
      flag = CVode(mem,tout,cvode->y,&t,CV_NORMAL);
    }
    if (flag)SETERRQ1(PETSC_ERR_LIB,"CVode() fails, flag %d",flag);
    if (t > ts->max_time && cvode->exact_final_time) { 
      /* interpolate to final requested time */
      ierr = CVodeGetDky(mem,tout,0,cvode->y);CHKERRQ(ierr);
      t = tout;
    }
    ts->time_step = t - ts->ptime;
    ts->ptime     = t; 

    /* copy the solution from cvode->y to cvode->update and sol */
    ierr = VecPlaceArray(cvode->w1,y_data); CHKERRQ(ierr);
    ierr = VecCopy(cvode->w1,cvode->update);CHKERRQ(ierr);
    ierr = VecResetArray(cvode->w1); CHKERRQ(ierr);
    ierr = VecCopy(cvode->update,sol);CHKERRQ(ierr);
    ierr = CVodeGetNumNonlinSolvIters(mem,&its);CHKERRQ(ierr);
    ts->nonlinear_its = its;
    ierr = CVSpilsGetNumLinIters(mem, &its);
    ts->linear_its = its; 
    ts->steps++;
    ierr = TSPostStep(ts);CHKERRQ(ierr);
    ierr = TSMonitor(ts,ts->steps,t,sol);CHKERRQ(ierr); 
  }
  *steps += ts->steps;
  *time   = t;
  PetscFunctionReturn(0);
}

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
  if (cvode->mem) {CVodeFree(&cvode->mem);}
  ierr = PetscFree(cvode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSSetUp_Sundials_Nonlinear"
PetscErrorCode TSSetUp_Sundials_Nonlinear(TS ts)
{
  TS_Sundials    *cvode = (TS_Sundials*)ts->data;
  PetscErrorCode ierr;
  PetscInt       glosize,locsize,i,flag;
  PetscScalar    *y_data,*parray;
  void           *mem;
  const PCType   pctype;
  PetscTruth     pcnone;
  Vec            sol = ts->vec_sol;

  PetscFunctionBegin;
  ierr = PCSetFromOptions(cvode->pc);CHKERRQ(ierr);
  /* get the vector size */
  ierr = VecGetSize(ts->vec_sol,&glosize);CHKERRQ(ierr);
  ierr = VecGetLocalSize(ts->vec_sol,&locsize);CHKERRQ(ierr);

  /* allocate the memory for N_Vec y */
  cvode->y = N_VNew_Parallel(cvode->comm_sundials,locsize,glosize);
  if (!cvode->y) SETERRQ(1,"cvode->y is not allocated");

  /* initialize N_Vec y: copy ts->vec_sol to cvode->y */
  ierr = VecGetArray(ts->vec_sol,&parray);CHKERRQ(ierr);
  y_data = (PetscScalar *) N_VGetArrayPointer(cvode->y);
  for (i = 0; i < locsize; i++) y_data[i] = parray[i];
  /*ierr = PetscMemcpy(y_data,parray,locsize*sizeof(PETSC_SCALAR)); CHKERRQ(ierr);*/
  ierr = VecRestoreArray(ts->vec_sol,PETSC_NULL);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&cvode->update);CHKERRQ(ierr);  
  ierr = VecDuplicate(ts->vec_sol,&cvode->func);CHKERRQ(ierr);  
  ierr = PetscLogObjectParent(ts,cvode->update);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(ts,cvode->func);CHKERRQ(ierr);

  /* 
    Create work vectors for the TSPSolve_Sundials() routine. Note these are
    allocated with zero space arrays because the actual array space is provided 
    by Sundials and set using VecPlaceArray().
  */
  ierr = VecCreateMPIWithArray(((PetscObject)ts)->comm,locsize,PETSC_DECIDE,0,&cvode->w1);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(((PetscObject)ts)->comm,locsize,PETSC_DECIDE,0,&cvode->w2);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(ts,cvode->w1);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(ts,cvode->w2);CHKERRQ(ierr);

  /* Call CVodeCreate to create the solver memory and the use of a Newton iteration */
  mem = CVodeCreate(cvode->cvode_type, CV_NEWTON); 
  if (!mem) SETERRQ(PETSC_ERR_MEM,"CVodeCreate() fails");
  cvode->mem = mem;

  /* Set the pointer to user-defined data */
  flag = CVodeSetUserData(mem, ts);
  if (flag) SETERRQ(PETSC_ERR_LIB,"CVodeSetUserData() fails");

  /* Call CVodeInit to initialize the integrator memory and specify the
   * user's right hand side function in u'=f(t,u), the inital time T0, and
   * the initial dependent variable vector cvode->y */
  flag = CVodeInit(mem,TSFunction_Sundials,ts->ptime,cvode->y);
  if (flag){
    SETERRQ1(PETSC_ERR_LIB,"CVodeInit() fails, flag %d",flag);
  }

  flag = CVodeSStolerances(mem,cvode->reltol,cvode->abstol);
  if (flag){
    SETERRQ1(PETSC_ERR_LIB,"CVodeSStolerances() fails, flag %d",flag);
  }

  /* initialize the number of steps */
  ierr   = TSMonitor(ts,ts->steps,ts->ptime,sol);CHKERRQ(ierr); 

  /* call CVSpgmr to use GMRES as the linear solver.        */
  /* setup the ode integrator with the given preconditioner */
  ierr = PCGetType(cvode->pc,&pctype);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)cvode->pc,PCNONE,&pcnone);CHKERRQ(ierr);
  if (pcnone){
    flag  = CVSpgmr(mem,PREC_NONE,0);
    if (flag) SETERRQ1(PETSC_ERR_LIB,"CVSpgmr() fails, flag %d",flag);
  } else {
    flag  = CVSpgmr(mem,PREC_LEFT,0);
    if (flag) SETERRQ1(PETSC_ERR_LIB,"CVSpgmr() fails, flag %d",flag);

    /* Set preconditioner and solve routines Precond and PSolve, 
     and the pointer to the user-defined block data */
    flag = CVSpilsSetPreconditioner(mem,TSPrecond_Sundials,TSPSolve_Sundials);
    if (flag) SETERRQ1(PETSC_ERR_LIB,"CVSpilsSetPreconditioner() fails, flag %d", flag);
  }

  flag = CVSpilsSetGSType(mem, MODIFIED_GS);
  if (flag) SETERRQ1(PETSC_ERR_LIB,"CVSpgmrSetGSType() fails, flag %d",flag);
  PetscFunctionReturn(0);
}

/* type of CVODE linear multistep method */
const char *TSSundialsLmmTypes[] = {"","adams","bdf","TSSundialsLmmType","SUNDIALS_",0};
/* type of G-S orthogonalization used by CVODE linear solver */
const char *TSSundialsGramSchmidtTypes[] = {"","modified","classical","TSSundialsGramSchmidtType","SUNDIALS_",0};

#undef __FUNCT__  
#define __FUNCT__ "TSSetFromOptions_Sundials_Nonlinear"
PetscErrorCode TSSetFromOptions_Sundials_Nonlinear(TS ts)
{
  TS_Sundials    *cvode = (TS_Sundials*)ts->data;
  PetscErrorCode ierr;
  int            indx;
  PetscTruth     flag;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("SUNDIALS ODE solver options");CHKERRQ(ierr);
    ierr = PetscOptionsEList("-ts_sundials_type","Scheme","TSSundialsSetType",TSSundialsLmmTypes,3,TSSundialsLmmTypes[cvode->cvode_type],&indx,&flag);CHKERRQ(ierr);
    if (flag) {
      ierr = TSSundialsSetType(ts,(TSSundialsLmmType)indx);CHKERRQ(ierr);
    }
    ierr = PetscOptionsEList("-ts_sundials_gramschmidt_type","Type of orthogonalization","TSSundialsSetGramSchmidtType",TSSundialsGramSchmidtTypes,3,TSSundialsGramSchmidtTypes[cvode->gtype],&indx,&flag);CHKERRQ(ierr);
    if (flag) {
      ierr = TSSundialsSetGramSchmidtType(ts,(TSSundialsGramSchmidtType)indx);CHKERRQ(ierr);
    }
    ierr = PetscOptionsReal("-ts_sundials_atol","Absolute tolerance for convergence","TSSundialsSetTolerance",cvode->abstol,&cvode->abstol,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ts_sundials_rtol","Relative tolerance for convergence","TSSundialsSetTolerance",cvode->reltol,&cvode->reltol,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ts_sundials_linear_tolerance","Convergence tolerance for linear solve","TSSundialsSetLinearTolerance",cvode->linear_tol,&cvode->linear_tol,&flag);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-ts_sundials_gmres_restart","Number of GMRES orthogonalization directions","TSSundialsSetGMRESRestart",cvode->restart,&cvode->restart,&flag);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-ts_sundials_exact_final_time","Interpolate output to stop exactly at the final time","TSSundialsSetExactFinalTime",cvode->exact_final_time,&cvode->exact_final_time,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-ts_sundials_monitor_steps","Monitor SUNDIALS internel steps","TSSundialsMonitorInternalSteps",cvode->monitorstep,&cvode->monitorstep,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSView_Sundials" 
PetscErrorCode TSView_Sundials(TS ts,PetscViewer viewer)
{
  TS_Sundials    *cvode = (TS_Sundials*)ts->data;
  PetscErrorCode ierr;
  char           *type;
  char           atype[] = "Adams";
  char           btype[] = "BDF: backward differentiation formula";
  PetscTruth     iascii,isstring;
  long int       nsteps,its,nfevals,nlinsetups,nfails,itmp;
  PetscInt       qlast,qcur;
  PetscReal      hinused,hlast,hcur,tcur,tolsfac;

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
    
    /* Outputs from CVODE, CVSPILS */
    ierr = CVodeGetTolScaleFactor(cvode->mem,&tolsfac);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Sundials suggested factor for tolerance scaling %g\n",tolsfac);CHKERRQ(ierr);
    ierr = CVodeGetIntegratorStats(cvode->mem,&nsteps,&nfevals,
                                   &nlinsetups,&nfails,&qlast,&qcur,
                                   &hinused,&hlast,&hcur,&tcur);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Sundials cumulative number of internal steps %D\n",nsteps);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Sundials no. of calls to rhs function %D\n",nfevals);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Sundials no. of calls to linear solver setup function %D\n",nlinsetups);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Sundials no. of error test failures %D\n",nfails);CHKERRQ(ierr);

    ierr = CVodeGetNonlinSolvStats(cvode->mem,&its,&nfails);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Sundials no. of nonlinear solver iterations %D\n",its);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Sundials no. of nonlinear convergence failure %D\n",nfails);CHKERRQ(ierr);

    ierr = CVSpilsGetNumLinIters(cvode->mem, &its);CHKERRQ(ierr); /* its = no. of calls to TSPrecond_Sundials() */
    ierr = PetscViewerASCIIPrintf(viewer,"Sundials no. of linear iterations %D\n",its);CHKERRQ(ierr);
    ierr = CVSpilsGetNumConvFails(cvode->mem,&itmp);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Sundials no. of linear convergence failures %D\n",itmp);CHKERRQ(ierr);
    ierr = CVSpilsGetNumPrecEvals(cvode->mem,&itmp);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Sundials no. of preconditioner evaluations %D\n",itmp);CHKERRQ(ierr);
    ierr = CVSpilsGetNumPrecSolves(cvode->mem,&itmp);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Sundials no. of preconditioner solves %D\n",itmp);CHKERRQ(ierr);
    ierr = CVSpilsGetNumJtimesEvals(cvode->mem,&itmp);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Sundials no. of Jacobian-vector product evaluations %D\n",itmp);CHKERRQ(ierr);
    ierr = CVSpilsGetNumRhsEvals(cvode->mem,&itmp);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Sundials no. of rhs calls for finite diff. Jacobian-vector evals %D\n",itmp);CHKERRQ(ierr);
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
PetscErrorCode PETSCTS_DLLEXPORT TSSundialsSetType_Sundials(TS ts,TSSundialsLmmType type)
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
  PetscFunctionBegin;
  if (nonlin) *nonlin = ts->nonlinear_its;
  if (lin)    *lin    = ts->linear_its;
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

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "TSSundialsMonitorInternalSteps_Sundials"
PetscErrorCode PETSCTS_DLLEXPORT TSSundialsMonitorInternalSteps_Sundials(TS ts,PetscTruth s)
{
  TS_Sundials *cvode = (TS_Sundials*)ts->data;
  
  PetscFunctionBegin;
  cvode->monitorstep = s;
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

   Level: intermediate

.keywords: Adams, backward differentiation formula

.seealso: TSSundialsGetIterations(),  TSSundialsSetGMRESRestart(),
          TSSundialsSetLinearTolerance(), TSSundialsSetGramSchmidtType(), TSSundialsSetTolerance(),
          TSSundialsGetIterations(), TSSundialsSetType(), TSSundialsSetGMRESRestart(),
          TSSundialsSetLinearTolerance(), TSSundialsSetTolerance(), TSSundialsGetPC(),
          TSSundialsSetExactFinalTime()
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSSundialsSetType(TS ts,TSSundialsLmmType type)
{
  PetscErrorCode ierr,(*f)(TS,TSSundialsLmmType);
  
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

#undef __FUNCT__  
#define __FUNCT__ "TSSundialsMonitorInternalSteps"
/*@
   TSSundialsMonitorInternalSteps - Monitor Sundials internal steps (Defaults to false).

   Input Parameter:
+   ts - the time-step context
-   ft - PETSC_TRUE if monitor, else PETSC_FALSE

   Level: beginner

.seealso:TSSundialsGetIterations(), TSSundialsSetType(), TSSundialsSetGMRESRestart(),
          TSSundialsSetLinearTolerance(), TSSundialsSetGramSchmidtType(), TSSundialsSetTolerance(),
          TSSundialsGetIterations(), TSSundialsSetType(), TSSundialsSetGMRESRestart(),
          TSSundialsSetLinearTolerance(), TSSundialsSetTolerance(), TSSundialsGetPC() 
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSSundialsMonitorInternalSteps(TS ts,PetscTruth ft)
{ 
  PetscErrorCode ierr,(*f)(TS,PetscTruth);  

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)ts,"TSSundialsMonitorInternalSteps_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ts,ft);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------------------------*/
/*MC
      TSSUNDIALS - ODE solver using the LLNL CVODE/SUNDIALS package (now called SUNDIALS)

   Options Database:
+    -ts_sundials_type <bdf,adams>
.    -ts_sundials_gramschmidt_type <modified, classical> - type of orthogonalization inside GMRES
.    -ts_sundials_atol <tol> - Absolute tolerance for convergence
.    -ts_sundials_rtol <tol> - Relative tolerance for convergence
.    -ts_sundials_linear_tolerance <tol> 
.    -ts_sundials_gmres_restart <restart> - Number of GMRES orthogonalization directions
.    -ts_sundials_exact_final_time - Interpolate output to stop exactly at the final time
-    -ts_sundials_monitor_steps - Monitor SUNDIALS internel steps


    Notes: This uses its own nonlinear solver and Krylov method so PETSc SNES and KSP options do not apply
           only PETSc PC options

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
  if (ts->problem_type != TS_NONLINEAR) {
    SETERRQ(PETSC_ERR_SUP,"Only support for nonlinear problems");
  }
  ts->ops->destroy        = TSDestroy_Sundials;
  ts->ops->view           = TSView_Sundials;
  ts->ops->setup          = TSSetUp_Sundials_Nonlinear;  
  ts->ops->step           = TSStep_Sundials_Nonlinear;
  ts->ops->setfromoptions = TSSetFromOptions_Sundials_Nonlinear;

  ierr = PetscNewLog(ts,TS_Sundials,&cvode);CHKERRQ(ierr);
  ierr = PCCreate(((PetscObject)ts)->comm,&cvode->pc);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(ts,cvode->pc);CHKERRQ(ierr);
  ts->data          = (void*)cvode;
  cvode->cvode_type = SUNDIALS_BDF;
  cvode->gtype      = SUNDIALS_CLASSICAL_GS;
  cvode->restart    = 5;
  cvode->linear_tol = .05;

  cvode->exact_final_time = PETSC_TRUE;
  cvode->monitorstep      = PETSC_FALSE;

  ierr = MPI_Comm_dup(((PetscObject)ts)->comm,&(cvode->comm_sundials));CHKERRQ(ierr);
  /* set tolerance for Sundials */
  cvode->reltol = 1e-6;
  cvode->abstol = 1e-6;

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
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSundialsMonitorInternalSteps_C",
                    "TSSundialsMonitorInternalSteps_Sundials",
                     TSSundialsMonitorInternalSteps_Sundials);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END










