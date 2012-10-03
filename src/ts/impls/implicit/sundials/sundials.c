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
  PC             pc;
  PetscErrorCode ierr;
  Mat            J,P;
  Vec            yy = cvode->w1,yydot = cvode->ydot;
  PetscReal      gm = (PetscReal)_gamma;
  MatStructure   str = DIFFERENT_NONZERO_PATTERN;
  PetscScalar    *y_data;

  PetscFunctionBegin;
  ierr = TSGetIJacobian(ts,&J,&P,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  y_data = (PetscScalar *) N_VGetArrayPointer(y);
  ierr = VecPlaceArray(yy,y_data); CHKERRQ(ierr);
  ierr = VecZeroEntries(yydot);CHKERRQ(ierr); /* The Jacobian is independent of Ydot for ODE which is all that CVode works for */
  /* compute the shifted Jacobian   (1/gm)*I + Jrest */
  ierr = TSComputeIJacobian(ts,ts->ptime,yy,yydot,1/gm,&J,&P,&str,PETSC_FALSE);CHKERRQ(ierr);
  ierr = VecResetArray(yy); CHKERRQ(ierr);
  ierr = MatScale(P,gm);CHKERRQ(ierr);  /* turn into I-gm*Jrest, J is not used by Sundials  */
  *jcurPtr = TRUE;
  ierr = TSSundialsGetPC(ts,&pc); CHKERRQ(ierr);
  ierr = PCSetOperators(pc,J,P,str);CHKERRQ(ierr);
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
  PC              pc;
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
  ierr = TSSundialsGetPC(ts,&pc); CHKERRQ(ierr);
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
  DM              dm;
  TSDM            tsdm;
  TSIFunction     ifunction;
  MPI_Comm        comm = ((PetscObject)ts)->comm;
  TS_Sundials     *cvode = (TS_Sundials*)ts->data;
  Vec             yy = cvode->w1,yyd = cvode->w2,yydot = cvode->ydot;
  PetscScalar     *y_data,*ydot_data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /* Make the PETSc work vectors yy and yyd point to the arrays in the SUNDIALS vectors y and ydot respectively*/
  y_data     = (PetscScalar *) N_VGetArrayPointer(y);
  ydot_data  = (PetscScalar *) N_VGetArrayPointer(ydot);
  ierr = VecPlaceArray(yy,y_data);CHKERRABORT(comm,ierr);
  ierr = VecPlaceArray(yyd,ydot_data); CHKERRABORT(comm,ierr);

  /* Now compute the right hand side function, via IFunction unless only the more efficient RHSFunction is set */
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMTSGetContext(dm,&tsdm);CHKERRQ(ierr);
  ierr = DMTSGetIFunction(dm,&ifunction,PETSC_NULL);CHKERRQ(ierr);
  if (!ifunction) {
    ierr = TSComputeRHSFunction(ts,t,yy,yyd);CHKERRQ(ierr);
  } else {                      /* If rhsfunction is also set, this computes both parts and shifts them to the right */
    ierr = VecZeroEntries(yydot);CHKERRQ(ierr);
    ierr = TSComputeIFunction(ts,t,yy,yydot,yyd,PETSC_FALSE); CHKERRABORT(comm,ierr);
    ierr = VecScale(yyd,-1.);CHKERRQ(ierr);
  }
  ierr = VecResetArray(yy); CHKERRABORT(comm,ierr);
  ierr = VecResetArray(yyd); CHKERRABORT(comm,ierr);
  PetscFunctionReturn(0);
}

/*
       TSStep_Sundials - Calls Sundials to integrate the ODE.
*/
#undef __FUNCT__
#define __FUNCT__ "TSStep_Sundials"
PetscErrorCode TSStep_Sundials(TS ts)
{
  TS_Sundials    *cvode = (TS_Sundials*)ts->data;
  PetscErrorCode ierr;
  PetscInt       flag;
  long int       its,nsteps;
  realtype       t,tout;
  PetscScalar    *y_data;
  void           *mem;

  PetscFunctionBegin;
  mem  = cvode->mem;
  tout = ts->max_time;
  ierr = VecGetArray(ts->vec_sol,&y_data);CHKERRQ(ierr);
  N_VSetArrayPointer((realtype *)y_data,cvode->y);
  ierr = VecRestoreArray(ts->vec_sol,PETSC_NULL);CHKERRQ(ierr);

  ierr = TSPreStep(ts);CHKERRQ(ierr);

  if (cvode->monitorstep) {
    /* We would like to call TSPreStep() when starting each step (including rejections) and TSPreStage() before each
     * stage solve, but CVode does not appear to support this. */
    flag = CVode(mem,tout,cvode->y,&t,CV_ONE_STEP);
  } else {
    flag = CVode(mem,tout,cvode->y,&t,CV_NORMAL);
  }

  if (flag){ /* display error message */
    switch (flag){
      case CV_ILL_INPUT:
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"CVode() fails, CV_ILL_INPUT");
        break;
      case CV_TOO_CLOSE:
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"CVode() fails, CV_TOO_CLOSE");
        break;
      case CV_TOO_MUCH_WORK: {
        PetscReal      tcur;
        ierr = CVodeGetNumSteps(mem,&nsteps);CHKERRQ(ierr);
        ierr = CVodeGetCurrentTime(mem,&tcur);CHKERRQ(ierr);
        SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_LIB,"CVode() fails, CV_TOO_MUCH_WORK. At t=%G, nsteps %D exceeds mxstep %D. Increase '-ts_max_steps <>' or modify TSSetDuration()",tcur,nsteps,ts->max_steps);
      } break;
      case CV_TOO_MUCH_ACC:
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"CVode() fails, CV_TOO_MUCH_ACC");
        break;
      case CV_ERR_FAILURE:
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"CVode() fails, CV_ERR_FAILURE");
        break;
      case CV_CONV_FAILURE:
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"CVode() fails, CV_CONV_FAILURE");
        break;
      case CV_LINIT_FAIL:
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"CVode() fails, CV_LINIT_FAIL");
        break;
      case CV_LSETUP_FAIL:
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"CVode() fails, CV_LSETUP_FAIL");
        break;
      case CV_LSOLVE_FAIL:
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"CVode() fails, CV_LSOLVE_FAIL");
        break;
      case CV_RHSFUNC_FAIL:
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"CVode() fails, CV_RHSFUNC_FAIL");
        break;
      case CV_FIRST_RHSFUNC_ERR:
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"CVode() fails, CV_FIRST_RHSFUNC_ERR");
        break;
      case CV_REPTD_RHSFUNC_ERR:
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"CVode() fails, CV_REPTD_RHSFUNC_ERR");
        break;
      case CV_UNREC_RHSFUNC_ERR:
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"CVode() fails, CV_UNREC_RHSFUNC_ERR");
        break;
      case CV_RTFUNC_FAIL:
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"CVode() fails, CV_RTFUNC_FAIL");
        break;
      default:
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CVode() fails, flag %d",flag);
    }
  }

  /* copy the solution from cvode->y to cvode->update and sol */
  ierr = VecPlaceArray(cvode->w1,y_data); CHKERRQ(ierr);
  ierr = VecCopy(cvode->w1,cvode->update);CHKERRQ(ierr);
  ierr = VecResetArray(cvode->w1); CHKERRQ(ierr);
  ierr = VecCopy(cvode->update,ts->vec_sol);CHKERRQ(ierr);
  ierr = CVodeGetNumNonlinSolvIters(mem,&its);CHKERRQ(ierr);
  ierr = CVSpilsGetNumLinIters(mem, &its);
  ts->snes_its = its; ts->ksp_its = its;

  ts->time_step = t - ts->ptime;
  ts->ptime     = t;
  ts->steps++;

  ierr = CVodeGetNumSteps(mem,&nsteps);CHKERRQ(ierr);
  if (!cvode->monitorstep) ts->steps = nsteps;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSInterpolate_Sundials"
static PetscErrorCode TSInterpolate_Sundials(TS ts,PetscReal t,Vec X)
{
  TS_Sundials     *cvode = (TS_Sundials*)ts->data;
  N_Vector        y;
  PetscErrorCode  ierr;
  PetscScalar     *x_data;
  PetscInt        glosize,locsize;

  PetscFunctionBegin;

  /* get the vector size */
  ierr = VecGetSize(X,&glosize);CHKERRQ(ierr);
  ierr = VecGetLocalSize(X,&locsize);CHKERRQ(ierr);

  /* allocate the memory for N_Vec y */
  y = N_VNew_Parallel(cvode->comm_sundials,locsize,glosize);
  if (!y) SETERRQ(PETSC_COMM_SELF,1,"Interpolated y is not allocated");

  ierr = VecGetArray(X,&x_data);CHKERRQ(ierr);
  N_VSetArrayPointer((realtype *)x_data,y);
  ierr = CVodeGetDky(cvode->mem,t,0,y);CHKERRQ(ierr);
  ierr = VecRestoreArray(X,&x_data);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSReset_Sundials"
PetscErrorCode TSReset_Sundials(TS ts)
{
  TS_Sundials    *cvode = (TS_Sundials*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&cvode->update);CHKERRQ(ierr);
  ierr = VecDestroy(&cvode->ydot);CHKERRQ(ierr);
  ierr = VecDestroy(&cvode->w1);CHKERRQ(ierr);
  ierr = VecDestroy(&cvode->w2);CHKERRQ(ierr);
  if (cvode->mem)    {CVodeFree(&cvode->mem);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSDestroy_Sundials"
PetscErrorCode TSDestroy_Sundials(TS ts)
{
  TS_Sundials    *cvode = (TS_Sundials*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSReset_Sundials(ts);CHKERRQ(ierr);
  ierr = MPI_Comm_free(&(cvode->comm_sundials));CHKERRQ(ierr);
  ierr = PetscFree(ts->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSundialsSetType_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSundialsSetMaxl_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSundialsSetLinearTolerance_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSundialsSetGramSchmidtType_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSundialsSetTolerance_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSundialsSetMinTimeStep_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSundialsSetMaxTimeStep_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSundialsGetPC_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSundialsGetIterations_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSundialsMonitorInternalSteps_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetUp_Sundials"
PetscErrorCode TSSetUp_Sundials(TS ts)
{
  TS_Sundials    *cvode = (TS_Sundials*)ts->data;
  PetscErrorCode ierr;
  PetscInt       glosize,locsize,i,flag;
  PetscScalar    *y_data,*parray;
  void           *mem;
  PC             pc;
  PCType         pctype;
  PetscBool      pcnone;

  PetscFunctionBegin;
  /* get the vector size */
  ierr = VecGetSize(ts->vec_sol,&glosize);CHKERRQ(ierr);
  ierr = VecGetLocalSize(ts->vec_sol,&locsize);CHKERRQ(ierr);

  /* allocate the memory for N_Vec y */
  cvode->y = N_VNew_Parallel(cvode->comm_sundials,locsize,glosize);
  if (!cvode->y) SETERRQ(PETSC_COMM_SELF,1,"cvode->y is not allocated");

  /* initialize N_Vec y: copy ts->vec_sol to cvode->y */
  ierr = VecGetArray(ts->vec_sol,&parray);CHKERRQ(ierr);
  y_data = (PetscScalar *) N_VGetArrayPointer(cvode->y);
  for (i = 0; i < locsize; i++) y_data[i] = parray[i];
  ierr = VecRestoreArray(ts->vec_sol,PETSC_NULL);CHKERRQ(ierr);

  ierr = VecDuplicate(ts->vec_sol,&cvode->update);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&cvode->ydot);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(ts,cvode->update);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(ts,cvode->ydot);CHKERRQ(ierr);

  /*
    Create work vectors for the TSPSolve_Sundials() routine. Note these are
    allocated with zero space arrays because the actual array space is provided
    by Sundials and set using VecPlaceArray().
  */
  ierr = VecCreateMPIWithArray(((PetscObject)ts)->comm,1,locsize,PETSC_DECIDE,0,&cvode->w1);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(((PetscObject)ts)->comm,1,locsize,PETSC_DECIDE,0,&cvode->w2);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(ts,cvode->w1);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(ts,cvode->w2);CHKERRQ(ierr);

  /* Call CVodeCreate to create the solver memory and the use of a Newton iteration */
  mem = CVodeCreate(cvode->cvode_type, CV_NEWTON);
  if (!mem) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"CVodeCreate() fails");
  cvode->mem = mem;

  /* Set the pointer to user-defined data */
  flag = CVodeSetUserData(mem, ts);
  if (flag) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"CVodeSetUserData() fails");

  /* Sundials may choose to use a smaller initial step, but will never use a larger step. */
  flag = CVodeSetInitStep(mem,(realtype)ts->time_step);
  if (flag) SETERRQ(((PetscObject)ts)->comm,PETSC_ERR_LIB,"CVodeSetInitStep() failed");
  if (cvode->mindt > 0) {
    flag = CVodeSetMinStep(mem,(realtype)cvode->mindt);
    if (flag){
      if (flag == CV_MEM_NULL){
        SETERRQ(((PetscObject)ts)->comm,PETSC_ERR_LIB,"CVodeSetMinStep() failed, cvode_mem pointer is NULL");
      } else if (flag == CV_ILL_INPUT){
        SETERRQ(((PetscObject)ts)->comm,PETSC_ERR_LIB,"CVodeSetMinStep() failed, hmin is nonpositive or it exceeds the maximum allowable step size");
      } else {
        SETERRQ(((PetscObject)ts)->comm,PETSC_ERR_LIB,"CVodeSetMinStep() failed");
      }
    }
  }
  if (cvode->maxdt > 0) {
    flag = CVodeSetMaxStep(mem,(realtype)cvode->maxdt);
    if (flag) SETERRQ(((PetscObject)ts)->comm,PETSC_ERR_LIB,"CVodeSetMaxStep() failed");
  }

  /* Call CVodeInit to initialize the integrator memory and specify the
   * user's right hand side function in u'=f(t,u), the inital time T0, and
   * the initial dependent variable vector cvode->y */
  flag = CVodeInit(mem,TSFunction_Sundials,ts->ptime,cvode->y);
  if (flag){
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CVodeInit() fails, flag %d",flag);
  }

  /* specifies scalar relative and absolute tolerances */
  flag = CVodeSStolerances(mem,cvode->reltol,cvode->abstol);
  if (flag){
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CVodeSStolerances() fails, flag %d",flag);
  }

  /* Specify max num of steps to be taken by cvode in its attempt to reach the next output time */
  flag = CVodeSetMaxNumSteps(mem,ts->max_steps);

  /* call CVSpgmr to use GMRES as the linear solver.        */
  /* setup the ode integrator with the given preconditioner */
  ierr = TSSundialsGetPC(ts,&pc); CHKERRQ(ierr);
  ierr = PCGetType(pc,&pctype);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)pc,PCNONE,&pcnone);CHKERRQ(ierr);
  if (pcnone){
    flag  = CVSpgmr(mem,PREC_NONE,0);
    if (flag) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CVSpgmr() fails, flag %d",flag);
  } else {
    flag  = CVSpgmr(mem,PREC_LEFT,cvode->maxl);
    if (flag) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CVSpgmr() fails, flag %d",flag);

    /* Set preconditioner and solve routines Precond and PSolve,
     and the pointer to the user-defined block data */
    flag = CVSpilsSetPreconditioner(mem,TSPrecond_Sundials,TSPSolve_Sundials);
    if (flag) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CVSpilsSetPreconditioner() fails, flag %d", flag);
  }

  flag = CVSpilsSetGSType(mem, MODIFIED_GS);
  if (flag) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CVSpgmrSetGSType() fails, flag %d",flag);
  PetscFunctionReturn(0);
}

/* type of CVODE linear multistep method */
const char *const TSSundialsLmmTypes[] = {"","ADAMS","BDF","TSSundialsLmmType","SUNDIALS_",0};
/* type of G-S orthogonalization used by CVODE linear solver */
const char *const TSSundialsGramSchmidtTypes[] = {"","MODIFIED","CLASSICAL","TSSundialsGramSchmidtType","SUNDIALS_",0};

#undef __FUNCT__
#define __FUNCT__ "TSSetFromOptions_Sundials"
PetscErrorCode TSSetFromOptions_Sundials(TS ts)
{
  TS_Sundials    *cvode = (TS_Sundials*)ts->data;
  PetscErrorCode ierr;
  int            indx;
  PetscBool      flag;
  PC             pc;

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
    ierr = PetscOptionsReal("-ts_sundials_mindt","Minimum step size","TSSundialsSetMinTimeStep",cvode->mindt,&cvode->mindt,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ts_sundials_maxdt","Maximum step size","TSSundialsSetMaxTimeStep",cvode->maxdt,&cvode->maxdt,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ts_sundials_linear_tolerance","Convergence tolerance for linear solve","TSSundialsSetLinearTolerance",cvode->linear_tol,&cvode->linear_tol,&flag);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-ts_sundials_maxl","Max dimension of the Krylov subspace","TSSundialsSetMaxl",cvode->maxl,&cvode->maxl,&flag);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-ts_sundials_monitor_steps","Monitor SUNDIALS internel steps","TSSundialsMonitorInternalSteps",cvode->monitorstep,&cvode->monitorstep,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  ierr = TSSundialsGetPC(ts,&pc);CHKERRQ(ierr);
  ierr = PCSetFromOptions(pc);CHKERRQ(ierr);
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
  PetscBool      iascii,isstring;
  long int       nsteps,its,nfevals,nlinsetups,nfails,itmp;
  PetscInt       qlast,qcur;
  PetscReal      hinused,hlast,hcur,tcur,tolsfac;
  PC             pc;

  PetscFunctionBegin;
  if (cvode->cvode_type == SUNDIALS_ADAMS) {type = atype;}
  else                                     {type = btype;}

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"Sundials integrater does not use SNES!\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Sundials integrater type %s\n",type);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Sundials abs tol %g rel tol %g\n",cvode->abstol,cvode->reltol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Sundials linear solver tolerance factor %g\n",cvode->linear_tol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Sundials max dimension of Krylov subspace %D\n",cvode->maxl);CHKERRQ(ierr);
    if (cvode->gtype == SUNDIALS_MODIFIED_GS) {
      ierr = PetscViewerASCIIPrintf(viewer,"Sundials using modified Gram-Schmidt for orthogonalization in GMRES\n");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"Sundials using unmodified (classical) Gram-Schmidt for orthogonalization in GMRES\n");CHKERRQ(ierr);
    }
    if (cvode->mindt > 0) {ierr = PetscViewerASCIIPrintf(viewer,"Sundials minimum time step %g\n",cvode->mindt);CHKERRQ(ierr);}
    if (cvode->maxdt > 0) {ierr = PetscViewerASCIIPrintf(viewer,"Sundials maximum time step %g\n",cvode->maxdt);CHKERRQ(ierr);}

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

    ierr = TSSundialsGetPC(ts,&pc); CHKERRQ(ierr);
    ierr = PCView(pc,viewer);CHKERRQ(ierr);
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
  }
  PetscFunctionReturn(0);
}


/* --------------------------------------------------------------------------*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "TSSundialsSetType_Sundials"
PetscErrorCode  TSSundialsSetType_Sundials(TS ts,TSSundialsLmmType type)
{
  TS_Sundials *cvode = (TS_Sundials*)ts->data;

  PetscFunctionBegin;
  cvode->cvode_type = type;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "TSSundialsSetMaxl_Sundials"
PetscErrorCode  TSSundialsSetMaxl_Sundials(TS ts,PetscInt maxl)
{
  TS_Sundials *cvode = (TS_Sundials*)ts->data;

  PetscFunctionBegin;
  cvode->maxl = maxl;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "TSSundialsSetLinearTolerance_Sundials"
PetscErrorCode  TSSundialsSetLinearTolerance_Sundials(TS ts,double tol)
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
PetscErrorCode  TSSundialsSetGramSchmidtType_Sundials(TS ts,TSSundialsGramSchmidtType type)
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
PetscErrorCode  TSSundialsSetTolerance_Sundials(TS ts,double aabs,double rel)
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
#define __FUNCT__ "TSSundialsSetMinTimeStep_Sundials"
PetscErrorCode  TSSundialsSetMinTimeStep_Sundials(TS ts,PetscReal mindt)
{
  TS_Sundials *cvode = (TS_Sundials*)ts->data;

  PetscFunctionBegin;
  cvode->mindt = mindt;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "TSSundialsSetMaxTimeStep_Sundials"
PetscErrorCode  TSSundialsSetMaxTimeStep_Sundials(TS ts,PetscReal maxdt)
{
  TS_Sundials *cvode = (TS_Sundials*)ts->data;

  PetscFunctionBegin;
  cvode->maxdt = maxdt;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "TSSundialsGetPC_Sundials"
PetscErrorCode  TSSundialsGetPC_Sundials(TS ts,PC *pc)
{
  SNES            snes;
  KSP             ksp;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,pc); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "TSSundialsGetIterations_Sundials"
PetscErrorCode  TSSundialsGetIterations_Sundials(TS ts,int *nonlin,int *lin)
{
  PetscFunctionBegin;
  if (nonlin) *nonlin = ts->snes_its;
  if (lin)    *lin    = ts->ksp_its;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "TSSundialsMonitorInternalSteps_Sundials"
PetscErrorCode  TSSundialsMonitorInternalSteps_Sundials(TS ts,PetscBool  s)
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

.seealso: TSSundialsSetType(), TSSundialsSetMaxl(),
          TSSundialsSetLinearTolerance(), TSSundialsSetGramSchmidtType(), TSSundialsSetTolerance(),
          TSSundialsGetIterations(), TSSundialsSetType(),
          TSSundialsSetLinearTolerance(), TSSundialsGetPC(), TSSetExactFinalTime()

@*/
PetscErrorCode  TSSundialsGetIterations(TS ts,int *nonlin,int *lin)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscUseMethod(ts,"TSSundialsGetIterations_C",(TS,int*,int*),(ts,nonlin,lin));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSundialsSetType"
/*@
   TSSundialsSetType - Sets the method that Sundials will use for integration.

   Logically Collective on TS

   Input parameters:
+    ts     - the time-step context
-    type   - one of  SUNDIALS_ADAMS or SUNDIALS_BDF

   Level: intermediate

.keywords: Adams, backward differentiation formula

.seealso: TSSundialsGetIterations(),  TSSundialsSetMaxl(),
          TSSundialsSetLinearTolerance(), TSSundialsSetGramSchmidtType(), TSSundialsSetTolerance(),
          TSSundialsGetIterations(), TSSundialsSetType(), 
          TSSundialsSetLinearTolerance(), TSSundialsSetTolerance(), TSSundialsGetPC(),
          TSSetExactFinalTime()
@*/
PetscErrorCode  TSSundialsSetType(TS ts,TSSundialsLmmType type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTryMethod(ts,"TSSundialsSetType_C",(TS,TSSundialsLmmType),(ts,type));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSundialsSetMaxl"
/*@
   TSSundialsSetMaxl - Sets the dimension of the Krylov space used by
       GMRES in the linear solver in SUNDIALS. SUNDIALS DOES NOT use restarted GMRES so
       this is the maximum number of GMRES steps that will be used.

   Logically Collective on TS

   Input parameters:
+    ts      - the time-step context
-    maxl - number of direction vectors (the dimension of Krylov subspace).

   Level: advanced

.keywords: GMRES

.seealso: TSSundialsGetIterations(), TSSundialsSetType(),
          TSSundialsSetLinearTolerance(), TSSundialsSetGramSchmidtType(), TSSundialsSetTolerance(),
          TSSundialsGetIterations(), TSSundialsSetType(),
          TSSundialsSetLinearTolerance(), TSSundialsSetTolerance(), TSSundialsGetPC(),
          TSSetExactFinalTime()

@*/
PetscErrorCode  TSSundialsSetMaxl(TS ts,PetscInt maxl)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveInt(ts,maxl,2);
  ierr = PetscTryMethod(ts,"TSSundialsSetMaxl_C",(TS,PetscInt),(ts,maxl));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSundialsSetLinearTolerance"
/*@
   TSSundialsSetLinearTolerance - Sets the tolerance used to solve the linear
       system by SUNDIALS.

   Logically Collective on TS

   Input parameters:
+    ts     - the time-step context
-    tol    - the factor by which the tolerance on the nonlinear solver is
             multiplied to get the tolerance on the linear solver, .05 by default.

   Level: advanced

.keywords: GMRES, linear convergence tolerance, SUNDIALS

.seealso: TSSundialsGetIterations(), TSSundialsSetType(), TSSundialsSetMaxl(),
          TSSundialsSetGramSchmidtType(), TSSundialsSetTolerance(),
          TSSundialsGetIterations(), TSSundialsSetType(),
          TSSundialsSetLinearTolerance(), TSSundialsSetTolerance(), TSSundialsGetPC(),
          TSSetExactFinalTime()

@*/
PetscErrorCode  TSSundialsSetLinearTolerance(TS ts,double tol)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveReal(ts,tol,2);
  ierr = PetscTryMethod(ts,"TSSundialsSetLinearTolerance_C",(TS,double),(ts,tol));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSundialsSetGramSchmidtType"
/*@
   TSSundialsSetGramSchmidtType - Sets type of orthogonalization used
        in GMRES method by SUNDIALS linear solver.

   Logically Collective on TS

   Input parameters:
+    ts  - the time-step context
-    type - either SUNDIALS_MODIFIED_GS or SUNDIALS_CLASSICAL_GS

   Level: advanced

.keywords: Sundials, orthogonalization

.seealso: TSSundialsGetIterations(), TSSundialsSetType(), TSSundialsSetMaxl(),
          TSSundialsSetLinearTolerance(),  TSSundialsSetTolerance(),
          TSSundialsGetIterations(), TSSundialsSetType(), 
          TSSundialsSetLinearTolerance(), TSSundialsSetTolerance(), TSSundialsGetPC(),
          TSSetExactFinalTime()

@*/
PetscErrorCode  TSSundialsSetGramSchmidtType(TS ts,TSSundialsGramSchmidtType type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTryMethod(ts,"TSSundialsSetGramSchmidtType_C",(TS,TSSundialsGramSchmidtType),(ts,type));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSundialsSetTolerance"
/*@
   TSSundialsSetTolerance - Sets the absolute and relative tolerance used by
                         Sundials for error control.

   Logically Collective on TS

   Input parameters:
+    ts  - the time-step context
.    aabs - the absolute tolerance
-    rel - the relative tolerance

     See the Cvode/Sundials users manual for exact details on these parameters. Essentially
    these regulate the size of the error for a SINGLE timestep.

   Level: intermediate

.keywords: Sundials, tolerance

.seealso: TSSundialsGetIterations(), TSSundialsSetType(), TSSundialsSetGMRESMaxl(),
          TSSundialsSetLinearTolerance(), TSSundialsSetGramSchmidtType(),
          TSSundialsGetIterations(), TSSundialsSetType(),
          TSSundialsSetLinearTolerance(), TSSundialsSetTolerance(), TSSundialsGetPC(),
          TSSetExactFinalTime()

@*/
PetscErrorCode  TSSundialsSetTolerance(TS ts,double aabs,double rel)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTryMethod(ts,"TSSundialsSetTolerance_C",(TS,double,double),(ts,aabs,rel));CHKERRQ(ierr);
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

.seealso: TSSundialsGetIterations(), TSSundialsSetType(), TSSundialsSetMaxl(),
          TSSundialsSetLinearTolerance(), TSSundialsSetGramSchmidtType(), TSSundialsSetTolerance(),
          TSSundialsGetIterations(), TSSundialsSetType(), 
          TSSundialsSetLinearTolerance(), TSSundialsSetTolerance()
@*/
PetscErrorCode  TSSundialsGetPC(TS ts,PC *pc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscUseMethod(ts,"TSSundialsGetPC_C",(TS,PC *),(ts,pc));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSundialsSetMinTimeStep"
/*@
   TSSundialsSetMinTimeStep - Smallest time step to be chosen by the adaptive controller.

   Input Parameter:
+   ts - the time-step context
-   mindt - lowest time step if positive, negative to deactivate

   Note:
   Sundials will error if it is not possible to keep the estimated truncation error below
   the tolerance set with TSSundialsSetTolerance() without going below this step size.

   Level: beginner

.seealso: TSSundialsSetType(), TSSundialsSetTolerance(),
@*/
PetscErrorCode  TSSundialsSetMinTimeStep(TS ts,PetscReal mindt)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTryMethod(ts,"TSSundialsSetMinTimeStep_C",(TS,PetscReal),(ts,mindt));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSundialsSetMaxTimeStep"
/*@
   TSSundialsSetMaxTimeStep - Largest time step to be chosen by the adaptive controller.

   Input Parameter:
+   ts - the time-step context
-   maxdt - lowest time step if positive, negative to deactivate

   Level: beginner

.seealso: TSSundialsSetType(), TSSundialsSetTolerance(),
@*/
PetscErrorCode  TSSundialsSetMaxTimeStep(TS ts,PetscReal maxdt)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTryMethod(ts,"TSSundialsSetMaxTimeStep_C",(TS,PetscReal),(ts,maxdt));CHKERRQ(ierr);
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

.seealso:TSSundialsGetIterations(), TSSundialsSetType(), TSSundialsSetMaxl(),
          TSSundialsSetLinearTolerance(), TSSundialsSetGramSchmidtType(), TSSundialsSetTolerance(),
          TSSundialsGetIterations(), TSSundialsSetType(), 
          TSSundialsSetLinearTolerance(), TSSundialsSetTolerance(), TSSundialsGetPC()
@*/
PetscErrorCode  TSSundialsMonitorInternalSteps(TS ts,PetscBool  ft)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTryMethod(ts,"TSSundialsMonitorInternalSteps_C",(TS,PetscBool),(ts,ft));CHKERRQ(ierr);
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
.    -ts_sundials_maxl <maxl> - Max dimension of the Krylov subspace
-    -ts_sundials_monitor_steps - Monitor SUNDIALS internel steps


    Notes: This uses its own nonlinear solver and Krylov method so PETSc SNES and KSP options do not apply
           only PETSc PC options

    Level: beginner

.seealso:  TSCreate(), TS, TSSetType(), TSSundialsSetType(), TSSundialsSetMaxl(), TSSundialsSetLinearTolerance(),
           TSSundialsSetGramSchmidtType(), TSSundialsSetTolerance(), TSSundialsGetPC(), TSSundialsGetIterations(), TSSetExactFinalTime()

M*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "TSCreate_Sundials"
PetscErrorCode  TSCreate_Sundials(TS ts)
{
  TS_Sundials    *cvode;
  PetscErrorCode ierr;
  PC             pc;

  PetscFunctionBegin;
  ts->ops->reset          = TSReset_Sundials;
  ts->ops->destroy        = TSDestroy_Sundials;
  ts->ops->view           = TSView_Sundials;
  ts->ops->setup          = TSSetUp_Sundials;
  ts->ops->step           = TSStep_Sundials;
  ts->ops->interpolate    = TSInterpolate_Sundials;
  ts->ops->setfromoptions = TSSetFromOptions_Sundials;

  ierr = PetscNewLog(ts,TS_Sundials,&cvode);CHKERRQ(ierr);
  ts->data                = (void*)cvode;
  cvode->cvode_type       = SUNDIALS_BDF;
  cvode->gtype            = SUNDIALS_CLASSICAL_GS;
  cvode->maxl             = 5;
  cvode->linear_tol       = .05;

  cvode->monitorstep      = PETSC_TRUE;

  ierr = MPI_Comm_dup(((PetscObject)ts)->comm,&(cvode->comm_sundials));CHKERRQ(ierr);

  cvode->mindt = -1.;
  cvode->maxdt = -1.;

  /* set tolerance for Sundials */
  cvode->reltol = 1e-6;
  cvode->abstol = 1e-6;

  /* set PCNONE as default pctype */
  ierr = TSSundialsGetPC_Sundials(ts,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);

  if (ts->exact_final_time == PETSC_DECIDE) ts->exact_final_time = PETSC_FALSE;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSundialsSetType_C","TSSundialsSetType_Sundials",
                    TSSundialsSetType_Sundials);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSundialsSetMaxl_C",
                    "TSSundialsSetMaxl_Sundials",
                    TSSundialsSetMaxl_Sundials);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSundialsSetLinearTolerance_C",
                    "TSSundialsSetLinearTolerance_Sundials",
                     TSSundialsSetLinearTolerance_Sundials);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSundialsSetGramSchmidtType_C",
                    "TSSundialsSetGramSchmidtType_Sundials",
                     TSSundialsSetGramSchmidtType_Sundials);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSundialsSetTolerance_C",
                    "TSSundialsSetTolerance_Sundials",
                     TSSundialsSetTolerance_Sundials);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSundialsSetMinTimeStep_C",
                    "TSSundialsSetMinTimeStep_Sundials",
                     TSSundialsSetMinTimeStep_Sundials);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSundialsSetMaxTimeStep_C",
                    "TSSundialsSetMaxTimeStep_Sundials",
                     TSSundialsSetMaxTimeStep_Sundials);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSundialsGetPC_C",
                    "TSSundialsGetPC_Sundials",
                     TSSundialsGetPC_Sundials);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSundialsGetIterations_C",
                    "TSSundialsGetIterations_Sundials",
                     TSSundialsGetIterations_Sundials);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSundialsMonitorInternalSteps_C",
                    "TSSundialsMonitorInternalSteps_Sundials",
                     TSSundialsMonitorInternalSteps_Sundials);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END
