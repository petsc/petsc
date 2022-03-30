/*
    Provides a PETSc interface to SUNDIALS/CVODE solver.
    The interface to PVODE (old version of CVODE) was originally contributed
    by Liyang Xu. It has been redone by Hong Zhang and Dinesh Kaushik.

    Reference: sundials-2.4.0/examples/cvode/parallel/cvDiurnal_kry_p.c
*/
#include <../src/ts/impls/implicit/sundials/sundials.h>  /*I "petscts.h" I*/

/*
      TSPrecond_Sundials - function that we provide to SUNDIALS to
                        evaluate the preconditioner.
*/
PetscErrorCode TSPrecond_Sundials(realtype tn,N_Vector y,N_Vector fy,booleantype jok,booleantype *jcurPtr,
                                  realtype _gamma,void *P_data,N_Vector vtemp1,N_Vector vtemp2,N_Vector vtemp3)
{
  TS             ts     = (TS) P_data;
  TS_Sundials    *cvode = (TS_Sundials*)ts->data;
  PC             pc;
  Mat            J,P;
  Vec            yy  = cvode->w1,yydot = cvode->ydot;
  PetscReal      gm  = (PetscReal)_gamma;
  PetscScalar    *y_data;

  PetscFunctionBegin;
  PetscCall(TSGetIJacobian(ts,&J,&P,NULL,NULL));
  y_data = (PetscScalar*) N_VGetArrayPointer(y);
  PetscCall(VecPlaceArray(yy,y_data));
  PetscCall(VecZeroEntries(yydot)); /* The Jacobian is independent of Ydot for ODE which is all that CVode works for */
  /* compute the shifted Jacobian   (1/gm)*I + Jrest */
  PetscCall(TSComputeIJacobian(ts,ts->ptime,yy,yydot,1/gm,J,P,PETSC_FALSE));
  PetscCall(VecResetArray(yy));
  PetscCall(MatScale(P,gm)); /* turn into I-gm*Jrest, J is not used by Sundials  */
  *jcurPtr = TRUE;
  PetscCall(TSSundialsGetPC(ts,&pc));
  PetscCall(PCSetOperators(pc,J,P));
  PetscFunctionReturn(0);
}

/*
     TSPSolve_Sundials -  routine that we provide to Sundials that applies the preconditioner.
*/
PetscErrorCode TSPSolve_Sundials(realtype tn,N_Vector y,N_Vector fy,N_Vector r,N_Vector z,
                                 realtype _gamma,realtype delta,int lr,void *P_data,N_Vector vtemp)
{
  TS             ts     = (TS) P_data;
  TS_Sundials    *cvode = (TS_Sundials*)ts->data;
  PC             pc;
  Vec            rr = cvode->w1,zz = cvode->w2;
  PetscScalar    *r_data,*z_data;

  PetscFunctionBegin;
  /* Make the PETSc work vectors rr and zz point to the arrays in the SUNDIALS vectors r and z respectively*/
  r_data = (PetscScalar*) N_VGetArrayPointer(r);
  z_data = (PetscScalar*) N_VGetArrayPointer(z);
  PetscCall(VecPlaceArray(rr,r_data));
  PetscCall(VecPlaceArray(zz,z_data));

  /* Solve the Px=r and put the result in zz */
  PetscCall(TSSundialsGetPC(ts,&pc));
  PetscCall(PCApply(pc,rr,zz));
  PetscCall(VecResetArray(rr));
  PetscCall(VecResetArray(zz));
  PetscFunctionReturn(0);
}

/*
        TSFunction_Sundials - routine that we provide to Sundials that applies the right hand side.
*/
int TSFunction_Sundials(realtype t,N_Vector y,N_Vector ydot,void *ctx)
{
  TS             ts = (TS) ctx;
  DM             dm;
  DMTS           tsdm;
  TSIFunction    ifunction;
  MPI_Comm       comm;
  TS_Sundials    *cvode = (TS_Sundials*)ts->data;
  Vec            yy     = cvode->w1,yyd = cvode->w2,yydot = cvode->ydot;
  PetscScalar    *y_data,*ydot_data;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)ts,&comm));
  /* Make the PETSc work vectors yy and yyd point to the arrays in the SUNDIALS vectors y and ydot respectively*/
  y_data    = (PetscScalar*) N_VGetArrayPointer(y);
  ydot_data = (PetscScalar*) N_VGetArrayPointer(ydot);
  PetscCallAbort(comm,VecPlaceArray(yy,y_data));
  PetscCallAbort(comm,VecPlaceArray(yyd,ydot_data));

  /* Now compute the right hand side function, via IFunction unless only the more efficient RHSFunction is set */
  PetscCall(TSGetDM(ts,&dm));
  PetscCall(DMGetDMTS(dm,&tsdm));
  PetscCall(DMTSGetIFunction(dm,&ifunction,NULL));
  if (!ifunction) {
    PetscCall(TSComputeRHSFunction(ts,t,yy,yyd));
  } else {                      /* If rhsfunction is also set, this computes both parts and shifts them to the right */
    PetscCall(VecZeroEntries(yydot));
    PetscCallAbort(comm,TSComputeIFunction(ts,t,yy,yydot,yyd,PETSC_FALSE));
    PetscCall(VecScale(yyd,-1.));
  }
  PetscCallAbort(comm,VecResetArray(yy));
  PetscCallAbort(comm,VecResetArray(yyd));
  PetscFunctionReturn(0);
}

/*
       TSStep_Sundials - Calls Sundials to integrate the ODE.
*/
PetscErrorCode TSStep_Sundials(TS ts)
{
  TS_Sundials    *cvode = (TS_Sundials*)ts->data;
  PetscInt       flag;
  long int       nits,lits,nsteps;
  realtype       t,tout;
  PetscScalar    *y_data;
  void           *mem;

  PetscFunctionBegin;
  mem  = cvode->mem;
  tout = ts->max_time;
  PetscCall(VecGetArray(ts->vec_sol,&y_data));
  N_VSetArrayPointer((realtype*)y_data,cvode->y);
  PetscCall(VecRestoreArray(ts->vec_sol,NULL));

  /* We would like to TSPreStage() and TSPostStage()
   * before each stage solve but CVode does not appear to support this. */
  if (cvode->monitorstep)
    flag = CVode(mem,tout,cvode->y,&t,CV_ONE_STEP);
  else
    flag = CVode(mem,tout,cvode->y,&t,CV_NORMAL);

  if (flag) { /* display error message */
    switch (flag) {
      case CV_ILL_INPUT:
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"CVode() fails, CV_ILL_INPUT");
        break;
      case CV_TOO_CLOSE:
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"CVode() fails, CV_TOO_CLOSE");
        break;
      case CV_TOO_MUCH_WORK: {
        PetscReal tcur;
        PetscCall(CVodeGetNumSteps(mem,&nsteps));
        PetscCall(CVodeGetCurrentTime(mem,&tcur));
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"CVode() fails, CV_TOO_MUCH_WORK. At t=%g, nsteps %D exceeds maxstep %D. Increase '-ts_max_steps <>' or modify TSSetMaxSteps()",(double)tcur,nsteps,ts->max_steps);
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
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"CVode() fails, flag %d",flag);
    }
  }

  /* log inner nonlinear and linear iterations */
  PetscCall(CVodeGetNumNonlinSolvIters(mem,&nits));
  PetscCall(CVSpilsGetNumLinIters(mem,&lits));
  ts->snes_its += nits; ts->ksp_its = lits;

  /* copy the solution from cvode->y to cvode->update and sol */
  PetscCall(VecPlaceArray(cvode->w1,y_data));
  PetscCall(VecCopy(cvode->w1,cvode->update));
  PetscCall(VecResetArray(cvode->w1));
  PetscCall(VecCopy(cvode->update,ts->vec_sol));

  ts->time_step = t - ts->ptime;
  ts->ptime = t;

  PetscCall(CVodeGetNumSteps(mem,&nsteps));
  if (!cvode->monitorstep) ts->steps += nsteps - 1; /* TSStep() increments the step counter by one */
  PetscFunctionReturn(0);
}

static PetscErrorCode TSInterpolate_Sundials(TS ts,PetscReal t,Vec X)
{
  TS_Sundials    *cvode = (TS_Sundials*)ts->data;
  N_Vector       y;
  PetscScalar    *x_data;
  PetscInt       glosize,locsize;

  PetscFunctionBegin;
  /* get the vector size */
  PetscCall(VecGetSize(X,&glosize));
  PetscCall(VecGetLocalSize(X,&locsize));
  PetscCall(VecGetArray(X,&x_data));

  /* Initialize N_Vec y with x_data */
  if (cvode->use_dense) {
    PetscMPIInt size;

    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
    PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"TSSUNDIALS only supports a dense solve in the serial case");
    y = N_VMake_Serial(locsize,(realtype*)x_data);
  } else {
    y = N_VMake_Parallel(cvode->comm_sundials,locsize,glosize,(realtype*)x_data);
  }

  PetscCheck(y,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Interpolated y is not allocated");

  PetscCall(CVodeGetDky(cvode->mem,t,0,y));
  PetscCall(VecRestoreArray(X,&x_data));
  PetscFunctionReturn(0);
}

PetscErrorCode TSReset_Sundials(TS ts)
{
  TS_Sundials    *cvode = (TS_Sundials*)ts->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&cvode->update));
  PetscCall(VecDestroy(&cvode->ydot));
  PetscCall(VecDestroy(&cvode->w1));
  PetscCall(VecDestroy(&cvode->w2));
  if (cvode->mem) CVodeFree(&cvode->mem);
  PetscFunctionReturn(0);
}

PetscErrorCode TSDestroy_Sundials(TS ts)
{
  TS_Sundials    *cvode = (TS_Sundials*)ts->data;

  PetscFunctionBegin;
  PetscCall(TSReset_Sundials(ts));
  PetscCallMPI(MPI_Comm_free(&(cvode->comm_sundials)));
  PetscCall(PetscFree(ts->data));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSSundialsSetType_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSSundialsSetMaxl_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSSundialsSetLinearTolerance_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSSundialsSetGramSchmidtType_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSSundialsSetTolerance_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSSundialsSetMinTimeStep_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSSundialsSetMaxTimeStep_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSSundialsGetPC_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSSundialsGetIterations_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSSundialsMonitorInternalSteps_C",NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode TSSetUp_Sundials(TS ts)
{
  TS_Sundials    *cvode = (TS_Sundials*)ts->data;
  PetscInt       glosize,locsize,i,flag;
  PetscScalar    *y_data,*parray;
  void           *mem;
  PC             pc;
  PCType         pctype;
  PetscBool      pcnone;

  PetscFunctionBegin;
  PetscCheckFalse(ts->exact_final_time == TS_EXACTFINALTIME_MATCHSTEP,PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for exact final time option 'MATCHSTEP' when using Sundials");

  /* get the vector size */
  PetscCall(VecGetSize(ts->vec_sol,&glosize));
  PetscCall(VecGetLocalSize(ts->vec_sol,&locsize));

  /* allocate the memory for N_Vec y */
  if (cvode->use_dense) {
    PetscMPIInt size;

    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
    PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"TSSUNDIALS only supports a dense solve in the serial case");
    cvode->y = N_VNew_Serial(locsize);
  } else {
    cvode->y = N_VNew_Parallel(cvode->comm_sundials,locsize,glosize);
  }
  PetscCheck(cvode->y,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"cvode->y is not allocated");

  /* initialize N_Vec y: copy ts->vec_sol to cvode->y */
  PetscCall(VecGetArray(ts->vec_sol,&parray));
  y_data = (PetscScalar*) N_VGetArrayPointer(cvode->y);
  for (i = 0; i < locsize; i++) y_data[i] = parray[i];
  PetscCall(VecRestoreArray(ts->vec_sol,NULL));

  PetscCall(VecDuplicate(ts->vec_sol,&cvode->update));
  PetscCall(VecDuplicate(ts->vec_sol,&cvode->ydot));
  PetscCall(PetscLogObjectParent((PetscObject)ts,(PetscObject)cvode->update));
  PetscCall(PetscLogObjectParent((PetscObject)ts,(PetscObject)cvode->ydot));

  /*
    Create work vectors for the TSPSolve_Sundials() routine. Note these are
    allocated with zero space arrays because the actual array space is provided
    by Sundials and set using VecPlaceArray().
  */
  PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)ts),1,locsize,PETSC_DECIDE,NULL,&cvode->w1));
  PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)ts),1,locsize,PETSC_DECIDE,NULL,&cvode->w2));
  PetscCall(PetscLogObjectParent((PetscObject)ts,(PetscObject)cvode->w1));
  PetscCall(PetscLogObjectParent((PetscObject)ts,(PetscObject)cvode->w2));

  /* Call CVodeCreate to create the solver memory and the use of a Newton iteration */
  mem = CVodeCreate(cvode->cvode_type, CV_NEWTON);
  PetscCheck(mem,PETSC_COMM_SELF,PETSC_ERR_MEM,"CVodeCreate() fails");
  cvode->mem = mem;

  /* Set the pointer to user-defined data */
  flag = CVodeSetUserData(mem, ts);
  PetscCheck(!flag,PETSC_COMM_SELF,PETSC_ERR_LIB,"CVodeSetUserData() fails");

  /* Sundials may choose to use a smaller initial step, but will never use a larger step. */
  flag = CVodeSetInitStep(mem,(realtype)ts->time_step);
  PetscCheck(!flag,PetscObjectComm((PetscObject)ts),PETSC_ERR_LIB,"CVodeSetInitStep() failed");
  if (cvode->mindt > 0) {
    flag = CVodeSetMinStep(mem,(realtype)cvode->mindt);
    if (flag) {
      PetscCheckFalse(flag == CV_MEM_NULL,PetscObjectComm((PetscObject)ts),PETSC_ERR_LIB,"CVodeSetMinStep() failed, cvode_mem pointer is NULL");
      else PetscCheckFalse(flag == CV_ILL_INPUT,PetscObjectComm((PetscObject)ts),PETSC_ERR_LIB,"CVodeSetMinStep() failed, hmin is nonpositive or it exceeds the maximum allowable step size");
      else SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_LIB,"CVodeSetMinStep() failed");
    }
  }
  if (cvode->maxdt > 0) {
    flag = CVodeSetMaxStep(mem,(realtype)cvode->maxdt);
    PetscCheck(!flag,PetscObjectComm((PetscObject)ts),PETSC_ERR_LIB,"CVodeSetMaxStep() failed");
  }

  /* Call CVodeInit to initialize the integrator memory and specify the
   * user's right hand side function in u'=f(t,u), the initial time T0, and
   * the initial dependent variable vector cvode->y */
  flag = CVodeInit(mem,TSFunction_Sundials,ts->ptime,cvode->y);
  PetscCheck(!flag,PETSC_COMM_SELF,PETSC_ERR_LIB,"CVodeInit() fails, flag %d",flag);

  /* specifies scalar relative and absolute tolerances */
  flag = CVodeSStolerances(mem,cvode->reltol,cvode->abstol);
  PetscCheck(!flag,PETSC_COMM_SELF,PETSC_ERR_LIB,"CVodeSStolerances() fails, flag %d",flag);

  /* Specify max order of BDF / ADAMS method */
  if (cvode->maxord != PETSC_DEFAULT) {
    flag = CVodeSetMaxOrd(mem,cvode->maxord);
    PetscCheck(!flag,PETSC_COMM_SELF,PETSC_ERR_LIB,"CVodeSetMaxOrd() fails, flag %d",flag);
  }

  /* Specify max num of steps to be taken by cvode in its attempt to reach the next output time */
  flag = CVodeSetMaxNumSteps(mem,ts->max_steps);
  PetscCheck(!flag,PETSC_COMM_SELF,PETSC_ERR_LIB,"CVodeSetMaxNumSteps() fails, flag %d",flag);

  if (cvode->use_dense) {
    /* call CVDense to use a dense linear solver. */
    PetscMPIInt size;

    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
    PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"TSSUNDIALS only supports a dense solve in the serial case");
    flag = CVDense(mem,locsize);
    PetscCheck(!flag,PETSC_COMM_SELF,PETSC_ERR_LIB,"CVDense() fails, flag %d",flag);
  } else {
    /* call CVSpgmr to use GMRES as the linear solver.        */
    /* setup the ode integrator with the given preconditioner */
    PetscCall(TSSundialsGetPC(ts,&pc));
    PetscCall(PCGetType(pc,&pctype));
    PetscCall(PetscObjectTypeCompare((PetscObject)pc,PCNONE,&pcnone));
    if (pcnone) {
      flag = CVSpgmr(mem,PREC_NONE,0);
      PetscCheck(!flag,PETSC_COMM_SELF,PETSC_ERR_LIB,"CVSpgmr() fails, flag %d",flag);
    } else {
      flag = CVSpgmr(mem,PREC_LEFT,cvode->maxl);
      PetscCheck(!flag,PETSC_COMM_SELF,PETSC_ERR_LIB,"CVSpgmr() fails, flag %d",flag);

      /* Set preconditioner and solve routines Precond and PSolve,
         and the pointer to the user-defined block data */
      flag = CVSpilsSetPreconditioner(mem,TSPrecond_Sundials,TSPSolve_Sundials);
      PetscCheck(!flag,PETSC_COMM_SELF,PETSC_ERR_LIB,"CVSpilsSetPreconditioner() fails, flag %d", flag);
    }
  }
  PetscFunctionReturn(0);
}

/* type of CVODE linear multistep method */
const char *const TSSundialsLmmTypes[] = {"","ADAMS","BDF","TSSundialsLmmType","SUNDIALS_",NULL};
/* type of G-S orthogonalization used by CVODE linear solver */
const char *const TSSundialsGramSchmidtTypes[] = {"","MODIFIED","CLASSICAL","TSSundialsGramSchmidtType","SUNDIALS_",NULL};

PetscErrorCode TSSetFromOptions_Sundials(PetscOptionItems *PetscOptionsObject,TS ts)
{
  TS_Sundials    *cvode = (TS_Sundials*)ts->data;
  int            indx;
  PetscBool      flag;
  PC             pc;

  PetscFunctionBegin;
  PetscCall(PetscOptionsHead(PetscOptionsObject,"SUNDIALS ODE solver options"));
  PetscCall(PetscOptionsEList("-ts_sundials_type","Scheme","TSSundialsSetType",TSSundialsLmmTypes,3,TSSundialsLmmTypes[cvode->cvode_type],&indx,&flag));
  if (flag) {
    PetscCall(TSSundialsSetType(ts,(TSSundialsLmmType)indx));
  }
  PetscCall(PetscOptionsEList("-ts_sundials_gramschmidt_type","Type of orthogonalization","TSSundialsSetGramSchmidtType",TSSundialsGramSchmidtTypes,3,TSSundialsGramSchmidtTypes[cvode->gtype],&indx,&flag));
  if (flag) {
    PetscCall(TSSundialsSetGramSchmidtType(ts,(TSSundialsGramSchmidtType)indx));
  }
  PetscCall(PetscOptionsReal("-ts_sundials_atol","Absolute tolerance for convergence","TSSundialsSetTolerance",cvode->abstol,&cvode->abstol,NULL));
  PetscCall(PetscOptionsReal("-ts_sundials_rtol","Relative tolerance for convergence","TSSundialsSetTolerance",cvode->reltol,&cvode->reltol,NULL));
  PetscCall(PetscOptionsReal("-ts_sundials_mindt","Minimum step size","TSSundialsSetMinTimeStep",cvode->mindt,&cvode->mindt,NULL));
  PetscCall(PetscOptionsReal("-ts_sundials_maxdt","Maximum step size","TSSundialsSetMaxTimeStep",cvode->maxdt,&cvode->maxdt,NULL));
  PetscCall(PetscOptionsReal("-ts_sundials_linear_tolerance","Convergence tolerance for linear solve","TSSundialsSetLinearTolerance",cvode->linear_tol,&cvode->linear_tol,NULL));
  PetscCall(PetscOptionsInt("-ts_sundials_maxord","Max Order for BDF/Adams method","TSSundialsSetMaxOrd",cvode->maxord,&cvode->maxord,NULL));
  PetscCall(PetscOptionsInt("-ts_sundials_maxl","Max dimension of the Krylov subspace","TSSundialsSetMaxl",cvode->maxl,&cvode->maxl,NULL));
  PetscCall(PetscOptionsBool("-ts_sundials_monitor_steps","Monitor SUNDIALS internal steps","TSSundialsMonitorInternalSteps",cvode->monitorstep,&cvode->monitorstep,NULL));
  PetscCall(PetscOptionsBool("-ts_sundials_use_dense","Use dense internal solver in SUNDIALS (serial only)","TSSundialsSetUseDense",cvode->use_dense,&cvode->use_dense,NULL));
  PetscCall(PetscOptionsTail());
  PetscCall(TSSundialsGetPC(ts,&pc));
  PetscCall(PCSetFromOptions(pc));
  PetscFunctionReturn(0);
}

PetscErrorCode TSView_Sundials(TS ts,PetscViewer viewer)
{
  TS_Sundials    *cvode = (TS_Sundials*)ts->data;
  char           *type;
  char           atype[] = "Adams";
  char           btype[] = "BDF: backward differentiation formula";
  PetscBool      iascii,isstring;
  long int       nsteps,its,nfevals,nlinsetups,nfails,itmp;
  PetscInt       qlast,qcur;
  PetscReal      hinused,hlast,hcur,tcur,tolsfac;
  PC             pc;

  PetscFunctionBegin;
  if (cvode->cvode_type == SUNDIALS_ADAMS) type = atype;
  else                                     type = btype;

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"Sundials integrator does not use SNES!\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer,"Sundials integrator type %s\n",type));
    PetscCall(PetscViewerASCIIPrintf(viewer,"Sundials maxord %D\n",cvode->maxord));
    PetscCall(PetscViewerASCIIPrintf(viewer,"Sundials abs tol %g rel tol %g\n",(double)cvode->abstol,(double)cvode->reltol));
    if (cvode->use_dense) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"Sundials integrator using a dense linear solve\n"));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer,"Sundials linear solver tolerance factor %g\n",(double)cvode->linear_tol));
      PetscCall(PetscViewerASCIIPrintf(viewer,"Sundials max dimension of Krylov subspace %D\n",cvode->maxl));
      if (cvode->gtype == SUNDIALS_MODIFIED_GS) {
        PetscCall(PetscViewerASCIIPrintf(viewer,"Sundials using modified Gram-Schmidt for orthogonalization in GMRES\n"));
      } else {
        PetscCall(PetscViewerASCIIPrintf(viewer,"Sundials using unmodified (classical) Gram-Schmidt for orthogonalization in GMRES\n"));
      }
    }
    if (cvode->mindt > 0) PetscCall(PetscViewerASCIIPrintf(viewer,"Sundials minimum time step %g\n",(double)cvode->mindt));
    if (cvode->maxdt > 0) PetscCall(PetscViewerASCIIPrintf(viewer,"Sundials maximum time step %g\n",(double)cvode->maxdt));

    /* Outputs from CVODE, CVSPILS */
    PetscCall(CVodeGetTolScaleFactor(cvode->mem,&tolsfac));
    PetscCall(PetscViewerASCIIPrintf(viewer,"Sundials suggested factor for tolerance scaling %g\n",tolsfac));
    PetscCall(CVodeGetIntegratorStats(cvode->mem,&nsteps,&nfevals,&nlinsetups,&nfails,&qlast,&qcur,&hinused,&hlast,&hcur,&tcur));
    PetscCall(PetscViewerASCIIPrintf(viewer,"Sundials cumulative number of internal steps %D\n",nsteps));
    PetscCall(PetscViewerASCIIPrintf(viewer,"Sundials no. of calls to rhs function %D\n",nfevals));
    PetscCall(PetscViewerASCIIPrintf(viewer,"Sundials no. of calls to linear solver setup function %D\n",nlinsetups));
    PetscCall(PetscViewerASCIIPrintf(viewer,"Sundials no. of error test failures %D\n",nfails));

    PetscCall(CVodeGetNonlinSolvStats(cvode->mem,&its,&nfails));
    PetscCall(PetscViewerASCIIPrintf(viewer,"Sundials no. of nonlinear solver iterations %D\n",its));
    PetscCall(PetscViewerASCIIPrintf(viewer,"Sundials no. of nonlinear convergence failure %D\n",nfails));
    if (!cvode->use_dense) {
      PetscCall(CVSpilsGetNumLinIters(cvode->mem, &its)); /* its = no. of calls to TSPrecond_Sundials() */
      PetscCall(PetscViewerASCIIPrintf(viewer,"Sundials no. of linear iterations %D\n",its));
      PetscCall(CVSpilsGetNumConvFails(cvode->mem,&itmp));
      PetscCall(PetscViewerASCIIPrintf(viewer,"Sundials no. of linear convergence failures %D\n",itmp));

      PetscCall(TSSundialsGetPC(ts,&pc));
      PetscCall(PCView(pc,viewer));
      PetscCall(CVSpilsGetNumPrecEvals(cvode->mem,&itmp));
      PetscCall(PetscViewerASCIIPrintf(viewer,"Sundials no. of preconditioner evaluations %D\n",itmp));
      PetscCall(CVSpilsGetNumPrecSolves(cvode->mem,&itmp));
      PetscCall(PetscViewerASCIIPrintf(viewer,"Sundials no. of preconditioner solves %D\n",itmp));
    }
    PetscCall(CVSpilsGetNumJtimesEvals(cvode->mem,&itmp));
    PetscCall(PetscViewerASCIIPrintf(viewer,"Sundials no. of Jacobian-vector product evaluations %D\n",itmp));
    PetscCall(CVSpilsGetNumRhsEvals(cvode->mem,&itmp));
    PetscCall(PetscViewerASCIIPrintf(viewer,"Sundials no. of rhs calls for finite diff. Jacobian-vector evals %D\n",itmp));
  } else if (isstring) {
    PetscCall(PetscViewerStringSPrintf(viewer,"Sundials type %s",type));
  }
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------*/
PetscErrorCode  TSSundialsSetType_Sundials(TS ts,TSSundialsLmmType type)
{
  TS_Sundials *cvode = (TS_Sundials*)ts->data;

  PetscFunctionBegin;
  cvode->cvode_type = type;
  PetscFunctionReturn(0);
}

PetscErrorCode  TSSundialsSetMaxl_Sundials(TS ts,PetscInt maxl)
{
  TS_Sundials *cvode = (TS_Sundials*)ts->data;

  PetscFunctionBegin;
  cvode->maxl = maxl;
  PetscFunctionReturn(0);
}

PetscErrorCode  TSSundialsSetLinearTolerance_Sundials(TS ts,PetscReal tol)
{
  TS_Sundials *cvode = (TS_Sundials*)ts->data;

  PetscFunctionBegin;
  cvode->linear_tol = tol;
  PetscFunctionReturn(0);
}

PetscErrorCode  TSSundialsSetGramSchmidtType_Sundials(TS ts,TSSundialsGramSchmidtType type)
{
  TS_Sundials *cvode = (TS_Sundials*)ts->data;

  PetscFunctionBegin;
  cvode->gtype = type;
  PetscFunctionReturn(0);
}

PetscErrorCode  TSSundialsSetTolerance_Sundials(TS ts,PetscReal aabs,PetscReal rel)
{
  TS_Sundials *cvode = (TS_Sundials*)ts->data;

  PetscFunctionBegin;
  if (aabs != PETSC_DECIDE) cvode->abstol = aabs;
  if (rel != PETSC_DECIDE)  cvode->reltol = rel;
  PetscFunctionReturn(0);
}

PetscErrorCode  TSSundialsSetMinTimeStep_Sundials(TS ts,PetscReal mindt)
{
  TS_Sundials *cvode = (TS_Sundials*)ts->data;

  PetscFunctionBegin;
  cvode->mindt = mindt;
  PetscFunctionReturn(0);
}

PetscErrorCode  TSSundialsSetMaxTimeStep_Sundials(TS ts,PetscReal maxdt)
{
  TS_Sundials *cvode = (TS_Sundials*)ts->data;

  PetscFunctionBegin;
  cvode->maxdt = maxdt;
  PetscFunctionReturn(0);
}

PetscErrorCode  TSSundialsSetUseDense_Sundials(TS ts,PetscBool use_dense)
{
  TS_Sundials *cvode = (TS_Sundials*)ts->data;

  PetscFunctionBegin;
  cvode->use_dense = use_dense;
  PetscFunctionReturn(0);
}

PetscErrorCode  TSSundialsGetPC_Sundials(TS ts,PC *pc)
{
  SNES           snes;
  KSP            ksp;

  PetscFunctionBegin;
  PetscCall(TSGetSNES(ts,&snes));
  PetscCall(SNESGetKSP(snes,&ksp));
  PetscCall(KSPGetPC(ksp,pc));
  PetscFunctionReturn(0);
}

PetscErrorCode  TSSundialsGetIterations_Sundials(TS ts,int *nonlin,int *lin)
{
  PetscFunctionBegin;
  if (nonlin) *nonlin = ts->snes_its;
  if (lin)    *lin    = ts->ksp_its;
  PetscFunctionReturn(0);
}

PetscErrorCode  TSSundialsMonitorInternalSteps_Sundials(TS ts,PetscBool s)
{
  TS_Sundials *cvode = (TS_Sundials*)ts->data;

  PetscFunctionBegin;
  cvode->monitorstep = s;
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------------------------*/

/*@C
   TSSundialsGetIterations - Gets the number of nonlinear and linear iterations used so far by Sundials.

   Not Collective

   Input Parameter:
.    ts     - the time-step context

   Output Parameters:
+   nonlin - number of nonlinear iterations
-   lin    - number of linear iterations

   Level: advanced

   Notes:
    These return the number since the creation of the TS object

.seealso: TSSundialsSetType(), TSSundialsSetMaxl(),
          TSSundialsSetLinearTolerance(), TSSundialsSetGramSchmidtType(), TSSundialsSetTolerance(),
          TSSundialsGetIterations(), TSSundialsSetType(),
          TSSundialsSetLinearTolerance(), TSSundialsGetPC(), TSSetExactFinalTime()

@*/
PetscErrorCode  TSSundialsGetIterations(TS ts,int *nonlin,int *lin)
{
  PetscFunctionBegin;
  PetscUseMethod(ts,"TSSundialsGetIterations_C",(TS,int*,int*),(ts,nonlin,lin));
  PetscFunctionReturn(0);
}

/*@
   TSSundialsSetType - Sets the method that Sundials will use for integration.

   Logically Collective on TS

   Input Parameters:
+    ts     - the time-step context
-    type   - one of  SUNDIALS_ADAMS or SUNDIALS_BDF

   Level: intermediate

.seealso: TSSundialsGetIterations(),  TSSundialsSetMaxl(),
          TSSundialsSetLinearTolerance(), TSSundialsSetGramSchmidtType(), TSSundialsSetTolerance(),
          TSSundialsGetIterations(), TSSundialsSetType(),
          TSSundialsSetLinearTolerance(), TSSundialsSetTolerance(), TSSundialsGetPC(),
          TSSetExactFinalTime()
@*/
PetscErrorCode  TSSundialsSetType(TS ts,TSSundialsLmmType type)
{
  PetscFunctionBegin;
  PetscTryMethod(ts,"TSSundialsSetType_C",(TS,TSSundialsLmmType),(ts,type));
  PetscFunctionReturn(0);
}

/*@
   TSSundialsSetMaxord - Sets the maximum order for BDF/Adams method used by SUNDIALS.

   Logically Collective on TS

   Input Parameters:
+    ts      - the time-step context
-    maxord  - maximum order of BDF / Adams method

   Level: advanced

.seealso: TSSundialsGetIterations(), TSSundialsSetType(),
          TSSundialsSetLinearTolerance(), TSSundialsSetGramSchmidtType(), TSSundialsSetTolerance(),
          TSSundialsGetIterations(), TSSundialsSetType(),
          TSSundialsSetLinearTolerance(), TSSundialsSetTolerance(), TSSundialsGetPC(),
          TSSetExactFinalTime()

@*/
PetscErrorCode  TSSundialsSetMaxord(TS ts,PetscInt maxord)
{
  PetscFunctionBegin;
  PetscValidLogicalCollectiveInt(ts,maxord,2);
  PetscTryMethod(ts,"TSSundialsSetMaxOrd_C",(TS,PetscInt),(ts,maxord));
  PetscFunctionReturn(0);
}

/*@
   TSSundialsSetMaxl - Sets the dimension of the Krylov space used by
       GMRES in the linear solver in SUNDIALS. SUNDIALS DOES NOT use restarted GMRES so
       this is the maximum number of GMRES steps that will be used.

   Logically Collective on TS

   Input Parameters:
+    ts      - the time-step context
-    maxl - number of direction vectors (the dimension of Krylov subspace).

   Level: advanced

.seealso: TSSundialsGetIterations(), TSSundialsSetType(),
          TSSundialsSetLinearTolerance(), TSSundialsSetGramSchmidtType(), TSSundialsSetTolerance(),
          TSSundialsGetIterations(), TSSundialsSetType(),
          TSSundialsSetLinearTolerance(), TSSundialsSetTolerance(), TSSundialsGetPC(),
          TSSetExactFinalTime()

@*/
PetscErrorCode  TSSundialsSetMaxl(TS ts,PetscInt maxl)
{
  PetscFunctionBegin;
  PetscValidLogicalCollectiveInt(ts,maxl,2);
  PetscTryMethod(ts,"TSSundialsSetMaxl_C",(TS,PetscInt),(ts,maxl));
  PetscFunctionReturn(0);
}

/*@
   TSSundialsSetLinearTolerance - Sets the tolerance used to solve the linear
       system by SUNDIALS.

   Logically Collective on TS

   Input Parameters:
+    ts     - the time-step context
-    tol    - the factor by which the tolerance on the nonlinear solver is
             multiplied to get the tolerance on the linear solver, .05 by default.

   Level: advanced

.seealso: TSSundialsGetIterations(), TSSundialsSetType(), TSSundialsSetMaxl(),
          TSSundialsSetGramSchmidtType(), TSSundialsSetTolerance(),
          TSSundialsGetIterations(), TSSundialsSetType(),
          TSSundialsSetLinearTolerance(), TSSundialsSetTolerance(), TSSundialsGetPC(),
          TSSetExactFinalTime()

@*/
PetscErrorCode  TSSundialsSetLinearTolerance(TS ts,PetscReal tol)
{
  PetscFunctionBegin;
  PetscValidLogicalCollectiveReal(ts,tol,2);
  PetscTryMethod(ts,"TSSundialsSetLinearTolerance_C",(TS,PetscReal),(ts,tol));
  PetscFunctionReturn(0);
}

/*@
   TSSundialsSetGramSchmidtType - Sets type of orthogonalization used
        in GMRES method by SUNDIALS linear solver.

   Logically Collective on TS

   Input Parameters:
+    ts  - the time-step context
-    type - either SUNDIALS_MODIFIED_GS or SUNDIALS_CLASSICAL_GS

   Level: advanced

.seealso: TSSundialsGetIterations(), TSSundialsSetType(), TSSundialsSetMaxl(),
          TSSundialsSetLinearTolerance(),  TSSundialsSetTolerance(),
          TSSundialsGetIterations(), TSSundialsSetType(),
          TSSundialsSetLinearTolerance(), TSSundialsSetTolerance(), TSSundialsGetPC(),
          TSSetExactFinalTime()

@*/
PetscErrorCode  TSSundialsSetGramSchmidtType(TS ts,TSSundialsGramSchmidtType type)
{
  PetscFunctionBegin;
  PetscTryMethod(ts,"TSSundialsSetGramSchmidtType_C",(TS,TSSundialsGramSchmidtType),(ts,type));
  PetscFunctionReturn(0);
}

/*@
   TSSundialsSetTolerance - Sets the absolute and relative tolerance used by
                         Sundials for error control.

   Logically Collective on TS

   Input Parameters:
+    ts  - the time-step context
.    aabs - the absolute tolerance
-    rel - the relative tolerance

     See the Cvode/Sundials users manual for exact details on these parameters. Essentially
    these regulate the size of the error for a SINGLE timestep.

   Level: intermediate

.seealso: TSSundialsGetIterations(), TSSundialsSetType(), TSSundialsSetGMRESMaxl(),
          TSSundialsSetLinearTolerance(), TSSundialsSetGramSchmidtType(),
          TSSundialsGetIterations(), TSSundialsSetType(),
          TSSundialsSetLinearTolerance(), TSSundialsSetTolerance(), TSSundialsGetPC(),
          TSSetExactFinalTime()

@*/
PetscErrorCode  TSSundialsSetTolerance(TS ts,PetscReal aabs,PetscReal rel)
{
  PetscFunctionBegin;
  PetscTryMethod(ts,"TSSundialsSetTolerance_C",(TS,PetscReal,PetscReal),(ts,aabs,rel));
  PetscFunctionReturn(0);
}

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
  PetscFunctionBegin;
  PetscUseMethod(ts,"TSSundialsGetPC_C",(TS,PC*),(ts,pc));
  PetscFunctionReturn(0);
}

/*@
   TSSundialsSetMinTimeStep - Smallest time step to be chosen by the adaptive controller.

   Input Parameters:
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
  PetscFunctionBegin;
  PetscTryMethod(ts,"TSSundialsSetMinTimeStep_C",(TS,PetscReal),(ts,mindt));
  PetscFunctionReturn(0);
}

/*@
   TSSundialsSetMaxTimeStep - Largest time step to be chosen by the adaptive controller.

   Input Parameters:
+   ts - the time-step context
-   maxdt - lowest time step if positive, negative to deactivate

   Level: beginner

.seealso: TSSundialsSetType(), TSSundialsSetTolerance(),
@*/
PetscErrorCode  TSSundialsSetMaxTimeStep(TS ts,PetscReal maxdt)
{
  PetscFunctionBegin;
  PetscTryMethod(ts,"TSSundialsSetMaxTimeStep_C",(TS,PetscReal),(ts,maxdt));
  PetscFunctionReturn(0);
}

/*@
   TSSundialsMonitorInternalSteps - Monitor Sundials internal steps (Defaults to false).

   Input Parameters:
+   ts - the time-step context
-   ft - PETSC_TRUE if monitor, else PETSC_FALSE

   Level: beginner

.seealso:TSSundialsGetIterations(), TSSundialsSetType(), TSSundialsSetMaxl(),
          TSSundialsSetLinearTolerance(), TSSundialsSetGramSchmidtType(), TSSundialsSetTolerance(),
          TSSundialsGetIterations(), TSSundialsSetType(),
          TSSundialsSetLinearTolerance(), TSSundialsSetTolerance(), TSSundialsGetPC()
@*/
PetscErrorCode  TSSundialsMonitorInternalSteps(TS ts,PetscBool ft)
{
  PetscFunctionBegin;
  PetscTryMethod(ts,"TSSundialsMonitorInternalSteps_C",(TS,PetscBool),(ts,ft));
  PetscFunctionReturn(0);
}

/*@
   TSSundialsSetUseDense - Set a flag to use a dense linear solver in SUNDIALS (serial only)

   Logically Collective

   Input Parameters:
+    ts         - the time-step context
-    use_dense  - PETSC_TRUE to use the dense solver

   Level: advanced

.seealso: TSSUNDIALS

@*/
PetscErrorCode  TSSundialsSetUseDense(TS ts,PetscBool use_dense)
{
  PetscFunctionBegin;
  PetscValidLogicalCollectiveInt(ts,use_dense,2);
  PetscTryMethod(ts,"TSSundialsSetUseDense_C",(TS,PetscBool),(ts,use_dense));
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------------*/
/*MC
      TSSUNDIALS - ODE solver using the LLNL CVODE/SUNDIALS package (now called SUNDIALS)

   Options Database:
+    -ts_sundials_type <bdf,adams> -
.    -ts_sundials_gramschmidt_type <modified, classical> - type of orthogonalization inside GMRES
.    -ts_sundials_atol <tol> - Absolute tolerance for convergence
.    -ts_sundials_rtol <tol> - Relative tolerance for convergence
.    -ts_sundials_linear_tolerance <tol> -
.    -ts_sundials_maxl <maxl> - Max dimension of the Krylov subspace
.    -ts_sundials_monitor_steps - Monitor SUNDIALS internal steps
-    -ts_sundials_use_dense - Use a dense linear solver within CVODE (serial only)

    Notes:
    This uses its own nonlinear solver and Krylov method so PETSc SNES and KSP options do not apply,
           only PETSc PC options.

    Level: beginner

.seealso:  TSCreate(), TS, TSSetType(), TSSundialsSetType(), TSSundialsSetMaxl(), TSSundialsSetLinearTolerance(),
           TSSundialsSetGramSchmidtType(), TSSundialsSetTolerance(), TSSundialsGetPC(), TSSundialsGetIterations(), TSSetExactFinalTime()

M*/
PETSC_EXTERN PetscErrorCode TSCreate_Sundials(TS ts)
{
  TS_Sundials    *cvode;
  PC             pc;

  PetscFunctionBegin;
  ts->ops->reset          = TSReset_Sundials;
  ts->ops->destroy        = TSDestroy_Sundials;
  ts->ops->view           = TSView_Sundials;
  ts->ops->setup          = TSSetUp_Sundials;
  ts->ops->step           = TSStep_Sundials;
  ts->ops->interpolate    = TSInterpolate_Sundials;
  ts->ops->setfromoptions = TSSetFromOptions_Sundials;
  ts->default_adapt_type  = TSADAPTNONE;

  PetscCall(PetscNewLog(ts,&cvode));

  ts->usessnes = PETSC_TRUE;

  ts->data           = (void*)cvode;
  cvode->cvode_type  = SUNDIALS_BDF;
  cvode->gtype       = SUNDIALS_CLASSICAL_GS;
  cvode->maxl        = 5;
  cvode->maxord      = PETSC_DEFAULT;
  cvode->linear_tol  = .05;
  cvode->monitorstep = PETSC_TRUE;
  cvode->use_dense   = PETSC_FALSE;

  PetscCallMPI(MPI_Comm_dup(PetscObjectComm((PetscObject)ts),&(cvode->comm_sundials)));

  cvode->mindt = -1.;
  cvode->maxdt = -1.;

  /* set tolerance for Sundials */
  cvode->reltol = 1e-6;
  cvode->abstol = 1e-6;

  /* set PCNONE as default pctype */
  PetscCall(TSSundialsGetPC_Sundials(ts,&pc));
  PetscCall(PCSetType(pc,PCNONE));

  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSSundialsSetType_C",TSSundialsSetType_Sundials));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSSundialsSetMaxl_C",TSSundialsSetMaxl_Sundials));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSSundialsSetLinearTolerance_C",TSSundialsSetLinearTolerance_Sundials));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSSundialsSetGramSchmidtType_C",TSSundialsSetGramSchmidtType_Sundials));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSSundialsSetTolerance_C",TSSundialsSetTolerance_Sundials));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSSundialsSetMinTimeStep_C",TSSundialsSetMinTimeStep_Sundials));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSSundialsSetMaxTimeStep_C",TSSundialsSetMaxTimeStep_Sundials));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSSundialsGetPC_C",TSSundialsGetPC_Sundials));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSSundialsGetIterations_C",TSSundialsGetIterations_Sundials));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSSundialsMonitorInternalSteps_C",TSSundialsMonitorInternalSteps_Sundials));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSSundialsSetUseDense_C",TSSundialsSetUseDense_Sundials));
  PetscFunctionReturn(0);
}
