
static char help[] = "Serial bouncing ball example to test TS event feature.\n";

/*
        u1_t = u2
        u2_t = -9.8
*/

#include <petscts.h>

#define MAXACTIVEEVENTS 2

typedef enum {EVENT_NONE,EVENT_DETECTED,EVENT_PROCESSING,EVENT_OVER,EVENT_RESET_NEXTSTEP} EventStatus;
typedef struct {
  PetscScalar    *fvalue;      /* value of event functions at the end of the step*/
  PetscScalar    *fvalue_prev; /* value of event function at start of the step */
  PetscReal       ptime;       /* time after step completion */
  PetscReal       ptime_prev;  /* time at step start */
  PetscErrorCode  (*monitor)(TS,PetscReal,Vec,PetscScalar*,PetscInt*,PetscBool*,void*);
  PetscErrorCode  (*postevent)(TS,PetscInt,PetscInt[],PetscReal,Vec,void*);
  PetscBool      *terminate;   /* 1 -> Terminate time stepping, 0 -> continue */
  PetscInt       *direction;   /* Zero crossing direction: 1 -> Going positive, -1 -> Going negative, 0 -> Any */ 
  PetscInt        nevents;     
  PetscInt        nevents_active; /* Number of active events */
  PetscInt        active_event_list[MAXACTIVEEVENTS];
  void           *monitorcontext;
  PetscReal       tol;
  EventStatus     status;        /* Event status */
  PetscReal       tstepend;      /* End time of step */
  PetscReal       initial_timestep; /* Initial time step */
} EventCtx;

EventCtx *event;

typedef struct {
  PetscInt maxbounces;
  PetscInt nbounces;
} AppCtx;

#undef __FUNCT__
#define __FUNCT__ "EventMonitorSet"
PetscErrorCode EventMonitorSet(TS ts,PetscInt nevents,PetscErrorCode (*eventmonitor)(TS,PetscReal,Vec,PetscScalar*,PetscInt*,PetscBool*,void*),PetscErrorCode (*postevent)(TS,PetscInt,PetscInt[],PetscReal,Vec,void*),void *mectx)
{
  PetscErrorCode ierr;
  PetscReal      t;
  Vec            U;

  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(EventCtx),&event);CHKERRQ(ierr);
  ierr = PetscMalloc(nevents*sizeof(PetscScalar),&event->fvalue);CHKERRQ(ierr);
  ierr = PetscMalloc(nevents*sizeof(PetscScalar),&event->fvalue_prev);CHKERRQ(ierr);
  ierr = PetscMalloc(nevents*sizeof(PetscBool),&event->terminate);CHKERRQ(ierr);
  ierr = PetscMalloc(nevents*sizeof(PetscInt),&event->direction);CHKERRQ(ierr);
  event->monitor = eventmonitor;
  event->postevent = postevent;
  event->monitorcontext = (void*)mectx;
  event->nevents = nevents;

  ierr = TSGetTime(ts,&t);CHKERRQ(ierr);
  ierr = TSGetTimeStep(ts,&event->initial_timestep);CHKERRQ(ierr);
  ierr = TSGetSolution(ts,&U);CHKERRQ(ierr);
  event->ptime_prev = t;
  ierr = (*event->monitor)(ts,t,U,event->fvalue_prev,NULL,NULL,mectx);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"TS Event options","");CHKERRQ(ierr);
  {
    event->tol = 1.0e-6;
    ierr = PetscOptionsReal("-event_tol","","",event->tol,&event->tol,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EventMonitorDestroy"
PetscErrorCode EventMonitorDestroy(EventCtx **event)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree((*event)->fvalue);CHKERRQ(ierr);
  ierr = PetscFree((*event)->fvalue_prev);CHKERRQ(ierr);
  ierr = PetscFree((*event)->terminate);CHKERRQ(ierr);
  ierr = PetscFree((*event)->direction);CHKERRQ(ierr);
  ierr = PetscFree(*event);CHKERRQ(ierr);
  *event = NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EventFunction"
PetscErrorCode EventFunction(TS ts,PetscReal t,Vec U,PetscScalar *fvalue,PetscInt *direction,PetscBool *terminate,void *ctx)
{
  AppCtx         *app=(AppCtx*)ctx;
  PetscErrorCode ierr;
  PetscScalar    *u;

  PetscFunctionBegin;
  /* Event for ball height */
  ierr = VecGetArray(U,&u);CHKERRQ(ierr);
  fvalue[0] = u[0];
  if (terminate) terminate[0] = PETSC_FALSE;
  if (direction) direction[0] = -1;
  /* Event for number of bounces */
  fvalue[1] = app->maxbounces - app->nbounces;
  if (direction) direction[1] = -1;
  if (terminate) terminate[1] = PETSC_TRUE;
  ierr = VecRestoreArray(U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PostEventFunction"
PetscErrorCode PostEventFunction(TS ts,PetscInt nevents,PetscInt event_list[],PetscReal t,Vec U,void* ctx)
{
  AppCtx         *app=(AppCtx*)ctx;
  PetscErrorCode ierr;
  PetscScalar    *u;

  PetscFunctionBegin;
  if (event_list[0] == 0) {
    ierr = VecGetArray(U,&u);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"Ball hit the ground at t = %f seconds\n",t);CHKERRQ(ierr);
    /* Set new initial conditions with .9 attenuation */
    u[0] = 0.0;
    u[1] = -0.9*u[1];
    ierr = VecRestoreArray(U,&u);CHKERRQ(ierr);
    ierr = TSSetSolution(ts,U);CHKERRQ(ierr);
  } else if (event_list[0] == 1) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"Ball bounced %d times\n",app->nbounces);CHKERRQ(ierr);
  } 
  app->nbounces++;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PostEvent"
PetscErrorCode PostEvent(TS ts,PetscInt nevents,PetscInt event_list[],PetscReal t,Vec U,void *ctx)
{
  PetscErrorCode ierr;
  PetscBool      terminate=PETSC_TRUE;
  PetscInt       i;

  PetscFunctionBegin;
  if (event->postevent) {
    ierr = (*event->postevent)(ts,nevents,event_list,t,U,ctx);CHKERRQ(ierr);
  }
  for(i = 0; i < nevents;i++) {
    terminate = terminate && event->terminate[event_list[i]];
  }
  if (terminate) {
    ierr = TSSetConvergedReason(ts,TS_CONVERGED_EVENT);CHKERRQ(ierr);
    event->status = EVENT_NONE;
  } else {
    event->status = EVENT_RESET_NEXTSTEP;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PostStep"
PetscErrorCode PostStep(TS ts)
{
  PetscErrorCode ierr;
  PetscReal      t;
  Vec            U;
  PetscInt       i;
  PetscReal      dt;

  PetscFunctionBegin;

  ierr = TSGetTime(ts,&t);CHKERRQ(ierr);
  ierr = TSGetSolution(ts,&U);CHKERRQ(ierr);

  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  if (event->status == EVENT_RESET_NEXTSTEP) {
    /* Take initial time step */
    dt = event->initial_timestep;
    ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
    event->status = EVENT_NONE;
  }

  /* Save the original time step before the event is detected */
  if (event->status == EVENT_NONE) {
    event->tstepend   = t;
  }

  event->nevents_active = 0;

  ierr = (*event->monitor)(ts,t,U,event->fvalue,event->direction,event->terminate,event->monitorcontext);CHKERRQ(ierr);
  for (i=0; i < event->nevents; i++) {
    if (PetscAbs(event->fvalue[i]) < event->tol) {
      event->status = EVENT_OVER;
      if (event->nevents_active >= MAXACTIVEEVENTS) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Cannot handle %d simultaneous events",event->nevents_active);
      event->active_event_list[event->nevents_active++] = i;
    }
  }
  if (event->status == EVENT_OVER) {
    ierr = TSSetTimeStep(ts,event->tstepend-t);CHKERRQ(ierr);
    ierr = PostEvent(ts,event->nevents_active,event->active_event_list,t,U,event->monitorcontext);CHKERRQ(ierr);
    for (i = 0; i < event->nevents; i++) {
      event->fvalue_prev[i] = event->fvalue[i];
    }
    event->ptime_prev  = t;
    PetscFunctionReturn(0);
  }

  for (i = 0; i < event->nevents; i++) {
    if ((event->direction[i] < 0 && PetscSign(event->fvalue[i]) <= 0 && PetscSign(event->fvalue_prev[i]) >= 0) || \
        (event->direction[i] > 0 && PetscSign(event->fvalue[i]) >= 0 && PetscSign(event->fvalue_prev[i]) <= 0) || \
        (event->direction[i] == 0 && PetscSign(event->fvalue[i])*PetscSign(event->fvalue_prev[i]) <= 0)) {

      event->status = EVENT_DETECTED;
      
      /* Compute linearly interpolated new time step */
      dt = PetscMin(dt,-event->fvalue_prev[i]*(t - event->ptime_prev)/(event->fvalue[i] - event->fvalue_prev[i]));
    }
  }
  if (event->status == EVENT_DETECTED) {
    ierr = TSRollBack(ts);CHKERRQ(ierr);
    event->status = EVENT_PROCESSING;
  } else {
    for (i = 0; i < event->nevents; i++) {
      event->fvalue_prev[i] = event->fvalue[i];
    }
    event->ptime_prev  = t;
    if (event->status == EVENT_PROCESSING) {
      dt = event->tstepend - event->ptime_prev;
    }
  }
  ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "IFunction"
/*
     Defines the ODE passed to the ODE solver
*/
static PetscErrorCode IFunction(TS ts,PetscReal t,Vec U,Vec Udot,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  PetscScalar    *u,*udot,*f;

  PetscFunctionBegin;
  /*  The next three lines allow us to access the entries of the vectors directly */
  ierr = VecGetArray(U,&u);CHKERRQ(ierr);
  ierr = VecGetArray(Udot,&udot);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);

  f[0] = udot[0] - u[1];
  f[1] = udot[1] + 9.8;

  ierr = VecRestoreArray(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArray(Udot,&udot);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "IJacobian"
/*
     Defines the Jacobian of the ODE passed to the ODE solver. See TSSetIJacobian() for the meaning of a and the Jacobian.
*/
static PetscErrorCode IJacobian(TS ts,PetscReal t,Vec U,Vec Udot,PetscReal a,Mat *A,Mat *B,MatStructure *flag,void *ctx)
{
  PetscErrorCode ierr;
  PetscInt       rowcol[] = {0,1};
  PetscScalar    *u,*udot,J[2][2];

  PetscFunctionBegin;
  ierr = VecGetArray(U,&u);CHKERRQ(ierr);
  ierr = VecGetArray(Udot,&udot);CHKERRQ(ierr);

  J[0][0] = a;                       J[0][1] = -1;
  J[1][0] = 0.0;                     J[1][1] = a;
  ierr    = MatSetValues(*B,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);

  ierr    = VecRestoreArray(U,&u);CHKERRQ(ierr);
  ierr    = VecRestoreArray(Udot,&udot);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (*A != *B) {
    ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  *flag = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}
  
#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  TS             ts;            /* ODE integrator */
  Vec            U;             /* solution will be stored here */
  Mat            A;             /* Jacobian matrix */
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscInt       n = 2;
  PetscScalar    *u;
  AppCtx         app;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Only for sequential runs");

  app.nbounces = 0;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"ex21 options","");CHKERRQ(ierr);
  {
    app.maxbounces = 10;
    ierr = PetscOptionsInt("-maxbounces","","",app.maxbounces,&app.maxbounces,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,n,n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(A,MATDENSE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  ierr = MatGetVecs(A,&U,NULL);CHKERRQ(ierr);

  ierr = VecGetArray(U,&u);CHKERRQ(ierr);
  u[0] = 0.0;
  u[1] = 20.0;
  ierr = VecRestoreArray(U,&u);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSROSW);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,(TSIFunction) IFunction,NULL);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,A,A,(TSIJacobian)IJacobian,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetSolution(ts,U);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetDuration(ts,1000,30.0);CHKERRQ(ierr);
  ierr = TSSetInitialTimeStep(ts,0.0,0.1);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSetPostStep(ts,PostStep);CHKERRQ(ierr);
  
  ierr = EventMonitorSet(ts,2,EventFunction,PostEventFunction,(void*)&app);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Run timestepping solver
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = TSSolve(ts,U);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
   
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = EventMonitorDestroy(&event);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  
  ierr = PetscFinalize();
  return(0);
}
