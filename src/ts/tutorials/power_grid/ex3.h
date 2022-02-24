typedef enum {SA_ADJ, SA_TLM} SAMethod;
static const char *const SAMethods[] = {"ADJ","TLM","SAMethod","SA_",0};

typedef struct {
  PetscScalar H,D,omega_b,omega_s,Pmax,Pmax_ini,Pm,E,V,X,u_s,c;
  PetscInt    beta;
  PetscReal   tf,tcl;
  /* Solver context */
  TS          ts,quadts;
  Vec         U;    /* solution will be stored here */
  Mat         Jac;  /* Jacobian matrix */
  Mat         Jacp; /* Jacobianp matrix */
  Mat         DRDU,DRDP;
  SAMethod    sa;
} AppCtx;

/* Event check */
PetscErrorCode EventFunction(TS ts,PetscReal t,Vec X,PetscScalar *fvalue,void *ctx)
{
  AppCtx *user=(AppCtx*)ctx;

  PetscFunctionBegin;
  /* Event for fault-on time */
  fvalue[0] = t - user->tf;
  /* Event for fault-off time */
  fvalue[1] = t - user->tcl;

  PetscFunctionReturn(0);
}

PetscErrorCode PostEventFunction(TS ts,PetscInt nevents,PetscInt event_list[],PetscReal t,Vec X,PetscBool forwardsolve,void* ctx)
{
  AppCtx *user=(AppCtx*)ctx;

  PetscFunctionBegin;
  if (event_list[0] == 0) {
    if (forwardsolve) user->Pmax = 0.0; /* Apply disturbance - this is done by setting Pmax = 0 */
    else user->Pmax = user->Pmax_ini; /* Going backward, reversal of event */
  } else if (event_list[0] == 1) {
    if (forwardsolve) user->Pmax = user->Pmax_ini; /* Remove the fault  - this is done by setting Pmax = Pmax_ini */
    else user->Pmax = 0.0; /* Going backward, reversal of event */
  }
  CHKERRQ(TSRestartStep(ts)); /* Must set restart flag to ture, otherwise methods with FSAL will fail */
  PetscFunctionReturn(0);
}

/*
     Defines the ODE passed to the ODE solver
*/
PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec U,Vec F,AppCtx *ctx)
{
  PetscScalar       *f,Pmax;
  const PetscScalar *u;

  PetscFunctionBegin;
  /*  The next three lines allow us to access the entries of the vectors directly */
  CHKERRQ(VecGetArrayRead(U,&u));
  CHKERRQ(VecGetArray(F,&f));
  Pmax = ctx->Pmax;
  f[0] = ctx->omega_b*(u[1] - ctx->omega_s);
  f[1] = ctx->omega_s/(2.0*ctx->H)*(ctx->Pm - Pmax*PetscSinScalar(u[0]) - ctx->D*(u[1] - ctx->omega_s));

  CHKERRQ(VecRestoreArrayRead(U,&u));
  CHKERRQ(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

/*
     Defines the Jacobian of the ODE passed to the ODE solver. See TSSetRHSJacobian() for the meaning of a and the Jacobian.
*/
PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec U,Mat A,Mat B,AppCtx *ctx)
{
  PetscInt          rowcol[] = {0,1};
  PetscScalar       J[2][2],Pmax;
  const PetscScalar *u;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(U,&u));
  Pmax    = ctx->Pmax;
  J[0][0] = 0;
  J[0][1] = ctx->omega_b;
  J[1][0] = -ctx->omega_s/(2.0*ctx->H)*Pmax*PetscCosScalar(u[0]);
  J[1][1] = -ctx->omega_s/(2.0*ctx->H)*ctx->D;
  CHKERRQ(MatSetValues(B,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES));
  CHKERRQ(VecRestoreArrayRead(U,&u));

  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  if (A != B) {
    CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

/*
     Defines the ODE passed to the ODE solver
*/
PetscErrorCode IFunction(TS ts,PetscReal t,Vec U,Vec Udot,Vec F,AppCtx *ctx)
{
  PetscScalar       *f,Pmax;
  const PetscScalar *u,*udot;

  PetscFunctionBegin;
  /*  The next three lines allow us to access the entries of the vectors directly */
  CHKERRQ(VecGetArrayRead(U,&u));
  CHKERRQ(VecGetArrayRead(Udot,&udot));
  CHKERRQ(VecGetArray(F,&f));
  Pmax = ctx->Pmax;
  f[0] = udot[0] - ctx->omega_b*(u[1] - ctx->omega_s);
  f[1] = 2.0*ctx->H/ctx->omega_s*udot[1] +  Pmax*PetscSinScalar(u[0]) + ctx->D*(u[1] - ctx->omega_s)- ctx->Pm;

  CHKERRQ(VecRestoreArrayRead(U,&u));
  CHKERRQ(VecRestoreArrayRead(Udot,&udot));
  CHKERRQ(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

/*
     Defines the Jacobian of the ODE passed to the ODE solver. See TSSetIJacobian() for the meaning of a and the Jacobian.
*/
PetscErrorCode IJacobian(TS ts,PetscReal t,Vec U,Vec Udot,PetscReal a,Mat A,Mat B,AppCtx *ctx)
{
  PetscInt          rowcol[] = {0,1};
  PetscScalar       J[2][2],Pmax;
  const PetscScalar *u,*udot;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(U,&u));
  CHKERRQ(VecGetArrayRead(Udot,&udot));
  Pmax = ctx->Pmax;
  J[0][0] = a;                       J[0][1] = -ctx->omega_b;
  J[1][1] = 2.0*ctx->H/ctx->omega_s*a + ctx->D;   J[1][0] = Pmax*PetscCosScalar(u[0]);

  CHKERRQ(MatSetValues(B,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES));
  CHKERRQ(VecRestoreArrayRead(U,&u));
  CHKERRQ(VecRestoreArrayRead(Udot,&udot));

  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  if (A != B) {
    CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode RHSJacobianP(TS ts,PetscReal t,Vec X,Mat A,void *ctx0)
{
  PetscInt       row[] = {0,1},col[] = {0};
  PetscScalar    *x,J[2][1];
  AppCtx         *ctx = (AppCtx*)ctx0;

  PetscFunctionBeginUser;
  CHKERRQ(VecGetArray(X,&x));
  J[0][0] = 0;
  J[1][0] = ctx->omega_s/(2.0*ctx->H);
  CHKERRQ(MatSetValues(A,2,row,1,col,&J[0][0],INSERT_VALUES));

  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

PetscErrorCode CostIntegrand(TS ts,PetscReal t,Vec U,Vec R,AppCtx *ctx)
{
  PetscScalar       *r;
  const PetscScalar *u;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(U,&u));
  CHKERRQ(VecGetArray(R,&r));
  r[0] = ctx->c*PetscPowScalarInt(PetscMax(0.,u[0]-ctx->u_s),ctx->beta);
  CHKERRQ(VecRestoreArray(R,&r));
  CHKERRQ(VecRestoreArrayRead(U,&u));
  PetscFunctionReturn(0);
}

/* Transpose of DRDU */
PetscErrorCode DRDUJacobianTranspose(TS ts,PetscReal t,Vec U,Mat DRDU,Mat B,AppCtx *ctx)
{
  PetscScalar       ru[2];
  PetscInt          row[] = {0,1},col[] = {0};
  const PetscScalar *u;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(U,&u));
  ru[0] = ctx->c*ctx->beta*PetscPowScalarInt(PetscMax(0.,u[0]-ctx->u_s),ctx->beta-1);
  ru[1] = 0;
  CHKERRQ(MatSetValues(DRDU,2,row,1,col,ru,INSERT_VALUES));
  CHKERRQ(VecRestoreArrayRead(U,&u));
  CHKERRQ(MatAssemblyBegin(DRDU,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(DRDU,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

PetscErrorCode DRDPJacobianTranspose(TS ts,PetscReal t,Vec U,Mat DRDP,void *ctx)
{
  PetscFunctionBegin;
  CHKERRQ(MatZeroEntries(DRDP));
  CHKERRQ(MatAssemblyBegin(DRDP,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(DRDP,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeSensiP(Vec lambda,Vec mu,AppCtx *ctx)
{
  PetscScalar       *y,sensip;
  const PetscScalar *x;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(lambda,&x));
  CHKERRQ(VecGetArray(mu,&y));
  sensip = 1./PetscSqrtScalar(1.-(ctx->Pm/ctx->Pmax)*(ctx->Pm/ctx->Pmax))/ctx->Pmax*x[0]+y[0];
  y[0] = sensip;
  CHKERRQ(VecRestoreArray(mu,&y));
  CHKERRQ(VecRestoreArrayRead(lambda,&x));
  PetscFunctionReturn(0);
}
