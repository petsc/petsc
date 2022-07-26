
static char help[] = "Solves 1D heat equation with FEM formulation.\n\
Input arguments are\n\
  -useAlhs: solve Alhs*U' =  (Arhs*U + g) \n\
            otherwise, solve U' = inv(Alhs)*(Arhs*U + g) \n\n";

/*--------------------------------------------------------------------------
  Solves 1D heat equation U_t = U_xx with FEM formulation:
                          Alhs*U' = rhs (= Arhs*U + g)
  We thank Chris Cox <clcox@clemson.edu> for contributing the original code
----------------------------------------------------------------------------*/

#include <petscksp.h>
#include <petscts.h>

/* special variable - max size of all arrays  */
#define num_z 10

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines.
*/
typedef struct {
  Mat         Amat;               /* left hand side matrix */
  Vec         ksp_rhs,ksp_sol;    /* working vectors for formulating inv(Alhs)*(Arhs*U+g) */
  int         max_probsz;         /* max size of the problem */
  PetscBool   useAlhs;            /* flag (1 indicates solving Alhs*U' = Arhs*U+g */
  int         nz;                 /* total number of grid points */
  PetscInt    m;                  /* total number of interio grid points */
  Vec         solution;           /* global exact ts solution vector */
  PetscScalar *z;                 /* array of grid points */
  PetscBool   debug;              /* flag (1 indicates activation of debugging printouts) */
} AppCtx;

extern PetscScalar exact(PetscScalar,PetscReal);
extern PetscErrorCode Monitor(TS,PetscInt,PetscReal,Vec,void*);
extern PetscErrorCode Petsc_KSPSolve(AppCtx*);
extern PetscScalar bspl(PetscScalar*,PetscScalar,PetscInt,PetscInt,PetscInt[][2],PetscInt);
extern PetscErrorCode femBg(PetscScalar[][3],PetscScalar*,PetscInt,PetscScalar*,PetscReal);
extern PetscErrorCode femA(AppCtx*,PetscInt,PetscScalar*);
extern PetscErrorCode rhs(AppCtx*,PetscScalar*, PetscInt, PetscScalar*,PetscReal);
extern PetscErrorCode RHSfunction(TS,PetscReal,Vec,Vec,void*);

int main(int argc,char **argv)
{
  PetscInt       i,m,nz,steps,max_steps,k,nphase=1;
  PetscScalar    zInitial,zFinal,val,*z;
  PetscReal      stepsz[4],T,ftime;
  TS             ts;
  SNES           snes;
  Mat            Jmat;
  AppCtx         appctx;   /* user-defined application context */
  Vec            init_sol; /* ts solution vector */
  PetscMPIInt    size;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_SELF,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only");

  /* initializations */
  zInitial  = 0.0;
  zFinal    = 1.0;
  nz        = num_z;
  m         = nz-2;
  appctx.nz = nz;
  max_steps = (PetscInt)10000;

  appctx.m          = m;
  appctx.max_probsz = nz;
  appctx.debug      = PETSC_FALSE;
  appctx.useAlhs    = PETSC_FALSE;

  PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"","");
  PetscCall(PetscOptionsName("-debug",NULL,NULL,&appctx.debug));
  PetscCall(PetscOptionsName("-useAlhs",NULL,NULL,&appctx.useAlhs));
  PetscCall(PetscOptionsRangeInt("-nphase",NULL,NULL,nphase,&nphase,NULL,1,3));
  PetscOptionsEnd();
  T = 0.014/nphase;

  /* create vector to hold ts solution */
  /*-----------------------------------*/
  PetscCall(VecCreate(PETSC_COMM_WORLD, &init_sol));
  PetscCall(VecSetSizes(init_sol, PETSC_DECIDE, m));
  PetscCall(VecSetFromOptions(init_sol));

  /* create vector to hold true ts soln for comparison */
  PetscCall(VecDuplicate(init_sol, &appctx.solution));

  /* create LHS matrix Amat */
  /*------------------------*/
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_WORLD, m, m, 3, NULL, &appctx.Amat));
  PetscCall(MatSetFromOptions(appctx.Amat));
  PetscCall(MatSetUp(appctx.Amat));
  /* set space grid points - interio points only! */
  PetscCall(PetscMalloc1(nz+1,&z));
  for (i=0; i<nz; i++) z[i]=(i)*((zFinal-zInitial)/(nz-1));
  appctx.z = z;
  femA(&appctx,nz,z);

  /* create the jacobian matrix */
  /*----------------------------*/
  PetscCall(MatCreate(PETSC_COMM_WORLD, &Jmat));
  PetscCall(MatSetSizes(Jmat,PETSC_DECIDE,PETSC_DECIDE,m,m));
  PetscCall(MatSetFromOptions(Jmat));
  PetscCall(MatSetUp(Jmat));

  /* create working vectors for formulating rhs=inv(Alhs)*(Arhs*U + g) */
  PetscCall(VecDuplicate(init_sol,&appctx.ksp_rhs));
  PetscCall(VecDuplicate(init_sol,&appctx.ksp_sol));

  /* set initial guess */
  /*-------------------*/
  for (i=0; i<nz-2; i++) {
    val  = exact(z[i+1], 0.0);
    PetscCall(VecSetValue(init_sol,i,(PetscScalar)val,INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(init_sol));
  PetscCall(VecAssemblyEnd(init_sol));

  /*create a time-stepping context and set the problem type */
  /*--------------------------------------------------------*/
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetProblemType(ts,TS_NONLINEAR));

  /* set time-step method */
  PetscCall(TSSetType(ts,TSCN));

  /* Set optional user-defined monitoring routine */
  PetscCall(TSMonitorSet(ts,Monitor,&appctx,NULL));
  /* set the right hand side of U_t = RHSfunction(U,t) */
  PetscCall(TSSetRHSFunction(ts,NULL,(PetscErrorCode (*)(TS,PetscScalar,Vec,Vec,void*))RHSfunction,&appctx));

  if (appctx.useAlhs) {
    /* set the left hand side matrix of Amat*U_t = rhs(U,t) */

    /* Note: this approach is incompatible with the finite differenced Jacobian set below because we can't restore the
     * Alhs matrix without making a copy.  Either finite difference the entire thing or use analytic Jacobians in both
     * places.
     */
    PetscCall(TSSetIFunction(ts,NULL,TSComputeIFunctionLinear,&appctx));
    PetscCall(TSSetIJacobian(ts,appctx.Amat,appctx.Amat,TSComputeIJacobianConstant,&appctx));
  }

  /* use petsc to compute the jacobian by finite differences */
  PetscCall(TSGetSNES(ts,&snes));
  PetscCall(SNESSetJacobian(snes,Jmat,Jmat,SNESComputeJacobianDefault,NULL));

  /* get the command line options if there are any and set them */
  PetscCall(TSSetFromOptions(ts));

#if defined(PETSC_HAVE_SUNDIALS2)
  {
    TSType    type;
    PetscBool sundialstype=PETSC_FALSE;
    PetscCall(TSGetType(ts,&type));
    PetscCall(PetscObjectTypeCompare((PetscObject)ts,TSSUNDIALS,&sundialstype));
    PetscCheck(!sundialstype || !appctx.useAlhs,PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot use Alhs formulation for TSSUNDIALS type");
  }
#endif
  /* Sets the initial solution */
  PetscCall(TSSetSolution(ts,init_sol));

  stepsz[0] = 1.0/(2.0*(nz-1)*(nz-1)); /* (mesh_size)^2/2.0 */
  ftime     = 0.0;
  for (k=0; k<nphase; k++) {
    if (nphase > 1) PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Phase %" PetscInt_FMT " initial time %g, stepsz %g, duration: %g\n",k,(double)ftime,(double)stepsz[k],(double)((k+1)*T)));
    PetscCall(TSSetTime(ts,ftime));
    PetscCall(TSSetTimeStep(ts,stepsz[k]));
    PetscCall(TSSetMaxSteps(ts,max_steps));
    PetscCall(TSSetMaxTime(ts,(k+1)*T));
    PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));

    /* loop over time steps */
    /*----------------------*/
    PetscCall(TSSolve(ts,init_sol));
    PetscCall(TSGetSolveTime(ts,&ftime));
    PetscCall(TSGetStepNumber(ts,&steps));
    stepsz[k+1] = stepsz[k]*1.5; /* change step size for the next phase */
  }

  /* free space */
  PetscCall(TSDestroy(&ts));
  PetscCall(MatDestroy(&appctx.Amat));
  PetscCall(MatDestroy(&Jmat));
  PetscCall(VecDestroy(&appctx.ksp_rhs));
  PetscCall(VecDestroy(&appctx.ksp_sol));
  PetscCall(VecDestroy(&init_sol));
  PetscCall(VecDestroy(&appctx.solution));
  PetscCall(PetscFree(z));

  PetscCall(PetscFinalize());
  return 0;
}

/*------------------------------------------------------------------------
  Set exact solution
  u(z,t) = sin(6*PI*z)*exp(-36.*PI*PI*t) + 3.*sin(2*PI*z)*exp(-4.*PI*PI*t)
--------------------------------------------------------------------------*/
PetscScalar exact(PetscScalar z,PetscReal t)
{
  PetscScalar val, ex1, ex2;

  ex1 = PetscExpReal(-36.*PETSC_PI*PETSC_PI*t);
  ex2 = PetscExpReal(-4.*PETSC_PI*PETSC_PI*t);
  val = PetscSinScalar(6*PETSC_PI*z)*ex1 + 3.*PetscSinScalar(2*PETSC_PI*z)*ex2;
  return val;
}

/*
   Monitor - User-provided routine to monitor the solution computed at
   each timestep.  This example plots the solution and computes the
   error in two different norms.

   Input Parameters:
   ts     - the timestep context
   step   - the count of the current step (with 0 meaning the
             initial condition)
   time   - the current time
   u      - the solution at this timestep
   ctx    - the user-provided context for this monitoring routine.
            In this case we use the application context which contains
            information about the problem size, workspace and the exact
            solution.
*/
PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal time,Vec u,void *ctx)
{
  AppCtx         *appctx = (AppCtx*)ctx;
  PetscInt       i,m=appctx->m;
  PetscReal      norm_2,norm_max,h=1.0/(m+1);
  PetscScalar    *u_exact;

  /* Compute the exact solution */
  PetscCall(VecGetArrayWrite(appctx->solution,&u_exact));
  for (i=0; i<m; i++) u_exact[i] = exact(appctx->z[i+1],time);
  PetscCall(VecRestoreArrayWrite(appctx->solution,&u_exact));

  /* Print debugging information if desired */
  if (appctx->debug) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"Computed solution vector at time %g\n",(double)time));
    PetscCall(VecView(u,PETSC_VIEWER_STDOUT_SELF));
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"Exact solution vector\n"));
    PetscCall(VecView(appctx->solution,PETSC_VIEWER_STDOUT_SELF));
  }

  /* Compute the 2-norm and max-norm of the error */
  PetscCall(VecAXPY(appctx->solution,-1.0,u));
  PetscCall(VecNorm(appctx->solution,NORM_2,&norm_2));

  norm_2 = PetscSqrtReal(h)*norm_2;
  PetscCall(VecNorm(appctx->solution,NORM_MAX,&norm_max));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"Timestep %" PetscInt_FMT ": time = %g, 2-norm error = %6.4f, max norm error = %6.4f\n",step,(double)time,(double)norm_2,(double)norm_max));

  /*
     Print debugging information if desired
  */
  if (appctx->debug) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"Error vector\n"));
    PetscCall(VecView(appctx->solution,PETSC_VIEWER_STDOUT_SELF));
  }
  return 0;
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      Function to solve a linear system using KSP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/

PetscErrorCode Petsc_KSPSolve(AppCtx *obj)
{
  KSP            ksp;
  PC             pc;

  /*create the ksp context and set the operators,that is, associate the system matrix with it*/
  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  PetscCall(KSPSetOperators(ksp,obj->Amat,obj->Amat));

  /*get the preconditioner context, set its type and the tolerances*/
  PetscCall(KSPGetPC(ksp,&pc));
  PetscCall(PCSetType(pc,PCLU));
  PetscCall(KSPSetTolerances(ksp,1.e-7,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT));

  /*get the command line options if there are any and set them*/
  PetscCall(KSPSetFromOptions(ksp));

  /*get the linear system (ksp) solve*/
  PetscCall(KSPSolve(ksp,obj->ksp_rhs,obj->ksp_sol));

  PetscCall(KSPDestroy(&ksp));
  return 0;
}

/***********************************************************************
  Function to return value of basis function or derivative of basis function.
 ***********************************************************************

        Arguments:
          x       = array of xpoints or nodal values
          xx      = point at which the basis function is to be
                      evaluated.
          il      = interval containing xx.
          iq      = indicates which of the two basis functions in
                      interval intrvl should be used
          nll     = array containing the endpoints of each interval.
          id      = If id ~= 2, the value of the basis function
                      is calculated; if id = 2, the value of the
                      derivative of the basis function is returned.
 ***********************************************************************/

PetscScalar bspl(PetscScalar *x, PetscScalar xx,PetscInt il,PetscInt iq,PetscInt nll[][2],PetscInt id)
{
  PetscScalar x1,x2,bfcn;
  PetscInt    i1,i2,iq1,iq2;

  /* Determine which basis function in interval intrvl is to be used in */
  iq1 = iq;
  if (iq1==0) iq2 = 1;
  else iq2 = 0;

  /*    Determine endpoint of the interval intrvl   */
  i1=nll[il][iq1];
  i2=nll[il][iq2];

  /*   Determine nodal values at the endpoints of the interval intrvl   */
  x1=x[i1];
  x2=x[i2];

  /*   Evaluate basis function   */
  if (id == 2) bfcn=(1.0)/(x1-x2);
  else bfcn=(xx-x2)/(x1-x2);
  return bfcn;
}

/*---------------------------------------------------------
  Function called by rhs function to get B and g
---------------------------------------------------------*/
PetscErrorCode femBg(PetscScalar btri[][3],PetscScalar *f,PetscInt nz,PetscScalar *z, PetscReal t)
{
  PetscInt    i,j,jj,il,ip,ipp,ipq,iq,iquad,iqq;
  PetscInt    nli[num_z][2],indx[num_z];
  PetscScalar dd,dl,zip,zipq,zz,b_z,bb_z,bij;
  PetscScalar zquad[num_z][3],dlen[num_z],qdwt[3];

  /*  initializing everything - btri and f are initialized in rhs.c  */
  for (i=0; i < nz; i++) {
    nli[i][0]   = 0;
    nli[i][1]   = 0;
    indx[i]     = 0;
    zquad[i][0] = 0.0;
    zquad[i][1] = 0.0;
    zquad[i][2] = 0.0;
    dlen[i]     = 0.0;
  } /*end for (i)*/

  /*  quadrature weights  */
  qdwt[0] = 1.0/6.0;
  qdwt[1] = 4.0/6.0;
  qdwt[2] = 1.0/6.0;

  /* 1st and last nodes have Dirichlet boundary condition -
     set indices there to -1 */

  for (i=0; i < nz-1; i++) indx[i] = i-1;
  indx[nz-1] = -1;

  ipq = 0;
  for (il=0; il < nz-1; il++) {
    ip           = ipq;
    ipq          = ip+1;
    zip          = z[ip];
    zipq         = z[ipq];
    dl           = zipq-zip;
    zquad[il][0] = zip;
    zquad[il][1] = (0.5)*(zip+zipq);
    zquad[il][2] = zipq;
    dlen[il]     = PetscAbsScalar(dl);
    nli[il][0]   = ip;
    nli[il][1]   = ipq;
  }

  for (il=0; il < nz-1; il++) {
    for (iquad=0; iquad < 3; iquad++) {
      dd = (dlen[il])*(qdwt[iquad]);
      zz = zquad[il][iquad];

      for (iq=0; iq < 2; iq++) {
        ip  = nli[il][iq];
        b_z = bspl(z,zz,il,iq,nli,2);
        i   = indx[ip];

        if (i > -1) {
          for (iqq=0; iqq < 2; iqq++) {
            ipp  = nli[il][iqq];
            bb_z = bspl(z,zz,il,iqq,nli,2);
            j    = indx[ipp];
            bij  = -b_z*bb_z;

            if (j > -1) {
              jj = 1+j-i;
              btri[i][jj] += bij*dd;
            } else {
              f[i] += bij*dd*exact(z[ipp], t);
              /* f[i] += 0.0; */
              /* if (il==0 && j==-1) { */
              /* f[i] += bij*dd*exact(zz,t); */
              /* }*/ /*end if*/
            } /*end else*/
          } /*end for (iqq)*/
        } /*end if (i>0)*/
      } /*end for (iq)*/
    } /*end for (iquad)*/
  } /*end for (il)*/
  return 0;
}

PetscErrorCode femA(AppCtx *obj,PetscInt nz,PetscScalar *z)
{
  PetscInt       i,j,il,ip,ipp,ipq,iq,iquad,iqq;
  PetscInt       nli[num_z][2],indx[num_z];
  PetscScalar    dd,dl,zip,zipq,zz,bb,bbb,aij;
  PetscScalar    rquad[num_z][3],dlen[num_z],qdwt[3],add_term;

  /*  initializing everything  */
  for (i=0; i < nz; i++) {
    nli[i][0]   = 0;
    nli[i][1]   = 0;
    indx[i]     = 0;
    rquad[i][0] = 0.0;
    rquad[i][1] = 0.0;
    rquad[i][2] = 0.0;
    dlen[i]     = 0.0;
  } /*end for (i)*/

  /*  quadrature weights  */
  qdwt[0] = 1.0/6.0;
  qdwt[1] = 4.0/6.0;
  qdwt[2] = 1.0/6.0;

  /* 1st and last nodes have Dirichlet boundary condition -
     set indices there to -1 */

  for (i=0; i < nz-1; i++) indx[i]=i-1;
  indx[nz-1]=-1;

  ipq = 0;

  for (il=0; il < nz-1; il++) {
    ip           = ipq;
    ipq          = ip+1;
    zip          = z[ip];
    zipq         = z[ipq];
    dl           = zipq-zip;
    rquad[il][0] = zip;
    rquad[il][1] = (0.5)*(zip+zipq);
    rquad[il][2] = zipq;
    dlen[il]     = PetscAbsScalar(dl);
    nli[il][0]   = ip;
    nli[il][1]   = ipq;
  } /*end for (il)*/

  for (il=0; il < nz-1; il++) {
    for (iquad=0; iquad < 3; iquad++) {
      dd = (dlen[il])*(qdwt[iquad]);
      zz = rquad[il][iquad];

      for (iq=0; iq < 2; iq++) {
        ip = nli[il][iq];
        bb = bspl(z,zz,il,iq,nli,1);
        i = indx[ip];
        if (i > -1) {
          for (iqq=0; iqq < 2; iqq++) {
            ipp = nli[il][iqq];
            bbb = bspl(z,zz,il,iqq,nli,1);
            j = indx[ipp];
            aij = bb*bbb;
            if (j > -1) {
              add_term = aij*dd;
              PetscCall(MatSetValue(obj->Amat,i,j,add_term,ADD_VALUES));
            }/*endif*/
          } /*end for (iqq)*/
        } /*end if (i>0)*/
      } /*end for (iq)*/
    } /*end for (iquad)*/
  } /*end for (il)*/
  PetscCall(MatAssemblyBegin(obj->Amat,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(obj->Amat,MAT_FINAL_ASSEMBLY));
  return 0;
}

/*---------------------------------------------------------
        Function to fill the rhs vector with
        By + g values ****
---------------------------------------------------------*/
PetscErrorCode rhs(AppCtx *obj,PetscScalar *y, PetscInt nz, PetscScalar *z, PetscReal t)
{
  PetscInt       i,j,js,je,jj;
  PetscScalar    val,g[num_z],btri[num_z][3],add_term;

  for (i=0; i < nz-2; i++) {
    for (j=0; j <= 2; j++) btri[i][j]=0.0;
    g[i] = 0.0;
  }

  /*  call femBg to set the tri-diagonal b matrix and vector g  */
  femBg(btri,g,nz,z,t);

  /*  setting the entries of the right hand side vector  */
  for (i=0; i < nz-2; i++) {
    val = 0.0;
    js  = 0;
    if (i == 0) js = 1;
    je = 2;
    if (i == nz-2) je = 1;

    for (jj=js; jj <= je; jj++) {
      j    = i+jj-1;
      val += (btri[i][jj])*(y[j]);
    }
    add_term = val + g[i];
    PetscCall(VecSetValue(obj->ksp_rhs,(PetscInt)i,(PetscScalar)add_term,INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(obj->ksp_rhs));
  PetscCall(VecAssemblyEnd(obj->ksp_rhs));
  return 0;
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Function to form the right hand side of the time-stepping problem.                       %%
%% -------------------------------------------------------------------------------------------%%
  if (useAlhs):
    globalout = By+g
  else if (!useAlhs):
    globalout = f(y,t)=Ainv(By+g),
      in which the ksp solver to transform the problem A*ydot=By+g
      to the problem ydot=f(y,t)=inv(A)*(By+g)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/

PetscErrorCode RHSfunction(TS ts,PetscReal t,Vec globalin,Vec globalout,void *ctx)
{
  AppCtx            *obj = (AppCtx*)ctx;
  PetscScalar       soln[num_z];
  const PetscScalar *soln_ptr;
  PetscInt          i,nz=obj->nz;
  PetscReal         time;

  /* get the previous solution to compute updated system */
  PetscCall(VecGetArrayRead(globalin,&soln_ptr));
  for (i=0; i < num_z-2; i++) soln[i] = soln_ptr[i];
  PetscCall(VecRestoreArrayRead(globalin,&soln_ptr));
  soln[num_z-1] = 0.0;
  soln[num_z-2] = 0.0;

  /* clear out the matrix and rhs for ksp to keep things straight */
  PetscCall(VecSet(obj->ksp_rhs,(PetscScalar)0.0));

  time = t;
  /* get the updated system */
  rhs(obj,soln,nz,obj->z,time); /* setup of the By+g rhs */

  /* do a ksp solve to get the rhs for the ts problem */
  if (obj->useAlhs) {
    /* ksp_sol = ksp_rhs */
    PetscCall(VecCopy(obj->ksp_rhs,globalout));
  } else {
    /* ksp_sol = inv(Amat)*ksp_rhs */
    PetscCall(Petsc_KSPSolve(obj));
    PetscCall(VecCopy(obj->ksp_sol,globalout));
  }
  return 0;
}

/*TEST

    build:
      requires: !complex

    test:
      suffix: euler
      output_file: output/ex3.out

    test:
      suffix: 2
      args:   -useAlhs
      output_file: output/ex3.out
      TODO: Broken because SNESComputeJacobianDefault is incompatible with TSComputeIJacobianConstant

TEST*/
