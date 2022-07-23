/*
       The Problem:
           Solve the convection-diffusion equation:

             u_t+a*(u_x+u_y)=epsilon*(u_xx+u_yy)
             u=0   at x=0, y=0
             u_x=0 at x=1
             u_y=0 at y=1
             u = exp(-20.0*(pow(x-0.5,2.0)+pow(y-0.5,2.0))) at t=0

       This program tests the routine of computing the Jacobian by the
       finite difference method as well as PETSc.

*/

static char help[] = "Solve the convection-diffusion equation. \n\n";

#include <petscts.h>

typedef struct
{
  PetscInt  m;          /* the number of mesh points in x-direction */
  PetscInt  n;          /* the number of mesh points in y-direction */
  PetscReal dx;         /* the grid space in x-direction */
  PetscReal dy;         /* the grid space in y-direction */
  PetscReal a;          /* the convection coefficient    */
  PetscReal epsilon;    /* the diffusion coefficient     */
  PetscReal tfinal;
} Data;

extern PetscErrorCode Monitor(TS,PetscInt,PetscReal,Vec,void*);
extern PetscErrorCode Initial(Vec,void*);
extern PetscErrorCode RHSFunction(TS,PetscReal,Vec,Vec,void*);
extern PetscErrorCode RHSJacobian(TS,PetscReal,Vec,Mat,Mat,void*);
extern PetscErrorCode PostStep(TS);

int main(int argc,char **argv)
{
  PetscInt       time_steps=100,iout,NOUT=1;
  Vec            global;
  PetscReal      dt,ftime,ftime_original;
  TS             ts;
  PetscViewer    viewfile;
  Mat            J = 0;
  Vec            x;
  Data           data;
  PetscInt       mn;
  PetscBool      flg;
  MatColoring    mc;
  ISColoring     iscoloring;
  MatFDColoring  matfdcoloring        = 0;
  PetscBool      fd_jacobian_coloring = PETSC_FALSE;
  SNES           snes;
  KSP            ksp;
  PC             pc;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));

  /* set data */
  data.m       = 9;
  data.n       = 9;
  data.a       = 1.0;
  data.epsilon = 0.1;
  data.dx      = 1.0/(data.m+1.0);
  data.dy      = 1.0/(data.n+1.0);
  mn           = (data.m)*(data.n);
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-time",&time_steps,NULL));

  /* set initial conditions */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&global));
  PetscCall(VecSetSizes(global,PETSC_DECIDE,mn));
  PetscCall(VecSetFromOptions(global));
  PetscCall(Initial(global,&data));
  PetscCall(VecDuplicate(global,&x));

  /* create timestep context */
  PetscCall(TSCreate(PETSC_COMM_WORLD,&ts));
  PetscCall(TSMonitorSet(ts,Monitor,&data,NULL));
  PetscCall(TSSetType(ts,TSEULER));
  dt   = 0.1;
  ftime_original = data.tfinal = 1.0;

  PetscCall(TSSetTimeStep(ts,dt));
  PetscCall(TSSetMaxSteps(ts,time_steps));
  PetscCall(TSSetMaxTime(ts,ftime_original));
  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSetSolution(ts,global));

  /* set user provided RHSFunction and RHSJacobian */
  PetscCall(TSSetRHSFunction(ts,NULL,RHSFunction,&data));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&J));
  PetscCall(MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,mn,mn));
  PetscCall(MatSetFromOptions(J));
  PetscCall(MatSeqAIJSetPreallocation(J,5,NULL));
  PetscCall(MatMPIAIJSetPreallocation(J,5,NULL,5,NULL));

  PetscCall(PetscOptionsHasName(NULL,NULL,"-ts_fd",&flg));
  if (!flg) {
    PetscCall(TSSetRHSJacobian(ts,J,J,RHSJacobian,&data));
  } else {
    PetscCall(TSGetSNES(ts,&snes));
    PetscCall(PetscOptionsHasName(NULL,NULL,"-fd_color",&fd_jacobian_coloring));
    if (fd_jacobian_coloring) { /* Use finite differences with coloring */
      /* Get data structure of J */
      PetscBool pc_diagonal;
      PetscCall(PetscOptionsHasName(NULL,NULL,"-pc_diagonal",&pc_diagonal));
      if (pc_diagonal) { /* the preconditioner of J is a diagonal matrix */
        PetscInt    rstart,rend,i;
        PetscScalar zero=0.0;
        PetscCall(MatGetOwnershipRange(J,&rstart,&rend));
        for (i=rstart; i<rend; i++) {
          PetscCall(MatSetValues(J,1,&i,1,&i,&zero,INSERT_VALUES));
        }
        PetscCall(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
      } else {
        /* Fill the structure using the expensive SNESComputeJacobianDefault. Temporarily set up the TS so we can call this function */
        PetscCall(TSSetType(ts,TSBEULER));
        PetscCall(TSSetUp(ts));
        PetscCall(SNESComputeJacobianDefault(snes,x,J,J,ts));
      }

      /* create coloring context */
      PetscCall(MatColoringCreate(J,&mc));
      PetscCall(MatColoringSetType(mc,MATCOLORINGSL));
      PetscCall(MatColoringSetFromOptions(mc));
      PetscCall(MatColoringApply(mc,&iscoloring));
      PetscCall(MatColoringDestroy(&mc));
      PetscCall(MatFDColoringCreate(J,iscoloring,&matfdcoloring));
      PetscCall(MatFDColoringSetFunction(matfdcoloring,(PetscErrorCode (*)(void))SNESTSFormFunction,ts));
      PetscCall(MatFDColoringSetFromOptions(matfdcoloring));
      PetscCall(MatFDColoringSetUp(J,iscoloring,matfdcoloring));
      PetscCall(SNESSetJacobian(snes,J,J,SNESComputeJacobianDefaultColor,matfdcoloring));
      PetscCall(ISColoringDestroy(&iscoloring));
    } else { /* Use finite differences (slow) */
      PetscCall(SNESSetJacobian(snes,J,J,SNESComputeJacobianDefault,NULL));
    }
  }

  /* Pick up a Petsc preconditioner */
  /* one can always set method or preconditioner during the run time */
  PetscCall(TSGetSNES(ts,&snes));
  PetscCall(SNESGetKSP(snes,&ksp));
  PetscCall(KSPGetPC(ksp,&pc));
  PetscCall(PCSetType(pc,PCJACOBI));
  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));

  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSSetUp(ts));

  /* Test TSSetPostStep() */
  PetscCall(PetscOptionsHasName(NULL,NULL,"-test_PostStep",&flg));
  if (flg) PetscCall(TSSetPostStep(ts,PostStep));

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-NOUT",&NOUT,NULL));
  for (iout=1; iout<=NOUT; iout++) {
    PetscCall(TSSetMaxSteps(ts,time_steps));
    PetscCall(TSSetMaxTime(ts,iout*ftime_original/NOUT));
    PetscCall(TSSolve(ts,global));
    PetscCall(TSGetSolveTime(ts,&ftime));
    PetscCall(TSSetTime(ts,ftime));
    PetscCall(TSSetTimeStep(ts,dt));
  }
  /* Interpolate solution at tfinal */
  PetscCall(TSGetSolution(ts,&global));
  PetscCall(TSInterpolate(ts,ftime_original,global));

  PetscCall(PetscOptionsHasName(NULL,NULL,"-matlab_view",&flg));
  if (flg) { /* print solution into a MATLAB file */
    PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD,"out.m",&viewfile));
    PetscCall(PetscViewerPushFormat(viewfile,PETSC_VIEWER_ASCII_MATLAB));
    PetscCall(VecView(global,viewfile));
    PetscCall(PetscViewerPopFormat(viewfile));
    PetscCall(PetscViewerDestroy(&viewfile));
  }

  /* free the memories */
  PetscCall(TSDestroy(&ts));
  PetscCall(VecDestroy(&global));
  PetscCall(VecDestroy(&x));
  PetscCall(MatDestroy(&J));
  if (fd_jacobian_coloring) PetscCall(MatFDColoringDestroy(&matfdcoloring));
  PetscCall(PetscFinalize());
  return 0;
}

/* -------------------------------------------------------------------*/
/* the initial function */
PetscReal f_ini(PetscReal x,PetscReal y)
{
  PetscReal f;

  f=PetscExpReal(-20.0*(PetscPowRealInt(x-0.5,2)+PetscPowRealInt(y-0.5,2)));
  return f;
}

PetscErrorCode Initial(Vec global,void *ctx)
{
  Data           *data = (Data*)ctx;
  PetscInt       m,row,col;
  PetscReal      x,y,dx,dy;
  PetscScalar    *localptr;
  PetscInt       i,mybase,myend,locsize;

  PetscFunctionBeginUser;
  /* make the local  copies of parameters */
  m  = data->m;
  dx = data->dx;
  dy = data->dy;

  /* determine starting point of each processor */
  PetscCall(VecGetOwnershipRange(global,&mybase,&myend));
  PetscCall(VecGetLocalSize(global,&locsize));

  /* Initialize the array */
  PetscCall(VecGetArrayWrite(global,&localptr));

  for (i=0; i<locsize; i++) {
    row         = 1+(mybase+i)-((mybase+i)/m)*m;
    col         = (mybase+i)/m+1;
    x           = dx*row;
    y           = dy*col;
    localptr[i] = f_ini(x,y);
  }

  PetscCall(VecRestoreArrayWrite(global,&localptr));
  PetscFunctionReturn(0);
}

PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal time,Vec global,void *ctx)
{
  VecScatter        scatter;
  IS                from,to;
  PetscInt          i,n,*idx,nsteps,maxsteps;
  Vec               tmp_vec;
  const PetscScalar *tmp;

  PetscFunctionBeginUser;
  PetscCall(TSGetStepNumber(ts,&nsteps));
  /* display output at selected time steps */
  PetscCall(TSGetMaxSteps(ts, &maxsteps));
  if (nsteps % 10 != 0) PetscFunctionReturn(0);

  /* Get the size of the vector */
  PetscCall(VecGetSize(global,&n));

  /* Set the index sets */
  PetscCall(PetscMalloc1(n,&idx));
  for (i=0; i<n; i++) idx[i]=i;

  /* Create local sequential vectors */
  PetscCall(VecCreateSeq(PETSC_COMM_SELF,n,&tmp_vec));

  /* Create scatter context */
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF,n,idx,PETSC_COPY_VALUES,&from));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF,n,idx,PETSC_COPY_VALUES,&to));
  PetscCall(VecScatterCreate(global,from,tmp_vec,to,&scatter));
  PetscCall(VecScatterBegin(scatter,global,tmp_vec,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter,global,tmp_vec,INSERT_VALUES,SCATTER_FORWARD));

  PetscCall(VecGetArrayRead(tmp_vec,&tmp));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"At t[%" PetscInt_FMT "] =%14.2e u= %14.2e at the center \n",nsteps,(double)time,(double)PetscRealPart(tmp[n/2])));
  PetscCall(VecRestoreArrayRead(tmp_vec,&tmp));

  PetscCall(PetscFree(idx));
  PetscCall(ISDestroy(&from));
  PetscCall(ISDestroy(&to));
  PetscCall(VecScatterDestroy(&scatter));
  PetscCall(VecDestroy(&tmp_vec));
  PetscFunctionReturn(0);
}

PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec x,Mat A,Mat BB,void *ptr)
{
  Data           *data = (Data*)ptr;
  PetscScalar    v[5];
  PetscInt       idx[5],i,j,row;
  PetscInt       m,n,mn;
  PetscReal      dx,dy,a,epsilon,xc,xl,xr,yl,yr;

  PetscFunctionBeginUser;
  m       = data->m;
  n       = data->n;
  mn      = m*n;
  dx      = data->dx;
  dy      = data->dy;
  a       = data->a;
  epsilon = data->epsilon;

  xc = -2.0*epsilon*(1.0/(dx*dx)+1.0/(dy*dy));
  xl = 0.5*a/dx+epsilon/(dx*dx);
  xr = -0.5*a/dx+epsilon/(dx*dx);
  yl = 0.5*a/dy+epsilon/(dy*dy);
  yr = -0.5*a/dy+epsilon/(dy*dy);

  row    = 0;
  v[0]   = xc;  v[1] = xr;  v[2] = yr;
  idx[0] = 0; idx[1] = 2; idx[2] = m;
  PetscCall(MatSetValues(A,1,&row,3,idx,v,INSERT_VALUES));

  row    = m-1;
  v[0]   = 2.0*xl; v[1] = xc;    v[2] = yr;
  idx[0] = m-2;  idx[1] = m-1; idx[2] = m-1+m;
  PetscCall(MatSetValues(A,1,&row,3,idx,v,INSERT_VALUES));

  for (i=1; i<m-1; i++) {
    row    = i;
    v[0]   = xl;    v[1] = xc;  v[2] = xr;    v[3] = yr;
    idx[0] = i-1; idx[1] = i; idx[2] = i+1; idx[3] = i+m;
    PetscCall(MatSetValues(A,1,&row,4,idx,v,INSERT_VALUES));
  }

  for (j=1; j<n-1; j++) {
    row    = j*m;
    v[0]   = xc;    v[1] = xr;    v[2] = yl;      v[3] = yr;
    idx[0] = j*m; idx[1] = j*m; idx[2] = j*m-m; idx[3] = j*m+m;
    PetscCall(MatSetValues(A,1,&row,4,idx,v,INSERT_VALUES));

    row    = j*m+m-1;
    v[0]   = xc;        v[1] = 2.0*xl;      v[2] = yl;          v[3] = yr;
    idx[0] = j*m+m-1; idx[1] = j*m+m-1-1; idx[2] = j*m+m-1-m; idx[3] = j*m+m-1+m;
    PetscCall(MatSetValues(A,1,&row,4,idx,v,INSERT_VALUES));

    for (i=1; i<m-1; i++) {
      row    = j*m+i;
      v[0]   = xc;      v[1] = xl;        v[2] = xr;        v[3] = yl; v[4]=yr;
      idx[0] = j*m+i; idx[1] = j*m+i-1; idx[2] = j*m+i+1; idx[3] = j*m+i-m;
      idx[4] = j*m+i+m;
      PetscCall(MatSetValues(A,1,&row,5,idx,v,INSERT_VALUES));
    }
  }

  row    = mn-m;
  v[0]   = xc;     v[1] = xr;       v[2] = 2.0*yl;
  idx[0] = mn-m; idx[1] = mn-m+1; idx[2] = mn-m-m;
  PetscCall(MatSetValues(A,1,&row,3,idx,v,INSERT_VALUES));

  row    = mn-1;
  v[0]   = xc;     v[1] = 2.0*xl; v[2] = 2.0*yl;
  idx[0] = mn-1; idx[1] = mn-2; idx[2] = mn-1-m;
  PetscCall(MatSetValues(A,1,&i,3,idx,v,INSERT_VALUES));

  for (i=1; i<m-1; i++) {
    row    = mn-m+i;
    v[0]   = xl;         v[1] = xc;       v[2] = xr;         v[3] = 2.0*yl;
    idx[0] = mn-m+i-1; idx[1] = mn-m+i; idx[2] = mn-m+i+1; idx[3] = mn-m+i-m;
    PetscCall(MatSetValues(A,1,&row,4,idx,v,INSERT_VALUES));
  }

  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  PetscFunctionReturn(0);
}

/* globalout = -a*(u_x+u_y) + epsilon*(u_xx+u_yy) */
PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec globalin,Vec globalout,void *ctx)
{
  Data              *data = (Data*)ctx;
  PetscInt          m,n,mn;
  PetscReal         dx,dy;
  PetscReal         xc,xl,xr,yl,yr;
  PetscReal         a,epsilon;
  PetscScalar       *outptr;
  const PetscScalar *inptr;
  PetscInt          i,j,len;
  IS                from,to;
  PetscInt          *idx;
  VecScatter        scatter;
  Vec               tmp_in,tmp_out;

  PetscFunctionBeginUser;
  m       = data->m;
  n       = data->n;
  mn      = m*n;
  dx      = data->dx;
  dy      = data->dy;
  a       = data->a;
  epsilon = data->epsilon;

  xc = -2.0*epsilon*(1.0/(dx*dx)+1.0/(dy*dy));
  xl = 0.5*a/dx+epsilon/(dx*dx);
  xr = -0.5*a/dx+epsilon/(dx*dx);
  yl = 0.5*a/dy+epsilon/(dy*dy);
  yr = -0.5*a/dy+epsilon/(dy*dy);

  /* Get the length of parallel vector */
  PetscCall(VecGetSize(globalin,&len));

  /* Set the index sets */
  PetscCall(PetscMalloc1(len,&idx));
  for (i=0; i<len; i++) idx[i]=i;

  /* Create local sequential vectors */
  PetscCall(VecCreateSeq(PETSC_COMM_SELF,len,&tmp_in));
  PetscCall(VecDuplicate(tmp_in,&tmp_out));

  /* Create scatter context */
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF,len,idx,PETSC_COPY_VALUES,&from));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF,len,idx,PETSC_COPY_VALUES,&to));
  PetscCall(VecScatterCreate(globalin,from,tmp_in,to,&scatter));
  PetscCall(VecScatterBegin(scatter,globalin,tmp_in,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter,globalin,tmp_in,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterDestroy(&scatter));

  /*Extract income array - include ghost points */
  PetscCall(VecGetArrayRead(tmp_in,&inptr));

  /* Extract outcome array*/
  PetscCall(VecGetArrayWrite(tmp_out,&outptr));

  outptr[0]   = xc*inptr[0]+xr*inptr[1]+yr*inptr[m];
  outptr[m-1] = 2.0*xl*inptr[m-2]+xc*inptr[m-1]+yr*inptr[m-1+m];
  for (i=1; i<m-1; i++) {
    outptr[i] = xc*inptr[i]+xl*inptr[i-1]+xr*inptr[i+1]+yr*inptr[i+m];
  }

  for (j=1; j<n-1; j++) {
    outptr[j*m] = xc*inptr[j*m]+xr*inptr[j*m+1]+ yl*inptr[j*m-m]+yr*inptr[j*m+m];
    outptr[j*m+m-1] = xc*inptr[j*m+m-1]+2.0*xl*inptr[j*m+m-1-1]+ yl*inptr[j*m+m-1-m]+yr*inptr[j*m+m-1+m];
    for (i=1; i<m-1; i++) {
      outptr[j*m+i] = xc*inptr[j*m+i]+xl*inptr[j*m+i-1]+xr*inptr[j*m+i+1]+yl*inptr[j*m+i-m]+yr*inptr[j*m+i+m];
    }
  }

  outptr[mn-m] = xc*inptr[mn-m]+xr*inptr[mn-m+1]+2.0*yl*inptr[mn-m-m];
  outptr[mn-1] = 2.0*xl*inptr[mn-2]+xc*inptr[mn-1]+2.0*yl*inptr[mn-1-m];
  for (i=1; i<m-1; i++) {
    outptr[mn-m+i] = xc*inptr[mn-m+i]+xl*inptr[mn-m+i-1]+xr*inptr[mn-m+i+1]+2*yl*inptr[mn-m+i-m];
  }

  PetscCall(VecRestoreArrayRead(tmp_in,&inptr));
  PetscCall(VecRestoreArrayWrite(tmp_out,&outptr));

  PetscCall(VecScatterCreate(tmp_out,from,globalout,to,&scatter));
  PetscCall(VecScatterBegin(scatter,tmp_out,globalout,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter,tmp_out,globalout,INSERT_VALUES,SCATTER_FORWARD));

  /* Destroy idx aand scatter */
  PetscCall(VecDestroy(&tmp_in));
  PetscCall(VecDestroy(&tmp_out));
  PetscCall(ISDestroy(&from));
  PetscCall(ISDestroy(&to));
  PetscCall(VecScatterDestroy(&scatter));

  PetscCall(PetscFree(idx));
  PetscFunctionReturn(0);
}

PetscErrorCode PostStep(TS ts)
{
  PetscReal      t;

  PetscFunctionBeginUser;
  PetscCall(TSGetTime(ts,&t));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"  PostStep, t: %g\n",(double)t));
  PetscFunctionReturn(0);
}

/*TEST

    test:
      args: -ts_fd -ts_type beuler
      output_file: output/ex4.out

    test:
      suffix: 2
      args: -ts_fd -ts_type beuler
      nsize: 2
      output_file: output/ex4.out

    test:
      suffix: 3
      args: -ts_fd -ts_type cn

    test:
      suffix: 4
      args: -ts_fd -ts_type cn
      output_file: output/ex4_3.out
      nsize: 2

    test:
      suffix: 5
      args: -ts_type beuler -ts_fd -fd_color -mat_coloring_type sl
      output_file: output/ex4.out

    test:
      suffix: 6
      args: -ts_type beuler -ts_fd -fd_color -mat_coloring_type sl
      output_file: output/ex4.out
      nsize: 2

    test:
      suffix: 7
      requires: !single
      args: -ts_fd -ts_type beuler -test_PostStep -ts_dt .1

    test:
      suffix: 8
      requires: !single
      args: -ts_type rk -ts_rk_type 5dp -ts_dt .01 -ts_adapt_type none -ts_view

TEST*/
