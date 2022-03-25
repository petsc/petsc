/*
       Formatted test for TS routines.

          Solves U_t=F(t,u)
          Where:

                  [2*u1+u2
          F(t,u)= [u1+2*u2+u3
                  [   u2+2*u3
       We can compare the solutions from euler, beuler and SUNDIALS to
       see what is the difference.

*/

static char help[] = "Solves a linear ODE. \n\n";

#include <petscts.h>
#include <petscpc.h>

extern PetscErrorCode RHSFunction(TS,PetscReal,Vec,Vec,void*);
extern PetscErrorCode RHSJacobian(TS,PetscReal,Vec,Mat,Mat,void*);
extern PetscErrorCode Monitor(TS,PetscInt,PetscReal,Vec,void*);
extern PetscErrorCode Initial(Vec,void*);
extern PetscErrorCode MyMatMult(Mat,Vec,Vec);

extern PetscReal solx(PetscReal);
extern PetscReal soly(PetscReal);
extern PetscReal solz(PetscReal);

int main(int argc,char **argv)
{
  PetscInt       time_steps = 100,steps;
  Vec            global;
  PetscReal      dt,ftime;
  TS             ts;
  Mat            A = 0,S;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-time",&time_steps,NULL));

  /* set initial conditions */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&global));
  PetscCall(VecSetSizes(global,PETSC_DECIDE,3));
  PetscCall(VecSetFromOptions(global));
  PetscCall(Initial(global,NULL));

  /* make timestep context */
  PetscCall(TSCreate(PETSC_COMM_WORLD,&ts));
  PetscCall(TSSetProblemType(ts,TS_NONLINEAR));
  PetscCall(TSMonitorSet(ts,Monitor,NULL,NULL));
  dt = 0.001;

  /*
    The user provides the RHS and Jacobian
  */
  PetscCall(TSSetRHSFunction(ts,NULL,RHSFunction,NULL));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,3,3));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(RHSJacobian(ts,0.0,global,A,A,NULL));
  PetscCall(TSSetRHSJacobian(ts,A,A,RHSJacobian,NULL));

  PetscCall(MatCreateShell(PETSC_COMM_WORLD,3,3,3,3,NULL,&S));
  PetscCall(MatShellSetOperation(S,MATOP_MULT,(void (*)(void))MyMatMult));
  PetscCall(TSSetRHSJacobian(ts,S,A,RHSJacobian,NULL));

  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetFromOptions(ts));

  PetscCall(TSSetTimeStep(ts,dt));
  PetscCall(TSSetMaxSteps(ts,time_steps));
  PetscCall(TSSetMaxTime(ts,1));
  PetscCall(TSSetSolution(ts,global));

  PetscCall(TSSolve(ts,global));
  PetscCall(TSGetSolveTime(ts,&ftime));
  PetscCall(TSGetStepNumber(ts,&steps));

  /* free the memories */

  PetscCall(TSDestroy(&ts));
  PetscCall(VecDestroy(&global));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&S));

  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode MyMatMult(Mat S,Vec x,Vec y)
{
  const PetscScalar  *inptr;
  PetscScalar        *outptr;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(x,&inptr));
  PetscCall(VecGetArrayWrite(y,&outptr));

  outptr[0] = 2.0*inptr[0]+inptr[1];
  outptr[1] = inptr[0]+2.0*inptr[1]+inptr[2];
  outptr[2] = inptr[1]+2.0*inptr[2];

  PetscCall(VecRestoreArrayRead(x,&inptr));
  PetscCall(VecRestoreArrayWrite(y,&outptr));
  PetscFunctionReturn(0);
}

/* this test problem has initial values (1,1,1).                      */
PetscErrorCode Initial(Vec global,void *ctx)
{
  PetscScalar    *localptr;
  PetscInt       i,mybase,myend,locsize;

  /* determine starting point of each processor */
  PetscCall(VecGetOwnershipRange(global,&mybase,&myend));
  PetscCall(VecGetLocalSize(global,&locsize));

  /* Initialize the array */
  PetscCall(VecGetArrayWrite(global,&localptr));
  for (i=0; i<locsize; i++) localptr[i] = 1.0;

  if (mybase == 0) localptr[0]=1.0;

  PetscCall(VecRestoreArrayWrite(global,&localptr));
  return 0;
}

PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal time,Vec global,void *ctx)
{
  VecScatter        scatter;
  IS                from,to;
  PetscInt          i,n,*idx;
  Vec               tmp_vec;
  PetscErrorCode    ierr;
  const PetscScalar *tmp;

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
  ierr = PetscPrintf(PETSC_COMM_WORLD,"At t =%14.6e u = %14.6e  %14.6e  %14.6e \n",
                     (double)time,(double)PetscRealPart(tmp[0]),(double)PetscRealPart(tmp[1]),(double)PetscRealPart(tmp[2]));PetscCall(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"At t =%14.6e errors = %14.6e  %14.6e  %14.6e \n",
                     (double)time,(double)PetscRealPart(tmp[0]-solx(time)),(double)PetscRealPart(tmp[1]-soly(time)),(double)PetscRealPart(tmp[2]-solz(time)));PetscCall(ierr);
  PetscCall(VecRestoreArrayRead(tmp_vec,&tmp));
  PetscCall(VecScatterDestroy(&scatter));
  PetscCall(ISDestroy(&from));
  PetscCall(ISDestroy(&to));
  PetscCall(PetscFree(idx));
  PetscCall(VecDestroy(&tmp_vec));
  return 0;
}

PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec globalin,Vec globalout,void *ctx)
{
  PetscScalar       *outptr;
  const PetscScalar *inptr;
  PetscInt          i,n,*idx;
  IS                from,to;
  VecScatter        scatter;
  Vec               tmp_in,tmp_out;

  /* Get the length of parallel vector */
  PetscCall(VecGetSize(globalin,&n));

  /* Set the index sets */
  PetscCall(PetscMalloc1(n,&idx));
  for (i=0; i<n; i++) idx[i]=i;

  /* Create local sequential vectors */
  PetscCall(VecCreateSeq(PETSC_COMM_SELF,n,&tmp_in));
  PetscCall(VecDuplicate(tmp_in,&tmp_out));

  /* Create scatter context */
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF,n,idx,PETSC_COPY_VALUES,&from));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF,n,idx,PETSC_COPY_VALUES,&to));
  PetscCall(VecScatterCreate(globalin,from,tmp_in,to,&scatter));
  PetscCall(VecScatterBegin(scatter,globalin,tmp_in,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter,globalin,tmp_in,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterDestroy(&scatter));

  /*Extract income array */
  PetscCall(VecGetArrayRead(tmp_in,&inptr));

  /* Extract outcome array*/
  PetscCall(VecGetArrayWrite(tmp_out,&outptr));

  outptr[0] = 2.0*inptr[0]+inptr[1];
  outptr[1] = inptr[0]+2.0*inptr[1]+inptr[2];
  outptr[2] = inptr[1]+2.0*inptr[2];

  PetscCall(VecRestoreArrayRead(tmp_in,&inptr));
  PetscCall(VecRestoreArrayWrite(tmp_out,&outptr));

  PetscCall(VecScatterCreate(tmp_out,from,globalout,to,&scatter));
  PetscCall(VecScatterBegin(scatter,tmp_out,globalout,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter,tmp_out,globalout,INSERT_VALUES,SCATTER_FORWARD));

  /* Destroy idx aand scatter */
  PetscCall(ISDestroy(&from));
  PetscCall(ISDestroy(&to));
  PetscCall(VecScatterDestroy(&scatter));
  PetscCall(VecDestroy(&tmp_in));
  PetscCall(VecDestroy(&tmp_out));
  PetscCall(PetscFree(idx));
  return 0;
}

PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec x,Mat A,Mat BB,void *ctx)
{
  PetscScalar       v[3];
  const PetscScalar *tmp;
  PetscInt          idx[3],i;

  idx[0]=0; idx[1]=1; idx[2]=2;
  PetscCall(VecGetArrayRead(x,&tmp));

  i    = 0;
  v[0] = 2.0; v[1] = 1.0; v[2] = 0.0;
  PetscCall(MatSetValues(BB,1,&i,3,idx,v,INSERT_VALUES));

  i    = 1;
  v[0] = 1.0; v[1] = 2.0; v[2] = 1.0;
  PetscCall(MatSetValues(BB,1,&i,3,idx,v,INSERT_VALUES));

  i    = 2;
  v[0] = 0.0; v[1] = 1.0; v[2] = 2.0;
  PetscCall(MatSetValues(BB,1,&i,3,idx,v,INSERT_VALUES));

  PetscCall(MatAssemblyBegin(BB,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(BB,MAT_FINAL_ASSEMBLY));

  if (A != BB) {
    PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  }
  PetscCall(VecRestoreArrayRead(x,&tmp));

  return 0;
}

/*
      The exact solutions
*/
PetscReal solx(PetscReal t)
{
  return PetscExpReal((2.0 - PetscSqrtReal(2.0))*t)/2.0 - PetscExpReal((2.0 - PetscSqrtReal(2.0))*t)/(2.0*PetscSqrtReal(2.0)) +
         PetscExpReal((2.0 + PetscSqrtReal(2.0))*t)/2.0 + PetscExpReal((2.0 + PetscSqrtReal(2.0))*t)/(2.0*PetscSqrtReal(2.0));
}

PetscReal soly(PetscReal t)
{
  return PetscExpReal((2.0 - PetscSqrtReal(2.0))*t)/2.0 - PetscExpReal((2.0 - PetscSqrtReal(2.0))*t)/PetscSqrtReal(2.0) +
         PetscExpReal((2.0 + PetscSqrtReal(2.0))*t)/2.0 + PetscExpReal((2.0 + PetscSqrtReal(2.0))*t)/PetscSqrtReal(2.0);
}

PetscReal solz(PetscReal t)
{
  return PetscExpReal((2.0 - PetscSqrtReal(2.0))*t)/2.0 - PetscExpReal((2.0 - PetscSqrtReal(2.0))*t)/(2.0*PetscSqrtReal(2.0)) +
         PetscExpReal((2.0 + PetscSqrtReal(2.0))*t)/2.0 + PetscExpReal((2.0 + PetscSqrtReal(2.0))*t)/(2.0*PetscSqrtReal(2.0));
}

/*TEST

    test:
      suffix: euler
      args: -ts_type euler
      requires: !single

    test:
      suffix: beuler
      args:   -ts_type beuler
      requires: !single

TEST*/
