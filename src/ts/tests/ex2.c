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
  PetscErrorCode ierr;
  PetscInt       time_steps = 100,steps;
  Vec            global;
  PetscReal      dt,ftime;
  TS             ts;
  Mat            A = 0,S;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-time",&time_steps,NULL));

  /* set initial conditions */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&global));
  CHKERRQ(VecSetSizes(global,PETSC_DECIDE,3));
  CHKERRQ(VecSetFromOptions(global));
  CHKERRQ(Initial(global,NULL));

  /* make timestep context */
  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&ts));
  CHKERRQ(TSSetProblemType(ts,TS_NONLINEAR));
  CHKERRQ(TSMonitorSet(ts,Monitor,NULL,NULL));
  dt = 0.001;

  /*
    The user provides the RHS and Jacobian
  */
  CHKERRQ(TSSetRHSFunction(ts,NULL,RHSFunction,NULL));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,3,3));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(RHSJacobian(ts,0.0,global,A,A,NULL));
  CHKERRQ(TSSetRHSJacobian(ts,A,A,RHSJacobian,NULL));

  CHKERRQ(MatCreateShell(PETSC_COMM_WORLD,3,3,3,3,NULL,&S));
  CHKERRQ(MatShellSetOperation(S,MATOP_MULT,(void (*)(void))MyMatMult));
  CHKERRQ(TSSetRHSJacobian(ts,S,A,RHSJacobian,NULL));

  CHKERRQ(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP));
  CHKERRQ(TSSetFromOptions(ts));

  CHKERRQ(TSSetTimeStep(ts,dt));
  CHKERRQ(TSSetMaxSteps(ts,time_steps));
  CHKERRQ(TSSetMaxTime(ts,1));
  CHKERRQ(TSSetSolution(ts,global));

  CHKERRQ(TSSolve(ts,global));
  CHKERRQ(TSGetSolveTime(ts,&ftime));
  CHKERRQ(TSGetStepNumber(ts,&steps));

  /* free the memories */

  CHKERRQ(TSDestroy(&ts));
  CHKERRQ(VecDestroy(&global));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&S));

  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode MyMatMult(Mat S,Vec x,Vec y)
{
  const PetscScalar  *inptr;
  PetscScalar        *outptr;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(x,&inptr));
  CHKERRQ(VecGetArrayWrite(y,&outptr));

  outptr[0] = 2.0*inptr[0]+inptr[1];
  outptr[1] = inptr[0]+2.0*inptr[1]+inptr[2];
  outptr[2] = inptr[1]+2.0*inptr[2];

  CHKERRQ(VecRestoreArrayRead(x,&inptr));
  CHKERRQ(VecRestoreArrayWrite(y,&outptr));
  PetscFunctionReturn(0);
}

/* this test problem has initial values (1,1,1).                      */
PetscErrorCode Initial(Vec global,void *ctx)
{
  PetscScalar    *localptr;
  PetscInt       i,mybase,myend,locsize;

  /* determine starting point of each processor */
  CHKERRQ(VecGetOwnershipRange(global,&mybase,&myend));
  CHKERRQ(VecGetLocalSize(global,&locsize));

  /* Initialize the array */
  CHKERRQ(VecGetArrayWrite(global,&localptr));
  for (i=0; i<locsize; i++) localptr[i] = 1.0;

  if (mybase == 0) localptr[0]=1.0;

  CHKERRQ(VecRestoreArrayWrite(global,&localptr));
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
  CHKERRQ(VecGetSize(global,&n));

  /* Set the index sets */
  CHKERRQ(PetscMalloc1(n,&idx));
  for (i=0; i<n; i++) idx[i]=i;

  /* Create local sequential vectors */
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,n,&tmp_vec));

  /* Create scatter context */
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,n,idx,PETSC_COPY_VALUES,&from));
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,n,idx,PETSC_COPY_VALUES,&to));
  CHKERRQ(VecScatterCreate(global,from,tmp_vec,to,&scatter));
  CHKERRQ(VecScatterBegin(scatter,global,tmp_vec,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(scatter,global,tmp_vec,INSERT_VALUES,SCATTER_FORWARD));

  CHKERRQ(VecGetArrayRead(tmp_vec,&tmp));
  ierr = PetscPrintf(PETSC_COMM_WORLD,"At t =%14.6e u = %14.6e  %14.6e  %14.6e \n",
                     (double)time,(double)PetscRealPart(tmp[0]),(double)PetscRealPart(tmp[1]),(double)PetscRealPart(tmp[2]));CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"At t =%14.6e errors = %14.6e  %14.6e  %14.6e \n",
                     (double)time,(double)PetscRealPart(tmp[0]-solx(time)),(double)PetscRealPart(tmp[1]-soly(time)),(double)PetscRealPart(tmp[2]-solz(time)));CHKERRQ(ierr);
  CHKERRQ(VecRestoreArrayRead(tmp_vec,&tmp));
  CHKERRQ(VecScatterDestroy(&scatter));
  CHKERRQ(ISDestroy(&from));
  CHKERRQ(ISDestroy(&to));
  CHKERRQ(PetscFree(idx));
  CHKERRQ(VecDestroy(&tmp_vec));
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
  CHKERRQ(VecGetSize(globalin,&n));

  /* Set the index sets */
  CHKERRQ(PetscMalloc1(n,&idx));
  for (i=0; i<n; i++) idx[i]=i;

  /* Create local sequential vectors */
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,n,&tmp_in));
  CHKERRQ(VecDuplicate(tmp_in,&tmp_out));

  /* Create scatter context */
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,n,idx,PETSC_COPY_VALUES,&from));
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,n,idx,PETSC_COPY_VALUES,&to));
  CHKERRQ(VecScatterCreate(globalin,from,tmp_in,to,&scatter));
  CHKERRQ(VecScatterBegin(scatter,globalin,tmp_in,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(scatter,globalin,tmp_in,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterDestroy(&scatter));

  /*Extract income array */
  CHKERRQ(VecGetArrayRead(tmp_in,&inptr));

  /* Extract outcome array*/
  CHKERRQ(VecGetArrayWrite(tmp_out,&outptr));

  outptr[0] = 2.0*inptr[0]+inptr[1];
  outptr[1] = inptr[0]+2.0*inptr[1]+inptr[2];
  outptr[2] = inptr[1]+2.0*inptr[2];

  CHKERRQ(VecRestoreArrayRead(tmp_in,&inptr));
  CHKERRQ(VecRestoreArrayWrite(tmp_out,&outptr));

  CHKERRQ(VecScatterCreate(tmp_out,from,globalout,to,&scatter));
  CHKERRQ(VecScatterBegin(scatter,tmp_out,globalout,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(scatter,tmp_out,globalout,INSERT_VALUES,SCATTER_FORWARD));

  /* Destroy idx aand scatter */
  CHKERRQ(ISDestroy(&from));
  CHKERRQ(ISDestroy(&to));
  CHKERRQ(VecScatterDestroy(&scatter));
  CHKERRQ(VecDestroy(&tmp_in));
  CHKERRQ(VecDestroy(&tmp_out));
  CHKERRQ(PetscFree(idx));
  return 0;
}

PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec x,Mat A,Mat BB,void *ctx)
{
  PetscScalar       v[3];
  const PetscScalar *tmp;
  PetscInt          idx[3],i;

  idx[0]=0; idx[1]=1; idx[2]=2;
  CHKERRQ(VecGetArrayRead(x,&tmp));

  i    = 0;
  v[0] = 2.0; v[1] = 1.0; v[2] = 0.0;
  CHKERRQ(MatSetValues(BB,1,&i,3,idx,v,INSERT_VALUES));

  i    = 1;
  v[0] = 1.0; v[1] = 2.0; v[2] = 1.0;
  CHKERRQ(MatSetValues(BB,1,&i,3,idx,v,INSERT_VALUES));

  i    = 2;
  v[0] = 0.0; v[1] = 1.0; v[2] = 2.0;
  CHKERRQ(MatSetValues(BB,1,&i,3,idx,v,INSERT_VALUES));

  CHKERRQ(MatAssemblyBegin(BB,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(BB,MAT_FINAL_ASSEMBLY));

  if (A != BB) {
    CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  }
  CHKERRQ(VecRestoreArrayRead(x,&tmp));

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
