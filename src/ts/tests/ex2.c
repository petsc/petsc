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
  ierr = PetscOptionsGetInt(NULL,NULL,"-time",&time_steps,NULL);CHKERRQ(ierr);

  /* set initial conditions */
  ierr = VecCreate(PETSC_COMM_WORLD,&global);CHKERRQ(ierr);
  ierr = VecSetSizes(global,PETSC_DECIDE,3);CHKERRQ(ierr);
  ierr = VecSetFromOptions(global);CHKERRQ(ierr);
  ierr = Initial(global,NULL);CHKERRQ(ierr);

  /* make timestep context */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSMonitorSet(ts,Monitor,NULL,NULL);CHKERRQ(ierr);
  dt = 0.001;

  /*
    The user provides the RHS and Jacobian
  */
  ierr = TSSetRHSFunction(ts,NULL,RHSFunction,NULL);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,3,3);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = RHSJacobian(ts,0.0,global,A,A,NULL);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,A,A,RHSJacobian,NULL);CHKERRQ(ierr);

  ierr = MatCreateShell(PETSC_COMM_WORLD,3,3,3,3,NULL,&S);CHKERRQ(ierr);
  ierr = MatShellSetOperation(S,MATOP_MULT,(void (*)(void))MyMatMult);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,S,A,RHSJacobian,NULL);CHKERRQ(ierr);

  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts,time_steps);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,1);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,global);CHKERRQ(ierr);

  ierr = TSSolve(ts,global);CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts,&ftime);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&steps);CHKERRQ(ierr);

  /* free the memories */

  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = VecDestroy(&global);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&S);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode MyMatMult(Mat S,Vec x,Vec y)
{
  PetscErrorCode     ierr;
  const PetscScalar  *inptr;
  PetscScalar        *outptr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(x,&inptr);CHKERRQ(ierr);
  ierr = VecGetArrayWrite(y,&outptr);CHKERRQ(ierr);

  outptr[0] = 2.0*inptr[0]+inptr[1];
  outptr[1] = inptr[0]+2.0*inptr[1]+inptr[2];
  outptr[2] = inptr[1]+2.0*inptr[2];

  ierr = VecRestoreArrayRead(x,&inptr);CHKERRQ(ierr);
  ierr = VecRestoreArrayWrite(y,&outptr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* this test problem has initial values (1,1,1).                      */
PetscErrorCode Initial(Vec global,void *ctx)
{
  PetscScalar    *localptr;
  PetscInt       i,mybase,myend,locsize;
  PetscErrorCode ierr;

  /* determine starting point of each processor */
  ierr = VecGetOwnershipRange(global,&mybase,&myend);CHKERRQ(ierr);
  ierr = VecGetLocalSize(global,&locsize);CHKERRQ(ierr);

  /* Initialize the array */
  ierr = VecGetArrayWrite(global,&localptr);CHKERRQ(ierr);
  for (i=0; i<locsize; i++) localptr[i] = 1.0;

  if (mybase == 0) localptr[0]=1.0;

  ierr = VecRestoreArrayWrite(global,&localptr);CHKERRQ(ierr);
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
  ierr = VecGetSize(global,&n);CHKERRQ(ierr);

  /* Set the index sets */
  ierr = PetscMalloc1(n,&idx);CHKERRQ(ierr);
  for (i=0; i<n; i++) idx[i]=i;

  /* Create local sequential vectors */
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&tmp_vec);CHKERRQ(ierr);

  /* Create scatter context */
  ierr = ISCreateGeneral(PETSC_COMM_SELF,n,idx,PETSC_COPY_VALUES,&from);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,n,idx,PETSC_COPY_VALUES,&to);CHKERRQ(ierr);
  ierr = VecScatterCreate(global,from,tmp_vec,to,&scatter);CHKERRQ(ierr);
  ierr = VecScatterBegin(scatter,global,tmp_vec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(scatter,global,tmp_vec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  ierr = VecGetArrayRead(tmp_vec,&tmp);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"At t =%14.6e u = %14.6e  %14.6e  %14.6e \n",
                     (double)time,(double)PetscRealPart(tmp[0]),(double)PetscRealPart(tmp[1]),(double)PetscRealPart(tmp[2]));CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"At t =%14.6e errors = %14.6e  %14.6e  %14.6e \n",
                     (double)time,(double)PetscRealPart(tmp[0]-solx(time)),(double)PetscRealPart(tmp[1]-soly(time)),(double)PetscRealPart(tmp[2]-solz(time)));CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(tmp_vec,&tmp);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&scatter);CHKERRQ(ierr);
  ierr = ISDestroy(&from);CHKERRQ(ierr);
  ierr = ISDestroy(&to);CHKERRQ(ierr);
  ierr = PetscFree(idx);CHKERRQ(ierr);
  ierr = VecDestroy(&tmp_vec);CHKERRQ(ierr);
  return 0;
}

PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec globalin,Vec globalout,void *ctx)
{
  PetscScalar       *outptr;
  const PetscScalar *inptr;
  PetscInt          i,n,*idx;
  PetscErrorCode    ierr;
  IS                from,to;
  VecScatter        scatter;
  Vec               tmp_in,tmp_out;

  /* Get the length of parallel vector */
  ierr = VecGetSize(globalin,&n);CHKERRQ(ierr);

  /* Set the index sets */
  ierr = PetscMalloc1(n,&idx);CHKERRQ(ierr);
  for (i=0; i<n; i++) idx[i]=i;

  /* Create local sequential vectors */
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&tmp_in);CHKERRQ(ierr);
  ierr = VecDuplicate(tmp_in,&tmp_out);CHKERRQ(ierr);

  /* Create scatter context */
  ierr = ISCreateGeneral(PETSC_COMM_SELF,n,idx,PETSC_COPY_VALUES,&from);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,n,idx,PETSC_COPY_VALUES,&to);CHKERRQ(ierr);
  ierr = VecScatterCreate(globalin,from,tmp_in,to,&scatter);CHKERRQ(ierr);
  ierr = VecScatterBegin(scatter,globalin,tmp_in,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(scatter,globalin,tmp_in,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&scatter);CHKERRQ(ierr);

  /*Extract income array */
  ierr = VecGetArrayRead(tmp_in,&inptr);CHKERRQ(ierr);

  /* Extract outcome array*/
  ierr = VecGetArrayWrite(tmp_out,&outptr);CHKERRQ(ierr);

  outptr[0] = 2.0*inptr[0]+inptr[1];
  outptr[1] = inptr[0]+2.0*inptr[1]+inptr[2];
  outptr[2] = inptr[1]+2.0*inptr[2];

  ierr = VecRestoreArrayRead(tmp_in,&inptr);CHKERRQ(ierr);
  ierr = VecRestoreArrayWrite(tmp_out,&outptr);CHKERRQ(ierr);

  ierr = VecScatterCreate(tmp_out,from,globalout,to,&scatter);CHKERRQ(ierr);
  ierr = VecScatterBegin(scatter,tmp_out,globalout,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(scatter,tmp_out,globalout,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  /* Destroy idx aand scatter */
  ierr = ISDestroy(&from);CHKERRQ(ierr);
  ierr = ISDestroy(&to);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&scatter);CHKERRQ(ierr);
  ierr = VecDestroy(&tmp_in);CHKERRQ(ierr);
  ierr = VecDestroy(&tmp_out);CHKERRQ(ierr);
  ierr = PetscFree(idx);CHKERRQ(ierr);
  return 0;
}

PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec x,Mat A,Mat BB,void *ctx)
{
  PetscScalar       v[3];
  const PetscScalar *tmp;
  PetscInt          idx[3],i;
  PetscErrorCode    ierr;

  idx[0]=0; idx[1]=1; idx[2]=2;
  ierr  = VecGetArrayRead(x,&tmp);CHKERRQ(ierr);

  i    = 0;
  v[0] = 2.0; v[1] = 1.0; v[2] = 0.0;
  ierr = MatSetValues(BB,1,&i,3,idx,v,INSERT_VALUES);CHKERRQ(ierr);

  i    = 1;
  v[0] = 1.0; v[1] = 2.0; v[2] = 1.0;
  ierr = MatSetValues(BB,1,&i,3,idx,v,INSERT_VALUES);CHKERRQ(ierr);

  i    = 2;
  v[0] = 0.0; v[1] = 1.0; v[2] = 2.0;
  ierr = MatSetValues(BB,1,&i,3,idx,v,INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(BB,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(BB,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (A != BB) {
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(x,&tmp);CHKERRQ(ierr);

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

