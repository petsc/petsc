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

static char help[] = "Solves a nonlinear ODE. \n\n";

#include <petscts.h>
#include <petscpc.h>

extern PetscErrorCode RHSFunction(TS,PetscReal,Vec,Vec,void*);
extern PetscErrorCode RHSJacobian(TS,PetscReal,Vec,Mat*,Mat*,MatStructure *,void*);
extern PetscErrorCode Monitor(TS,PetscInt,PetscReal,Vec,void *);
extern PetscErrorCode Initial(Vec,void *);

extern PetscReal solx(PetscReal);
extern PetscReal soly(PetscReal);
extern PetscReal solz(PetscReal);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       time_steps = 100,steps;
  PetscMPIInt    size;
  Vec            global;
  PetscReal      dt,ftime;
  TS             ts;
  MatStructure   A_structure;
  Mat            A = 0;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  ierr = PetscOptionsGetInt(PETSC_NULL,"-time",&time_steps,PETSC_NULL);CHKERRQ(ierr);

  /* set initial conditions */
  ierr = VecCreate(PETSC_COMM_WORLD,&global);CHKERRQ(ierr);
  ierr = VecSetSizes(global,PETSC_DECIDE,3);CHKERRQ(ierr);
  ierr = VecSetFromOptions(global);CHKERRQ(ierr);
  ierr = Initial(global,PETSC_NULL);CHKERRQ(ierr);

  /* make timestep context */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSMonitorSet(ts,Monitor,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

  dt = 0.1;

  /*
    The user provides the RHS and Jacobian
  */
  ierr = TSSetRHSFunction(ts,PETSC_NULL,RHSFunction,NULL);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,3,3);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = RHSJacobian(ts,0.0,global,&A,&A,&A_structure,NULL);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,A,A,RHSJacobian,NULL);CHKERRQ(ierr);

  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = TSSetInitialTimeStep(ts,0.0,dt);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,time_steps,1);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,global);CHKERRQ(ierr);

  ierr = TSSolve(ts,global,&ftime);CHKERRQ(ierr);
  ierr = TSGetTimeStepNumber(ts,&steps);CHKERRQ(ierr);


  /* free the memories */

  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = VecDestroy(&global);CHKERRQ(ierr);
  ierr= MatDestroy(&A);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return 0;
}

/* -------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "Initial"
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
  ierr = VecGetArray(global,&localptr);CHKERRQ(ierr);
  for (i=0; i<locsize; i++) {
    localptr[i] = 1.0;
  }
  
  if (mybase == 0) localptr[0]=1.0;

  ierr = VecRestoreArray(global,&localptr);CHKERRQ(ierr);
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "Monitor"
PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal time,Vec global,void *ctx)
{
  VecScatter     scatter;
  IS             from,to;
  PetscInt       i,n,*idx;
  Vec            tmp_vec;
  PetscErrorCode ierr;
  PetscScalar    *tmp;

  /* Get the size of the vector */
  ierr = VecGetSize(global,&n);CHKERRQ(ierr);

  /* Set the index sets */
  ierr = PetscMalloc(n*sizeof(PetscInt),&idx);CHKERRQ(ierr);
  for(i=0; i<n; i++) idx[i]=i;
 
  /* Create local sequential vectors */
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&tmp_vec);CHKERRQ(ierr);

  /* Create scatter context */
  ierr = ISCreateGeneral(PETSC_COMM_SELF,n,idx,PETSC_COPY_VALUES,&from);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,n,idx,PETSC_COPY_VALUES,&to);CHKERRQ(ierr);
  ierr = VecScatterCreate(global,from,tmp_vec,to,&scatter);CHKERRQ(ierr);
  ierr = VecScatterBegin(scatter,global,tmp_vec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(scatter,global,tmp_vec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  ierr = VecGetArray(tmp_vec,&tmp);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"At t =%14.6e u = %14.6e  %14.6e  %14.6e \n",
                     time,PetscRealPart(tmp[0]),PetscRealPart(tmp[1]),PetscRealPart(tmp[2]));CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"At t =%14.6e errors = %14.6e  %14.6e  %14.6e \n",
                     time,PetscRealPart(tmp[0]-solx(time)),PetscRealPart(tmp[1]-soly(time)),PetscRealPart(tmp[2]-solz(time)));CHKERRQ(ierr);
  ierr = VecRestoreArray(tmp_vec,&tmp);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&scatter);CHKERRQ(ierr);
  ierr = ISDestroy(&from);CHKERRQ(ierr);
  ierr = ISDestroy(&to);CHKERRQ(ierr);
  ierr = PetscFree(idx);CHKERRQ(ierr);
  ierr = VecDestroy(&tmp_vec);CHKERRQ(ierr);
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "RHSFunction"
PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec globalin,Vec globalout,void *ctx)
{
  PetscScalar    *inptr,*outptr;
  PetscInt       i,n,*idx;
  PetscErrorCode ierr;
  IS             from,to;
  VecScatter     scatter;
  Vec            tmp_in,tmp_out;

  /* Get the length of parallel vector */
  ierr = VecGetSize(globalin,&n);CHKERRQ(ierr);

  /* Set the index sets */
  ierr = PetscMalloc(n*sizeof(PetscInt),&idx);CHKERRQ(ierr);
  for(i=0; i<n; i++) idx[i]=i;
  
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
  ierr = VecGetArray(tmp_in,&inptr);CHKERRQ(ierr);

  /* Extract outcome array*/
  ierr = VecGetArray(tmp_out,&outptr);CHKERRQ(ierr);

  outptr[0] = 2.0*inptr[0]+inptr[1];
  outptr[1] = inptr[0]+2.0*inptr[1]+inptr[2];
  outptr[2] = inptr[1]+2.0*inptr[2];

  ierr = VecRestoreArray(tmp_in,&inptr);CHKERRQ(ierr);
  ierr = VecRestoreArray(tmp_out,&outptr);CHKERRQ(ierr);

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

#undef __FUNCT__
#define __FUNCT__ "RHSJacobian"
PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec x,Mat *AA,Mat *BB,MatStructure *str,void *ctx)
{
  Mat            A = *AA;
  PetscScalar    v[3],*tmp;
  PetscInt       idx[3],i;
  PetscErrorCode ierr;
 
  *str = SAME_NONZERO_PATTERN;

  idx[0]=0; idx[1]=1; idx[2]=2;
  ierr = VecGetArray(x,&tmp);CHKERRQ(ierr);

  i = 0;
  v[0] = 2.0; v[1] = 1.0; v[2] = 0.0; 
  ierr = MatSetValues(A,1,&i,3,idx,v,INSERT_VALUES);CHKERRQ(ierr);

  i = 1;
  v[0] = 1.0; v[1] = 2.0; v[2] = 1.0; 
  ierr = MatSetValues(A,1,&i,3,idx,v,INSERT_VALUES);CHKERRQ(ierr);
 
  i = 2;
  v[0]= 0.0; v[1] = 1.0; v[2] = 2.0;
  ierr = MatSetValues(A,1,&i,3,idx,v,INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = VecRestoreArray(x,&tmp);CHKERRQ(ierr);

  return 0;
}

/*
      The exact solutions 
*/
PetscReal solx(PetscReal t) 
{
  return exp((2.0 - PetscSqrtReal(2.0))*t)/2.0 - exp((2.0 - PetscSqrtReal(2.0))*t)/(2.0*PetscSqrtReal(2.0)) + 
         exp((2.0 + PetscSqrtReal(2.0))*t)/2.0 + exp((2.0 + PetscSqrtReal(2.0))*t)/(2.0*PetscSqrtReal(2.0));
}

PetscReal soly(PetscReal t) 
{
  return exp((2.0 - PetscSqrtReal(2.0))*t)/2.0 - exp((2.0 - PetscSqrtReal(2.0))*t)/PetscSqrtReal(2.0) + 
         exp((2.0 + PetscSqrtReal(2.0))*t)/2.0 + exp((2.0 + PetscSqrtReal(2.0))*t)/PetscSqrtReal(2.0);
}
 
PetscReal solz(PetscReal t) 
{
  return exp((2.0 - PetscSqrtReal(2.0))*t)/2.0 - exp((2.0 - PetscSqrtReal(2.0))*t)/(2.0*PetscSqrtReal(2.0)) + 
         exp((2.0 + PetscSqrtReal(2.0))*t)/2.0 + exp((2.0 + PetscSqrtReal(2.0))*t)/(2.0*PetscSqrtReal(2.0));
}



