#ifndef lint
static char vcid[] = "$Id: ex3.c,v 1.18 1997/06/3 01:37:07 lixu Exp $";
#endif
/*
       Formatted test for TS routines.

          Solves U_t=F(t,u)
	  Where:
          
	          [2*u1+u2
	  F(t,u)= [u1+2*u2+u3
	          [   u2+2*u3
       We can compare the solutions from euler, beuler and PVODE to
       see what is the difference.

*/

static char help[] = "Solves a nonlinear ODE \n\n";

#include <malloc.h>
#include "sys.h"
#include <math.h>
#include "ts.h"
#include "pc.h"

int RHSFunction(TS,double,Vec,Vec,void*);
int RHSJacobian(TS,double,Vec,Mat*,Mat*,MatStructure *,void*);
int Monitor(TS, int, double, Vec, void *);
int Initial(Vec, void *);

#define linear_no_matrix       0
#define linear_no_time         1
#define linear                 2
#define nonlinear_no_jacobian  3
#define nonlinear              4

int main(int argc,char **argv)
{
  int           ierr,  time_steps = 100, steps, flg, size;
  Vec           global;
  double        dt,ftime;
  TS            ts;
  TSType        type;
  Viewer	viewer;
  MatStructure  A_structure;

  PC		pc;

  Mat           A = 0;
  TSProblemType tsproblem = TS_NONLINEAR; /* Need to be TS_NONLINEAR */
  char          pcinfo[120], tsinfo[120];
 
  PetscInitialize(&argc,&argv,(char*)0,help);
  MPI_Comm_size(PETSC_COMM_WORLD, &size);
 
  ierr = OptionsGetInt(PETSC_NULL,"-time",&time_steps,&flg);CHKERRA(ierr);
    
  /* set initial conditions */
  ierr = VecCreate(MPI_COMM_WORLD,3,&global); CHKERRQ(ierr);
  ierr = Initial(global,NULL); CHKERRA(ierr);
 
  /* make timestep context */
  ierr = TSCreate(PETSC_COMM_WORLD,tsproblem,&ts); CHKERRA(ierr);
  ierr = TSSetMonitor(ts,Monitor,NULL); CHKERRA(ierr);

  dt = 0.1;

  /*
    The user provides the RHS and Jacobian
  */
  ierr = TSSetRHSFunction(ts,RHSFunction,NULL); CHKERRA(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,3,3,&A); CHKERRA(ierr);
  ierr = RHSJacobian(ts,0.0,global,&A,&A,&A_structure,NULL); CHKERRA(ierr);
  ierr = TSSetRHSJacobian(ts,A,A,RHSJacobian,NULL); CHKERRA(ierr);  
 
  /* Use CVODE */
  ierr = TSSetType(ts, TS_CVODE); CHKERRA(ierr);

  ierr = TSSetFromOptions(ts);CHKERRA(ierr);
  ierr = TSGetType(ts,&type,PETSC_NULL); CHKERRA(ierr);

  ierr = TSSetInitialTimeStep(ts,0.0,dt); CHKERRA(ierr);
  ierr = TSSetDuration(ts,time_steps,1); CHKERRA(ierr);
  ierr = TSSetSolution(ts,global); CHKERRA(ierr);


  /* Pick up a Petsc preconditioner */
  /* one can always set method or preconditioner during the run time */
  /*
  ierr = TSPVodePCSetType(ts, PCJACOBI); CHKERRA(ierr);
  ierr = TSPVodeSetMethod(ts, PVODE_BDF); CHKERRA(ierr);
  ierr = TSPVodeSetMethodFromOptions(ts); CHKERRA(ierr);
  */

  ierr = TSSetUp(ts); CHKERRA(ierr);
  ierr = TSStep(ts,&steps,&ftime); CHKERRA(ierr);

  /* extracts the PC  from ts */
  ierr = TSPVodeGetPC(ts, &pc); CHKERRA(ierr);

  ierr = ViewerStringOpen(PETSC_COMM_WORLD,tsinfo,120,&viewer); CHKERRA(ierr);
  ierr = TSView(ts,viewer); CHKERRQ(ierr);

  ierr = ViewerStringOpen(PETSC_COMM_WORLD,pcinfo,120,&viewer); CHKERRA(ierr);
  ierr = PCView(pc,viewer); CHKERRQ(ierr);

  PetscPrintf(PETSC_COMM_WORLD,"%d Procs, %s Preconditioner, %s\n",
                size,tsinfo,pcinfo);

  /* free the memories */
  ierr = TSDestroy(ts); CHKERRA(ierr);
  ierr = VecDestroy(global); CHKERRA(ierr);
  if (A) {ierr= MatDestroy(A); CHKERRA(ierr);}
  ierr = PCDestroy(pc); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}

/* -------------------------------------------------------------------*/
/* this test problem has initial values (1,1,1).                      */
int Initial(Vec global, void *ctx)
{
  Scalar *localptr;
  int    i,mybase,myend,ierr,locsize;

  /* determine starting point of each processor */
  ierr = VecGetOwnershipRange(global,&mybase,&myend); CHKERRQ(ierr);
  ierr = VecGetLocalSize(global,&locsize); CHKERRQ(ierr);

  /* Initialize the array */
  ierr = VecGetArray(global,&localptr); CHKERRQ(ierr);
  for (i=0; i<locsize; i++) {
    localptr[i] = 1.0;
  }
  
  if (mybase == 0) localptr[0]=1.0;

  ierr = VecRestoreArray(global,&localptr); CHKERRQ(ierr);
  return 0;
}

int Monitor(TS ts, int step, double time,Vec global, void *ctx)
{
  VecScatter scatter;
  IS from, to;
  int i, n, *idx;
  Vec tmp_vec;
  int      ierr;
  Scalar   *tmp;

  /* Get the size of the vector */
  ierr = VecGetSize(global, &n); CHKERRQ(ierr);

  /* Set the index sets */
  idx=(int *) calloc(n,sizeof(int));
  for(i=0; i<n; i++) idx[i]=i;
 
  /* Create local sequential vectors */
  ierr = VecCreateSeq(MPI_COMM_SELF,n,&tmp_vec); CHKERRQ(ierr);

  /* Create scatter context */
  ierr = ISCreateGeneral(MPI_COMM_SELF,n,idx,&from); CHKERRQ(ierr);
  ierr = ISCreateGeneral(MPI_COMM_SELF,n,idx,&to); CHKERRQ(ierr);
  ierr = VecScatterCreate(global,from,tmp_vec,to,&scatter); CHKERRQ(ierr);
  ierr = VecScatterBegin(global,tmp_vec,INSERT_VALUES,SCATTER_FORWARD,scatter);
  CHKERRA(ierr);
  ierr = VecScatterEnd(global,tmp_vec,INSERT_VALUES,SCATTER_FORWARD,scatter);
  CHKERRA(ierr);

  ierr = VecGetArray(tmp_vec,&tmp); CHKERRQ(ierr);
  PetscPrintf(MPI_COMM_WORLD,"At t =%14.6e u = %14.6e  %14.6e  %14.6e \n",
    time,tmp[0],tmp[1],tmp[2]);
  ierr = VecRestoreArray(tmp_vec,&tmp);
  return 0;
}

int RHSFunction(TS ts, double t,Vec globalin, Vec globalout, void *ctx)
{
  Scalar *inptr, *outptr;
  int i, n, ierr;

  IS from, to;
  int *idx;
  VecScatter scatter;
  Vec tmp_in, tmp_out;

  /* Get the length of parallel vector */
  ierr = VecGetSize(globalin, &n); CHKERRQ(ierr);

  /* Set the index sets */
  idx=(int *) calloc(n,sizeof(int));
  for(i=0; i<n; i++) idx[i]=i;
  
  /* Create local sequential vectors */
  ierr = VecCreateSeq(MPI_COMM_SELF,n,&tmp_in); CHKERRQ(ierr);
  ierr = VecDuplicate(tmp_in, &tmp_out); CHKERRQ(ierr);

  /* Create scatter context */
  ierr = ISCreateGeneral(MPI_COMM_SELF,n,idx,&from); CHKERRQ(ierr);
  ierr = ISCreateGeneral(MPI_COMM_SELF,n,idx,&to); CHKERRQ(ierr);
  ierr = VecScatterCreate(globalin,from,tmp_in,to,&scatter); CHKERRQ(ierr);
  ierr = VecScatterBegin(globalin,tmp_in,INSERT_VALUES,SCATTER_FORWARD,scatter);
  CHKERRA(ierr);
  ierr = VecScatterEnd(globalin,tmp_in,INSERT_VALUES,SCATTER_FORWARD,scatter);
  CHKERRA(ierr);

  /*Extract income array */ 
  ierr = VecGetArray(tmp_in,&inptr); CHKERRQ(ierr);

  /* Extract outcome array*/
  ierr = VecGetArray(tmp_out,&outptr); CHKERRQ(ierr);

  outptr[0] = 2*inptr[0]+inptr[1];
  outptr[1] = inptr[0]+2*inptr[1]+inptr[2];
  outptr[2] = inptr[1]+2*inptr[2];

  ierr = VecRestoreArray(globalin,&inptr);
  ierr = VecRestoreArray(tmp_out,&outptr);

  ierr = VecScatterCreate(tmp_out,from,globalout,to,&scatter); CHKERRQ(ierr);
  ierr = VecScatterBegin(tmp_out,globalout,INSERT_VALUES,SCATTER_FORWARD,scatter);
  CHKERRA(ierr);
  ierr = VecScatterEnd(tmp_out,globalout,INSERT_VALUES,SCATTER_FORWARD,scatter);
  CHKERRA(ierr);

  /* Destroy idx aand scatter */
  ierr = ISDestroy(from); CHKERRQ(ierr);
  ierr = ISDestroy(to); CHKERRQ(ierr);
  ierr = VecScatterDestroy(scatter); CHKERRQ(ierr);

  return 0;
}

int RHSJacobian(TS ts,double t,Vec x,Mat *AA,Mat *BB, MatStructure *str,void *ctx)
{
  Mat A = *AA;
  Scalar v[3], *tmp;
  int idx[3], i, ierr;
 
  *str = SAME_NONZERO_PATTERN;

  idx[0]=0; idx[1]=1; idx[2]=2;
  ierr = VecGetArray(x,&tmp); CHKERRQ(ierr);

  i = 0;
  v[0] = 2.0; v[1] = 1.0; v[2] = 0.0; 
  ierr = MatSetValues(A,1,&i,3,idx,v,INSERT_VALUES); CHKERRQ(ierr);

  i = 1;
  v[0] = 1.0; v[1] = 2.0; v[2] = 1.0; 
  ierr = MatSetValues(A,1,&i,3,idx,v,INSERT_VALUES); CHKERRQ(ierr);
 
  i = 2;
  v[0]= 0.0; v[1] = 1.0; v[2] = 2.0;
  ierr = MatSetValues(A,1,&i,3,idx,v,INSERT_VALUES); CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  ierr = VecRestoreArray(x,&tmp); CHKERRQ(ierr);

  return 0;
}





