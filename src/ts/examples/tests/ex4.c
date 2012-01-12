/*
       The Problem:
           Solve the convection-diffusion equation:

             u_t+a*(u_x+u_y)=epsilon*(u_xx+u_yy)
             u=0   at x=0, y=0
             u_x=0 at x=1
             u_y=0 at y=1
             u = exp(-20.0*(pow(x-0.5,2.0)+pow(y-0.5,2.0))) at t=0

       This program tests the routine of computing the Jacobian by the
       finite difference method as well as PETSc with SUNDIALS.

*/

static char help[] = "Solve the convection-diffusion equation. \n\n";

#include <petscts.h>

typedef struct
{
  PetscInt      m;      /* the number of mesh points in x-direction */
  PetscInt      n;      /* the number of mesh points in y-direction */
  PetscReal     dx;     /* the grid space in x-direction */
  PetscReal     dy;     /* the grid space in y-direction */
  PetscReal     a;      /* the convection coefficient    */
  PetscReal     epsilon; /* the diffusion coefficient    */
  PetscReal     tfinal;  
} Data;

extern PetscErrorCode Monitor(TS,PetscInt,PetscReal,Vec,void *);
extern PetscErrorCode Initial(Vec,void *);
extern PetscErrorCode RHSFunction(TS,PetscReal,Vec,Vec,void*);
extern PetscErrorCode RHSJacobian(TS,PetscReal,Vec,Mat*,Mat*,MatStructure *,void*);
extern PetscErrorCode PostStep(TS);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       time_steps=100,iout,NOUT=1;
  PetscMPIInt    size;
  Vec            global;
  PetscReal      dt,ftime,ftime_original;
  TS             ts;
  PetscViewer    viewfile;
  MatStructure   J_structure;
  Mat            J = 0;
  Vec            x;
  Data           data;
  PetscInt       mn;
  PetscBool      flg;
  ISColoring     iscoloring;
  MatFDColoring  matfdcoloring = 0;
  PetscBool      fd_jacobian_coloring = PETSC_FALSE;
  SNES           snes;
  KSP            ksp;
  PC             pc;
  PetscViewer    viewer;
  char           pcinfo[120],tsinfo[120];
  const TSType   tstype;
  PetscBool      sundials;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  /* set data */
  data.m = 9;
  data.n = 9;
  data.a = 1.0;
  data.epsilon = 0.1;
  data.dx      = 1.0/(data.m+1.0);
  data.dy      = 1.0/(data.n+1.0);
  mn = (data.m)*(data.n);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-time",&time_steps,PETSC_NULL);CHKERRQ(ierr);

  /* set initial conditions */
  ierr = VecCreate(PETSC_COMM_WORLD,&global);CHKERRQ(ierr);
  ierr = VecSetSizes(global,PETSC_DECIDE,mn);CHKERRQ(ierr);
  ierr = VecSetFromOptions(global);CHKERRQ(ierr);
  ierr = Initial(global,&data);CHKERRQ(ierr);
  ierr = VecDuplicate(global,&x);CHKERRQ(ierr);

  /* create timestep context */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSMonitorSet(ts,Monitor,&data,PETSC_NULL);CHKERRQ(ierr);
#if defined(PETSC_HAVE_SUNDIALS)
  ierr = TSSetType(ts,TSSUNDIALS);CHKERRQ(ierr);
#else
  ierr = TSSetType(ts,TSEULER);CHKERRQ(ierr);
#endif
  dt             = 0.1;
  ftime_original = data.tfinal = 1.0; 
  ierr = TSSetInitialTimeStep(ts,0.0,dt);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,time_steps,ftime_original);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,global);CHKERRQ(ierr);

  /* set user provided RHSFunction and RHSJacobian */
  ierr = TSSetRHSFunction(ts,PETSC_NULL,RHSFunction,&data);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&J);CHKERRQ(ierr);
  ierr = MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,mn,mn);CHKERRQ(ierr);
  ierr = MatSetFromOptions(J);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(J,5,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(J,5,PETSC_NULL,5,PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(PETSC_NULL,"-ts_fd",&flg);CHKERRQ(ierr);
  if (!flg){
    ierr = TSSetRHSJacobian(ts,J,J,RHSJacobian,&data);CHKERRQ(ierr);
  } else {
    ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
    ierr = PetscOptionsHasName(PETSC_NULL,"-fd_color",&fd_jacobian_coloring);CHKERRQ(ierr);
    if (fd_jacobian_coloring){ /* Use finite differences with coloring */
      /* Get data structure of J */
      PetscBool  pc_diagonal;
      ierr = PetscOptionsHasName(PETSC_NULL,"-pc_diagonal",&pc_diagonal);CHKERRQ(ierr);
      if (pc_diagonal){ /* the preconditioner of J is a diagonal matrix */
        PetscInt rstart,rend,i;
        PetscScalar  zero=0.0;
        ierr = MatGetOwnershipRange(J,&rstart,&rend);CHKERRQ(ierr);
        for (i=rstart; i<rend; i++){
          ierr = MatSetValues(J,1,&i,1,&i,&zero,INSERT_VALUES);CHKERRQ(ierr);
        }
        ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      } else {
        /* Fill the structure using the expensive SNESDefaultComputeJacobian. Temporarily set up the TS so we can call this function */
        ierr = TSSetType(ts,TSBEULER);CHKERRQ(ierr);
        ierr = TSSetUp(ts);CHKERRQ(ierr);
        ierr = SNESDefaultComputeJacobian(snes,x,&J,&J,&J_structure,ts);CHKERRQ(ierr);
      }

      /* create coloring context */
      ierr = MatGetColoring(J,MATCOLORINGSL,&iscoloring);CHKERRQ(ierr);
      ierr = MatFDColoringCreate(J,iscoloring,&matfdcoloring);CHKERRQ(ierr);
      ierr = MatFDColoringSetFunction(matfdcoloring,(PetscErrorCode (*)(void))SNESTSFormFunction,ts);CHKERRQ(ierr);
      ierr = MatFDColoringSetFromOptions(matfdcoloring);CHKERRQ(ierr);
      ierr = SNESSetJacobian(snes,J,J,SNESDefaultComputeJacobianColor,matfdcoloring);CHKERRQ(ierr);
      ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);
    } else { /* Use finite differences (slow) */
      ierr = SNESSetJacobian(snes,J,J,SNESDefaultComputeJacobian,PETSC_NULL);CHKERRQ(ierr);
    }
  }

  /* Pick up a Petsc preconditioner */
  /* one can always set method or preconditioner during the run time */
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCJACOBI);CHKERRQ(ierr);

  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSetUp(ts);CHKERRQ(ierr);
  
  /* Test TSSetPostStep() */
  ierr = PetscOptionsHasName(PETSC_NULL,"-test_PostStep",&flg);CHKERRQ(ierr);
  if (flg){
    ierr = TSSetPostStep(ts,PostStep);CHKERRQ(ierr);
  }

  ierr = PetscOptionsGetInt(PETSC_NULL,"-NOUT",&NOUT,PETSC_NULL);CHKERRQ(ierr);
  for (iout=1; iout<=NOUT; iout++){
    ierr = TSSetDuration(ts,time_steps,iout*ftime_original/NOUT);CHKERRQ(ierr);
    ierr = TSSolve(ts,global,&ftime);CHKERRQ(ierr);
    ierr = TSSetInitialTimeStep(ts,ftime,dt);CHKERRQ(ierr);
  }
  /* Interpolate solution at tfinal */
  ierr = TSGetSolution(ts,&global);CHKERRQ(ierr);
  ierr = TSInterpolate(ts,ftime_original,global);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(PETSC_NULL,"-matlab_view",&flg);CHKERRQ(ierr);
  if (flg){ /* print solution into a MATLAB file */
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"out.m",&viewfile);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(viewfile,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    ierr = VecView(global,viewfile);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewfile);CHKERRQ(ierr);
  }

  /* display solver info for Sundials */ 
  ierr = TSGetType(ts,&tstype);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)ts,TSSUNDIALS,&sundials);CHKERRQ(ierr);
  if (sundials){
    ierr = PetscViewerStringOpen(PETSC_COMM_WORLD,tsinfo,120,&viewer);CHKERRQ(ierr);
    ierr = TSView(ts,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = PetscViewerStringOpen(PETSC_COMM_WORLD,pcinfo,120,&viewer);CHKERRQ(ierr);
    ierr = PCView(pc,viewer);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%d Procs,%s TSType, %s Preconditioner\n",
                     size,tsinfo,pcinfo);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }

  /* free the memories */
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = VecDestroy(&global);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr= MatDestroy(&J);CHKERRQ(ierr);
  if (fd_jacobian_coloring){ierr = MatFDColoringDestroy(&matfdcoloring);CHKERRQ(ierr);}
  ierr = PetscFinalize();
  return 0;
}

/* -------------------------------------------------------------------*/
/* the initial function */
PetscReal f_ini(PetscReal x,PetscReal y)
{
  PetscReal f;

  f=exp(-20.0*(pow(x-0.5,2.0)+pow(y-0.5,2.0)));
  return f;
}

#undef __FUNCT__
#define __FUNCT__ "Initial"
PetscErrorCode Initial(Vec global,void *ctx)
{
  Data           *data = (Data*)ctx;
  PetscInt       m,row,col;
  PetscReal      x,y,dx,dy;
  PetscScalar    *localptr;
  PetscInt       i,mybase,myend,locsize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* make the local  copies of parameters */
  m = data->m;
  dx = data->dx;
  dy = data->dy;

  /* determine starting point of each processor */
  ierr = VecGetOwnershipRange(global,&mybase,&myend);CHKERRQ(ierr);
  ierr = VecGetLocalSize(global,&locsize);CHKERRQ(ierr);

  /* Initialize the array */
  ierr = VecGetArray(global,&localptr);CHKERRQ(ierr);

  for (i=0; i<locsize; i++) {
    row = 1+(mybase+i)-((mybase+i)/m)*m;
    col = (mybase+i)/m+1;
    x = dx*row;
    y = dy*col;
    localptr[i] = f_ini(x,y);
  }

  ierr = VecRestoreArray(global,&localptr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Monitor"
PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal time,Vec global,void *ctx)
{
  VecScatter     scatter;
  IS             from,to;
  PetscInt       i,n,*idx,nsteps,maxsteps;
  Vec            tmp_vec;
  PetscErrorCode ierr;
  PetscScalar    *tmp;
  PetscReal      maxtime;
  Data           *data = (Data*)ctx;
  PetscReal      tfinal = data->tfinal;

  PetscFunctionBegin;
  if (time > tfinal) PetscFunctionReturn(0);

  ierr = TSGetTimeStepNumber(ts,&nsteps);CHKERRQ(ierr);
  /* display output at selected time steps */
  ierr = TSGetDuration(ts, &maxsteps, &maxtime);CHKERRQ(ierr);
  if (nsteps % 10 != 0 && time < maxtime) PetscFunctionReturn(0);

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
  ierr = PetscPrintf(PETSC_COMM_WORLD,"At t[%d] =%14.2e u= %14.2e at the center \n",nsteps,time,PetscRealPart(tmp[n/2]));CHKERRQ(ierr);
  ierr = VecRestoreArray(tmp_vec,&tmp);CHKERRQ(ierr);

  ierr = PetscFree(idx);CHKERRQ(ierr);
  ierr = ISDestroy(&from);CHKERRQ(ierr);
  ierr = ISDestroy(&to);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&scatter);CHKERRQ(ierr);
  ierr = VecDestroy(&tmp_vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RHSJacobian"
PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec x,Mat *AA,Mat *BB,MatStructure *flag,void *ptr)
{
  Data           *data = (Data*)ptr;
  Mat            A = *AA;
  PetscScalar    v[5];
  PetscInt       idx[5],i,j,row;
  PetscErrorCode ierr;
  PetscInt       m,n,mn;
  PetscReal      dx,dy,a,epsilon,xc,xl,xr,yl,yr;

  PetscFunctionBegin;
  m = data->m;
  n = data->n;
  mn = m*n;
  dx = data->dx;
  dy = data->dy;
  a = data->a;
  epsilon = data->epsilon;

  xc = -2.0*epsilon*(1.0/(dx*dx)+1.0/(dy*dy));
  xl = 0.5*a/dx+epsilon/(dx*dx);
  xr = -0.5*a/dx+epsilon/(dx*dx);
  yl = 0.5*a/dy+epsilon/(dy*dy);
  yr = -0.5*a/dy+epsilon/(dy*dy);

  row=0;
  v[0] = xc; v[1]=xr; v[2]=yr;
  idx[0]=0; idx[1]=2; idx[2]=m;
  ierr = MatSetValues(A,1,&row,3,idx,v,INSERT_VALUES);CHKERRQ(ierr);

  row=m-1;
  v[0]=2.0*xl; v[1]=xc; v[2]=yr;
  idx[0]=m-2; idx[1]=m-1; idx[2]=m-1+m;
  ierr = MatSetValues(A,1,&row,3,idx,v,INSERT_VALUES);CHKERRQ(ierr);

  for (i=1; i<m-1; i++) {
    row=i;
    v[0]=xl; v[1]=xc; v[2]=xr; v[3]=yr;
    idx[0]=i-1; idx[1]=i; idx[2]=i+1; idx[3]=i+m;
    ierr = MatSetValues(A,1,&row,4,idx,v,INSERT_VALUES);CHKERRQ(ierr);
  }

  for (j=1; j<n-1; j++) {
    row=j*m;
    v[0]=xc; v[1]=xr; v[2]=yl; v[3]=yr;
    idx[0]=j*m; idx[1]=j*m; idx[2]=j*m-m; idx[3]=j*m+m;
    ierr = MatSetValues(A,1,&row,4,idx,v,INSERT_VALUES);CHKERRQ(ierr);

    row=j*m+m-1;
    v[0]=xc; v[1]=2.0*xl; v[2]=yl; v[3]=yr;
    idx[0]=j*m+m-1; idx[1]=j*m+m-1-1; idx[2]=j*m+m-1-m; idx[3]=j*m+m-1+m;
    ierr = MatSetValues(A,1,&row,4,idx,v,INSERT_VALUES);CHKERRQ(ierr);

    for(i=1; i<m-1; i++) {
      row=j*m+i;
      v[0]=xc; v[1]=xl; v[2]=xr; v[3]=yl; v[4]=yr;
      idx[0]=j*m+i; idx[1]=j*m+i-1; idx[2]=j*m+i+1; idx[3]=j*m+i-m;
      idx[4]=j*m+i+m;
      ierr = MatSetValues(A,1,&row,5,idx,v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  row=mn-m;
  v[0] = xc; v[1]=xr; v[2]=2.0*yl;
  idx[0]=mn-m; idx[1]=mn-m+1; idx[2]=mn-m-m;
  ierr = MatSetValues(A,1,&row,3,idx,v,INSERT_VALUES);CHKERRQ(ierr);

  row=mn-1;
  v[0] = xc; v[1]=2.0*xl; v[2]=2.0*yl;
  idx[0]=mn-1; idx[1]=mn-2; idx[2]=mn-1-m;
  ierr = MatSetValues(A,1,&i,3,idx,v,INSERT_VALUES);CHKERRQ(ierr);

  for (i=1; i<m-1; i++) {
    row=mn-m+i;
    v[0]=xl; v[1]=xc; v[2]=xr; v[3]=2.0*yl;
    idx[0]=mn-m+i-1; idx[1]=mn-m+i; idx[2]=mn-m+i+1; idx[3]=mn-m+i-m;
    ierr = MatSetValues(A,1,&row,4,idx,v,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* *flag = SAME_NONZERO_PATTERN; */
  *flag = DIFFERENT_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

/* globalout = -a*(u_x+u_y) + epsilon*(u_xx+u_yy) */
#undef __FUNCT__
#define __FUNCT__ "RHSFunction"
PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec globalin,Vec globalout,void *ctx)
{
  Data           *data = (Data*)ctx;
  PetscInt       m,n,mn;
  PetscReal      dx,dy;
  PetscReal      xc,xl,xr,yl,yr;
  PetscReal      a,epsilon;
  PetscScalar    *inptr,*outptr;
  PetscInt       i,j,len;
  PetscErrorCode ierr;
  IS             from,to;
  PetscInt       *idx;
  VecScatter     scatter;
  Vec            tmp_in,tmp_out;

  PetscFunctionBegin;
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
  ierr = VecGetSize(globalin,&len);CHKERRQ(ierr);

  /* Set the index sets */
  ierr = PetscMalloc(len*sizeof(PetscInt),&idx);CHKERRQ(ierr);
  for(i=0; i<len; i++) idx[i]=i;

  /* Create local sequential vectors */
  ierr = VecCreateSeq(PETSC_COMM_SELF,len,&tmp_in);CHKERRQ(ierr);
  ierr = VecDuplicate(tmp_in,&tmp_out);CHKERRQ(ierr);

  /* Create scatter context */
  ierr = ISCreateGeneral(PETSC_COMM_SELF,len,idx,PETSC_COPY_VALUES,&from);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,len,idx,PETSC_COPY_VALUES,&to);CHKERRQ(ierr);
  ierr = VecScatterCreate(globalin,from,tmp_in,to,&scatter);CHKERRQ(ierr);
  ierr = VecScatterBegin(scatter,globalin,tmp_in,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(scatter,globalin,tmp_in,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&scatter);CHKERRQ(ierr);

  /*Extract income array - include ghost points */
  ierr = VecGetArray(tmp_in,&inptr);CHKERRQ(ierr);

  /* Extract outcome array*/
  ierr = VecGetArray(tmp_out,&outptr);CHKERRQ(ierr);

  outptr[0] = xc*inptr[0]+xr*inptr[1]+yr*inptr[m];
  outptr[m-1] = 2.0*xl*inptr[m-2]+xc*inptr[m-1]+yr*inptr[m-1+m];
  for (i=1; i<m-1; i++) {
    outptr[i] = xc*inptr[i]+xl*inptr[i-1]+xr*inptr[i+1]
      +yr*inptr[i+m];
  }

  for (j=1; j<n-1; j++) {
    outptr[j*m] = xc*inptr[j*m]+xr*inptr[j*m+1]+
      yl*inptr[j*m-m]+yr*inptr[j*m+m];
    outptr[j*m+m-1] = xc*inptr[j*m+m-1]+2.0*xl*inptr[j*m+m-1-1]+
      yl*inptr[j*m+m-1-m]+yr*inptr[j*m+m-1+m];
    for(i=1; i<m-1; i++) {
      outptr[j*m+i] = xc*inptr[j*m+i]+xl*inptr[j*m+i-1]+xr*inptr[j*m+i+1]
        +yl*inptr[j*m+i-m]+yr*inptr[j*m+i+m];
    }
  }

  outptr[mn-m] = xc*inptr[mn-m]+xr*inptr[mn-m+1]+2.0*yl*inptr[mn-m-m];
  outptr[mn-1] = 2.0*xl*inptr[mn-2]+xc*inptr[mn-1]+2.0*yl*inptr[mn-1-m];
  for (i=1; i<m-1; i++) {
    outptr[mn-m+i] = xc*inptr[mn-m+i]+xl*inptr[mn-m+i-1]+xr*inptr[mn-m+i+1]
      +2*yl*inptr[mn-m+i-m];
  }

  ierr = VecRestoreArray(tmp_in,&inptr);
  ierr = VecRestoreArray(tmp_out,&outptr);

  ierr = VecScatterCreate(tmp_out,from,globalout,to,&scatter);CHKERRQ(ierr);
  ierr = VecScatterBegin(scatter,tmp_out,globalout,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(scatter,tmp_out,globalout,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  /* Destroy idx aand scatter */
  ierr = VecDestroy(&tmp_in);CHKERRQ(ierr);
  ierr = VecDestroy(&tmp_out);CHKERRQ(ierr);
  ierr = ISDestroy(&from);CHKERRQ(ierr);
  ierr = ISDestroy(&to);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&scatter);CHKERRQ(ierr);

  ierr = PetscFree(idx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PostStep"
PetscErrorCode PostStep(TS ts)
{
  PetscErrorCode  ierr;
  PetscReal       t;

  PetscFunctionBegin;
  ierr = TSGetTime(ts,&t); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"  PostStep, t: %g\n",t);
  PetscFunctionReturn(0);
}
