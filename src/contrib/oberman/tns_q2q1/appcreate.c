


#include "appctx.h"

/*
         -  Generates the "global" parallel vector to contain the 
	    right hand side and solution.
         -  Generates "ghosted" local vectors for local computations etc.
         -  Generates scatter context for updating ghost points etc.
*/
#undef __FUNC__
#define __FUNC__ "AppCxtCreateVector"
int AppCtxCreateVector(AppCtx* appctx)
{
  /* Want everything here to be DF driven, not vertex driven 

Output of CreateVec:
  Global vectors:
    - b for rhs
    - f for nonlinear function
    - g for solution, initial guess
    - *solnv for time step solutions
  Local Vectors:
    - f_local for local part of nonlinear function
  Scatter:
    - dfgtol
    */

/********* Collect context informatrion ***********/
  AppGrid                *grid = &appctx->grid;
  AppAlgebra             *algebra = &appctx->algebra;
  MPI_Comm               comm = appctx->comm;
 AppEquations *equations = &appctx->equations;

 /************* Variables to set ************/
/* global to local mapping for vectors */
 Vec x, z;  /* x,z for nonzero structure,*/
 Vec x_local, w_local, z_local; /* used for determining nonzero structure of matriz */
Vec b;  /* b for holding rhs */
Vec f; /* nonlinear function */
Vec g; /*for solution, and initial guess */
Vec *solnv; /* for solution at each time step */
 Vec f_local; /* used for nonlinear function */
IS is1, is2; /* temporary */
 /********** Internal Variables **********/
int ierr,its, i;
int n1,n2;
int df_v_count, *dfv_indices;
double one = 1.0, zero = 0.0;
  PetscFunctionBegin;
  /*  Create vector to contain load, nonlinear function, and initial guess  
   This is driven by the DFs: local size should be number of  DFs on this proc.  */
  ierr = VecCreateMPI(comm, grid->df_local_count, PETSC_DECIDE, &algebra->b);CHKERRQ(ierr);
  ierr = VecSetLocalToGlobalMapping(algebra->b, grid->dfltog);CHKERRQ(ierr);
 
  /* create vector to contain the first and second velocity components */
  ISGetSize(grid->df_v1, &n1);  /* THIS MAY INCLUDE GHOSTED DFs */
  ISGetSize(grid->df_v2, &n2);
  ierr = VecCreateMPI(comm, n1, PETSC_DECIDE, &algebra->v1);CHKERRQ(ierr);
  ierr = VecCreateMPI(comm, n2, PETSC_DECIDE, &algebra->v2);CHKERRQ(ierr);
  ierr = VecSetLocalToGlobalMapping(algebra->v1, grid->dfltog);CHKERRQ(ierr);
  ierr = VecSetLocalToGlobalMapping(algebra->v2, grid->dfltog);CHKERRQ(ierr);

  ISCreateStride(comm,n1,0,1,&is1);
 ISCreateStride(comm,n2,0,1,&is2);

  ierr = VecScatterCreate(algebra->b, grid->df_v1, algebra->v1, is1, &algebra->dfvtov1);CHKERRQ(ierr);
  ierr = VecScatterCreate(algebra->b, grid->df_v2, algebra->v2, is2, &algebra->dfvtov2);CHKERRQ(ierr);

  /* The size of f_local should be the number of dfs including those corresponding to ghosted vertices on this proc */
   ierr = ISGetSize(grid->df_v1, &n1); CHKERRQ(ierr);
   ierr = ISGetSize(grid->df_v2, &n2); CHKERRQ(ierr);
 
 ierr = VecCreateSeq(PETSC_COMM_SELF, n1, &algebra->v1_local);CHKERRQ(ierr);
 ierr = VecCreateSeq(PETSC_COMM_SELF, n2, &algebra->v2_local);CHKERRQ(ierr);
ierr = VecScatterCreate(algebra->v1, grid->df_v1, algebra->v1_local, is1, &algebra->dfv1gtol);CHKERRQ(ierr);
ierr = VecScatterCreate(algebra->v2, grid->df_v2, algebra->v2_local, is1, &algebra->dfv2gtol);CHKERRQ(ierr);

  ierr = VecDuplicate(algebra->v1,&algebra->v1a);CHKERRQ(ierr);
  ierr = VecDuplicate(algebra->v2,&algebra->v2a);CHKERRQ(ierr);
  ierr = VecDuplicate(algebra->v1,&algebra->v1b);CHKERRQ(ierr);
  ierr = VecDuplicate(algebra->v2,&algebra->v2b);CHKERRQ(ierr);
  
  /* For load and solution vectors.  Duplicated vectors inherit the blocking */
  ierr = VecDuplicate(algebra->b,&algebra->f);CHKERRQ(ierr);/*  the nonlinear function */
  ierr = VecDuplicate(algebra->b,&algebra->g);CHKERRQ(ierr);/*  the initial guess  */
  ierr = VecDuplicate(algebra->b,&algebra->conv);CHKERRQ(ierr);/*  convection  */
  ierr = VecDuplicate(algebra->b,&algebra->convl);CHKERRQ(ierr);/*  the old convection  */
  ierr = VecDuplicate(algebra->b,&algebra->convll);CHKERRQ(ierr);/*  old old convection  */

  ierr = VecDuplicate(algebra->b,&algebra->soln);CHKERRQ(ierr);/*  old old convection  */
  ierr = VecDuplicate(algebra->b,&algebra->soln1);CHKERRQ(ierr);/*  old old convection  */
  ierr = VecDuplicate(algebra->b,&algebra->soln2);CHKERRQ(ierr);/*  old old convection  */

  /* vector for the matrix multiplication in dynamic jacobian */
  ierr = VecDuplicate(algebra->b,&algebra->dtvec);CHKERRQ(ierr);

  /* The size of f_local should be the number of dfs including those corresponding to ghosted vertices on this proc */
  ierr = VecCreateSeq(PETSC_COMM_SELF, grid->df_count, &algebra->f_local);CHKERRQ(ierr);
  ierr = VecScatterCreate(algebra->f, grid->df_global, algebra->f_local, 0, &algebra->dfgtol);CHKERRQ(ierr);





 /* here create a vector and a scatter for each type of the boundary vertices */
  if (equations->vin_flag){
    ierr = VecCreateSeq(PETSC_COMM_SELF, grid->inlet_vcount, &algebra->f_vinlet);CHKERRQ(ierr);
    ierr = VecScatterCreate(algebra->f, grid->isinlet_vdf, algebra->f_vinlet, 0, &algebra->gtol_vinlet);CHKERRQ(ierr);
  }
  if(equations->vout_flag){
    ierr = VecCreateSeq(PETSC_COMM_SELF, grid->outlet_vcount, &algebra->f_voutlet);CHKERRQ(ierr);
    ierr = VecScatterCreate(algebra->f, grid->isoutlet_vdf, algebra->f_voutlet, 0, &algebra->gtol_voutlet);CHKERRQ(ierr);
  }

  if(equations->wall_flag){
  ierr = VecCreateSeq(PETSC_COMM_SELF, grid->wall_vcount, &algebra->f_wall);CHKERRQ(ierr);
  ierr = VecScatterCreate(algebra->f, grid->iswall_vdf, algebra->f_wall, 0, &algebra->gtol_wall);CHKERRQ(ierr);
  }

  if(equations->ywall_flag){
  ierr = VecCreateSeq(PETSC_COMM_SELF, grid->ywall_vcount, &algebra->f_ywall);CHKERRQ(ierr);
  ierr = VecScatterCreate(algebra->f, grid->isywall_vdf, algebra->f_ywall, 0, &algebra->gtol_ywall);CHKERRQ(ierr);
  }

 if(equations->pout_flag){
   ierr = VecCreateSeq(PETSC_COMM_SELF, grid->outlet_pcount, &algebra->f_poutlet);CHKERRQ(ierr);
   ierr = VecScatterCreate(algebra->f, grid->isoutlet_pdf, algebra->f_poutlet, 0, &algebra->gtol_poutlet);CHKERRQ(ierr);
 }
 if (equations->pin_flag){
   ierr = VecCreateSeq(PETSC_COMM_SELF, grid->inlet_pcount, &algebra->f_pinlet);CHKERRQ(ierr);
   ierr = VecScatterCreate(algebra->f, grid->isinlet_pdf, algebra->f_pinlet, 0, &algebra->gtol_pinlet);CHKERRQ(ierr);
 }
  /************ STOP here, figure out the vectors needed for MatCreate later on ***********/

  PetscFunctionReturn(0);
}

/*
     Creates the sparse matrix (with near- correct nonzero pattern, at least enough on each pplace) 
We need a matrix for, e.g the stiffness, and one for the jacobian of the nonlinear map.
Generally, the size of the matrix is df_total x df_total, but we need to compute the nonzero structure
for efficiency purposes.
DO LATER
*/
#undef __FUNC__
#define __FUNC__ "AppCxtCreateMatrix"
int AppCtxCreateMatrix(AppCtx* appctx)
{
/********* Collect context informatrion ***********/
  AppAlgebra             *algebra = &appctx->algebra;
  AppGrid                *grid    = &appctx->grid;
  MPI_Comm               comm = appctx->comm;
int ierr; 
  PetscFunctionBegin;
  /* now create the matrix */
  /* using very rough estimate for nonzeros on and off the diagonal */
  ierr = MatCreateMPIAIJ(comm, grid->df_local_count, grid->df_local_count, PETSC_DETERMINE, PETSC_DETERMINE, 69 , 0, 5 , 0, &algebra->A); CHKERRQ(ierr);
  /* Set the local to global mapping */
  ierr = MatSetLocalToGlobalMapping(algebra->A, grid->dfltog);  CHKERRQ(ierr);

  /* ditto for MassMAtrix */
  ierr = MatCreateMPIAIJ(comm, grid->df_local_count, grid->df_local_count, PETSC_DETERMINE, PETSC_DETERMINE, 69 , 0, 5 , 0, &algebra->M); CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(algebra->M, grid->dfltog);  CHKERRQ(ierr);

  /* ditto for jac */
  ierr = MatCreateMPIAIJ(comm, grid->df_local_count, grid->df_local_count, PETSC_DETERMINE, PETSC_DETERMINE, 69 , 0, 5 , 0, &algebra->J); CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(algebra->J, grid->dfltog);  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}




