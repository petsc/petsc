
#include "appctx.h"
#include "math.h"

/*
         Sample right hand side and boundary conditions
*/
#undef __FUNCT__
#define __FUNCT__ "pde_rhs"
int pde_rhs(void *dummy,PetscInt n,PetscReal *xx,PetscReal *f)
{
  PetscReal pi = PETSC_PI, x = xx[0], y = xx[1];
  PetscFunctionBegin;
  *f = 8*pi*pi*sin(2*pi*x)*sin(2*pi*y)-20*pi*cos(2*pi*x)*sin(2*pi*y);
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "pde_bc"
int pde_bc(void *dummy,PetscInt n,PetscReal *xx,PetscReal *f)
{
  PetscReal pi = 3.1415927, x = xx[0], y = xx[1];
  PetscFunctionBegin;
  *f = sin(2*pi*x)*sin(2*pi*y);
  PetscFunctionReturn(0);
}


/*
         Sets up the linear system associated with the PDE and solves it
*/
#undef __FUNCT__
#define __FUNCT__ "AppCxtSolve"
PetscErrorCode AppCtxSolve(AppCtx* appctx)
{
  AppAlgebra  *algebra = &appctx->algebra;
  MPI_Comm    comm = appctx->comm;
  KSP        ksp;
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  /*  Set the functions to use for the right hand side and Dirichlet boundary */
  ierr = PFCreate(comm,2,1,&appctx->element.rhs);CHKERRQ(ierr);
  ierr = PFSetOptionsPrefix(appctx->element.rhs,"rhs_");CHKERRQ(ierr);
  ierr = PFSetType(appctx->element.rhs,PFQUICK,(void*)pde_rhs);CHKERRQ(ierr);
  ierr = PFSetFromOptions(appctx->element.rhs);CHKERRQ(ierr);

  ierr = PFCreate(comm,2,1,&appctx->bc);CHKERRQ(ierr);
  ierr = PFSetOptionsPrefix(appctx->bc,"bc_");CHKERRQ(ierr);
  ierr = PFSetType(appctx->bc,PFQUICK,(void*)pde_bc);CHKERRQ(ierr);
  ierr = PFSetFromOptions(appctx->bc);CHKERRQ(ierr);

  /*     A) Set the quadrature values for the reference element  */
  ierr = SetReferenceElement(appctx);CHKERRQ(ierr);

  /*     1) Create vector to contain load and various work vectors  */
  ierr = AppCtxCreateRhs(appctx);CHKERRQ(ierr);

  /*     2)  Create the sparse matrix,with correct nonzero pattern  */
  ierr = AppCtxCreateMatrix(appctx);CHKERRQ(ierr);

  /*     3)  Set the right hand side values into the load vector   */
  ierr = AppCtxSetRhs(appctx);CHKERRQ(ierr);

  /*     4)  Set the matrix entries   */
  ierr = AppCtxSetMatrix(appctx);CHKERRQ(ierr);

  /* view sparsity structure of the matrix */
  if (appctx->view.show_matrix) {  
    ierr = PetscPrintf(PETSC_COMM_WORLD,"The stiffness matrix, before bc applied\n");CHKERRQ(ierr);
    ierr = MatView(appctx->algebra.A,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  }


  /*     6) Set the matrix boundary conditions */
  ierr = SetMatrixBoundaryConditions(appctx);CHKERRQ(ierr);

  /* view sparsity structure of the matrix */
  if(appctx->view.show_matrix) {  
    ierr = PetscPrintf(PETSC_COMM_WORLD,"The stiffness matrix, after bc applied\n");CHKERRQ(ierr);
    ierr = MatView(appctx->algebra.A,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  }

  PreLoadBegin(PETSC_TRUE,"Solver setup");  

    /*     5) Set the rhs boundary conditions - this also creates initial guess that satisfies boundary conditions */
    ierr = SetBoundaryConditions(appctx);CHKERRQ(ierr);

    ierr = KSPCreate(comm,&ksp);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,algebra->A,algebra->A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    {
      ierr = KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);CHKERRQ(ierr);
    }
    ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
    ierr = KSPSetUp(ksp);CHKERRQ(ierr);

    PreLoadStage("Solve");  
    ierr = KSPSolve(ksp,appctx->algebra.b,appctx->algebra.x);CHKERRQ(ierr);

    {
      PetscTruth flg;
      ierr = PetscOptionsHasName(PETSC_NULL,"-save_global_preconditioner",&flg);CHKERRQ(ierr);
      if (flg) {
	PC          pc;
	Mat         mat,mat2;
	PetscViewer viewer;
	ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
	ierr = PCComputeExplicitOperator(pc,&mat);CHKERRQ(ierr);
	ierr = KSPComputeExplicitOperator(ksp,&mat2);CHKERRQ(ierr);
	ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"pc.m",&viewer);CHKERRQ(ierr);
	ierr = PetscViewerSetFormat(viewer,PETSC_VIEWER_ASCII_MATLAB);
	ierr = MatView(mat,viewer);CHKERRQ(ierr);
	ierr = PetscViewerSetFormat(viewer,PETSC_VIEWER_ASCII_MATLAB);
	ierr = MatView(mat2,viewer);CHKERRQ(ierr);
	ierr = MatDestroy(mat);CHKERRQ(ierr);
	ierr = MatDestroy(mat2);CHKERRQ(ierr);
	ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
      }
    }

    /*      Free the solver data structures */
    ierr = KSPDestroy(ksp);CHKERRQ(ierr);
  PreLoadEnd();

  ierr = PFDestroy(appctx->bc);CHKERRQ(ierr);
  ierr = PFDestroy(appctx->element.rhs);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*----------------------------------------------------------------
       1  -  Generates the "global" parallel vector to contain the 
	     right hand side and solution.
*/
#undef __FUNCT__
#define __FUNCT__ "AppCxtCreateRhs"
PetscErrorCode AppCtxCreateRhs(AppCtx *appctx)
{
  AppGrid     *grid = &appctx->grid;
  AppAlgebra  *algebra = &appctx->algebra;
  MPI_Comm    comm = appctx->comm;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  /*  Create vector to contain load,  local size should be number of  vertices  on this proc.  */
  ierr = VecCreateMPI(comm,grid->vertex_local_n,PETSC_DETERMINE,&algebra->b);CHKERRQ(ierr);

  /* This allows one to set entries into the vector using the LOCAL numbering: via VecSetValuesLocal() */
  ierr = VecSetLocalToGlobalMapping(algebra->b,grid->ltog);CHKERRQ(ierr);

  /* Generate the vector to contain the solution */
  ierr = VecDuplicate(algebra->b,&algebra->x);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*---------------------------------------------------------------
      2  - Generates the "global" parallel matrix
*/
#undef __FUNCT__
#define __FUNCT__ "AppCxtCreateMatrix"
PetscErrorCode AppCtxCreateMatrix(AppCtx* appctx)
{

  AppAlgebra  *algebra = &appctx->algebra;
  AppGrid     *grid    = &appctx->grid;
  MPI_Comm    comm = appctx->comm;
  PetscErrorCode         ierr;
 
  PetscFunctionBegin;
  ierr = MatCreate(comm,&algebra->A);CHKERRQ(ierr);
  ierr = MatSetSizes(algebra->A,grid->vertex_local_n,grid->vertex_local_n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(algebra->A);CHKERRQ(ierr);

  /* Allows one to set values into the matrix using the LOCAL numbering, via MatSetValuesLocal() */
  ierr = MatSetLocalToGlobalMapping(algebra->A,grid->ltog);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*---------------------------------------------------------------------------
     3 - Computes the entries in the right hand side and sets them into the parallel vector
         Uses B and C
*/
#undef __FUNCT__
#define __FUNCT__ "AppCxtSetRhs"
PetscErrorCode AppCtxSetRhs(AppCtx* appctx)
{
  /********* Context informatrion ***********/
  AppGrid    *grid = &appctx->grid;
  AppAlgebra *algebra = &appctx->algebra;
  AppElement *phi = &appctx->element;

  /****** Local Variables ***********/
  PetscInt        i;
  PetscErrorCode ierr;
  PetscInt        *vertex_ptr;
  PetscInt        bn = 4; /* number of basis functions */
  PetscInt        vertexn = 4; /* number of degrees of freedom */

  PetscFunctionBegin;
  /* loop over local cells */
  for(i=0;i<grid->cell_n;i++){

    /* coords_ptr points to the coordinates of the current cell */
    phi->coords = grid->cell_coords + 2*bn*i;  /*number of cell coords */

    /* compute the values of basis functions on this element */
    ierr = SetLocalElement(phi);CHKERRQ(ierr); 

    /* compute the  element load (integral of f with the 4 basis elements)  */
    /* values get put into phi->rhsresult  */
    ierr = ComputeRHSElement(phi);CHKERRQ(ierr);

    /*********  Set Values *************/
    /* vertex_ptr points to place in the vector to set the values */
    vertex_ptr = grid->cell_vertex + vertexn*i; 

    ierr = VecSetValuesLocal(algebra->b,bn,vertex_ptr,phi->rhsresult,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(algebra->b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(algebra->b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}  

/*------------------------------------------------------------------
      4 - Computes the element stiffness matrices and stick into 
   global stiffness matrix. Uses B and D.
*/
#undef __FUNCT__
#define __FUNCT__ "AppCxtSetMatrix"
PetscErrorCode AppCtxSetMatrix(AppCtx* appctx)
{
  /********* Contex information ***********/
  AppAlgebra *algebra = &appctx->algebra;
  AppGrid    *grid    = &appctx->grid;
  AppElement *phi = &appctx->element; 

  /****** Local Variables ***********/
  PetscInt        i;
  PetscErrorCode ierr;
  PetscInt        *vertex_ptr;
  PetscInt        bn = 4; /* number of basis functions */
  PetscInt        vertexn = 4; /* number of degrees of freedom */

  PetscFunctionBegin;

  /* loop over local cells */
  for(i=0;i<grid->cell_n;i++){

    /* coords_ptr points to the coordinates of the current cell */
    phi->coords = grid->cell_coords + 2*bn*i;/*number of cell coords */

    /* compute the values of basis functions on this element */
    ierr = SetLocalElement(phi);CHKERRQ(ierr);
   
    /*    Compute the element stiffness  */  
    /* result is returned in phi->stiffnessresult */
    ierr = ComputeStiffnessElement(phi);CHKERRQ(ierr);

    /*********  Set Values *************/
    /* vertex_ptr points to place in the matrix to set the values */
    vertex_ptr = grid->cell_vertex + vertexn*i;

    ierr = MatSetValuesLocal(algebra->A,vertexn,vertex_ptr,vertexn,vertex_ptr,(PetscReal*)phi->stiffnessresult,ADD_VALUES);CHKERRQ(ierr);
    /* ierr = MatSetValues(algebra->localA,vertexn,vertex_ptr,vertexn,vertex_ptr,(double*)phi->stiffnessresult,ADD_VALUES);CHKERRQ(ierr);*/
  }
  ierr = MatAssemblyBegin(algebra->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(algebra->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*----------------------------------------------------------------
      5   - Apply the Dirichlet boundary conditions (see 6 also).
     This places the Dirichlet function value on the right hand side
     and 6 sticks a row of the identity matrix on the left side 
     thus forcing the solution at the given points to match the 
     Dirichlet function.
*/
#undef __FUNCT__
#define __FUNCT__ "SetBoundaryConditions"
PetscErrorCode SetBoundaryConditions(AppCtx *appctx)
{
 /********* Context informatrion ***********/
  AppAlgebra *algebra = &appctx->algebra;
  AppGrid    *grid = &appctx->grid;

  /****** Local Variables ***********/
  PetscInt          i;
  PetscErrorCode ierr;
  PetscInt          *vertex_ptr; 
  PetscScalar  zero = 0.0;

  PetscFunctionBegin;

  /*  -------------------------------------------------------------
         Apply Dirichlet boundary conditions
      -----------------------------------------------------------*/
  
  /* need to set the points on RHS corresponding to vertices on the boundary to
     the desired value. */

  /* get list of vertices on the bounday */
  ierr = ISGetIndices(grid->vertex_boundary,&vertex_ptr);CHKERRQ(ierr);
  for(i=0;i<grid->boundary_n;i++){
    /* evaluate boundary condition function at point */
    ierr = PFApply(appctx->bc,1,&grid->boundary_coords[2*i],&grid->boundary_values[i]);CHKERRQ(ierr);
  }

  /* set the right hand side values at those points */
  ierr = VecSetValuesLocal(algebra->b,grid->boundary_n,vertex_ptr,grid->boundary_values,INSERT_VALUES);CHKERRQ(ierr);

  /* set initial guess satisfying boundary conditions */
  ierr = VecSet(algebra->x,zero);CHKERRQ(ierr);
  ierr = VecSetValuesLocal(algebra->x,grid->boundary_n,vertex_ptr,grid->boundary_values,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(algebra->x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(algebra->x);CHKERRQ(ierr);

  ierr = ISRestoreIndices(grid->vertex_boundary,&vertex_ptr);CHKERRQ(ierr);
 
  ierr = VecAssemblyBegin(algebra->b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(algebra->b);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*-----------------------------------------------------------------------
     6 - Set the matrix boundary conditions (see also 5). Replace the corresponding 
         rows in the matrix with the identity.
*/
#undef __FUNCT__
#define __FUNCT__ "SetMatrixBoundaryConditions"
PetscErrorCode SetMatrixBoundaryConditions(AppCtx *appctx)
{
  /********* Context informatrion ***********/
  AppAlgebra *algebra = &appctx->algebra;
  AppGrid    *grid = &appctx->grid;

  /****** Local Variables ***********/
  PetscReal     one = 1.0;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  ierr = MatZeroRowsLocal(algebra->A,grid->vertex_boundary,&one);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}
























