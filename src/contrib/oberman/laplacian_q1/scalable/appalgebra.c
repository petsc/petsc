
#include "appctx.h"
#include "math.h"

/*
         Sample right hand side and boundary conditions
*/
#undef __FUNCT__
#define __FUNCT__ "pde_rhs"
int pde_rhs(void *dummy,int n,double *xx,double *f)
{
  double pi = PETSC_PI, x = xx[0], y = xx[1];
  PetscFunctionBegin;
  *f = 8*pi*pi*sin(2*pi*x)*sin(2*pi*y)-20*pi*cos(2*pi*x)*sin(2*pi*y);
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "pde_bc"
int pde_bc(void *dummy,int n,double *xx,double *f)
{
  double pi = 3.1415927, x = xx[0], y = xx[1];
  PetscFunctionBegin;
  *f = sin(2*pi*x)*sin(2*pi*y);
  PetscFunctionReturn(0);
}


/*
         Sets up the linear system associated with the PDE and solves it
*/
#undef __FUNCT__
#define __FUNCT__ "AppCxtSolve"
int AppCtxSolve(AppCtx* appctx)
{
  AppAlgebra   *algebra = &appctx->algebra;
  MPI_Comm     comm = appctx->comm;
  KSP         ksp;
  int          ierr;

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
  /*     2)  Create the sparse matrix,with correct nonzero pattern  */
  ierr = AppCtxCreateRhsAndMatrix(appctx);CHKERRQ(ierr);

  /*     3)  Set the right hand side values into the load vector   */
  ierr = AppCtxSetRhs(appctx);CHKERRQ(ierr);

  /*     4)  Set the matrix entries   */
  ierr = AppCtxSetMatrix(appctx);CHKERRQ(ierr);

  /*     6) Set the matrix boundary conditions */
  ierr = SetMatrixBoundaryConditions(appctx);CHKERRQ(ierr);

  PreLoadBegin(PETSC_TRUE,"Solver setup");  

    /*     5) Set the rhs boundary conditions - this also creates initial guess that satisfies boundary conditions */
    ierr = SetBoundaryConditions(appctx);CHKERRQ(ierr);

    ierr = KSPCreate(comm,&ksp);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,algebra->A,algebra->A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    {
      PetscTruth flg;
      ierr = PetscOptionsHasName(PETSC_NULL,"-use_zero_initial_guess",&flg);CHKERRQ(ierr);
      if (!flg) {
        ierr = KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);CHKERRQ(ierr);
      }
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
       1,2  -  Generates the "global" parallel vector to contain the 
	       right hand side and solution.
               Generates the "global" parallel matrix.
*/
#undef __FUNCT__
#define __FUNCT__ "AppCxtCreateRhsAndMatrix"
int AppCtxCreateRhsAndMatrix (AppCtx *appctx)
{
  AppAlgebra   *algebra = &appctx->algebra;
  AppPartition *part = &appctx->part;
  MPI_Comm     comm = appctx->comm;
  int          ierr;
  PetscTruth   flg;

  PetscFunctionBegin;
  /*  Create vector to contain load,  local size should be number of  vertices  on this proc.  */
  ierr = PetscOptionsHasName(PETSC_NULL,"-vec_type",&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = VecCreateMPI(comm,part->m,PETSC_DETERMINE,&algebra->b);CHKERRQ(ierr);
  } else {
    ierr = VecCreate(comm,&algebra->b);CHKERRQ(ierr);
    ierr = VecSetSizes(algebra->b,part->m,PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = VecSetFromOptions(algebra->b);CHKERRQ(ierr);
  }
  /* Create the structure for the stiffness matrix */
  ierr = MatCreate(comm,part->m,part->m,PETSC_DETERMINE,PETSC_DETERMINE,&algebra->A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(algebra->A);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(algebra->A,9,0,3,0);CHKERRQ(ierr);

  /* This allows one to set entries into the vector and matrix using the LOCAL numbering */
  {
    ISLocalToGlobalMapping mapping;
    Mat                    local;

    ierr = AppPartitionCreateLocalToGlobalMapping(part,&mapping);CHKERRQ(ierr);
    ierr = VecSetLocalToGlobalMapping(algebra->b,mapping);CHKERRQ(ierr);
    ierr = MatSetLocalToGlobalMapping(algebra->A,mapping);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingDestroy(mapping);CHKERRQ(ierr);
    ierr = MatISGetLocalMat(algebra->A,&local);CHKERRQ(ierr);
    if (local) {
      ierr = MatSeqAIJSetPreallocation(local,9,0);CHKERRQ(ierr);
    }
  }

  /* Generate the vector to contain the solution */
  ierr = VecDuplicate(algebra->b,&algebra->x);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*---------------------------------------------------------------------------
     3 - Computes the entries in the right hand side and sets them into the parallel vector
         Uses B and C
*/
#undef __FUNCT__
#define __FUNCT__ "AppCxtSetRhs"
int AppCtxSetRhs(AppCtx* appctx)
{
  /********* Context informatrion ***********/
  AppPartition *part = &appctx->part;
  AppAlgebra   *algebra = &appctx->algebra;
  AppElement   *phi = &appctx->element;

  /****** Local Variables ***********/
  int        ierr,i;
  int        *vertex_ptr;

  PetscFunctionBegin;
  /* loop over local cells */
  for(i=0;i<((part->nelx)*(part->nely));i++){

    /* coords_ptr points to the coordinates of the current cell */
    ierr = AppPartitionGetCoords(part,i,&phi->coords);CHKERRQ(ierr);

    /* compute the values of basis functions on this element */
    ierr = SetLocalElement(phi);CHKERRQ(ierr); 

    /* compute the  element load (integral of f with the 4 basis elements)  */
    /* values get put into phi->rhsresult  */
    ierr = ComputeRHSElement(phi);CHKERRQ(ierr);

    /*********  Set Values *************/
    /* vertex_ptr points to place in the vector to set the values */
    ierr = AppPartitionGetNodes(part,i,&vertex_ptr);CHKERRQ(ierr);

    ierr = VecSetValuesLocal(algebra->b,4,vertex_ptr,phi->rhsresult,ADD_VALUES);CHKERRQ(ierr);
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
int AppCtxSetMatrix(AppCtx* appctx)
{
  /********* Contex information ***********/
  AppAlgebra   *algebra = &appctx->algebra;
  AppPartition *part    = &appctx->part;
  AppElement   *phi     = &appctx->element; 

  /****** Local Variables ***********/
  int        i,ierr;
  int        *vertex_ptr;

  PetscFunctionBegin;

  /* loop over local cells */
  for(i=0;i<((part->nelx)*(part->nely));i++){

    /* coords_ptr points to the coordinates of the current cell */
    ierr = AppPartitionGetCoords(part,i,&phi->coords);CHKERRQ(ierr);

    /* compute the values of basis functions on this element */
    ierr = SetLocalElement(phi);CHKERRQ(ierr);
   
    /*    Compute the element stiffness  */  
    /* result is returned in phi->stiffnessresult */
    ierr = ComputeStiffnessElement(phi);CHKERRQ(ierr);

    /*********  Set Values *************/
    /* vertex_ptr points to place in the matrix to set the values */
    ierr = AppPartitionGetNodes(part,i,&vertex_ptr);CHKERRQ(ierr);

    ierr = MatSetValuesLocal(algebra->A,4,vertex_ptr,4,vertex_ptr,(double*)phi->stiffnessresult,ADD_VALUES);CHKERRQ(ierr);
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
int SetBoundaryConditions(AppCtx *appctx)
{
 /********* Context informatrion ***********/
  AppAlgebra   *algebra = &appctx->algebra;
  AppPartition *part    = &appctx->part;

  /****** Local Variables ***********/
  int          ierr,i;
  int          *vertex_ptr; 
  double       *coord_ptr;
  double       *values;
  int          n;
  PetscScalar  zero = 0.0;

  PetscFunctionBegin;

  /*  -------------------------------------------------------------
         Apply Dirichlet boundary conditions
      -----------------------------------------------------------*/
  
  /* need to set the points on RHS corresponding to vertices on the boundary to
     the desired value. */

  /* get list of vertices on the boundary and their coordinates */
  ierr = AppPartitionGetBoundaryNodesAndCoords(part,&n,&vertex_ptr,&coord_ptr);CHKERRQ(ierr);
  ierr = PetscMalloc((n+1)*sizeof(double),&values);CHKERRQ(ierr);
  for(i=0;i<n;i++){
    /* evaluate boundary condition function at point */
    ierr = PFApply(appctx->bc,1,coord_ptr+(2*i),values+i);CHKERRQ(ierr);
  }

  /* set the right hand side values at those points */
  ierr = VecSetValuesLocal(algebra->b,n,vertex_ptr,values,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(algebra->b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(algebra->b);CHKERRQ(ierr);

  /* set initial guess satisfying boundary conditions */
  ierr = VecSet(algebra->x,zero);CHKERRQ(ierr);
  ierr = VecSetValuesLocal(algebra->x,n,vertex_ptr,values,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(algebra->x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(algebra->x);CHKERRQ(ierr);

  ierr = PetscFree(vertex_ptr);CHKERRQ(ierr);
  ierr = PetscFree(coord_ptr);CHKERRQ(ierr);
  ierr = PetscFree(values);CHKERRQ(ierr);
 
  PetscFunctionReturn(0);
}

/*-----------------------------------------------------------------------
     6 - Set the matrix boundary conditions (see also 5). Replace the corresponding 
         rows in the matrix with the identity.
*/
#undef __FUNCT__
#define __FUNCT__ "SetMatrixBoundaryConditions"
int SetMatrixBoundaryConditions(AppCtx *appctx)
{
  /********* Context informatrion ***********/
  AppAlgebra   *algebra = &appctx->algebra;
  AppPartition *part    = &appctx->part;

  /****** Local Variables ***********/
  int        *vertex_ptr; 
  double     *coord_ptr;
  int        n;
  double     one = 1.0;
  int        ierr;
  IS         is;

  PetscFunctionBegin;

  ierr = AppPartitionGetBoundaryNodesAndCoords(part,&n,&vertex_ptr,&coord_ptr);CHKERRQ(ierr);

  ierr = PetscFree(coord_ptr);CHKERRQ(ierr);

  ierr = ISCreateGeneral(appctx->comm,n,vertex_ptr,&is);CHKERRQ(ierr);

  ierr = PetscFree(vertex_ptr);CHKERRQ(ierr);

  ierr = MatZeroRowsLocal(algebra->A,is,&one);CHKERRQ(ierr); 

  ierr = ISDestroy(is);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}
























