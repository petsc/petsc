
static char help[] ="Solves the 2d time-dependant burgers equation.\ndu/dt + u*du/dx + v*du/dy - c(lap(u)) = f.\n  dv/dt + u*dv/dv + v*dv/dy - c(lap(v)) =g.  This has exact solution, see fletcher.\n  This version has new indexing of Degrees of Freedom";

#include "appctx.h"

#undef __FUNC__
#define __FUNC__ "main"
int main( int argc, char **argv )
{
  int ierr;
  AppCtx         *appctx;
  AppAlgebra     *algebra;
  AppGrid        *grid;
 
  /* ---------------------------------------------------------------------
     Initialize PETSc
     --------------------- ---------------------------------------------------*/
  PetscInitialize(&argc,&argv,(char *)0,help);

  /*  Load the grid database*/
  ierr = AppCtxCreate(PETSC_COMM_WORLD,&appctx); CHKERRA(ierr);
  algebra = &appctx->algebra; grid = &appctx->grid;

  /*      Initialize graphics */
  ierr = AppCtxGraphics(appctx); CHKERRA(ierr);

  /*   Setup the linear system and solve it*/
  ierr = AppCtxSolve(appctx);CHKERRQ(ierr);

  /* Send to  matlab viewer */
  if (appctx->view.matlabgraphics) {AppCtxViewMatlab(appctx);  }

  /*      Destroy all datastructures  */
  ierr = AppCtxDestroy(appctx); CHKERRA(ierr);

  PetscFinalize();
  PetscFunctionReturn(0);
}

/*
         Sets up the non-linear system associated with the PDE and solves it
*/
#undef __FUNC__
#define __FUNC__ "AppCxtSolve"
int AppCtxSolve(AppCtx* appctx)
{
  AppGrid                *grid = &appctx->grid;
  AppAlgebra             *algebra = &appctx->algebra;
  MPI_Comm               comm = appctx->comm;
  SLES                   sles;
  SNES                   snes;
  int ierr, its;
 
  PetscFunctionBegin;

  /*        Create vector to contain load and various work vectors  */
  ierr = AppCtxCreateVector(appctx); CHKERRQ(ierr);
  /*      Create the sparse matrix, with correct nonzero pattern  */
  ierr = AppCtxCreateMatrix(appctx); CHKERRQ(ierr);
  /*     Set the quadrature values for the reference square element  */
  ierr = AppCtxSetReferenceElement(appctx);CHKERRQ(ierr);
  /*      Set the right hand side values into the vectors   */
  ierr = AppCtxSetRhs(appctx); CHKERRQ(ierr);

  /* The coeff of diffusivity.  LATER call a function set equations */
  appctx->equations.eta =-0.04;  
  /*      Set the matrix entries   */
  ierr = AppCtxSetMatrix(appctx); CHKERRQ(ierr);

/* MatView(algebra->A, VIEWER_STDOUT_SELF); */

  /*     Create the nonlinear solver context  */
  ierr = SNESCreate(PETSC_COMM_WORLD,SNES_NONLINEAR_EQUATIONS,&snes); CHKERRQ(ierr);

  /********* solve the stationary problem with boundary conditions.  ************/
  /*      Set function evaluation rountine and vector */
  ierr = SNESSetFunction(snes,algebra->f ,FormStationaryFunction,(void *)appctx); CHKERRQ(ierr);
  /*      Set Jacobian   */ 
  ierr = SNESSetJacobian(snes, algebra->J, algebra->J, FormStationaryJacobian,(void *)appctx);CHKERRQ(ierr);
  /*      Set Solver Options, could put internal options here      */
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  /* initial guess */
  ierr = FormInitialGuess(appctx);CHKERRQ(ierr); 
  /*       Solve the non-linear system  */
  ierr = SNESSolve(snes, algebra->g, &its);CHKERRQ(ierr);

  if(0){VecView(algebra->g, VIEWER_STDOUT_SELF);}

  printf("the number of its, %d\n", its);

  ierr = SNESDestroy(snes); CHKERRQ(ierr);  
  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "FormInitialGuess"
int FormInitialGuess(AppCtx* appctx)
{
    AppAlgebra *algebra = &appctx->algebra;
    int ierr;
    double onep1 = 1.234;
    ierr = VecSet(algebra->g,onep1); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

/* FormStationaryFunction - Evaluates the nonlinear function, F(x), which is the discretised equations, 
     Input Parameters:
    - the vector x, corresponding to u values at each vertex
    - snes, the SNES context
    - appctx
   Output Parameter:
    - f, the value of the function
*/
#undef __FUNC__
#define __FUNC__ "FormStationaryFunction"
int FormStationaryFunction(SNES snes, Vec x, Vec f, void *dappctx)
{
/********* Collect context informatrion ***********/
 AppCtx *appctx = (AppCtx *)dappctx;
  AppElement phi = appctx->element;
  AppAlgebra *algebra = &appctx->algebra;

/* Later may want to have these computed from here, with a flag passed 
to see if they need to be recomputed */
  /* A is the (already computed) linear part*/
  Mat A = algebra->A;

  /* b is the (already computed) rhs */ 
  Vec  b = algebra->b;
  /* Internal Variables */
  int ierr;
  double zero = 0.0, mone = -1.0;

/****** Perform computation ***********/
  /* need to zero f */
  ierr = VecSet(f,zero); CHKERRQ(ierr); 
  /* add rhs to get constant part */
  ierr = VecAXPY(f,mone,b); CHKERRQ(ierr); /* this says f = f - 1*b */
  /*apply matrix to the input vector x, to get linear part */
  /* Assuming matrix doesn't need to be recomputed */
  ierr = MatMultAdd(A, x, f, f); CHKERRQ(ierr);  /* f = A*x + f */
 /* create nonlinear part */
  ierr = SetNonlinearFunction(x, appctx, f);CHKERRQ(ierr);

/*  printf("output of nonlinear fun (before bc imposed)\n");    */
/*   ierr = VecView(f, VIEWER_STDOUT_WORLD);CHKERRQ(ierr);     */

  ierr = SetBoundaryConditions(x, appctx, f);CHKERRQ(ierr);
 /* printf("output of nonlinear fun \n");    */
/*    ierr = VecView(f, VIEWER_STDOUT_WORLD);CHKERRQ(ierr);     */
  

  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "FormStationaryJacobian"
int FormStationaryJacobian(SNES snes, Vec g, Mat *jac, Mat *B, MatStructure *flag, void *dappctx)
{
  AppCtx *appctx = (AppCtx *)dappctx;
  AppAlgebra *algebra = &appctx->algebra;
  int ierr;

  /* copy the linear part into jac.*/
  ierr= MatCopy(algebra->A, *jac);CHKERRQ(ierr);
  /* the nonlinear part */
  ierr = SetJacobian(g, appctx, jac);CHKERRQ(ierr);
  /* Set flag */
  *flag = DIFFERENT_NONZERO_PATTERN;  /*  is this right? */
  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "SetNonlinearFunction"
/* input vector is g, output is f.  Loop over elements, getting coords of each vertex and 
computing load vertex by vertex.  Set the values into f.  */
int SetNonlinearFunction(Vec g, AppCtx *appctx, Vec f)
{
/********* Collect context informatrion ***********/
  AppElement *phi = &appctx->element;
  AppAlgebra *algebra = &appctx->algebra;
  AppGrid *grid = &appctx->grid;
 
 /* The array of vertices/Degrees of Freedom  in the local numbering for each cell */
  int *cell_vertex = grid->cell_vertex;
  int  *cell_df = grid->cell_df;

/****** Internal Variables ***********/
  double result[8];
  double *coords_ptr;
  double cell_values[8],  *uvvals;
  int ierr, i, j;
  int *df_ptr;

  /* Scatter the input values from the global vector g, to those on this processor */
  ierr = VecScatterBegin( g, algebra->f_local, INSERT_VALUES, SCATTER_FORWARD, algebra->dfgtol); CHKERRQ(ierr);
  ierr = VecScatterEnd( g,  algebra->f_local, INSERT_VALUES, SCATTER_FORWARD, algebra->dfgtol); CHKERRQ(ierr);
  ierr = VecGetArray( algebra->f_local, &uvvals); CHKERRQ(ierr);

  /* set a flag in computation of local elements */
  phi->dorhs = 0;
  /* loop over cells */
  for(i=0;i<grid->cell_n;i++){
    /* loop over degrees of freedom and cell coords */
    df_ptr = grid->cell_df + 8*i;
    coords_ptr = grid->cell_coords + 8*i;
      /* Need to point to the uvvals associated to the vertices */
    for ( j=0; j<8; j++){
      cell_values[j] = uvvals[df_ptr[j]];   
    }
    /* compute the values of basis functions on this element */
     ierr = SetLocalElement(phi, coords_ptr);CHKERRQ(ierr);
    /* do the integrals */
    ierr = ComputeNonlinear(phi, cell_values, result);CHKERRQ(ierr);
    /* put result in */
    ierr = VecSetValuesLocal(f, 8, df_ptr, result, ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(f);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNC__
#define __FUNC__ "SetBoundaryConditions"
int SetBoundaryConditions(Vec g, AppCtx *appctx, Vec f)
{
 /********* Collect context informatrion ***********/
  AppElement *phi = &appctx->element;
  AppAlgebra *algebra = &appctx->algebra;
  AppGrid *grid = &appctx->grid;

  int ierr, i;
  double  *bvs, xval, yval, *uvvals; 

  /* Scatter the input values from the global vector g, to those on this processor */
  ierr = VecScatterBegin( g, algebra->f_boundary, INSERT_VALUES, SCATTER_FORWARD, algebra->dfbgtol); CHKERRQ(ierr);
  ierr = VecScatterEnd( g,  algebra->f_boundary, INSERT_VALUES, SCATTER_FORWARD, algebra->dfbgtol); CHKERRQ(ierr);
  ierr = VecGetArray( algebra->f_boundary, &uvvals); CHKERRQ(ierr);

  /* create space for the array of boundary values */
  bvs = grid->bvs;

  for( i = 0; i < grid->vertex_boundary_count; i++ ){
    /* get the vertex_value corresponding to element of vertices
       then evaluate bc(vertex value) and put this in bvs(i) */
    xval = grid->bvc[2*i];
    yval = grid->bvc[2*i+1];
    bvs[2*i] = uvvals[2*i] - bc1(xval, yval);
    bvs[2*i+1] = uvvals[2*i+1] - bc2(xval, yval);    
  }

 /*********  Set Values *************/
  ierr = VecSetValues(f, 2*grid->vertex_boundary_count, grid->boundary_df, bvs, INSERT_VALUES);CHKERRQ(ierr);

  /********* Assemble Data **************/
 ierr = VecAssemblyBegin(f);CHKERRQ(ierr);
 ierr = VecAssemblyEnd(f);CHKERRQ(ierr);
 
 /* Destroy stuff */
  ierr = VecRestoreArray(algebra->f_boundary, &uvvals);  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* input is the input vector , output is the jacobian jac */
#undef __FUNC__
#define __FUNC__ "SetJacobian"
int SetJacobian(Vec g, AppCtx *appctx, Mat* jac)
{
/********* Collect context informatrion ***********/
  AppAlgebra *algebra = &appctx->algebra;
  AppGrid    *grid    = &appctx->grid;
  AppElement *phi = &appctx->element;
  
/****** Internal Variables ***********/
  int  i,j, ierr;
  int  *df_ptr; 
  double *coords_ptr;
  double   *uvvals, cell_values[8];
  double values[8*8];  /* the integral of the combination of phi's */
 double one = 1.0;

  PetscFunctionBegin;
  /* Matrix is set to the linear part already, so just ADD_VALUES the nonlinear part  */ 
  ierr = VecScatterBegin(g, algebra->f_local, INSERT_VALUES, SCATTER_FORWARD, algebra->dfgtol); CHKERRQ(ierr);
  ierr = VecScatterEnd(g, algebra->f_local, INSERT_VALUES, SCATTER_FORWARD, algebra->dfgtol); CHKERRQ(ierr);
  ierr = VecGetArray(algebra->f_local, &uvvals);
 
  /* loop over cells */
  for(i=0;i<grid->cell_n;i++){
   /* loop over degrees of freedom and cell coords */
    df_ptr = grid->cell_df + 8*i;
    coords_ptr = grid->cell_coords + 8*i;
    /* Need to point to the uvvals associated to the vertices */
    for ( j=0; j<8; j++){
      cell_values[j] = uvvals[df_ptr[j]];
    }
    /* compute the values of basis functions on this element */
    ierr = SetLocalElement(phi, coords_ptr);CHKERRQ(ierr);
    /*    Compute the partial derivatives of the nonlinear map    */  
    ierr = ComputeJacobian( phi, cell_values,  values );CHKERRQ(ierr);
    /*  Set the values in the matrix */
    ierr  = MatSetValuesLocal(*jac,8,df_ptr,8,df_ptr,values,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /********** The process repeats for setting boundary conditions ************/

  ierr = MatZeroRows(*jac, grid->isboundary_df,&one);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);

}

#undef __FUNC__
#define __FUNC__ "AppCxtSetRhs"
int AppCtxSetRhs(AppCtx* appctx)
{
  /********* Collect context informatrion ***********/
  AppGrid    *grid = &appctx->grid;
  AppAlgebra *algebra = &appctx->algebra;
  AppElement *phi = &appctx->element;

  int ierr, i;
  int *df_ptr;
  double *coords_ptr;
  double  values[8]; 

  /* set flag for element computation */
  phi->dorhs = 1;

 /* loop over cells */
  for(i=0;i<grid->cell_n;i++){
    /* loop over degrees of freedom and cell coords */
    df_ptr = grid->cell_df + 8*i;
    coords_ptr = grid->cell_coords + 8*i;
    /* compute the values of basis functions on this element */
    SetLocalElement(phi, coords_ptr); 
    /* compute the  element load (integral of f with the 4 basis elements)  */
    ComputeRHS( f, g, phi, values );/* f,g are rhs functions */
    /*********  Set Values *************/
    ierr = VecSetValuesLocal(algebra->b, 8, df_ptr, values, ADD_VALUES);CHKERRQ(ierr);
  }
  
  /********* Assemble Data **************/
  ierr = VecAssemblyBegin(algebra->b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(algebra->b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}  


#undef __FUNC__
#define __FUNC__ "AppCxtSetMatrix"
int AppCtxSetMatrix(AppCtx* appctx)
{
/********* Collect contex informatrion ***********/
  AppAlgebra *algebra = &appctx->algebra;
  AppGrid    *grid    = &appctx->grid;
  AppElement *phi = &appctx->element; 
  AppEquations *equations = &appctx->equations;

/****** Internal Variables ***********/
  int i, ierr;
  int *df_ptr;
  double *coords_ptr;
  double values[8*8];
 
  PetscFunctionBegin;
  /************ Set Up **************/ 
  /* set flag for phi computation */
  phi->dorhs = 0;
  /* loop over cells */
  for(i=0;i<grid->cell_n;i++){
    /* loop over degrees of freedom and cell coords */
    df_ptr = grid->cell_df + 8*i;
    coords_ptr = grid->cell_coords + 8*i;
    /* compute the values of basis functions on this element */
    ierr = SetLocalElement(phi, coords_ptr); CHKERRQ(ierr);
    /*    Compute the element stiffness    */  
    ierr = ComputeMatrix( phi, values ); CHKERRQ(ierr);
    /*********  Set Values *************/
    ierr = MatSetValuesLocal(algebra->A, 8, df_ptr, 8, df_ptr, values, ADD_VALUES);CHKERRQ(ierr);
  }
  /********* Assemble Data **************/
  ierr = MatAssemblyBegin(algebra->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(algebra->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  /********** Multiply by the viscosity coeff ***************/
  ierr = MatScale(&equations->eta, algebra->A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}






