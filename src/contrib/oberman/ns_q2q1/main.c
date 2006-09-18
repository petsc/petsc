static char help[] ="Solves the 2d incompressible Navier-Stokes  equations.\n    This version uses biquadratic velocity elements and bilinear pressure elements\n";

#include "appctx.h"

#undef __FUNC__
#define __FUNC__ "main"
int main( int argc, char **argv )
{
  int ierr;
  AppCtx         *appctx;
  /* ---------------------------------------------------------------------
     Initialize PETSc
     --------------------- ---------------------------------------------------*/
  PetscInitialize(&argc,&argv,(char *)0,help);

  /*  Load the grid database and set up the equations from options */
  ierr = AppCtxCreate(PETSC_COMM_WORLD,&appctx); CHKERRA(ierr);
  /*   Setup the linear system and solve it*/
  ierr = AppCtxSolve(appctx);CHKERRQ(ierr);
  /*      Destroy all datastructures  */
  ierr = AppCtxDestroy(appctx); CHKERRA(ierr);
  PetscFinalize();
  PetscFunctionReturn(0);
}

/*  Sets up the non-linear system associated with the PDE and solves it */
#undef __FUNC__
#define __FUNC__ "AppCxtSolve"
int AppCtxSolve(AppCtx* appctx)
{

  /* need to create these guys somewhere */
  AppGrid                *grid = &appctx->grid;
  AppAlgebra             *algebra = &appctx->algebra;
  AppElement          *element = &appctx->element;
  MPI_Comm               comm = appctx->comm;
  SLES                   sles;
  SNES                   snes;
  int ierr, its;
  int zero = 0;
  PetscFunctionBegin;

  /*        Create vector to contain load and various work vectors  */
  ierr = AppCtxCreateVector(appctx); CHKERRQ(ierr);
  /*      Create the sparse matrix, (later with correct nonzero pattern)  */
  ierr = AppCtxCreateMatrix(appctx); CHKERRQ(ierr);
  /*     Set the quadrature values for the reference square element  */
  ierr =SetReferenceElement(appctx);CHKERRQ(ierr);
  /*      Set the right hand side values into the vectors   */
  ierr = AppCtxSetRhs(appctx); CHKERRQ(ierr);
  /*      Set the matrix entries   (just the linear part, do this only once )*/
  ierr = AppCtxSetMatrix(appctx); CHKERRQ(ierr);

  /*       set the boundary conditions in the linear part */
  /*--- this sets the final nonzero structure of the matrix ----  */
  ierr = SetJacobianBoundaryConditions(appctx, &algebra->A);CHKERRQ(ierr); 
  ierr = MatSetOption(algebra->A, MAT_NO_NEW_NONZERO_LOCATIONS);CHKERRQ(ierr); 
  ierr= MatDuplicate(algebra->A, MAT_DO_NOT_COPY_VALUES, &algebra->J);CHKERRQ(ierr);  
  ierr = MatSetLocalToGlobalMapping(algebra->J, grid->dfltog);  CHKERRQ(ierr);
  ierr = MatSetOption(algebra->J, MAT_NO_NEW_NONZERO_LOCATIONS);CHKERRQ(ierr); 

  /* view the matrix */
  if( appctx->view.show_matrix ) { /*  MatView(algebra->A, VIEWER_STDOUT_SELF);  */}
  /*     Create the nonlinear solver context  */
  ierr = SNESCreate(PETSC_COMM_WORLD,SNES_NONLINEAR_EQUATIONS,&snes); CHKERRQ(ierr);

  /********* solve the stationary problem with boundary conditions.  ************/
  /*      Set function evaluation rountine and vector */
  ierr = SNESSetFunction(snes,algebra->f ,FormStationaryFunction,(void *)appctx); CHKERRQ(ierr);
  /*      Set Jacobian   */ 
  ierr = SNESSetJacobian(snes, algebra->J, algebra->J, FormStationaryJacobian,(void *)appctx);CHKERRQ(ierr);
  /* set monintor functions */
  if(appctx->view.monitor) {ierr = SNESMonitorSet(snes, MonitorFunction, (void *)appctx,0);CHKERRQ(ierr);}

/* Need this call, otherwise the defaults don't get set, and solve won't work */
 /*      Set Solver Options, could put internal options here      */
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  /* initial guess */
  ierr = FormInitialGuess(appctx);CHKERRQ(ierr); 

  /* Setup matlab viewer */
   if (appctx->view.matlabgraphics) {AppCtxViewMatlab(appctx);  } 

   
  /*       Solve the non-linear system  */
  ierr = SNESSolve(snes, PETSC_NULL, algebra->g);CHKERRQ(ierr);
  ierr = SNESGetIteratioNumber(snes, &its);CHKERRQ(ierr);

  /* send solution to matlab */
  if (appctx->view.matlabgraphics){
    ierr = VecView(appctx->algebra.g,VIEWER_SOCKET_WORLD); CHKERRQ(ierr);
    /* send the done signal */
    ierr = PetscIntView(1, &zero, VIEWER_SOCKET_WORLD);CHKERRQ(ierr);
  }
  /* show solution vector */
  if (appctx->view.show_vector ){ 
    printf("the current soln vector\n");
    VecView(algebra->g, VIEWER_STDOUT_SELF);}
  /* output number of its */
  printf("the number of its, %d\n", its);

  ierr = SNESDestroy(snes); CHKERRQ(ierr);  
  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "FormInitialGuess"
int FormInitialGuess(AppCtx* appctx)
{
    AppAlgebra *algebra = &appctx->algebra;  
    AppGrid *grid = &appctx->grid;
    AppEquations *equations = &appctx->equations;
   AppElement *phi = &appctx->element;

 int ierr, i,j;
  int *df_ptr;
  double *coords_ptr;
  double result[22];

  /* loop over cells */
  for(i=0;i<grid->cell_n;i++){
    /* loop over degrees of freedom and cell coords */
    df_ptr = grid->cell_df + 22*i;
    coords_ptr = grid->cell_vcoords + 2*9*i;  /*number of cell coords */
    for(j=0;j<9;j++){
      equations->xval = coords_ptr[2*j];
      equations->yval = coords_ptr[2*j+1];
      result[2*j] = bc1( equations);
      result[2*j+1] = bc2(equations);
    }
    for(j=0;j<4;j++){
      equations->xval = coords_ptr[4*j];
      equations->yval = coords_ptr[4*j+1];
      result[18+j] = bc3(equations);
    }
    ierr = VecSetValuesLocal(algebra->g, 22, df_ptr, result, INSERT_VALUES);CHKERRQ(ierr);
  }
  /********* Assemble Data **************/
  ierr = VecAssemblyBegin(algebra->g);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(algebra->g);CHKERRQ(ierr);
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
  /* Internal Variables */
  int ierr;
  double zero = 0.0, mone = -1.0;

/****** Perform computation ***********/
  /* A is the (already computed) linear part*/
   /* b is the (already computed) rhs */ 

/****** Perform computation ***********/
  /* need to zero f */
  ierr = VecSet(f,zero); CHKERRQ(ierr); 
  /* add rhs to get constant part */
  ierr = VecAXPY(f,mone,algebra->b); CHKERRQ(ierr); /* this says f = f - 1*b */

  if (appctx->view.show_vector ){  printf("f-rhs\n"); VecView(f, VIEWER_STDOUT_SELF);}

  /*apply matrix to the input vector x, to get linear part */
  ierr = MatMultAdd(algebra->A, x, f, f); CHKERRQ(ierr);  /* f = A*x - b */

  if (appctx->view.show_vector ){  
    printf("apply matrix to f\n"); 
    VecView(f, VIEWER_STDOUT_SELF);
      };

 /* create nonlinear part */
if( appctx->equations.stokes_flag != 1 ){
  ierr = SetNonlinearFunction(x, appctx, f);CHKERRQ(ierr);
  if (appctx->view.show_vector ){ printf("add nonlinear part to  f\n"); VecView(f, VIEWER_STDOUT_SELF);}
}

  /* apply boundary conditions */
  ierr = SetBoundaryConditions(x, appctx, f);CHKERRQ(ierr);

  if (appctx->view.show_vector ){ printf("set bc to  f\n"); VecView(f, VIEWER_STDOUT_SELF);}

  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "FormStationaryJacobian"
int FormStationaryJacobian(SNES snes, Vec g, Mat *jac, Mat *B, MatStructure *flag, void *dappctx)
{
  static int cnt = 0;
  Viewer viewer;
  AppCtx *appctx = (AppCtx *)dappctx;
  AppAlgebra *algebra = &appctx->algebra;
  int ierr;
  /* copy the linear part into jac.*/
  ierr = MatCopy(algebra->A, *jac, SAME_NONZERO_PATTERN); CHKERRQ(ierr);


/* if(1){  if(!cnt){ */
/*   PLogStagePush(1); */
/*   ierr= MatCopy(algebra->A, *jac);CHKERRQ(ierr);    */
/*   ierr = MatSetOption(*jac, MAT_NO_NEW_NONZERO_LOCATIONS);CHKERRQ(ierr);  */
/*   PLogStagePop(); */

/*     ierr= MatDuplicate(algebra->A, jac);CHKERRQ(ierr);   */
/*      ierr = MatSetLocalToGlobalMapping(*jac, appctx->grid.dfltog);  CHKERRQ(ierr);  */
/*    algebra->J = *jac;  */
/*   ierr = MatAssemblyBegin(algebra->J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr); */
/*   ierr = MatAssemblyEnd(algebra->J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr); */


/*  cnt++; */
/*   } */
/*   else{ */
/*    ierr= MatCopy_SeqAIJ(algebra->A, *jac);CHKERRQ(ierr); */
/*   } */
/* } */

if( appctx->view.show_matrix ) {  
  printf("in jac, stiffness and pressure\n");
  ierr = MatView(*jac, VIEWER_DRAW_WORLD );CHKERRQ(ierr);
}
  /* the nonlinear part */
  if( appctx->equations.stokes_flag != 1 ){
    ierr = SetJacobian(g, appctx, jac);CHKERRQ(ierr);
  }

  ierr = SetJacobianBoundaryConditions(appctx, jac);CHKERRQ(ierr);
if( appctx->view.show_matrix ) {  

  printf("set boundary conditions\n");
  ierr = MatView(*jac, VIEWER_DRAW_WORLD );CHKERRQ(ierr);
}
  /* Set flag */
  *flag = SAME_NONZERO_PATTERN;


  if( 0 ) {  
 printf("about to send to the file\n");
  ierr = MatView(*jac, VIEWER_DRAW_WORLD );CHKERRQ(ierr);
    ierr = ViewerASCIIOpen(PETSC_COMM_WORLD,"mat.output",&viewer);CHKERRQ(ierr);
    ierr = ViewerSetFormat(viewer ,   VIEWER_FORMAT_ASCII_MATLAB, 0);CHKERRQ(ierr);
    ierr = MatView(*jac, viewer);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "MonitorFunction"
int MonitorFunction(SNES snes, int its, double norm, void *dappctx)
{ 
  AppCtx *appctx = (AppCtx *)dappctx;
  int ierr, one = 1; 
  Vec x;  
  if (appctx->view.matlabgraphics ){
    ierr = SNESGetSolution(snes,&x); CHKERRQ(ierr);
    ierr = VecView(x,VIEWER_SOCKET_WORLD); CHKERRQ(ierr);
    ierr = PetscIntView(1, &one, VIEWER_SOCKET_WORLD);CHKERRQ(ierr);}
    PetscFunctionReturn(0);
}

