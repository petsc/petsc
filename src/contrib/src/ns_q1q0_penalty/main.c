static char help[] ="Solves the 2d incompressible Navier-Stokes  equations.\n           du/dt + u*du/dx + v*du/dy + dp/dx - c(lap(u)) - f = 0.\n             dv/dt + u*dv/dv + v*dv/dy + dp/dy - c(lap(v)) - g = 0.\n                     du/dx + dv/dy = 0.\n  This version has new indexing of Degrees of Freedom";

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
  AppEquations *equations = &appctx->equations;
  MPI_Comm               comm = appctx->comm;
  SLES                   sles;
  SNES                   snes;
  int ierr, its;
 int zero = 0;
  PetscFunctionBegin;

  /* set the number of degrees of freedom per cell */
  equations->DF = 9 - equations->penalty_flag;

  /*        Create vector to contain load and various work vectors  */
  ierr = AppCtxCreateVector(appctx); CHKERRQ(ierr);
  /*      Create the sparse matrix, with correct nonzero pattern  */
  ierr = AppCtxCreateMatrix(appctx); CHKERRQ(ierr);
  /*     Set the quadrature values for the reference square element  */
  ierr = AppCtxSetReferenceElement(appctx);CHKERRQ(ierr);
  /*      Set the right hand side values into the vectors   */
  ierr = AppCtxSetRhs(appctx); CHKERRQ(ierr);

  /* The coeff of diffusivity.  LATER call a function set equations */
  /* set in appload.c */
  /*      Set the matrix entries   */
  ierr = AppCtxSetMatrix(appctx); CHKERRQ(ierr);

/*    MatView(algebra->A, VIEWER_STDOUT_SELF);   */

  /*     Create the nonlinear solver context  */
  ierr = SNESCreate(PETSC_COMM_WORLD,SNES_NONLINEAR_EQUATIONS,&snes); CHKERRQ(ierr);

  /********* solve the stationary problem with boundary conditions.  ************/
  /*      Set function evaluation rountine and vector */
  ierr = SNESSetFunction(snes,algebra->f ,FormStationaryFunction,(void *)appctx); CHKERRQ(ierr);
  /*      Set Jacobian   */ 
  ierr = SNESSetJacobian(snes, algebra->J, algebra->J, FormStationaryJacobian,(void *)appctx);CHKERRQ(ierr);

  /* set monintor functions */
  ierr = SNESSetMonitor(snes, MonitorFunction, PETSC_NULL);CHKERRQ(ierr);

  /*      Set Solver Options, could put internal options here      */
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  /* initial guess */
  ierr = FormInitialGuess(appctx);CHKERRQ(ierr); 

/* printf("the initital guess\n"); */
/*  VecView(algebra->g, VIEWER_STDOUT_SELF); */

 /* Send to  matlab viewer */
   if (appctx->view.matlabgraphics) {AppCtxViewMatlab(appctx);  } 

 
  /*       Solve the non-linear system  */
   ierr = SNESSolve(snes, PETSC_NULL, algebra->g);CHKERRQ(ierr);
  ierr = SNESGetIteratioNumber(snes, &its);CHKERRQ(ierr);


  /* First, send solution vector to Matlab */
/* Send to  matlab viewer */
   if (appctx->view.matlabgraphics) {
     ierr = VecView(appctx->algebra.g,VIEWER_MATLAB_WORLD); CHKERRQ(ierr);

  /* send the done signal */
     ierr = PetscIntView(1, &zero, VIEWER_MATLAB_WORLD);CHKERRQ(ierr);}
/*    VecView(algebra->g, VIEWER_STDOUT_SELF);  */

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

    int *df_ptr;
    double tweak;
    double *coords_ptr;
    int ierr,i,j;
    double xval,yval;
    double values[9];
    double val = 1.23;
    tweak = equations->tweak;
    if (0){
 /* loop over cells */
  for(i=0;i<grid->cell_n;i++){
    /* loop over degrees of freedom and cell coords */
    df_ptr = grid->cell_df + equations->DF*i;
    coords_ptr = grid->cell_coords + 8*i;

    /* now assign to the vector, the values of the solution */
    for(j=0;j<4;j++){
      xval = coords_ptr[2*j];
      yval = coords_ptr[2*j + 1];
      values[2*j] = bc1(xval, yval) + tweak;
      values[2*j+1] = bc2(xval, yval) + tweak;
    }
    xval = coords_ptr[0]; yval = coords_ptr[1];
    if(!equations->penalty_flag){
      values[8] = bc3(xval, yval);}
    ierr = VecSetValuesLocal(algebra->g, equations->DF, df_ptr, values, INSERT_VALUES ); CHKERRQ(ierr);
  }
    }
 ierr = VecSet(algebra->g,val);CHKERRQ(ierr); 
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
  ierr = MatMultAdd(A, x, f, f); CHKERRQ(ierr);  /* f = A*x - b */
 /* create nonlinear part */
  ierr = SetNonlinearFunction(x, appctx, f);CHKERRQ(ierr);

 /*  printf("output of nonlinear fun (before bc imposed)\n");     */
/*    ierr = VecView(f, VIEWER_STDOUT_WORLD);CHKERRQ(ierr);      */

  ierr = SetBoundaryConditions(x, appctx, f);CHKERRQ(ierr);
/*   printf("output of nonlinear fun \n");     */
/*     ierr = VecView(f, VIEWER_STDOUT_WORLD);CHKERRQ(ierr);      */
  
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
#define __FUNC__ "MonitorFunction"
int MonitorFunction(SNES snes, int its, double norm, void *metcx)
{
  int ierr, one = 1; 
  Vec x;  
  if(0){
  ierr = SNESGetSolution(snes,&x); CHKERRQ(ierr);
  ierr = VecView(x,VIEWER_MATLAB_WORLD); CHKERRQ(ierr); 
   ierr = PetscIntView(1, &one, VIEWER_MATLAB_WORLD);CHKERRQ(ierr); 
}
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
   AppEquations *equations = &appctx->equations;
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
    df_ptr = grid->cell_df + equations->DF*i;
    coords_ptr = grid->cell_coords + 8*i;
      /* Need to point to the uvvals associated to the velocity degrees of freedom. 
       Can ignore pressure, since it is linear */
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
  AppEquations *equations = &appctx->equations;
  int ierr, i;
  double   xval, yval; 
  double *inlet_vvals, *outlet_vvals, *wall_vals, *outlet_pvals,  *inlet_pvals; 


  /* Fix one pressure node */

 /* Pressure */
 /*    ierr = VecScatterBegin( g, algebra->f_poutlet, INSERT_VALUES, SCATTER_FORWARD, algebra->gtol_poutlet);   CHKERRQ(ierr); */
/*     ierr = VecScatterEnd( g,  algebra->f_poutlet, INSERT_VALUES, SCATTER_FORWARD, algebra->gtol_poutlet);   CHKERRQ(ierr); */
/*     ierr = VecGetArray( algebra->f_poutlet, &outlet_pvals); CHKERRQ(ierr); */
/*     ierr = VecSetValuesLocal(f, 1, grid->outlet_pdf, outlet_pvals, INSERT_VALUES); CHKERRQ(ierr); */

  /* Velocity */
  /* INLET */

  if (equations->vin_flag){
    ierr = VecScatterBegin( g, algebra->f_vinlet, INSERT_VALUES, SCATTER_FORWARD, algebra->gtol_vinlet);     CHKERRQ(ierr);
    ierr = VecScatterEnd( g,  algebra->f_vinlet, INSERT_VALUES, SCATTER_FORWARD, algebra->gtol_vinlet);   CHKERRQ(ierr);
    ierr = VecGetArray( algebra->f_vinlet, &inlet_vvals); CHKERRQ(ierr);
    /* need to be very careful here, for each 2 df, have 2 coords */
    for( i=0; i < grid->inlet_vcount; i = i+2){
      xval = grid->inlet_coords[i];
      yval = grid->inlet_coords[i+1];
      grid->inlet_values[i] = inlet_vvals[i]  - bc1(xval, yval)  ;
      grid->inlet_values[i+1] = inlet_vvals[i+1]  - bc2(xval, yval); 
    }
    ierr = VecSetValuesLocal(f, grid->inlet_vcount, grid->inlet_vdf, grid->inlet_values, INSERT_VALUES);  
  }
   /*VOUTLET*/
  if(equations->vout_flag){
    ierr = VecScatterBegin( g, algebra->f_voutlet, INSERT_VALUES, SCATTER_FORWARD, algebra->gtol_voutlet); CHKERRQ(ierr);
    ierr = VecScatterEnd( g,  algebra->f_voutlet, INSERT_VALUES, SCATTER_FORWARD, algebra->gtol_voutlet);  CHKERRQ(ierr);
    ierr = VecGetArray( algebra->f_voutlet, &outlet_vvals); CHKERRQ(ierr);
    /* need to be very careful here, for each 2 df, have 2 coords */
    for( i=0; i < grid->outlet_vcount; i = i+2){
      xval = grid->outlet_coords[i];
      yval = grid->outlet_coords[i+1];
      grid->outlet_values[i] = outlet_vvals[i]  - bc1(xval, yval)  ;
      grid->outlet_values[i+1] = outlet_vvals[i+1]  - bc2(xval, yval); 
    }
    ierr = VecSetValuesLocal(f, grid->outlet_vcount, grid->outlet_vdf, grid->outlet_values, INSERT_VALUES);  
  }

 /* Pressure */
   /* POUTLET */
  if(equations->pout_flag){
    ierr = VecScatterBegin( g, algebra->f_poutlet, INSERT_VALUES, SCATTER_FORWARD, algebra->gtol_poutlet);   CHKERRQ(ierr);
    ierr = VecScatterEnd( g,  algebra->f_poutlet, INSERT_VALUES, SCATTER_FORWARD, algebra->gtol_poutlet);   CHKERRQ(ierr);
    ierr = VecGetArray( algebra->f_poutlet, &outlet_pvals); CHKERRQ(ierr);
    ierr = VecSetValuesLocal(f, grid->outlet_pcount, grid->outlet_pdf, outlet_pvals, INSERT_VALUES); CHKERRQ(ierr);
  }

 /* PINLET  */
  if (equations->pin_flag){
    ierr = VecScatterBegin( g, algebra->f_pinlet, INSERT_VALUES, SCATTER_FORWARD, algebra->gtol_pinlet);     CHKERRQ(ierr);
  ierr = VecScatterEnd( g,  algebra->f_pinlet, INSERT_VALUES, SCATTER_FORWARD, algebra->gtol_pinlet);   CHKERRQ(ierr);
  ierr = VecGetArray( algebra->f_pinlet, &inlet_pvals); CHKERRQ(ierr);
  /* Here Setting Pressure to 1 */
  for( i=0; i < grid->inlet_pcount; i++){ 
    grid->inlet_pvalues[i] = inlet_pvals[i] - 10; 
  } 
  ierr = VecSetValuesLocal(f, grid->inlet_pcount, grid->inlet_pdf, grid->inlet_pvalues, INSERT_VALUES);      CHKERRQ(ierr); 
  }

  /* WALL */
  if (equations->wall_flag){
  ierr = VecScatterBegin( g, algebra->f_wall, INSERT_VALUES, SCATTER_FORWARD, algebra->gtol_wall); CHKERRQ(ierr);
  ierr = VecScatterEnd( g,  algebra->f_wall, INSERT_VALUES, SCATTER_FORWARD, algebra->gtol_wall); CHKERRQ(ierr);
  ierr = VecGetArray( algebra->f_wall, &wall_vals); CHKERRQ(ierr);
  /* just need to set these values to f */
  ierr = VecSetValuesLocal(f, grid->wall_vcount, grid->wall_vdf, wall_vals, INSERT_VALUES);   CHKERRQ(ierr);
  }
  /* YWALL */
  if (equations->ywall_flag){
  ierr = VecScatterBegin( g, algebra->f_ywall, INSERT_VALUES, SCATTER_FORWARD, algebra->gtol_ywall); CHKERRQ(ierr);
  ierr = VecScatterEnd( g,  algebra->f_ywall, INSERT_VALUES, SCATTER_FORWARD, algebra->gtol_ywall); CHKERRQ(ierr);
  ierr = VecGetArray( algebra->f_ywall, &wall_vals); CHKERRQ(ierr);
  /* just need to set these values to f */
  ierr = VecSetValuesLocal(f, grid->ywall_vcount, grid->ywall_vdf, wall_vals, INSERT_VALUES);   CHKERRQ(ierr);
  }

  /********* Assemble Data **************/
 ierr = VecAssemblyBegin(f);CHKERRQ(ierr);
 ierr = VecAssemblyEnd(f);CHKERRQ(ierr);
 
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
  AppEquations *equations = &appctx->equations;
/****** Internal Variables ***********/
  int  i,j, ierr;
  int  *df_ptr; 
  double *coords_ptr;
  double   *uvvals, cell_values[8];
  double values[8*8];  /* the integral of the combination of phi's */
 double one = 1.0;
int nine = 9;
IS is_single_pressure;

  PetscFunctionBegin;
  /* Matrix is set to the linear part already, so just ADD_VALUES the nonlinear part  */ 
  ierr = VecScatterBegin(g, algebra->f_local, INSERT_VALUES, SCATTER_FORWARD, algebra->dfgtol); CHKERRQ(ierr);
  ierr = VecScatterEnd(g, algebra->f_local, INSERT_VALUES, SCATTER_FORWARD, algebra->dfgtol); CHKERRQ(ierr);
  ierr = VecGetArray(algebra->f_local, &uvvals);
 
  /* loop over cells */
  for(i=0;i<grid->cell_n;i++){
   /* loop over degrees of freedom and cell coords */
    df_ptr = grid->cell_df + equations->DF*i;
    coords_ptr = grid->cell_coords + 8*i;
    /* Need to point to the uvvals associated to the velocity dfs (can ignore pressure) */
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

  /**********  boundary conditions ************/
  /* here should zero rows corresponding to dfs where bc imposed */
  if(equations->wall_flag){
    ierr = MatZeroRowsLocalIS(*jac, grid->iswall_vdf,one);CHKERRQ(ierr); }
  if(equations->ywall_flag){
    ierr = MatZeroRowsLocalIS(*jac, grid->isywall_vdf,one);CHKERRQ(ierr); }
  if(equations->vin_flag){
    ierr = MatZeroRowsLocalIS(*jac, grid->isinlet_vdf,one);CHKERRQ(ierr);}
  if(equations->vout_flag){
    ierr = MatZeroRowsLocalIS(*jac, grid->isoutlet_vdf,one);CHKERRQ(ierr);}
 if(equations->pout_flag){
   ierr = MatZeroRowsLocalIS(*jac, grid->isoutlet_pdf,one);CHKERRQ(ierr);}
 if(equations->pin_flag){
   ierr = MatZeroRowsLocalIS(*jac, grid->isinlet_pdf,one);CHKERRQ(ierr);}

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
  AppEquations *equations = &appctx->equations;
  int ierr, i;
  int *df_ptr;
  double *coords_ptr;
  double  values[9]; 

  /* set flag for element computation */
  phi->dorhs = 1;

 /* loop over cells */
  for(i=0;i<grid->cell_n;i++){
    /* loop over degrees of freedom and cell coords */
    df_ptr = grid->cell_df + equations->DF*i;
    coords_ptr = grid->cell_coords + 8*i;
    /* compute the values of basis functions on this element */
    SetLocalElement(phi, coords_ptr); 
    /* compute the  element load (integral of f with the 4 basis elements)  */
    ComputeRHS( f, g, phi, values );/* f,g are rhs functions */
    values[8] = 0; /* no forcing for pressure */
    /*********  Set Values *************/
    ierr = VecSetValuesLocal(algebra->b, equations->DF, df_ptr, values, ADD_VALUES);CHKERRQ(ierr);
  }
  
  /********* Assemble Data **************/
  ierr = VecAssemblyBegin(algebra->b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(algebra->b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}  

 /* we want to compute C*M_inverse*C^t, where C is the incompressbility matrix (minus what is computed by computepressure) and M is the mass matrix, which works out to be the absolute value of the jacobian (times the quadrature weight = 4). */
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
  int i, j, ierr;
  int *df_ptr;
  double mass_scale;
  double *coords_ptr;
  double values[4*4], pvalues[8], result[9*9];
 
  PetscFunctionBegin;
  /************ Set Up **************/ 
  /* set flag for phi computation */
  phi->dorhs = 0;
  /* loop over cells */
  for(i=0;i<grid->cell_n;i++){
    /* loop over degrees of freedom and cell coords */
    df_ptr = grid->cell_df + equations->DF*i;
    coords_ptr = grid->cell_coords + 8*i;
    /* compute the values of basis functions on this element */
    ierr = SetLocalElement(phi, coords_ptr); CHKERRQ(ierr);
    ierr = SetCentrElement(phi, coords_ptr); CHKERRQ(ierr);
    
    /*    Compute the element stiffness */  
    /* returns 4*4 values which need to be spread*/
    ierr = ComputeMatrix( phi, values ); CHKERRQ(ierr);
    /*scale the stiffnes by -1*viscosity*/
    for(j =0; j<16;j++){ values[j] = -equations->eta*values[j];}
    ierr = ComputePressure( phi, pvalues ); CHKERRQ(ierr);
    if(equations->penalty_flag)
      {
      /* scale by the penalty parameter */
      for(j =0; j<8;j++){ pvalues[j] = -pvalues[j]/equations->penalty;}
      mass_scale = 4*PetscAbsDouble(phi->detDhCentr); 
      /* this takes care of the mass matrix in here */
      PenaltySpreadValues( values, pvalues, mass_scale, result );
      }
    else
      { SpreadValues(values, pvalues, result);}
    
    ierr = MatSetValuesLocal(algebra->A, equations->DF, df_ptr, equations->DF, df_ptr, result, ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(algebra->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(algebra->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int PenaltySpreadValues (double  values[16], double pvalues[8], double mass_scale, double result[8*8]){
  /* for the stiffness: 
     first row : alternate every first 4 values , 0,  
     second row alternate 0, first 4 values, 
     */ 
  int i,j;
  for(i=0;i<4;i++){ /* rows */
    /* even numbered rows */
    for(j=0;j<4;j++){ /* columns */
      result[8*2*i + 2*j] = values[4*i + j];
      result[8*2*i + 2*j+1] = 0;
    }
    /* odd numbered rows */
    for(j=0;j<4;j++){ /* columns */
      result[8*2*i + 8 + 2*j] = 0;
      result[8*2*i + 8 + 2*j + 1] = values[4*i + j];
    }
  }
/* for the penalty matrix:
   take the product of the pvalues[i]*pvalues[j]/mass_scale*/
  for(i=0;i<8;i++){/*rows*/
    for(j=0;j<8;j++){/*columns */
      result[8*i+j] += pvalues[i]*pvalues[j]/mass_scale;}
  }
  PetscFunctionReturn(0);
}

int SpreadValues(double  values[16], double pvalues[8], double result[9*9]){
  /* 
 first row : alternate every first 4 values , 0,  then finish with pvalues
  second row alternate 0, first 4 values, finish pvalues
 and so on
 las row, pvalues, then 0;
 */
  int i,j;
  for(i=0;i<4;i++){ /* rows */
    /* even numbered rows */
    for(j=0;j<4;j++){ /* columns */
      result[9*2*i + 2*j] = values[4*i + j];
      result[9*2*i + 2*j + 1] = 0;
    }
    result[9*2*i + 8] = pvalues[2*i];
    /* odd numbered rows */
    for(j=0;j<4;j++){ /* columns */
      result[9*2*i + 9 + 2*j] = 0;
      result[9*2*i + 9 + 2*j + 1] = values[4*i + j];
    }
    result[9*2*i + 9 + 8] = pvalues[2*i+1];
  }
  for(i=0;i<8;i++){ result[9*8 + i] = pvalues[i];}
  result[9*8 + 8] = 0;
PetscFunctionReturn(0);
}

