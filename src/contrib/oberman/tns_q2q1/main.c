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

  AppGrid                *grid = &appctx->grid;
  AppAlgebra             *algebra = &appctx->algebra;
  AppElement          *element = &appctx->element;
  AppEquations *equations = &appctx->equations;
  SLES                   sles;
  SNES                   snes;
  int ierr, its;
  int flag;
  int i;
  int zero = 0, one = 1;
  double dzero = 0.0;
  double ddt;
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
  ierr = AppCtxSetMassMatrix(appctx); CHKERRQ(ierr);/* for the time stepping */

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
ierr = SolveStationary(appctx, snes);CHKERRQ(ierr);  
/* the solution goes into algebra->g


/************* now begin timestepping, with computed soln as initial values *************/

  /* time step for accuracy is determined by courant number:
     u*dt/dx in convection dominated flows, or by mu*dt/dx*dx in diffusion dominated */

ierr = SolveTimeDependant(appctx, snes);CHKERRQ(ierr);  


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
  double zero = 0.0, mone = -1.0, one = 1.0;

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

  if (appctx->view.show_vector ){     printf("apply matrix to f\n");     VecView(f, VIEWER_STDOUT_SELF); };

 /* create nonlinear part */
if( !appctx->equations.stokes_flag ){
  ierr = VecSet(algebra->conv,zero); CHKERRQ(ierr); 
  ierr = SetNonlinearFunction(x, appctx, algebra->conv);CHKERRQ(ierr);
  if (appctx->view.show_vector ){ printf("put nonlinear part in conv\n"); VecView(algebra->conv, VIEWER_STDOUT_SELF);} 
  /* add the real convected term */
    ierr = VecAXPY(f,one,algebra->conv); CHKERRQ(ierr); 
}

  /* apply boundary conditions */
  ierr = SetBoundaryConditions(x, appctx, f);CHKERRQ(ierr);

  if (appctx->view.show_vector ){ printf("set bc to  f\n"); VecView(f, VIEWER_STDOUT_SELF);}

  PetscFunctionReturn(0);
}


#undef __FUNC__
#define __FUNC__ "FormDynamicFunction"
int FormDynamicFunction(SNES snes, Vec x, Vec f, void *dappctx)
{
  /********* Collect context informatrion ***********/
  AppCtx *appctx = (AppCtx *)dappctx;
  AppAlgebra *algebra = &appctx->algebra;
  AppEquations *equations = &appctx->equations;
  AppGrid *grid = &appctx->grid;
 
  /* Internal Variables */
  int ierr,i;
  double zero = 0.0, mone = -1.0, one = 1.0, onep5 = 1.5, mhalf = -0.5;
  double alpha, malpha;
  int df_v_count, *dfv_indices, *df_ptr;
  double *fvals;
  double newval;

/****** Perform computation ***********/
  /* M is the already computed mass matrix */
  /* the old solution is g, the input is x, compute the output to be f. */


  /* on the velocity components, need to compute  M*(x - g)/dt + computed stuff
     on the pressure, just compute pressure on the input. */
  /* 1 compute the linear & nolinear stuff on the input values.
     2 multiply the velocity components by dt. TIMES MASS MATRIX
     3 add to velocity computed input - xold
   */
/* need to zero f */
  ierr = VecSet(f,zero); CHKERRQ(ierr); 

  /*apply matrix to the input vector x, to get linear part */
  ierr = MatMultAdd(algebra->A, x, f, f); CHKERRQ(ierr);  /* f = A*x + f */

  /* add rhs to get constant part */
  ierr = VecAXPY(f,mone,algebra->b); CHKERRQ(ierr); /* this says f = f - 1*b */

 /*  nonlinear part */
/* create nonlinear part */
if( !appctx->equations.stokes_flag ){
  ierr = VecSet(algebra->conv,zero); CHKERRQ(ierr); 
  ierr = SetNonlinearFunction(x, appctx, algebra->conv);CHKERRQ(ierr);
  if (appctx->view.show_vector ){ printf("put nonlinear part in conv\n"); VecView(algebra->conv, VIEWER_STDOUT_SELF);}
}

/* in explicit convection, do Adams-Bashford on the old convection vector, add add to rhs, then compute the convection vector and increment the old guys */
if(appctx->equations.convection_flag){
  /* current convection term = 3/2conv(1 time-step ago) - 1/2conv(2 time steps) */
   ierr = VecAXPY(f,onep5,algebra->convl); CHKERRQ(ierr); 
   ierr = VecAXPY(f,mhalf,algebra->convll); CHKERRQ(ierr); 
}
else
  { 
    /* add the real convected term */
    ierr = VecAXPY(f,one,algebra->conv); CHKERRQ(ierr); 
  }  

  /* view mass matrix */
  if( appctx->view.show_matrix ){    printf("here comes the mass matrix\n");
    ierr = MatView(algebra->M, VIEWER_DRAWX_WORLD );CHKERRQ(ierr);
  }

  /* Now mass matrix times estimate of du/dt */
  /* Mass Matrix is zeros on the pressure df */
  /* so want M*(x-g)/dt */
  
  /* set dtvec */  
  alpha = 1.0/equations->dt;  malpha = -alpha;
  ierr = VecCopy(x, algebra->dtvec);CHKERRQ(ierr); /* dtvec = x */
  ierr = VecAXPBY(algebra->dtvec, malpha, alpha, algebra->g); /* dtvec = (x-g)/dt */

  /* apply mass matrix to dt and add to f */
  ierr = MatMultAdd(algebra->M, algebra->dtvec, f, f); CHKERRQ(ierr);  /* f = A*x + f */

  /* boundary conditions */
  ierr = SetBoundaryConditions(x, appctx, f);CHKERRQ(ierr);

  ierr = VecAssemblyBegin(f);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(f);CHKERRQ(ierr);  
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

if( appctx->view.show_matrix ) {  
  printf("in jac, stiffness and pressure\n");
  ierr = MatView(*jac, VIEWER_DRAWX_WORLD );CHKERRQ(ierr);
}
  /* the nonlinear part */
  if( !appctx->equations.stokes_flag){
    ierr = SetJacobian(g, appctx, jac);CHKERRQ(ierr);
  }

  ierr = SetJacobianBoundaryConditions(appctx, jac);CHKERRQ(ierr);
if( appctx->view.show_matrix ) {  

  printf("set boundary conditions\n");
  ierr = MatView(*jac, VIEWER_DRAWX_WORLD );CHKERRQ(ierr);
}
  /* Set flag */
  *flag = SAME_NONZERO_PATTERN;

  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "FormDynamicJacobian"
int FormDynamicJacobian(SNES snes, Vec g, Mat *jac, Mat *B, MatStructure *flag, void *dappctx)
{
  AppCtx *appctx = (AppCtx *)dappctx;
  AppAlgebra *algebra = &appctx->algebra;
  AppGrid *grid = &appctx->grid;
  AppEquations *equations = &appctx->equations;
  int ierr,i;
  int df_v_count, *dfv_indices;
  double one = 1.0, zero = 0.0;
  double idt;
  /* copy the linear part into jac.*/
  ierr= MatCopy(algebra->A, *jac, SAME_NONZERO_PATTERN);CHKERRQ(ierr);

  /* the nonlinear part */

  if(!(appctx->equations.stokes_flag+appctx->equations.convection_flag) ){
    ierr = SetJacobian(g, appctx, jac);CHKERRQ(ierr);
  }

  /* Now add the part part corresponding to dt, in this case the mass matrix */
  idt = 1.0/equations->dt;
  ierr = MatAXPY(*jac, algebra->M, idt); CHKERRQ(ierr);  /* jac = idt*M + jac */

  /* Apply boundary conditions*/
  ierr = SetJacobianBoundaryConditions(appctx, jac); CHKERRQ(ierr);

  ierr = MatAssemblyBegin(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Set flag */
  *flag = SAME_NONZERO_PATTERN; 
  PetscFunctionReturn(0);
}
#undef __FUNC__
#define __FUNC__ "SolveTimeDependant"
int SolveTimeDependant(AppCtx* appctx, SNES snes)
{

  AppGrid                *grid = &appctx->grid;
  AppAlgebra             *algebra = &appctx->algebra;
  AppElement          *element = &appctx->element;
  AppEquations *equations = &appctx->equations;
 
  int ierr, its;
  int flag;
  int i;
  int zero = 0, one = 1;
  double dzero = 0.0;
  double ddt;

  ierr = SNESSetFunction(snes,algebra->f,FormDynamicFunction,(void *)appctx); CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,algebra->J,algebra->J,FormDynamicJacobian,(void *)appctx);CHKERRQ(ierr);

  if(equations->convection_flag){    ierr = ExplicitConvectionSolve(appctx, snes);    }


  /* Determine time increment*/
  equations->dt = (equations->final_time - equations->initial_time)/equations->Nsteps;

  /* Use the old guy for first step */
  ierr = VecCopy(algebra->g, algebra->soln); CHKERRQ(ierr);

    
  /* now loop and solve */
  while( equations->current_time < equations->final_time)
    { 

      /* determine whether to change dt
	 have previous solution, generate the next, get the iteration data, 
	 and the accuracy, do a two step guy, and compare the differences.
	 use that to determine the time-step

	 */

      equations->current_time += equations->dt;
      ierr = TimeStep(appctx, snes);  CHKERRQ(ierr);

      /* send this step to the matlab viewer */  
      if( i % equations->Nplot == 0){
	ierr = ProcessSolution(appctx);  CHKERRQ(ierr);}
      i++;

    }


  if (appctx->view.matlabgraphics){
    /* finish up with matlab */
    ierr = VecView(appctx->algebra.soln,VIEWER_MATLAB_WORLD); CHKERRQ(ierr);
    /* send the done signal */
    ierr = PetscIntView(1, &zero, VIEWER_MATLAB_WORLD);CHKERRQ(ierr); 
  }

 PetscFunctionReturn(0);
}
#undef __FUNC__
#define __FUNC__ "ProcessSolution"
int ProcessSolution(AppCtx* appctx)
{ 
  AppGrid                *grid = &appctx->grid;
  AppAlgebra             *algebra = &appctx->algebra;
  int ierr;
  int one = 1;
  double mone = -1.0;

  if (appctx->view.matlabgraphics){
	ierr = VecView(appctx->algebra.soln, VIEWER_MATLAB_WORLD); CHKERRQ(ierr); 
	ierr = PetscIntView(1, &one, VIEWER_MATLAB_WORLD);CHKERRQ(ierr);
      }
ierr = VecView(algebra->soln, 0);

 /* compute the vorticity, and output it */
 /* need to scatter to the v1 and v2 guys, then compute the partials */
  ierr = VecScatterBegin(algebra->soln, algebra->v1, INSERT_VALUES,SCATTER_FORWARD, algebra->dfvtov1);CHKERRQ(ierr); 
  ierr = VecScatterBegin(algebra->soln, algebra->v2, INSERT_VALUES,SCATTER_FORWARD, algebra->dfvtov2);CHKERRQ(ierr); 

  ierr = VecView(algebra->v1, 0);

  /* now compute a bunch of partials */
  ierr = VecSet(algebra->v1a,mone); CHKERRQ(ierr);
  ierr = SetPartialDx(algebra->v1, algebra->v1a); CHKERRQ(ierr);
  ierr = SetPartialDy(algebra->v1, algebra->v1b); CHKERRQ(ierr);
  ierr = SetPartialDx(algebra->v2, algebra->v2a); CHKERRQ(ierr);
  ierr = SetPartialDy(algebra->v2, algebra->v2b); CHKERRQ(ierr);
 
  /* now do the vector arithmetic */
  ierr = VecAXPY(algebra->v2b,mone,algebra->v1a);CHKERRQ(ierr);
 
  /* now output the solution */
ierr = VecView(algebra->v2b, 0);

 PetscFunctionReturn(0);
}


#undef __FUNC__
#define __FUNC__ "SolveStationary"
int SolveStationary(AppCtx* appctx, SNES snes)
{ 

  AppGrid                *grid = &appctx->grid;
  AppAlgebra             *algebra = &appctx->algebra;
  AppElement          *element = &appctx->element;
  AppEquations *equations = &appctx->equations;
 
  int ierr, its;
  int flag;
  int i;
  int zero = 0, one = 1;
  double dzero = 0.0;
  double ddt;

  /*      Set function evaluation rountine and vector */
  ierr = SNESSetFunction(snes,algebra->f ,FormStationaryFunction,(void *)appctx); CHKERRQ(ierr);
  /*      Set Jacobian   */ 
  ierr = SNESSetJacobian(snes, algebra->J, algebra->J, FormStationaryJacobian,(void *)appctx);CHKERRQ(ierr);
  /* set monintor functions */
  if(appctx->view.monitor) {ierr = SNESSetMonitor(snes, MonitorFunction, (void *)appctx);CHKERRQ(ierr);}

/* Need this call, otherwise the defaults don't get set, and solve won't work */
 /*      Set Solver Options, could put internal options here      */
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  /* initial guess */
  ierr = FormInitialGuess(appctx);CHKERRQ(ierr); 

  /* Setup matlab viewer */
   if (appctx->view.matlabgraphics) {AppCtxViewMatlab(appctx);  } 

equations->current_time = equations->initial_time;

  /*       Solve the non-linear system  */
 ierr = SNESSolve(snes, PETSC_NULL, algebra->g);CHKERRQ(ierr);
  ierr = SNESGetIteratioNumber(snes, &its);CHKERRQ(ierr);

  /* send solution to matlab */
  if (appctx->view.matlabgraphics){
    ierr = VecView(appctx->algebra.g,VIEWER_MATLAB_WORLD); CHKERRQ(ierr);
    /* send the not done signal */
     ierr = PetscIntView(1, &one, VIEWER_MATLAB_WORLD);CHKERRQ(ierr); 
  }

  /* show solution vector */
  if (appctx->view.show_vector ){ 
    printf("the current soln vector\n");
    VecView(algebra->g, VIEWER_STDOUT_SELF);}

  /* output number of its */
  printf("the number of its, %d\n", its);
  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "TimeStep"
int TimeStep(AppCtx* appctx, SNES snes)
{
 AppGrid                *grid = &appctx->grid;
  AppAlgebra             *algebra = &appctx->algebra;
  AppElement          *element = &appctx->element;
  AppEquations *equations = &appctx->equations;
 
  int ierr, its;
  int flag;
  int i;
  int zero = 0, one = 1;
  double dzero = 0.0;
  double ddt;

   /* last solution becomes initial guess */
      ierr = VecCopy( algebra->soln, algebra->soln1); CHKERRQ(ierr);
      /* pop xold into g, to be computed in the  nonlinear function */
      ierr = VecCopy( algebra->soln, algebra->g); CHKERRQ(ierr);
 
      ierr = SNESSolve(snes, PETSC_NULL, algebra->soln1);CHKERRQ(ierr); 
      ierr = SNESGetIteratioNumber(snes, &its);CHKERRQ(ierr);
   /* now update the solution vectors */
      ierr = VecCopy( algebra->soln1, algebra->soln); CHKERRQ(ierr);
 
 PetscFunctionReturn(0);
}


#undef __FUNC__
#define __FUNC__ "ExplicitConvectionSolve"
int ExplicitConvectionSolve(AppCtx* appctx, SNES snes)
{

  AppGrid                *grid = &appctx->grid;
  AppAlgebra             *algebra = &appctx->algebra;
  AppElement          *element = &appctx->element;
  AppEquations *equations = &appctx->equations;
 
  int ierr, its;
  int flag;
  int i;
  int zero = 0, one = 1;
  double dzero = 0.0;
  double ddt;
  /* zero the old convections vectors */
    ierr = VecCopy(algebra->conv, algebra->convl); CHKERRQ(ierr); 
    ierr = VecSet(algebra->convll,dzero); CHKERRQ(ierr); 
 
    ierr = SNESSetFunction(snes,algebra->f,FormDynamicFunction,(void *)appctx); CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes,algebra->J,algebra->J,FormDynamicJacobian,(void *)appctx);CHKERRQ(ierr);
  /* Determine time increment*/
  equations->dt = (equations->final_time - equations->initial_time)/equations->Nsteps;
  /* Use the old guy for first step */
  ierr = VecCopy(algebra->g, algebra->soln); CHKERRQ(ierr);

   /* do 2 steps implicit, to get the convection vectors */
  flag = equations->convection_flag;

  /* now loop and solve */
  for(i=0;i<equations->Nsteps;i++)
    {
      /* last solution becomes initial guess */
      ierr = VecCopy( algebra->soln, algebra->soln1); CHKERRQ(ierr);
      /* pop xold into g, to be computed in the  nonlinear function */
      ierr = VecCopy( algebra->soln, algebra->g); CHKERRQ(ierr);
      /* put the dt in the context */
      equations->current_time += equations->dt;
      /* now solve */
      /* do 2 steps implicit, to get the convection vectors */
      if(i < 2){equations->convection_flag=0;}
      else {equations->convection_flag = flag;}

      ierr = SNESSolve(snes, PETSC_NULL, algebra->soln1);CHKERRQ(ierr); 
      ierr = SNESGetIteratioNumber(snes, &its);CHKERRQ(ierr);

      printf("time step %d: the number of its, %d\n", i, its);
      if(equations->preconconv_flag*equations->convection_flag){
	/* here we use the convection solution as an initial guess for the fully implicit */
	equations->convection_flag = 0;

	ierr = SNESSolve(snes, PETSC_NULL, algebra->soln1);CHKERRQ(ierr); 
        ierr = SNESGetIteratioNumber(snes, &its);CHKERRQ(ierr);

	equations->convection_flag = 1;
      }
      /* send this step to the matlab viewer */  
      if (appctx->view.matlabgraphics){
	ierr = VecView(algebra->soln1, VIEWER_MATLAB_WORLD); CHKERRQ(ierr); 
	ierr = PetscIntView(1, &one, VIEWER_MATLAB_WORLD);CHKERRQ(ierr);
      }
      /* now update the solution vectors */
      ierr = VecCopy( algebra->soln1, algebra->soln); CHKERRQ(ierr);

      if(equations->convection_flag){
	/*once solved, evolve the vectors */ 
	ierr = VecCopy(algebra->convl, algebra->convll); CHKERRQ(ierr); 
	ierr = VecCopy(algebra->conv, algebra->convl); CHKERRQ(ierr); 
      } 
    }

  if (appctx->view.matlabgraphics){
    /* finish up with matlab */
    ierr = VecView(appctx->algebra.soln,VIEWER_MATLAB_WORLD); CHKERRQ(ierr);
    /* send the done signal */
    ierr = PetscIntView(1, &zero, VIEWER_MATLAB_WORLD);CHKERRQ(ierr); 
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
    ierr = VecView(x,VIEWER_MATLAB_WORLD); CHKERRQ(ierr);
    ierr = PetscIntView(1, &one, VIEWER_MATLAB_WORLD);CHKERRQ(ierr);}
    PetscFunctionReturn(0);
}
