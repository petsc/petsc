
static char help[] ="Solves the 2d incompressible Navier-Stokes  equations.\ndu/dt + u*du/dx + v*du/dy + dp/dx - c(lap(u)) - f = 0.\n  dv/dt + u*dv/dv + v*dv/dy + dp/dy - c(lap(v)) - g = 0.\n";

#include "appctx.h"

#undef __FUNC__
#define __FUNC__ "main"
int main( int argc, char **argv )
{
  int            ierr;
  AppCtx         *appctx;
  AppAlgebra     *algebra;
  AppGrid        *grid;
  double         *vertex_value, *values, xval, yval, val;
  int            i, cell_n;
  int            *cell_vertex; /* vertex lists in local numbering */
  int            *vertex_global;

  /* ---------------------------------------------------------------------
     Initialize PETSc
     --------------------- ---------------------------------------------------*/

  PetscInitialize(&argc,&argv,(char *)0,help);

  /*  Load the grid database*/
  ierr = AppCtxCreate(PETSC_COMM_WORLD,&appctx); CHKERRA(ierr);

  /*      Initialize graphics */
  ierr = AppCtxGraphics(appctx); CHKERRA(ierr);
  algebra = &appctx->algebra;
  algebra = &appctx->algebra; grid = &appctx->grid;

  /*   Setup the linear system and solve it*/
  ierr = AppCtxSolve(appctx);CHKERRQ(ierr);

  /*    Output solution and grid coords to a plotter*/
  ierr = ISGetIndices(grid->vertex_global, &vertex_global); CHKERRQ(ierr);
  cell_vertex = grid->cell_vertex; 
  vertex_value = grid->vertex_value;
  ierr = VecGetArray(algebra->g,&values); CHKERRQ(ierr);
  cell_n = grid->cell_n;

  /*      Visualize solution   */
  /* Send to  matlab viewer */
  if (appctx->view.matlabgraphics) { AppCtxViewMatlab(appctx);  }

  /*      Destroy all datastructures  */
  ierr = AppCtxDestroy(appctx); CHKERRA(ierr);

  PetscFinalize();
  PetscFunctionReturn(0);
}

/*
         Set up the non-linear system associated with the PDE and solve it
*/
#undef __FUNC__
#define __FUNC__ "AppCxtSolve"
int AppCtxSolve(AppCtx* appctx)
{
  AppGrid                *grid = &appctx->grid;
  AppAlgebra             *algebra = &appctx->algebra;
  MPI_Comm               comm = appctx->comm;
  ISLocalToGlobalMapping ltog = grid->ltog;
  IS                     vertex_global = grid->vertex_global;
  SLES                   sles;
  SNES                   snes;
  PC pc;
  KSP ksp;
  Mat J;  /* Jacobian */
  Vec f;
  Vec g;  /* f is for the nonlinear function evaluation, g is the initial guess, solution */
  Vec *solnv;
 int its, ierr, size;
  double zero = 0.0; double onep2 = 1.2345;
  /* Time stepping stuff */
int dynamic = 0; /* flag to comment out the timestepping */
  int i;  double deltat, idt;
  double initial_time = 0.0;
  double final_time = .4;  /* Later get this from a context, from run time option */
  double times[NSTEPS];

  PetscFunctionBegin;

  /*        Create vector to contain load and various work vectors  */
  ierr = AppCtxCreateVector(appctx); CHKERRQ(ierr);
  /*      Create the sparse matrix, with correct nonzero pattern, create the Jacobian  */
  ierr = AppCtxCreateMatrix(appctx); CHKERRQ(ierr);
  /*     Set the quadrature values for the reference square element  */
  ierr = AppCtxSetReferenceElement(appctx);CHKERRQ(ierr);
  /*      Set the right hand side values into the vectors   */
  ierr = AppCtxSetRhs(appctx); CHKERRQ(ierr);
  /* The coeff of diffusivity.  LATER call a function set equations */
appctx->equations.eta =-0.1;  
  /*      Set the matrix entries   */
  ierr = AppCtxSetMatrix(appctx); CHKERRQ(ierr);
  /*     Create the nonlinear solver context  */
  ierr = SNESCreate(PETSC_COMM_WORLD,SNES_NONLINEAR_EQUATIONS,&snes); CHKERRQ(ierr);

  /********* solve the stationary problem with boundary conditions once.  ************/
  /*      Set function evaluation rountine and vector */
  f = algebra->f;
  ierr = SNESSetFunction(snes,f,FormStationaryFunction,(void *)appctx); CHKERRQ(ierr);
  /*      Set Jacobian   */ 
  J = algebra->J;
  ierr = SNESSetJacobian(snes,J,J,FormStationaryJacobian,(void *)appctx);CHKERRQ(ierr);
  /*      Set Solver Options, could put internal options here      */
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  /* initial guess */
  ierr = FormInitialGuess(appctx);CHKERRQ(ierr); 
  g = algebra->g;
  /*       Solve the non-linear system  */
  ierr = SNESSolve(snes, g, &its);CHKERRQ(ierr);
  printf("the number of its, %d\n", its);
  if(dynamic){
/************* now begin timestepping, with computed soln as initial values *************/
  ierr = SNESSetFunction(snes,f,FormDynamicFunction,(void *)appctx); CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,J,J,FormDynamicJacobian,(void *)appctx);CHKERRQ(ierr);
  /* extract solution vector */
  solnv = algebra->solnv;
  /* Determine times */
  for(i=0; i<NSTEPS; i++){ times[i] = i*(final_time - initial_time)/NSTEPS; }
   
  /* Use the old guy, loop over time steps, solve */
  ierr = VecCopy(g, solnv[0]); CHKERRQ(ierr);
  for(i=0;i<NSTEPS;i++)
    {
      /* last solution becomes initial guess */
    ierr = VecCopy( solnv[i], solnv[i+1]); CHKERRQ(ierr);
    /* last solution is used to compute nonlinear function */
    ierr = VecCopy( solnv[i], g); CHKERRQ(ierr);
    /* put the dt in the context */
    deltat = times[i+1]-times[i];
    appctx->dt = deltat;
    ierr = SNESSolve(snes, solnv[i+1], &its);CHKERRQ(ierr); 
    printf("time step %d: the number of its, %d\n", i, its);
    }
  ierr = SNESDestroy(snes); CHKERRQ(ierr);  
  }
  PetscFunctionReturn(0);
}
#undef __FUNC__
#define __FUNC__ "FormInitialGuess"
int FormInitialGuess(AppCtx* appctx)
{
/********* Collect context informatrion ***********/
    AppAlgebra             *algebra = &appctx->algebra;
    Vec g = algebra->g;
    int ierr;
    double zero = 0.0;
    double onep1 = 1.234;
    ierr = VecSet(g,&onep1); CHKERRQ(ierr);
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
  /* Assuming mattrix doesn't need to be recomputed */
  ierr = MatMultAdd(A, x, f, f); CHKERRQ(ierr);  /* f = A*x + f */
 /* create nonlinear part */
  ierr = SetNonlinearFunction(x, appctx, f);CHKERRQ(ierr);
  /* set boundary conditions : done in setNonlinearFunction */
  ierr = SetBoundaryConditions(x,appctx, f);CHKERRQ(ierr);

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
  /* Global to Local scatter (the blocked version) */
  VecScatter dgtol = algebra->dgtol;         
 /* The geometrical values of the vertices */
  double  *vertex_values = grid->vertex_value;
 /* The array of vertices in the local numbering for each cell */
  int  *cell_vertex = grid->cell_vertex;
 /* the number of cells on this processor */
  int  cell_n = grid->cell_n;

/****** Internal Variables ***********/
  /* need a local vector of size DF*(vertex_n_ghosted)*/
  Vec f_local = algebra->f_local;  
  double rresult[8], coors[8];
  double spreadresult[4*3]; /* in case DF = 3 */
  double cell_values[4*2];  /* just those for u,v */
  double *uvvals;
  int ierr, i, j, k, size;
  int *vertex_ptr;
  double xval, yval;
 /*  Loop over local elements, extracting the values from g  and add them into f  */

  /* Scatter the input values from the global vector g, to those on this processor */
  ierr = VecScatterBegin(g, f_local, INSERT_VALUES, SCATTER_FORWARD, dgtol); CHKERRQ(ierr);
  ierr = VecScatterEnd(g, f_local, INSERT_VALUES, SCATTER_FORWARD, dgtol); CHKERRQ(ierr);
 /* ierr = VecView(f_local,VIEWER_STDOUT_SELF);  */
 ierr = VecGetArray(f_local, &uvvals); CHKERRQ(ierr);

  /* set a flag in computation of local elements */
  phi->dorhs = 0;
  
  for(i=0;i<cell_n;i++)
    {
    vertex_ptr = cell_vertex + 4*i; 
    for ( j=0; j<4; j++) 
      {
	/* Compute Nonlinear just wants the u,v values */
	/* Need to point to the uvvals associated to the vertices */

	cell_values[2*j] = uvvals[3*vertex_ptr[j]];
	cell_values[2*j+1] = uvvals[3*vertex_ptr[j]+1];
	/*	cell_values[3*j+2] = uvvals[3*vertex_ptr[j]+2];*/
	/* get geometrical coordinates */
	coors[2*j] = vertex_values[2*vertex_ptr[j]];
	coors[2*j+1] = vertex_values[2*vertex_ptr[j]+1];
      }

    /* compute the values of basis functions on this element */
    SetLocalElement(phi, coors);
    /* do the integrals */
    ierr = ComputeNonlinear(phi, cell_values, rresult);CHKERRQ(ierr);    
 /*    printf("result of computeNonlinear\n"); for(i=0;i<8;i++) printf("%f\t", rresult[i]); */
    /* put result in */
    SpreadVector(rresult, spreadresult);
   ierr = VecSetValuesBlockedLocal(f, 4, vertex_ptr, spreadresult, ADD_VALUES);CHKERRQ(ierr);
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
  AppGrid *grid = &appctx->grid; 
  AppAlgebra *algebra = &appctx->algebra;
/* Global to Local scatter (the blocked version) */
  VecScatter dgtol = algebra->dgtol;         
/* The index set of the vertices on the boundary */  
  IS  boundary_wall = grid->boundary_wall;
  IS  boundary_inlet = grid->boundary_inlet; 
  IS  boundary_outlet = grid->boundary_outlet; 
  IS vertex_wall_blocked = grid->vertex_wall_blocked;
  IS vertex_inlet_blocked = grid->vertex_inlet_blocked;
 IS vertex_outlet_blocked = grid->vertex_outlet_blocked;

  double *uvvals;
  int ierr, i, j, k;
  int  *blocked_wall_indices, *inlet_indices, *inlet_indices_blocked, *outlet_indices, blocked_wall_count, inlet_count, inlet_blocked_count, outlet_count,outlet_blocked_count, *outlet_indices_blocked;

  double  *bvswall, *bvsinlet, *bvsoutlet;
  Vec f_local = algebra->f_local;  
 double xval, yval;

  /* Scatter the input values from the global vector g, to those on this processor */
  ierr = VecScatterBegin(g, f_local, INSERT_VALUES, SCATTER_FORWARD, dgtol); CHKERRQ(ierr);
  ierr = VecScatterEnd(g, f_local, INSERT_VALUES, SCATTER_FORWARD, dgtol); CHKERRQ(ierr);
 ierr = VecGetArray(f_local, &uvvals); CHKERRQ(ierr);

  /* need to set the points on RHS corresponding to vertices on the boundary to
     the desired value.  Since we are solving f = 0 , need to give them the values u_b - bc_value */

 /******** Get Context Data ***************/ 

  ierr = ISGetIndices(boundary_inlet, &inlet_indices);CHKERRQ(ierr);
  ierr = ISGetSize(boundary_inlet, &inlet_count); CHKERRQ(ierr);

  ierr = ISGetIndices(vertex_inlet_blocked, &inlet_indices_blocked);CHKERRQ(ierr);
  ierr = ISGetSize(vertex_inlet_blocked, &inlet_blocked_count); CHKERRQ(ierr);

  ierr = ISGetIndices(boundary_outlet, &outlet_indices);CHKERRQ(ierr);
  ierr = ISGetSize(boundary_outlet, &outlet_count); CHKERRQ(ierr);

  ierr = ISGetIndices(vertex_outlet_blocked, &outlet_indices_blocked);CHKERRQ(ierr);
  ierr = ISGetSize(vertex_outlet_blocked, &outlet_blocked_count); CHKERRQ(ierr);

  ierr = ISGetIndices(vertex_wall_blocked, &blocked_wall_indices);CHKERRQ(ierr);
  ierr = ISGetSize(vertex_wall_blocked, &blocked_wall_count); CHKERRQ(ierr);
/* create space for the array of boundary values */
  bvswall = (double*)PetscMalloc(blocked_wall_count*sizeof(double)); CHKPTRQ(bvswall);
  bvsinlet = (double*)PetscMalloc(2*inlet_count*sizeof(double)); CHKPTRQ(bvsinlet);
  bvsoutlet = (double*)PetscMalloc(outlet_count*sizeof(double)); CHKPTRQ(bvsoutlet);
  /* Later reuse this space, by creating once in appload.c */

  /* Decide which bc to apply.  Presssure is Neumann at the wall, Dirichlet at opening.
Velocity is : Zero on wall, Dirichlet at opening.
/* First go throught the local wall indices */
  for(i = 0; i < blocked_wall_count; i++ ){
  bvswall[i] = uvvals[blocked_wall_indices[i]];
    /* pressure is neumann */
    /* need to set these values in a non-blocked way, so that pressure isn't affected */
  }
  ierr = VecSetValuesLocal(f, blocked_wall_count,  blocked_wall_indices, bvswall, INSERT_VALUES); CHKERRQ(ierr);
  PetscFree(bvswall);

/* Now the inlet  indices */
 for( i = 0; i <inlet_count; i++ )
   { 
     xval = grid->vertex_value[2*inlet_indices[i]];
     yval = grid->vertex_value[2*inlet_indices[i]+1];
     bvsinlet[2*i] = uvvals[3*inlet_indices[i]] -  bc1(xval, yval);
     bvsinlet[2*i+1] = uvvals[3*inlet_indices[i]+1] -  bc2(xval, yval);
     /* skip the pressure */
   }
ierr = VecSetValuesLocal(f, inlet_blocked_count, inlet_indices_blocked, bvsinlet, INSERT_VALUES); CHKERRQ(ierr);
PetscFree(bvsinlet);

/* Now the outlet  indices */
 for( i = 0; i <outlet_count; i++ )
   { 
     xval = grid->vertex_value[2*outlet_indices[i]];
     yval = grid->vertex_value[2*outlet_indices[i]+1];
     bvsoutlet[i] = uvvals[3*inlet_indices[i]+2] -  bc3(xval, yval);   
     /* just the pressure */
   }
ierr = VecSetValuesLocal(f, outlet_blocked_count, outlet_indices_blocked, bvsoutlet, INSERT_VALUES); CHKERRQ(ierr);
PetscFree(bvsoutlet);


  /* need to free up the indices */
  ierr = ISRestoreIndices(boundary_inlet,&inlet_indices);CHKERRQ(ierr); 
  ierr = ISRestoreIndices(boundary_outlet,&outlet_indices);CHKERRQ(ierr); 
  ierr = ISRestoreIndices(vertex_outlet_blocked, &outlet_indices_blocked);CHKERRQ(ierr); 
  ierr = ISRestoreIndices(vertex_inlet_blocked, &inlet_indices_blocked);CHKERRQ(ierr); 
  ierr = ISRestoreIndices(vertex_wall_blocked,&blocked_wall_indices);CHKERRQ(ierr);
  ierr = VecRestoreArray(f_local, &uvvals);  CHKERRQ(ierr);
  /********* Assemble Data **************/
 ierr = VecAssemblyBegin(f);CHKERRQ(ierr);
 ierr = VecAssemblyEnd(f);CHKERRQ(ierr);
PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "FormStationaryJacobian"
int FormStationaryJacobian(SNES snes, Vec g, Mat *jac, Mat *B, MatStructure *flag, void *dappctx)
{
  AppCtx *appctx = (AppCtx *)dappctx;
  AppAlgebra *algebra = &appctx->algebra;
  Mat A = algebra->A;
  AppGrid    *grid    = &appctx->grid;

  int ierr;

  /* copy the linear part into jac.*/
/* Mat Copy just zeros jac, and copies in the values.  The blocked structure and ltog is preserved */
ierr= MatCopy(A, *jac);CHKERRQ(ierr);
MatView(*jac, VIEWER_STDOUT_SELF);
  /* the nonlinear part */
  /* Will be putting in lots of values. Check on the nonzero structure.   */
  ierr = SetJacobian(g, appctx, jac);CHKERRQ(ierr);
MatView(*jac, VIEWER_STDOUT_SELF);

  /* Set flag */
  *flag = DIFFERENT_NONZERO_PATTERN;  /*  is this right? */
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
 
/* neighbours of the cell */
  int  *cell_cell = grid->cell_cell;
  /* vertices of the cell (in local numbering) */
  int  *cell_vertex = grid->cell_vertex;
/* number of vertices on this processor */
  int   vertex_n = grid->vertex_n;
 /* number of vertices including ghosted ones */
 int vertex_count= grid->vertex_n_ghosted;
  /* The geometrical values of the vertices */
  double     *vertex_values = grid->vertex_value;
  /* the number of cells on this processor */
  int  cell_n = grid->cell_n;
 
  /* the global to local scatter */
 VecScatter dgtol = algebra->dgtol;
 /* The local vector to work with */
 Vec f_local = algebra->f_local;
/* The index set of the vertices on the boundary */
 IS vertex_boundary_blocked = grid->vertex_boundary_blocked;    
 IS vertex_inlet_blocked = grid->vertex_inlet_blocked;    
 IS vertex_outlet_blocked = grid->vertex_outlet_blocked;    

 IS vertex_wall_blocked = grid->vertex_wall_blocked;    


 /****** Internal Variables ***********/
 int  i,j,k, ierr;
 int    *vert_ptr; 
  double   *uvvals, cell_values[8];
  double values[4*4*2*2];  /* the integral of the combination of phi's */
  double spreadvalues[4*4*3*3]; /* the full size of Jac on an element for DF = 3 */
  
  double coors[8]; /* the coordinates of one element */
 double one = 1.0, zero = 0.0;

  PetscFunctionBegin;
  /* Matrix is set to the linear part already, so just ADD_VALUES the nonlinear part  */
 
  /* The scatter will have the pressure values in it if DF = 3 */
  ierr = VecScatterBegin(g, f_local, INSERT_VALUES, SCATTER_FORWARD, dgtol); CHKERRQ(ierr);
  ierr = VecScatterEnd(g, f_local, INSERT_VALUES, SCATTER_FORWARD, dgtol); CHKERRQ(ierr);
  ierr = VecGetArray(f_local, &uvvals);

  /*   loop over local elements, putting values into matrix -*/
  for ( i=0; i<cell_n; i++ ){
    vert_ptr = cell_vertex + 4*i;   
 
    for ( j=0; j<4; j++) {
      coors[2*j] = vertex_values[2*vert_ptr[j]];
      coors[2*j+1] = vertex_values[2*vert_ptr[j]+1];
#if DF == 2
      cell_values[2*j] = uvvals[2*vert_ptr[j]];
      cell_values[2*j+1] = uvvals[2*vert_ptr[j]+1];
#elif DF == 3
      cell_values[2*j] = uvvals[3*vert_ptr[j]];
      cell_values[2*j+1] = uvvals[3*vert_ptr[j]+1];
#endif

    }
    /* compute the values of basis functions on this element */
    ierr = SetLocalElement(phi, coors);CHKERRQ(ierr);
    /*    Compute the partial derivatives of the nonlinear map    */  
    ierr = ComputeJacobian( phi, cell_values,  values );CHKERRQ(ierr);
#if DF == 3
    SpreadMatrix(values, spreadvalues);
   ierr  = MatSetValuesBlockedLocal(*jac,4,vert_ptr,4,vert_ptr,spreadvalues,ADD_VALUES);CHKERRQ(ierr);
#else
    ierr  = MatSetValuesBlockedLocal(*jac,4,vert_ptr,4,vert_ptr,values,ADD_VALUES);CHKERRQ(ierr);
#endif
  }
  ierr = MatAssemblyBegin(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /********** The process repeats for setting boundary conditions ************/
  /*  -------------------------------------------------------------
         Apply Dirichlet boundary conditions
      -----------------------------------------------------------*/
  /* inlet */
  ierr = MatZeroRowsLocal(*jac,vertex_inlet_blocked,&one);CHKERRQ(ierr);
  /* outlet */
  ierr = MatZeroRowsLocal(*jac,vertex_outlet_blocked,&one);CHKERRQ(ierr);
  /* Wall */
  ierr = MatZeroRowsLocal(*jac,vertex_wall_blocked,&one);CHKERRQ(ierr);

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
  AppEquations *equations = &appctx->equations;
  /* The index set of the vertices on the boundary */
  IS         vertex_boundary = grid->vertex_boundary;
   /* The Variable for the quadrature */
  AppElement *phi = &appctx->element;
  /* The vector we use */
  Vec        b = algebra->b;
  /* the number of cells on this processor */
  int  cell_n = grid->cell_n;
  /* The array of vertices in the local numbering for each cell */
  int        *cell_vertex = grid->cell_vertex;
  /* The geometrical values of the vertices */
  double     *vertex_values = grid->vertex_value;
 /* The number of vertices per cell (4 in the case of billinear) */
  int NV = grid->NV;
  /* extract the rhs functions */
/* DFP f = equations->f; */
/* DFP g = equations->g; */

  /********* Declare Local Variables ******************/
  /* Room to hold the coordinates of a single cell, plus the RHS generated from a single cell.  */
  double coors[4*2]; /* quad cell */
  double values[4*2]; /* number of elements * number of variables */  
#if DF == 3
  double spreadvalues[4*3]; /* in case DF = 3 */
#endif

 int ierr, i, nindices,*vertices, *indices, j;
  double  *bvs, xval, yval;

  /* set flag for element computation */
  phi->dorhs = 1;

/********* The Loop over Elements Begins **************/
  for ( i=0; i<cell_n; i++ )
    {
      vertices = cell_vertex + NV*i;
      /*  Load the cell vertex coordinates */
      for ( j=0; j<4; j++) {
	coors[2*j] = vertex_values[2*vertices[j]];
	coors[2*j+1] = vertex_values[2*vertices[j]+1];    }
      /****** Perform computation ***********/
      /* compute the values of basis functions on this element */
      SetLocalElement(phi, coors);
      
      /* compute the  element load (integral of f with the 4 basis elements)  */
      ComputeRHS( f, g, phi, values );

      /*********  Set Values *************/
#if DF == 3
      SpreadVector(values, spreadvalues);
      ierr = VecSetValuesBlockedLocal(b,NV,vertices,spreadvalues,ADD_VALUES);CHKERRQ(ierr);
#else
      ierr = VecSetValuesBlockedLocal(b,NV,vertices,values,ADD_VALUES);CHKERRQ(ierr);
#endif
    }
  
  /********* Assemble Data **************/
  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);

  /* Boundary conditions can be set by the total (nonlinear) function.   This is just one part */
  PetscFunctionReturn(0);
}


#undef __FUNC__
#define __FUNC__ "AppCxtSetMatrix"
int AppCtxSetMatrix(AppCtx* appctx)
{
/********* Collect contex informatrion ***********/
  AppAlgebra *algebra = &appctx->algebra;
  AppGrid    *grid    = &appctx->grid;
  MPI_Comm   comm = appctx->comm;
  AppEquations *equations = &appctx->equations;
/* The index set of the vertices on the boundary */
  IS         vertex_boundary = grid->vertex_boundary;  /* for blocked calls */
  IS vertex_boundary_blocked = grid->vertex_boundary_blocked;      /* for by hand blocking */
  /* the blocked ltod, for by hand blocking */
  ISLocalToGlobalMapping dltog = grid->dltog;
  /* The Variable for the quadrature */
  AppElement *phi = &appctx->element; 
  /* The number of vertices per cell (4 in the case of billinear) */
  int NV = grid->NV;
 /* the number of cells on this processor */
  int  cell_n = grid->cell_n;
 /* The array of vertices in the local numbering for each cell */
  int        *cell_vertex = grid->cell_vertex;
  /* The geometrical values of the vertices */
  double     *vertex_values = grid->vertex_value;
  /* The number of vertices on this processor */
  int vertex_count = grid->vertex_n_ghosted;
  /* The viscosity */
  double eta = equations->eta;
  
  /* The matrix we are working with */
  Mat        A = algebra->A;

/****** Internal Variables ***********/
  int i,j, ierr;
  int  *vert_ptr;
  int *vertex_boundary_ptr, *vertex_boundary_array;  
  int vertex_boundary_n;

  double one = 1.0, zero = 0.0;
  double values[4*4*2*2];  /* the integral of the combination of phi's */
#if DF == 3
  double spreadvalues[4*4*3*3]; /* in case DF = 3 */
  double pvalues[8*4];
#endif

  double coors[2*4]; /* the coordinates of one element */

  PetscFunctionBegin;
  /************ Set Up **************/ 
  /* set flag for phi computation */
    phi->dorhs = 0;

/********* The Loop over Elements Begins **************/

  for ( i=0; i<cell_n; i++ ) 
    {
      vert_ptr = cell_vertex + NV*i;    
      /*  Load the cell vertex coordinates */
      for ( j=0; j<NV; j++) 
	{
	  coors[2*j] = vertex_values[2*vert_ptr[j]];
	  coors[2*j+1] = vertex_values[2*vert_ptr[j]+1];
	}
      /****** Perform computation ***********/
      /* compute the values of basis functions on this element */
      ierr = SetLocalElement(phi, coors); CHKERRQ(ierr);
      ierr = SetCentrElement(phi, coors); CHKERRQ(ierr);
      /*    Compute the element stiffness    */  
      ierr = ComputeMatrix( phi, values ); CHKERRQ(ierr);
      /*********  Set Values *************/
      ierr = ComputePressure( phi, pvalues ); CHKERRQ(ierr);
      
      SpreadMatrix(values, spreadvalues);
      PopPressure(pvalues, spreadvalues);
      ierr = MatSetValuesBlockedLocal(A,NV,vert_ptr,NV,vert_ptr,spreadvalues,ADD_VALUES);CHKERRQ(ierr);
    }

  /********* Assemble Data **************/
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  /********** Multiply by the viscosity coeff ***************/
  ierr = MatScale(&eta, A);CHKERRQ(ierr);
  /* Boundary conditions are set by the total function. This is just the linear part */
  PetscFunctionReturn(0);
}


int SpreadVector(double a[8], double b[12])
{
  int i;
  for(i=0;i<4;i++){
    b[3*i] = a[2*i];
    b[3*i+1] = a[2*i+1];
    b[3*i + 2] = 0.0;
  }
 PetscFunctionReturn(0);
}
int PopPressure(double M[32], double N[144])
{
  /* M is an 8x4 matrix, first pop each row into the 3rd position of each quarter of
     the first and second each each 3 rows of N.  */
  /* Then take the transpose of M and put it in to the first and second position of 
     the third rows of each 3 rows of N. */
  int i,j;
  for(j=0;j<4;j++){
      for(i=0;i<4;i++){
	/* put in C */
	N[3*12*j + 3*i + 3]          = M[8*j + i];
	N[3*12*j + 12 + 3*i + 3] = M[8*j + 4 + i];
	/* put in C transpose */
	N[3*12*j + 24 + 3*i]        = M[8*i + j ];
	N[3*12*j + 24 + 3*i + 1] = M[8*i + 4 + j];
      }
  }
}

int SpreadMatrix(double M[64], double N[144])
{
  int i,j;
  for(j=0;j<4;j++){
    for(i=0;i<4;i++){
      N[3*12*j + 3*i]      = M[2*8*j + 2*i];
      N[3*12*j + 3*i+1] = M[2*8*j + 2*i+1];
      N[3*12*j + 3*i+2] = 0;
    }
    for(i=0;i<4;i++){
      N[3*12*j + 12 + 3*i]      = M[2*8*j + 8 + 2*i];
      N[3*12*j + 12 + 3*i+1] = M[2*8*j + 8 + 2*i+1];
      N[3*12*j + 12 + 3*i+2] = 0;
    }
    for(i=0;i<4;i++){
      N[3*12*j + 24 + 3*i]      = 0;
      N[3*12*j + 24 + 3*i+1] = 0;
      N[3*12*j + 24 + 3*i+2] = 0;
    }
 
  }
PetscFunctionReturn(0);
}

/* Input Vector is x */
#undef __FUNC__
#define __FUNC__ "FormDynamicFunction"
int FormDynamicFunction(SNES snes, Vec x, Vec f, void *dappctx)
{
/********* Collect context informatrion ***********/
 AppCtx *appctx = (AppCtx *)dappctx;
  AppElement phi = appctx->element;
  AppAlgebra *algebra = &appctx->algebra;
  double dt = appctx->dt;
double idt = 1/dt;
Vec xold = algebra->g;

  /* A is the (already computed) linear part*/
  Mat A = algebra->A;
  /* b is the (already computed) rhs */ 
  Vec  b = algebra->b;
  /* Internal Variables */
  int ierr;  double zero = 0.0, mone = -1.0;

/****** Perform computation ***********/
  /* and input scaled by 1/dt */
  ierr = VecCopy(x, f); CHKERRQ(ierr); 
  ierr = VecScale(f,idt);CHKERRQ(ierr); /* f = input/dt */
  /* subtract old solution  */
  ierr = VecScale(xold,idt);CHKERRQ(ierr); 
  ierr = VecAXPY(f,mone,xold);/* f = f - xold/dt */
  /* add rhs to get constant part */
  ierr = VecAXPY(f,mone,b); CHKERRQ(ierr); /* f = f - 1*b */
  /*apply matrix to the input vector x, to get linear part */
  ierr = MatMultAdd(A, x, f, f); CHKERRQ(ierr);  /* f = A*x + f */
 /* create nonlinear part */
  ierr = SetNonlinearFunction(x, appctx, f);CHKERRQ(ierr); /* f = Nonlin + f */
  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "FormDynamicJacobian"
int FormDynamicJacobian(SNES snes, Vec g, Mat *jac_ptr, Mat *B, MatStructure *flag, void *dappctx)
{
  AppCtx *appctx = (AppCtx *)dappctx;
  AppAlgebra *algebra = &appctx->algebra;
  Mat A = algebra->A;
  AppGrid    *grid    = &appctx->grid;
  double dt = appctx->dt;
double dti;
  int ierr;
  
  /***** Just add a multiple of id matrix to result of formStationaryJacobian *******/
  /* copy the linear part into jac.*/
ierr= MatCopy(A, *jac_ptr);CHKERRQ(ierr);
  /* the nonlinear part */
 /* Will be putting in lots of values. Check on the nonzero structure.   */
  ierr = SetJacobian(g, appctx, jac_ptr);CHKERRQ(ierr);
  /* add scaled Id */
dti = 1/dt;
ierr = MatShift(&dti, *jac_ptr); CHKERRQ(ierr);
  /* Set flag */
  *flag = SAME_NONZERO_PATTERN;  /*  is this right? */
  PetscFunctionReturn(0);
}
