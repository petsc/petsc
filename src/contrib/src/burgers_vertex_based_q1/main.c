
static char help[] ="Solves the 2d burgers equation.   u*du/dx + v*du/dy - c(lap(u)) = f.  u*dv/dv + v*dv/dy - c(lap(v)) =g.  This has exact solution, see fletcher.";


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

  /******* This needs to be fixed.  g is the solution now, not x ******************/
  if (appctx->view.showsomething) {
    ierr = VecScatterBegin(algebra->x,algebra->w_local,INSERT_VALUES,SCATTER_FORWARD,algebra->gtol);CHKERRQ(ierr);
    ierr = VecScatterEnd(algebra->x,algebra->w_local,INSERT_VALUES,SCATTER_FORWARD,algebra->gtol);CHKERRQ(ierr);
    ierr = DrawZoom(appctx->view.drawglobal,AppCtxViewSolution,appctx); CHKERRA(ierr);
  }

  /* Send to  matlab viewer */
  if (appctx->view.matlabgraphics) {
    AppCtxViewMatlab(appctx);
  }

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
  ISLocalToGlobalMapping ltog = grid->ltog;
  int its, ierr, size;
  double zero = 0.0; double onep2 = 1.2345;
  IS                     vertex_global = grid->vertex_global;
  SLES                   sles;
  SNES                   snes;
  PC pc;
  KSP ksp;
  Mat J;  /* Jacobian */
  Vec f;
  Vec g;  /* f is for the nonlinear function evaluation, x is the initial guess, solution */
  
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

  if(0){ ierr = MatView(algebra->A, VIEWER_STDOUT_WORLD);CHKERRQ(ierr); }

  /*     Create the nonlinear solver context  */
  ierr = SNESCreate(PETSC_COMM_WORLD,SNES_NONLINEAR_EQUATIONS,&snes); CHKERRQ(ierr);

  /*      Set function evaluation rountine and vector */
  f = algebra->f;
  ierr = SNESSetFunction(snes,f,FormFunction,(void *)appctx); CHKERRQ(ierr);


  /*      Set Jacobian   */ 
  J = algebra->J;
  ierr = SNESSetJacobian(snes,J,J,FormJacobian,(void *)appctx);CHKERRQ(ierr);
  
  /*      Set Solver Options, could put internal options here      */
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  /* initial guess */
  ierr = FormInitialGuess(appctx);CHKERRQ(ierr); 
  g = algebra->g;
/* printf("the initial guess\n"); */
/*   ierr = VecView(g, VIEWER_STDOUT_WORLD);CHKERRQ(ierr);   */

  /*       Solve the non-linear system  */
  ierr = SNESSolve(snes, PETSC_NULL, g);CHKERRQ(ierr);
  ierr = SNESGetIteratioNumber(snes, &its);CHKERRQ(ierr);

/* printf("the final solution vector\n"); */

/*  ierr = VecView(g, VIEWER_STDOUT_WORLD);CHKERRQ(ierr);   */
printf("the number of its, %d\n", its);

  ierr = SNESDestroy(snes); CHKERRQ(ierr);  
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
    ierr = VecSet(g,onep1); CHKERRQ(ierr);
 PetscFunctionReturn(0);
}

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
/********* Collect context informatrion ***********/
  AppGrid                *grid = &appctx->grid;
  AppAlgebra             *algebra = &appctx->algebra;
  MPI_Comm               comm = appctx->comm;
  /* The local to global mapping */
 ISLocalToGlobalMapping ltog = grid->ltog;
 /* number of vertices on this processor */
  int   vertex_n = grid->vertex_n;
  /* number of vertices including ghosted ones */
 int vertex_n_ghosted = grid->vertex_n_ghosted;
 /* global number of each vertex on the processor */
 IS  vertex_global = grid->vertex_global;
/* blocked global number of each vertex on the processor */
 IS  vertex_global_blocked   = grid->vertex_global_blocked;
 /************* Variables to set ************/
/* global to local mapping for vectors */
 VecScatter    gtol;
 VecScatter    dgtol; /* for the nonlinear funtion */
 Vec x, z;  /* x,z for nonzero structure,*/
 Vec x_local, w_local, z_local; /* used for determining nonzero structure of matriz */
Vec b;  /* b for holding rhs */
Vec f; /* nonlinear function */
Vec g; /*for solution, and initial guess */
 Vec f_local; /* used for nonlinear function */
 /********** Internal Variables **********/
int ierr,its, i;
const int two = 2;

  PetscFunctionBegin;

  /*  Create vector to contain load, nonlinear function, and initial guess  */
  ierr = VecCreateMPI(comm,two*vertex_n,PETSC_DECIDE,&b);CHKERRQ(ierr);
  ierr = VecSetBlockSize(b , two);  CHKERRQ(ierr);
  ierr = VecSetLocalToGlobalMappingBlocked(b,ltog);CHKERRQ(ierr);
 
  /* duplicated vectors inherit the blocking */
  ierr = VecDuplicate(b,&f);/*  the nonlinear function */
  ierr = VecDuplicate(b,&g);/*  the initial guess  */

  /* set it */
  algebra->b       = b;
  algebra->f       = f;
  algebra->g       = g;
 
  ierr = VecCreateSeq(PETSC_COMM_SELF,2*vertex_n_ghosted,&f_local);CHKERRQ(ierr);
  ierr = VecScatterCreate(f,vertex_global_blocked,f_local,0,&dgtol);CHKERRQ(ierr);
  /* for vecscatter, second argument is the IS to scatter.  
 Use the  blocked form created in appload.c */  

  /* set variables */
  algebra->f_local = f_local;
  algebra->dgtol = dgtol;

  /* Create work vectors for MatCreate */
  ierr = VecCreateMPI(comm,vertex_n,PETSC_DECIDE,&x);CHKERRQ(ierr);
  ierr = VecSetLocalToGlobalMapping(x,ltog);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&z);
 
  /* Create local work vectors for MatCreate */
  ierr = VecCreateSeq(PETSC_COMM_SELF,vertex_n_ghosted,&w_local);CHKERRQ(ierr);
  ierr = VecDuplicate(w_local,&x_local);CHKERRQ(ierr);
  ierr = VecDuplicate(w_local,&z_local);CHKERRQ(ierr);
  /* only need to create one scatter, it works on duplicate  vectors */ 
  ierr = VecScatterCreate(x,vertex_global,w_local,0,&gtol);CHKERRQ(ierr);

  /* set variables */
  algebra->x       = x;
  algebra->z       = z;
  algebra->w_local = w_local;
  algebra->x_local = x_local;
  algebra->z_local = z_local;
  algebra->gtol    = gtol;

  PetscFunctionReturn(0);
}

/*
     Creates the sparse matrix (with the correct nonzero pattern) that will
  be later filled with the stiffness matrix
*/
#undef __FUNC__
#define __FUNC__ "AppCxtCreateMatrix"
int AppCtxCreateMatrix(AppCtx* appctx)
{
/********* Collect context informatrion ***********/
  AppAlgebra             *algebra = &appctx->algebra;
  AppGrid                *grid    = &appctx->grid;
  /* these vectors should all have one space per node */ 
 Vec x = algebra->x,  z = algebra->z;
  Vec w_local = algebra->w_local, x_local = algebra->x_local; 
  Vec z_local = algebra->z_local;
  VecScatter             gtol = algebra->gtol;
  MPI_Comm               comm = appctx->comm;
  /* The local to global mapping */
 ISLocalToGlobalMapping ltog = grid->ltog;
/* the blocked ltod, for by hand blocking */
  ISLocalToGlobalMapping dltog = grid->dltog;

 /* number of vertices on this processor */
  int   vertex_n = grid->vertex_n;
  /* number of cells on this processor */
  int    cell_n = grid->cell_n;
  /* neighbours of the cell */
  int  *cell_cell = grid->cell_cell;
  /* vertices of the cell (in local numbering) */
  int  *cell_vertex = grid->cell_vertex;

 /************* Variables to set ************/
  Mat A;
  Mat J;
  
 /********** Internal Variables **********/
  double *sdnz, *sonz;  /* non-zero entries on this processor, non-zero entries off this processor */
   int *onz, *dnz;
   int rank; double srank;  /* copies of the integer variables */
   const int four = 4, two = 2;
   double *procs; 
   double  wght,  zero = 0.0, one = 1.0;
   int   ierr, cproc, i, j;
   int  *cells,*vertices; 

   /* We are doing everything blocked, so just use the vectors with one values
 per vertex, same for one degree of freedom per node, then CreateBlocked.  */
  PetscFunctionBegin;
 /* ------------------------------------------------
      Determine non-zero structure of the matrix 
      --------------------------------------------*/

  /* 1) make proc[] contain the processor number of each ghosted vertex */

  MPI_Comm_rank(comm,&rank);    /* Get the index of this processor */
  srank = rank;
   /* set all values of x to the index of this processor */
  ierr = VecSet(x,srank);CHKERRQ(ierr);           

  /* w_local contains all vertices, including ghosted that this processor uses */
  ierr = VecScatterBegin(x,w_local,INSERT_VALUES,SCATTER_FORWARD,gtol);CHKERRQ(ierr);
  ierr = VecScatterEnd(x,w_local,INSERT_VALUES,SCATTER_FORWARD,gtol);CHKERRQ(ierr);

  /* copy w_local into the array procs */
  ierr = VecGetArray(w_local,&procs);CHKERRQ(ierr);
  /* make an array the size x_local ( total number of vertices, including ghosted) ,
 this is for the elements on this processor */ 
  ierr = VecSet(x_local,zero);CHKERRQ(ierr);   
  ierr = VecGetArray(x_local,&sdnz);CHKERRQ(ierr);  
  /* make an array of appropriate size, for the  vertices off this processor */
  ierr = VecSet(z_local,zero);CHKERRQ(ierr); 
  ierr = VecGetArray(z_local,&sonz);CHKERRQ(ierr);

  /* 2) loop over local elements; count matrix nonzeros */

  /*  For each vertex, we count the number of nonzero entries in the matrix.  This is done by looking at how many other vertices are adjacent,  at least in the current case of billinear elements we only have elements on the vertices.  We compute this efficiently, by looping over cells, the vertices, and weighting with .5 those vertices which are adjacen and have nieghbouring element and so will be counted twice.  For data management purposes we need to know if the elements are on or off - processor, so we put the count into sdnz, or donz respectively.  */

  /* loop over cells */
  for ( i=0; i<cell_n; i++ ) {
    vertices = cell_vertex + four*i;
    cells    = cell_cell   + four*i;
    
    /* loop over vertices */
    for ( j=0; j<four; j += 1 ) {
      cproc = PetscReal(procs[vertices[j]]);
      
      /* 1st neighbor, -adjacent */
      if (cells[j] >= 0) wght = .5; else wght = 1.0;
      if (cproc == procs[vertices[(j+1) % four ]]) { /* on diagonal part */
        sdnz[vertices[j]] += wght;} 
      else { sonz[vertices[j]] += wght;}
      /* 2nd neighbor - diagonally opposite*/
     if (cproc == procs[vertices[(j+2) % four]]) { /* on diagonal part */
        sdnz[vertices[j]] += 1.0;}
     else{ sonz[vertices[j]] += 1.0;}
      /* 3rd neighbor  - adjacent */
      if (cells[(j+3)% four ] >= 0) wght = .5; else wght = 1.0; /* check if it has an adjacent cell */
      if (cproc == procs[vertices[(j+3) % four]]) { /* on diagonal part */
        sdnz[vertices[j]] += wght; } 
      else {sonz[vertices[j]] += wght;}
    }
  }

  /* Put the values back into the vectors */
  ierr = VecRestoreArray(x_local,&sdnz);CHKERRQ(ierr);
  ierr = VecRestoreArray(z_local,&sonz);CHKERRQ(ierr);
  ierr = VecRestoreArray(w_local,&procs);CHKERRQ(ierr);

  /* copy the local values up into x. */
  ierr = VecSet(x,zero);CHKERRQ(ierr);
  ierr = VecScatterBegin(x_local,x,ADD_VALUES,SCATTER_REVERSE,gtol);CHKERRQ(ierr);
  ierr = VecScatterEnd(x_local,x,ADD_VALUES,SCATTER_REVERSE,gtol);CHKERRQ(ierr);
  ierr = VecGetArray(x,&sdnz);CHKERRQ(ierr);
  /* copy the local values up into z. */
  ierr = VecSet(z,zero);CHKERRQ(ierr);
  ierr = VecScatterBegin(z_local,z,ADD_VALUES,SCATTER_REVERSE,gtol);CHKERRQ(ierr);
  ierr = VecScatterEnd(z_local,z,ADD_VALUES,SCATTER_REVERSE,gtol);CHKERRQ(ierr);
  ierr = VecGetArray(z,&sonz);CHKERRQ(ierr);

  /* now copy values into and integer array, adding one for the diagonal entry */
  dnz  = (int *) PetscMalloc((vertex_n+1)*sizeof(int));CHKPTRQ(dnz);
  onz  = (int *) PetscMalloc((vertex_n+1)*sizeof(int));CHKPTRQ(onz);
  for ( i=0; i<vertex_n; i++ ) {
    dnz[i] = 1 + (int) PetscReal(sdnz[i]);
    onz[i] = (int) PetscReal(sonz[i]);
  }
  ierr = VecRestoreArray(x,&sdnz);CHKERRQ(ierr);
  ierr = VecRestoreArray(z,&sonz);CHKERRQ(ierr);

  /* now create the matrix */
  ierr = MatCreateMPIBAIJ(comm, 2, 2*vertex_n,2*vertex_n,PETSC_DETERMINE,PETSC_DETERMINE,0,dnz,0,onz,&A); CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMappingBlocked(A,ltog);CHKERRQ(ierr);

  /* Dupicate the matrix for now.  Later the Jacobian will not have the same nonzero structure  */
   ierr = MatCreateMPIBAIJ(comm, 2, 2*vertex_n,2*vertex_n,PETSC_DETERMINE,PETSC_DETERMINE,0,dnz,0,onz,&J); CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMappingBlocked(J,ltog);CHKERRQ(ierr); 
 ierr = MatSetLocalToGlobalMapping(J,dltog);CHKERRQ(ierr); 

  PetscFree(dnz);
  PetscFree(onz);
  algebra->A = A;
  algebra->J = J;

  PetscFunctionReturn(0);
}

/* FormFunction - Evaluates the nonlinear function, F(x), which is the discretised equations, 
     Input Parameters:
    - the vector x, corresponding to u values at each vertex
    - snes, the SNES context
    - appctx
   Output Parameter:
    - f, the value of the function
*/
#undef __FUNC__
#define __FUNC__ "FormFunction"
int FormFunction(SNES snes, Vec x, Vec f, void *dappctx)
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
/*  printf("input to nonlinear fun) \n");    */
/* ierr = VecView(x, VIEWER_STDOUT_WORLD);CHKERRQ(ierr);    */

  /* need to zero f */
  ierr = VecSet(f,zero); CHKERRQ(ierr); /* don't need to assemble for VecSet */
 
  /* add rhs to get constant part */
  ierr = VecAXPY(f,mone,b); CHKERRQ(ierr); /* this says f = f - 1*b */
/*  printf("zero f, add rhs (should be zero) \n");   */
/*  ierr = VecView(f, VIEWER_STDOUT_WORLD);CHKERRQ(ierr);   */

 /*  printf("input vector to the function \n");   */
/*  ierr = VecView(x, VIEWER_STDOUT_WORLD);CHKERRQ(ierr);   */

  /*apply matrix to the input vector x, to get linear part */
  /* Assuming mattrix doesn't need to be recomputed */
  ierr = MatMultAdd(A, x, f, f); CHKERRQ(ierr);  /* f = A*x + f */

 /* create nonlinear part */
  /* Need to call SetNonlinear on the input vector  */
  ierr = SetNonlinearFunction(x, appctx, f);CHKERRQ(ierr);
 
 if(0){  printf("output of nonlinear fun \n");   
   ierr = VecView(f, VIEWER_STDOUT_WORLD);CHKERRQ(ierr);    }
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
  double     *vertex_values = grid->vertex_value;
 /* The array of vertices in the local numbering for each cell */
  int  *cell_vertex = grid->cell_vertex;
 /* the number of cells on this processor */
  int  cell_n = grid->cell_n;
/* The index set of the vertices on the boundary */
  IS         vertex_boundary = grid->vertex_boundary;

/****** Internal Variables ***********/
  /* need a local vector of size 2*(vertex_n_ghosted)*/
  Vec f_local = algebra->f_local;
  
  double result[8], coors[8];
  double cell_values[8],  *uvvals;
  int ierr, i, j, k;
  int *vertex_ptr;
  int  nindices, *indices;
  double  *bvs, xval, yval;
int size;
 /*  Loop over local elements, extracting the values from g  and add them into f  */

  /* Scatter the input values from the global vector g, to those on this processor */
  ierr = VecScatterBegin(g, f_local, INSERT_VALUES, SCATTER_FORWARD, dgtol); CHKERRQ(ierr);
  ierr = VecScatterEnd(g, f_local, INSERT_VALUES, SCATTER_FORWARD, dgtol); CHKERRQ(ierr);

  /* put the values into an array */
  ierr = VecGetArray(f_local, &uvvals); CHKERRQ(ierr);
 
  /* set a flag in computation of local elements */
  phi->dorhs = 0;
  
  for(i=0;i<cell_n;i++){
    vertex_ptr = cell_vertex + 4*i; 
   for ( j=0; j<4; j++) {
     /*  Need to point to the uvvals associated to the vertices */
      cell_values[2*j] = uvvals[2*vertex_ptr[j]];
      cell_values[2*j+1] = uvvals[2*vertex_ptr[j]+1];
      /* get geometrical coordinates */
      coors[2*j] = vertex_values[2*vertex_ptr[j]];
      coors[2*j+1] = vertex_values[2*vertex_ptr[j]+1];
    }

    /* compute the values of basis functions on this element */
    SetLocalElement(phi, coors);
    /* do the integrals */
    ierr = ComputeNonlinear(phi, cell_values, result);CHKERRQ(ierr);

/*   printf("result of compute nonlinear\n");   */
/*   for(k=0;k<8;k++){ printf("%f\n", result[k]); }   */

    /* put result in */
    ierr = VecSetValuesBlockedLocal(f, 4, vertex_ptr, result, ADD_VALUES);CHKERRQ(ierr);
  }

  ierr = VecAssemblyBegin(f);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(f);CHKERRQ(ierr);
  if(0){
 printf("output of nonlinear fun (before bc imposed)\n");   
 ierr = VecView(f, VIEWER_STDOUT_WORLD);CHKERRQ(ierr); }
  /********** The process repeats for setting boundary conditions ************/
  /*  -------------------------------------------------------------
         Apply Dirichlet boundary conditions
      -----------------------------------------------------------*/  
  /* need to set the points on RHS corresponding to vertices on the boundary to
     the desired value.  Since we are solving f = 0 , need to give them the values u_b - bc_value */
 
  /******** Get Context Data ***************/ 
  ierr = ISGetIndices(vertex_boundary, &indices);CHKERRQ(ierr);
  ierr = ISGetSize(vertex_boundary, &nindices); CHKERRQ(ierr);

  /* create space for the array of boundary values */
  bvs = (double*)PetscMalloc(2*(nindices+1)*sizeof(double)); CHKPTRQ(bvs);

 /****** Perform computation ***********/
   
  for( i = 0; i < nindices; i++ ){
    /* get the vertex_value corresponding to element of indices
       then evaluate bc(vertex value) and put this in bvs(i) */
    xval = grid->vertex_value[2*indices[i]];
    yval = grid->vertex_value[2*indices[i]+1];

    bvs[2*i] = uvvals[2*indices[i]] - bc1(xval, yval);
    bvs[2*i+1] = uvvals[2*indices[i]+1] - bc2(xval, yval);
  }

 /*********  Set Values *************/
  ierr = VecSetValuesBlockedLocal(f, nindices, indices, bvs, INSERT_VALUES);CHKERRQ(ierr);
  PetscFree(bvs);
  ierr = ISRestoreIndices(vertex_boundary,&indices);CHKERRQ(ierr);
  ierr = VecRestoreArray(f_local, &uvvals);  CHKERRQ(ierr);

  /********* Assemble Data **************/
 ierr = VecAssemblyBegin(f);CHKERRQ(ierr);
 ierr = VecAssemblyEnd(f);CHKERRQ(ierr);
 
PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "FormJacobian"
int FormJacobian(SNES snes, Vec g, Mat *jac, Mat *B, MatStructure *flag, void *dappctx)
{
  AppCtx *appctx = (AppCtx *)dappctx;
  AppAlgebra *algebra = &appctx->algebra;
  Mat A = algebra->A;
  AppGrid    *grid    = &appctx->grid;

  int ierr;

  /* copy the linear part into jac.*/
/* Mat Copy just zeros jac, and copies in the values.  The blocked structure and ltog is preserved */
ierr= MatCopy(A, *jac);CHKERRQ(ierr);

  /* the nonlinear part */
  /* Will be putting in lots of values. Check on the nonzero structure.   */
  ierr = SetJacobian(g, appctx, jac);CHKERRQ(ierr);

  /* Set flag */
  *flag = DIFFERENT_NONZERO_PATTERN;  /*  is this right? */

/*  printf("about to view jac from insize form jacobian \n");  */
/*    ierr = MatView(*jac, VIEWER_STDOUT_WORLD);CHKERRQ(ierr);  */
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

/****** Internal Variables ***********/
  int  i,j,k, ierr;
  int    *vert_ptr; 
  double   *uvvals, cell_values[8];
  double values[4*4*2*2];  /* the integral of the combination of phi's */
  double coors[8]; /* the coordinates of one element */
 double one = 1.0, zero = 0.0;

  PetscFunctionBegin;
  /* Matrix is set to the linear part already, so just ADD_VALUES the nonlinear part  */
 
  ierr = VecScatterBegin(g, f_local, INSERT_VALUES, SCATTER_FORWARD, dgtol); CHKERRQ(ierr);
  ierr = VecScatterEnd(g, f_local, INSERT_VALUES, SCATTER_FORWARD, dgtol); CHKERRQ(ierr);
  ierr = VecGetArray(f_local, &uvvals);

  /*   loop over local elements, putting values into matrix -*/
  for ( i=0; i<cell_n; i++ ){
    vert_ptr = cell_vertex + 4*i;   
 
    for ( j=0; j<4; j++) {
      coors[2*j] = vertex_values[2*vert_ptr[j]];
      coors[2*j+1] = vertex_values[2*vert_ptr[j]+1];

      cell_values[2*j] = uvvals[2*vert_ptr[j]];
      cell_values[2*j+1] = uvvals[2*vert_ptr[j]+1];
    }
    /* compute the values of basis functions on this element */
    ierr = SetLocalElement(phi, coors);CHKERRQ(ierr);
    /*    Compute the partial derivatives of the nonlinear map    */  
    ierr = ComputeJacobian( phi, cell_values,  values );CHKERRQ(ierr);

    /*  Set the values in the matrix */
 /*    printf("values computed\n"); */
/*     for ( k=0; k<8*8; k++) {printf("%e\n", values[k]);} */
    ierr  = MatSetValuesBlockedLocal(*jac,4,vert_ptr,4,vert_ptr,values,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /********** The process repeats for setting boundary conditions ************/
  /*  -------------------------------------------------------------
         Apply Dirichlet boundary conditions
      -----------------------------------------------------------*/

  ierr = MatZeroRowsLocalIS(*jac,vertex_boundary_blocked,one);CHKERRQ(ierr);
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
  int NVs = grid->NVs;
  /* extract the rhs functions */
/* DFP f = equations->f; */
/* DFP g = equations->g; */

  /********* Declare Local Variables ******************/
  /* Room to hold the coordinates of a single cell, plus the RHS generated from a single cell.  */
  double coors[4*2]; /* quad cell */
  double values[4*2]; /* number of elements * number of variables */  
  int ierr, i, nindices,*vertices, *indices, j;
  double  *bvs, xval, yval;

  /* set flag for element computation */
  phi->dorhs = 1;

/********* The Loop over Elements Begins **************/
  for ( i=0; i<cell_n; i++ )
    {
      vertices = cell_vertex+NVs*i;
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
      ierr = VecSetValuesBlockedLocal(b,NVs,vertices,values,ADD_VALUES);CHKERRQ(ierr);
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
  /* The number of degrees of freedom at each Vertex */
  int DFs = 2;
  /* The number of vertices per cell (4 in the case of billinear) */
  int NVs = grid->NVs;
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
  double coors[2*4]; /* the coordinates of one element */

  PetscFunctionBegin;
  /************ Set Up **************/ 
  /* set flag for phi computation */
    phi->dorhs = 0;
  
/********* The Loop over Elements Begins **************/

  for ( i=0; i<cell_n; i++ ) {
    vert_ptr = cell_vertex + NVs*i;    
    /*  Load the cell vertex coordinates */
    for ( j=0; j<NVs; j++) {
      coors[2*j] = vertex_values[2*vert_ptr[j]];
      coors[2*j+1] = vertex_values[2*vert_ptr[j]+1];
    }
 /****** Perform computation ***********/
    /* compute the values of basis functions on this element */
    ierr = SetLocalElement(phi, coors); CHKERRQ(ierr);
    /*    Compute the element stiffness    */  
    ierr = ComputeMatrix( phi, values ); CHKERRQ(ierr);
  /*********  Set Values *************/
    ierr     = MatSetValuesBlockedLocal(A,NVs,vert_ptr,NVs,vert_ptr,values,ADD_VALUES);CHKERRQ(ierr);
  }

 /********* Assemble Data **************/
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  /********** Multiply by the viscosity coeff ***************/
  ierr = MatScale(A,eta);CHKERRQ(ierr);
/* Boundary conditions are set by the total function. This is just the linear part */
  PetscFunctionReturn(0);
}



