
static char help[] ="Solves the 2d burgers equation.   u*du/dx + v*du/dy - c(lap(u)) = f.  u*dv/dv + v*dv/dy - c(lap(v)) =g.  This has exact solution, see fletcher.";

#include "appctx.h"

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
  ierr = VecGetArray(algebra->x,&values); CHKERRQ(ierr);
  cell_n = grid->cell_n;

  /*      Visualize solution   */
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
  double zero = 0.0;
  IS                     vertex_global = grid->vertex_global;
  SLES                   sles;
  SNES                   snes;
  PC pc;
  KSP ksp;
  Mat J;
  Vec f, g, x;
  
  PetscFunctionBegin;

  /*        Create vector to contain load and various work vectors  */
  ierr = AppCtxCreateVector(appctx); CHKERRQ(ierr);

  /*      Create the sparse matrix, with correct nonzero pattern  */
  ierr = AppCtxCreateMatrix(appctx); CHKERRQ(ierr);

  /*     Set the quadrature values for the reference square element  */
  ierr = AppCtxSetElement(appctx);CHKERRQ(ierr);

  /*      Set the right hand side values into the vectors   */
  ierr = AppCtxSetRhs(appctx); CHKERRQ(ierr);

  /*      Set the matrix entries   */
  ierr = AppCtxSetMatrix(appctx); CHKERRQ(ierr);

  /*     Create the nonlinear solver context  */
  ierr = SNESCreate(PETSC_COMM_WORLD,SNES_NONLINEAR_EQUATIONS,&snes); CHKERRQ(ierr);

  /*      Set function evaluation rountine and vector */
  f = algebra->f;
  ierr = SNESSetFunction(snes,f,FormFunction,(void *)appctx); CHKERRQ(ierr);

  /***** Forgot to create matrix J.  Later will need nonzero strucure  ************/
  ierr = VecGetSize(f, &size); CHKERRQ(ierr);
  ierr = MatCreate(comm, &J); CHKERRQ(ierr);
  ierr = MatSetSizes(J, PETSC_DECIDE, PETSC_DECIDE, size, size); CHKERRQ(ierr);

  /*      Set Jacobian   */
  ierr = SNESSetJacobian(snes,J,J,FormJacobian,(void *)appctx);CHKERRQ(ierr);
  
  /*      Set Solver Options, could put internal options here      */
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  /* initial guess */
  /* Instead call FormInitialGuess  */

  x = algebra->x;
  ierr = VecSet(x,zero); CHKERRQ(ierr);

printf("about to call the solver\n");

  /*       Solve the non-linear system  */
 ierr = SNESSolve(snes, PETSC_NULL, x);CHKERRQ(ierr);
  ierr = SNESGetIteratioNumber(snes, &its);CHKERRQ(ierr);
  
  ierr = SNESDestroy(snes); CHKERRQ(ierr);  

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
 
  int ierr;
  double zero = 0.0, mone = -1.0;

  AppCtx *appctx = (AppCtx *)dappctx;
  AppElement phi = appctx->element;
  AppAlgebra *algebra = &appctx->algebra;
  Mat A = algebra->A;
  Vec  b = algebra->b;

  /* need to zero f */
  ierr = VecSet(f,zero); CHKERRQ(ierr); /* don't need to assemble for VecSet */

  /* create nonlinear part */
/* Need to call SetNonlinear on the input vector  */
  ierr = SetNonlinear(x, appctx, f);CHKERRQ(ierr);

 /* add rhs to get constant part */
  ierr = VecAXPY(f,mone,b); CHKERRQ(ierr); /* this says f = f + 1*b */

  /*apply matrix to the input vector x, to get linear part */
  /* Assuming mattrix doesn't need to be recomputed */
  ierr = MatMultAdd(A, x, f, f); CHKERRQ(ierr);  /* f = A*x + f */
printf("about to view the vector f\n");
ierr = VecView(f, VIEWER_STDOUT_WORLD);CHKERRQ(ierr); 

PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "FormJacobian"
int FormJacobian(SNES snes, Vec x, Mat *jac, Mat *B, MatStructure *flag,
void *dappctx)
{
  AppCtx *appctx = (AppCtx *)dappctx;
  AppAlgebra *algebra = &appctx->algebra;
  Mat A = algebra->A;
  int ierr;

  /* put in the linear part */
  /* could have a flag later to recompute this */
    ierr = MatConvert(A, MATSAME, MAT_INITIAL_MATRIX, jac);CHKERRQ(ierr);
    ierr = MatConvert(A, MATSAME, MAT_INITIAL_MATRIX, B);CHKERRQ(ierr); 

  /* the nonlinear part */
  /* Will be putting in lots of values. Think about changing the structure.  Check with MatConvert */
  ierr = SetJacobian(x, appctx, jac);CHKERRQ(ierr);





  *flag = DIFFERENT_NONZERO_PATTERN;
printf("about to view jac from insize form jacobian \n");
  ierr = MatView(*jac, VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* for now don't worry about B */

  PetscFunctionReturn(0);
}


#undef __FUNC__
#define __FUNC__ "SetJacobian"
int SetJacobian(Vec x, AppCtx *appctx, Mat* jac)
{

  AppAlgebra *algebra = &appctx->algebra;
  AppGrid    *grid    = &appctx->grid;
  int cell_n = grid->cell_n; 
  int i,j,k, ierr;
  int vertex_n = grid->vertex_n;
  int *vert_ptr;
  int *cell_vertex = grid->cell_vertex;
  int        *cell_cell = grid->cell_cell;
  AppElement *phi = &appctx->element;
  double  *vertex_values = grid->vertex_value *uvvals;
  int vertex_count = grid->vertex_n_ghosted;
  double values[4*4*2*2];  /* the integral of the combination of phi's */
  double coors[4][2]; /* the coordinates of one element */
 VecScatter gtol = algebra->gtol;
  Vec w_local = algebra->w_local;

  PetscFunctionBegin;
  /* Matrix is set to the linear part already, so just ADD_VALUES the nonlinear part  */

  
ierr = VecScatterBegin(x, w_local, INSERT_VALUES, SCATTER_FORWARD, gtol); CHKERRQ(ierr);
ierr = VecScatterEnd(x, w_local, INSERT_VALUES, SCATTER_FORWARD, gtol); CHKERRQ(ierr);

ierr = VecGetArray(w_local, &uvvals);


/*   loop over local elements, putting values into matrix -*/

  for ( i=0; i<cell_n; i++ ) {
    vert_ptr = cell_vertex + 4*i;   
    value_ptr = uvvals + 8*i;
    for ( j=0; j<4; j++) {
      coors[j][0] = vertex_values[2*vert_ptr[j]];
      coors[j][1] = vertex_values[2*vert_ptr[j]+1];
    }
    /*    Compute the partial derivatives of the nonlinear map    */  
    ierr = ComputeJacobian( coors, phi, value_ptr,  values );CHKERRQ(ierr);

    ierr     = MatSetValuesBlockedLocal(*jac,4,vert_ptr,4,vert_ptr,values,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
}

#undef __FUNC__
#define __FUNC__ "SetNonlinear"
/* input vector is x, output is f.  Loop over elements, getting coords of each vertex and 
computing load vertex by vertex.  Set the values into f.  */
int SetNonlinear(Vec x, AppCtx *appctx, Vec f)
{

  AppElement *phi = &appctx->element;
  AppAlgebra *algebra = &appctx->algebra;
  AppGrid *grid = &appctx->grid;
  VecScatter gtol = algebra->gtol;
  Vec w_local = algebra->w_local;
  double result[8], *value_ptr, coors[4][2], *vertex_values = grid->vertex_value,  *uvvals;
  int ierr, i, j, cell_n = grid->cell_n, *vertex_ptr, *cell_vertex = grid->cell_vertex;

  /*  Loop over local elements, extracting the values from x  and add them into f  */
  
ierr = VecScatterBegin(x, w_local, INSERT_VALUES, SCATTER_FORWARD, gtol); CHKERRQ(ierr);
ierr = VecScatterEnd(x, w_local, INSERT_VALUES, SCATTER_FORWARD, gtol); CHKERRQ(ierr);

ierr = VecGetArray(w_local, &uvvals);

  for(i=0;i<cell_n;i++){
    vertex_ptr = cell_vertex + 4*i; 
    value_ptr = uvvals + 8*i;
    for ( j=0; j<4; j++) {
      coors[j][0] = vertex_values[2*vert_ptr[j]];
      coors[j][1] = vertex_values[2*vert_ptr[j]+1];
    }
    ierr = ComputeNonlinear(coors, phi, value_ptr, result);CHKERRQ(ierr);
    /* put result in */
    ierr = VecSetValuesBlockedLocal(f, 4, vertex_ptr, result, ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(f);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(f);CHKERRQ(ierr);

}
#undef __FUNC__
#define __FUNC__ "ComputeJacobian"

/* input is x, output the nonlinear part into f for a particulat element */
/* Much of the code is dublicated from ComputeMatrix; the integral is different */
int ComputeJacobian(double coords[4][2], AppElement *phi, double *uvvals, double *result)
{
  /*
    Dh        Jacobian of h at each of 4 gauss pts
    Dhinv     its inverse
    phi_x, phi_y  the partials (of each basis fn at each point)
    detDh     determinant of Dh
    gauss     the image of each gauss pt
    u         the 4 values of u
    v         the 4 values of v
  */
  
  int i,j,k,ii ;
  double Dh[4][2][2], Dhinv[4][2][2],  phi_x[4][4],  phi_y[4][4], detDh[4], gauss[4][2];
  double u[4],v[4];

  /* copy array into more convenient form */
  for(i=0;i<4;i++){
    u[i] = uvvals[2*i]; v[i] = uvvals[2*i+1];
  }

  /* the image of the reference element is given by sum (coord i)*phi_i */
  for(j=0;j<4;j++){ /* loop over points */
    gauss[j][0] = 0; gauss[j][1] = 0;
    for( k=0;k<4;k++ ){
      gauss[j][0] += coords[k][0]*phi->Values[k][j];
      gauss[j][1] += coords[k][1]*phi->Values[k][j];
    }
  }

  /* Jacobian */
  for(i=0;i<4;i++){ /* loop over Gauss points */
    Dh[i][0][0] = 0; Dh[i][0][1] = 0; Dh[i][1][0] = 0; Dh[i][1][1] = 0;
    for(k=0; k<4; k++ ){
      Dh[i][0][0] += coords[k][0]*phi->DxValues[k][i];
      Dh[i][0][1] += coords[k][0]*phi->DyValues[k][i];
      Dh[i][1][0] += coords[k][1]*phi->DxValues[k][i];
      Dh[i][1][0] += coords[k][1]*phi->DyValues[k][i];    
    }
  }

  /* Determinant of the Jacobian */
  for( j=0; j<4; j++){   /* loop over Gauss points */
    detDh[j] = Dh[j][0][0]*Dh[j][1][1] - Dh[j][0][1]*Dh[j][1][0];
  }

  /* Inverse of the Jacobian */
  for( j=0; j<4; j++){   /* loop over Gauss points */
    Dhinv[j][0][0] = Dh[j][1][1]/detDh[j];
    Dhinv[j][0][1] = -Dh[j][0][1]/detDh[j];
    Dhinv[j][1][0] = -Dh[j][1][0]/detDh[j];
    Dhinv[j][1][1] = Dh[j][0][0]/detDh[j];
  }

  /* Notice that phi~ = phi(h), so Dphi~ = Dphi*Dh, so Dphi~ = Dphi*(Dh)inv.
 
  /* partial of phi at h(gauss pt) times Dhinv */
  /* loop over gauss, the basis fns, then d/dx or d/dy */
  for( i=0;i<4;i++ ){  /* loop over Gauss points */
    for( j=0;j<4;j++ ){ /* loop over basis functions */
      phi_x[i][j] = phi->DxValues[j][i]*Dhinv[i][0][0] +  phi->DyValues[j][i]*Dhinv[i][1][0];
      phi_y[i][j] = phi->DxValues[j][i]*Dhinv[i][0][1] + phi->DyValues[j][i]*Dhinv[i][1][1];
    }
  }

  /* INTEGRAL */
  /* Figure this part out, and look at MatComputValuse to see the indexing.  */
************
Right here
************

   PetscFunctionReturn(0);
}


#undef __FUNC__
#define __FUNC__ "ComputeNonlinear"

/* input is x, output the nonlinear part into f for a particulat element */
/* Much of the code is dublicated from ComputeMatrix; the integral is different */
int ComputeNonlinear(double coords[4][2], AppElement *phi, double *uvvals, double *result)
{
  /*
    Dh        Jacobian of h at each of 4 gauss pts
    Dhinv     its inverse
    phi_x, phi_y  the partials (of each basis fn at each point)
    detDh     determinant of Dh
    gauss     the image of each gauss pt
    u         the 4 values of u
    v         the 4 values of v
  */
  
  int i,j,k,ii ;
  double Dh[4][2][2], Dhinv[4][2][2],  phi_x[4][4],  phi_y[4][4], detDh[4], gauss[4][2];
  double u[4],v[4];

  /* copy array into more convenient form */
  for(i=0;i<4;i++){
    u[i] = uvvals[2*i]; v[i] = uvvals[2*i+1];
  }

  /* the image of the reference element is given by sum (coord i)*phi_i */
  for(j=0;j<4;j++){ /* loop over points */
    gauss[j][0] = 0; gauss[j][1] = 0;
    for( k=0;k<4;k++ ){
      gauss[j][0] += coords[k][0]*phi->Values[k][j];
      gauss[j][1] += coords[k][1]*phi->Values[k][j];
    }
  }

  /* Jacobian */
  for(i=0;i<4;i++){ /* loop over Gauss points */
    Dh[i][0][0] = 0; Dh[i][0][1] = 0; Dh[i][1][0] = 0; Dh[i][1][1] = 0;
    for(k=0; k<4; k++ ){
      Dh[i][0][0] += coords[k][0]*phi->DxValues[k][i];
      Dh[i][0][1] += coords[k][0]*phi->DyValues[k][i];
      Dh[i][1][0] += coords[k][1]*phi->DxValues[k][i];
      Dh[i][1][0] += coords[k][1]*phi->DyValues[k][i];    
    }
  }

  /* Determinant of the Jacobian */
  for( j=0; j<4; j++){   /* loop over Gauss points */
    detDh[j] = Dh[j][0][0]*Dh[j][1][1] - Dh[j][0][1]*Dh[j][1][0];
  }

  /* Inverse of the Jacobian */
  for( j=0; j<4; j++){   /* loop over Gauss points */
    Dhinv[j][0][0] = Dh[j][1][1]/detDh[j];
    Dhinv[j][0][1] = -Dh[j][0][1]/detDh[j];
    Dhinv[j][1][0] = -Dh[j][1][0]/detDh[j];
    Dhinv[j][1][1] = Dh[j][0][0]/detDh[j];
  }

  /* Notice that phi~ = phi(h), so Dphi~ = Dphi*Dh, so Dphi~ = Dphi*(Dh)inv.
 
  /* partial of phi at h(gauss pt) times Dhinv */
  /* loop over gauss, the basis fns, then d/dx or d/dy */
  for( i=0;i<4;i++ ){  /* loop over Gauss points */
    for( j=0;j<4;j++ ){ /* loop over basis functions */
      phi_x[i][j] = phi->DxValues[j][i]*Dhinv[i][0][0] +  phi->DyValues[j][i]*Dhinv[i][1][0];
      phi_y[i][j] = phi->DxValues[j][i]*Dhinv[i][0][1] + phi->DyValues[j][i]*Dhinv[i][1][1];
    }
  }

  /* INTEGRAL */
  /* terms are u*du/dx + v*du/dy, u*dv/dx + v*dv/dy */
  /* Go element by element.  
Compute 
 u_i * phi_i * u_j * phi_j_x + v_i*phi_i*u_j*phi_j_y * phi_k
and
 u_i * phi_i * v_j * phi_j_x + v_i*phi_i*v_j*phi_j_y * phi_k.

Put the result in index k.  Add all possibilities up to get contribution to k, and loop over k.
*/

/* Can exploit a little symetry to cut iterations from 4*4*4 to 2*4*4  */

   for( k=0;k<4;k++ ){ /* loop over first basis fn */
     result[2*k] = 0; result[2*k+1] = 0;
     for( i=0; i<4; i++){ /* loop over second */
       for( j=0; j<4; j++){/* loop over third */
	 for(ii=0;ii<4;ii++){ /* loop over gauss points */
	 result[2*k] += 
	   (u[i]*u[j]*phi->Values[i][ii]*phi_x[j][ii] +
	    v[i]*v[j]*phi->Values[i][ii]*phi_y[j][ii])*phi->Values[k][ii]*PetscAbsDouble(detDh[ii]); 
	 result[2*k+1] +=
	   (u[i]*v[j]*phi->Values[i][ii]*phi_x[j][ii] +
	    v[i]*v[j]*phi->Values[i][ii]*phi_y[j][ii])*phi->Values[k][ii]*PetscAbsDouble(detDh[ii]);
	 }
       }
     }
   }
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
  AppGrid                *grid = &appctx->grid;
  AppAlgebra             *algebra = &appctx->algebra;
  MPI_Comm               comm = appctx->comm;
  Vec                    f,g,x,b,x_local,w_local,z,z_local;
  ISLocalToGlobalMapping ltog = grid->ltog;
  int                    vertex_n = grid->vertex_n,vertex_n_ghosted = grid->vertex_n_ghosted,ierr,its;
  VecScatter             gtol;
  IS                     vertex_global = grid->vertex_global;
  SLES                   sles;
  const int two = 2;
  PetscFunctionBegin;

  /*        Create vector to contain load and various work vectors  */
  ierr = VecCreateMPI(comm,two*vertex_n,PETSC_DECIDE,&b);CHKERRQ(ierr);
  ierr = VecSetBlockSize(b , two);  CHKERRQ(ierr);
  ierr = VecSetLocalToGlobalMappingBlocked(b,ltog);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&x);
  ierr = VecDuplicate(b,&z);
  ierr = VecDuplicate(b,&f);
  ierr = VecDuplicate(b,&g);


 /*   ierr = VecCreateMPI(comm, two*vertex_n, PETSC_DECIDE, &f);CHKERRQ(ierr); */
/*   ierr = VecDuplicate(f,&g); */


  ierr = VecCreateSeq(PETSC_COMM_SELF,vertex_n_ghosted,&w_local);CHKERRQ(ierr);
  ierr = VecDuplicate(w_local,&x_local);CHKERRQ(ierr);
  ierr = VecDuplicate(w_local,&z_local);CHKERRQ(ierr);
  ierr = VecScatterCreate(b,vertex_global,w_local,0,&gtol);CHKERRQ(ierr);

  algebra->x       = x;
  algebra->b       = b;
  algebra->z       = z;
  algebra->f       = f;
  algebra->g       = g;
  algebra->w_local = w_local;
  algebra->x_local = x_local;
  algebra->z_local = z_local;
  algebra->gtol    = gtol;

  PetscFunctionReturn(0);
}


#undef __FUNC__
#define __FUNC__ "AppCxtSetRhs"
int AppCtxSetRhs(AppCtx* appctx)
{
  AppGrid    *grid = &appctx->grid;
  AppAlgebra *algebra = &appctx->algebra;
  IS         vertex_boundary = grid->vertex_boundary;
  AppElement *phi  = &appctx->element;
  int        ierr, i, nindices, cell_n = grid->cell_n;
  const int two = 2; /* number of variables we carry */
  const int four = 4; /* square element - four sides  */
  int        *cell_vertex = grid->cell_vertex,*vertices, *indices, j;
  double     *vertex_values = grid->vertex_value, *bvs, xval, yval;
  Vec        b = algebra->b;

  /*      Room to hold the coordinates of a single cell, plus the 
     RHS generated from a single cell.  */

  double coors[4][2]; /* quad cell */
  double values[8]; /* number of elements * number of variables */  


  /*     Loop over elements computing load one element at a time 
        and putting into right-hand-side-*/
  for ( i=0; i<cell_n; i++ ) {
    vertices = cell_vertex+four*i;

    /*        Load the cell vertex coordinates     */
    for ( j=0; j<four; j++) {
      coors[j][0] = vertex_values[2*vertices[j]];
      coors[j][1] = vertex_values[2*vertices[j]+1];    }

    /* compute the  element load (integral of f with the 4 basis elements)  */
     ComputeRHS( coors, f, g, phi, values );

    ierr = VecSetValuesBlockedLocal(b,four,vertices,values,ADD_VALUES);CHKERRQ(ierr);
  }

  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
/*   printf("View the vector b"); */
/*    ierr = VecView(b,VIEWER_STDOUT_WORLD);CHKERRQ(ierr);   */
 
  /*  -------------------------------------------------------------
         Apply Dirichlet boundary conditions
      -----------------------------------------------------------*/
  
  /* need to set the points on RHS corresponding to vertices on the boundary to
     the desired value.  In this case, if the b.c. is zero, then set these
     values to zero */
  
  ierr = ISGetIndices(vertex_boundary, &indices);CHKERRQ(ierr);
  ierr = ISGetSize(vertex_boundary, &nindices); CHKERRQ(ierr);

  bvs = (double*)PetscMalloc(two*(nindices+1)*sizeof(double)); CHKPTRQ(bvs);

  /* if I want different bc, put in later  */

  for( i = 0; i < nindices; i++ ){
    /* get the vertex_value corresponding to element of indices
       then evaluate bc(vertex value) and put this in bvs(i) */
    xval = grid->vertex_value[2*indices[i]];
    yval = grid->vertex_value[2*indices[i]+1];
    /*printf("xval = %f, yval = %f \n", xval, yval);*/
    bvs[2*i] = bc(xval, yval);
    bvs[2*i+1] = bc(xval, yval);
      }

  ierr = VecSetValuesBlockedLocal(b, nindices, indices, bvs, INSERT_VALUES);CHKERRQ(ierr);
  PetscFree(bvs);
  ierr = ISRestoreIndices(vertex_boundary,&indices);CHKERRQ(ierr);
 
  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);

  printf("View the vector b, bc applied"); 

    ierr = VecView(b, VIEWER_STDOUT_WORLD);   
  
  PetscFunctionReturn(0);
}


void ComputeRHS(double coords[4][2], DFP f, DFP g, AppElement *phi, double *integrals){
  int i,j,k;
  double detDh[4];     /* the determinant of the jacobian */
  double Dh[4][2][2];   /* The Jacobian of h at each of 4 Gauss pts */
  double gauss[4][2];   /* The image of the gauss points in the given quad */

  /* The image of the reference element is given by sum (coord i)*phi_i */
  for (j = 0; j < 4; j++ ){ /* loop over the points */
    gauss[j][0] = 0;
    gauss[j][1] = 0;
    for ( k = 0; k<4; k++ ){
       gauss[j][0] += coords[k][0]*phi->Values[k][j]; 
       gauss[j][1] += coords[k][1]*phi->Values[k][j]; 
    }
  }
 
/* Need to compute the determinant of the Jacobian of the map at each point */
/*(sum(x_i)*d(phi_i)/dx)*(sum(y_i)*d(phi_i)/dy) - sum(y d/dx)*(sum(x d/dy) */

  /* Jacobian */
  for (i = 0; i < 4; i++){  /* loop over the points */
   Dh[i][0][0] = 0.0; Dh[i][0][1] = 0.0;
   Dh[i][1][0] = 0.0; Dh[i][1][1] = 0.0;
   for (k = 0; k < 4; k++){ /* loop for the sum */
     Dh[i][0][0] += coords[k][0]*phi->DxValues[k][i];
     Dh[i][0][1] += coords[k][0]*phi->DyValues[k][i];
     Dh[i][1][0] += coords[k][1]*phi->DxValues[k][i];
     Dh[i][1][1] += coords[k][1]*phi->DyValues[k][i];
   }
  }

  /* Determinant of the Jacobian */
  for( j = 0; j < 4; j++ ){
   detDh[j] = Dh[j][0][0]*Dh[j][1][1] - Dh[j][0][1]*Dh[j][1][0];
   /*    printf("detDh %f\n", detDh[j]); */
  };
 

 /* integral over given element of f()*Phi_j =
        integral over reference element of f()*phi_j*Det(JacH),
      where Phi_j is the basis function on the given element, and 
            phi_j is the reference basis element.
    Compute the second integral by quadrature, which gives,
  ~= sum over gauss points of
         f(image of gauss point)*phi_j(gauss point)*Det(JacH)
   The Gaussian quadrature should be exact to second (third?) order
 */ 

  /* need to go over each element , then each variable */
 for( i = 0; i < 4; i++ ){ /* loop over basis functions */
   integrals[2*i] = 0.0; 
   integrals[2*i+1] = 0.0; 
   for( j = 0; j < 4; j++ ){ /* loop over Gauss points */
     integrals[2*i] +=  f(gauss[j][0], gauss[j][1])*PetscAbsDouble(detDh[j])*(phi->Values[i][j]);
     integrals[2*i+1] +=  f(gauss[j][0], gauss[j][1])*PetscAbsDouble(detDh[j])*(phi->Values[i][j]);
   }
 }
}

int AppCtxSetElement(AppCtx* appctx){

  AppElement *phi = &appctx->element;
  double psi, psi_m, psi_p, psi_pp, psi_mp, psi_pm, psi_mm;

  psi = sqrt(3.0)/3.0;
  psi_p = 0.25*(1.0 + psi);   psi_m = 0.25*(1.0 - psi);
  psi_pp = 0.25*(1.0 + psi)*(1.0 + psi);  psi_pm = 0.25*(1.0 + psi)*(1.0 - psi); 
  psi_mp = 0.25*(1.0 - psi)*(1.0 + psi);  psi_mm = 0.25*(1.0 - psi)*(1.0 - psi);

phi->Values[0][0] = psi_pp; phi->Values[0][1] = psi_pm;phi->Values[0][2] = psi_mm;
phi->Values[0][3] = psi_mp;phi->Values[1][0] = psi_mp; phi->Values[1][1] = psi_pp;
phi->Values[1][2] = psi_pm;phi->Values[1][3] = psi_mm;phi->Values[2][0] = psi_mm; 
phi->Values[2][1] = psi_pm;phi->Values[2][2] = psi_pp;phi->Values[2][3] = psi_mp;
phi->Values[3][0] = psi_pm; phi->Values[3][1] = psi_mm;phi->Values[3][2] = psi_mp;
phi->Values[3][3] = psi_pp;

phi->DxValues[0][0] = -psi_p; phi->DxValues[0][1] = -psi_p;phi->DxValues[0][2] = -psi_m;
phi->DxValues[0][3] = -psi_m;phi->DxValues[1][0] = psi_p; phi->DxValues[1][1] = psi_p;
phi->DxValues[1][2] = psi_m;phi->DxValues[1][3] = psi_m;phi->DxValues[2][0] = psi_m; 
phi->DxValues[2][1] = psi_m;phi->DxValues[2][2] = psi_p;phi->DxValues[2][3] = psi_p;
phi->DxValues[3][0] = -psi_m; phi->DxValues[3][1] = -psi_m;phi->DxValues[3][2] = -psi_p;
phi->DxValues[3][3] = -psi_p;

phi->DyValues[0][0] = -psi_p; phi->DyValues[0][1] = -psi_m;phi->DyValues[0][2] = -psi_m;
phi->DyValues[0][3] = -psi_p;phi->DyValues[1][0] = -psi_m; phi->DyValues[1][1] = -psi_p;
phi->DyValues[1][2] = -psi_p;phi->DyValues[1][3] = -psi_m;phi->DyValues[2][0] = psi_m; 
phi->DyValues[2][1] = psi_p;phi->DyValues[2][2] = psi_p;phi->DyValues[2][3] = psi_m;
phi->DyValues[3][0] = psi_p; phi->DyValues[3][1] = psi_m;phi->DyValues[3][2] = psi_m;
phi->DyValues[3][3] = psi_p;
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
  AppAlgebra             *algebra = &appctx->algebra;
  AppGrid                *grid    = &appctx->grid;
  Vec                    w_local = algebra->w_local, x = algebra->x, x_local = algebra->x_local;
  Vec                    z_local = algebra->z_local;
  Vec                    z = algebra->z;
  VecScatter             gtol = algebra->gtol;
  MPI_Comm               comm = appctx->comm;
  double *sdnz, *sonz;  /* non-zero entries on this processor, non-zero entries off this processor */
  double                 srank,*procs,zero = 0.0,wght,one = 1.0;
  int                    ierr, rank,*vertices,cproc,i,j,*dnz,vertex_n = grid->vertex_n;
  int                    cell_n = grid->cell_n, *cell_vertex = grid->cell_vertex;
 const int four = 4, two = 2;
  int                    *cell_cell = grid->cell_cell,*cells,*onz;
  Mat                    A;
  Mat J;
  ISLocalToGlobalMapping ltog = grid->ltog;

  PetscFunctionBegin;
  MPI_Comm_rank(comm,&rank);    /* Get the index of this processor */

  /* ------------------------------------------------
      Determine non-zero structure of the matrix 
      --------------------------------------------*/
  
  /* 1) make proc[] contain the processor number of each ghosted vertex */
  srank = rank;

  /* set all values of x to the index of this processor */
  ierr = VecSet(x,srank);CHKERRQ(ierr);            

  /* w_local contains all vertices, including ghosted that this processor uses */
  ierr = VecScatterBegin(x,w_local,INSERT_VALUES,SCATTER_FORWARD,gtol);CHKERRQ(ierr);
  ierr = VecScatterEnd(x,w_local,INSERT_VALUES,SCATTER_FORWARD,gtol);CHKERRQ(ierr);

  /* copy w_local into the array procs */
  ierr = VecGetArray(w_local,&procs);CHKERRQ(ierr);
  /* make an array the size x_local ( total number of vertices, including ghosted) , fill with zeros */ 
 ierr = VecSet(x_local,zero);CHKERRQ(ierr);   
  ierr = VecGetArray(x_local,&sdnz);CHKERRQ(ierr);  
  /* make an array of appropriate size, for the off-diagonals whic corresponds to vertices off this processor */
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
        sdnz[vertices[j]] += wght;
      } else {
        sonz[vertices[j]] += wght;
      }
      /* 2nd neighbor - diagonally opposite*/
     if (cproc == procs[vertices[(j+2) % four]]) { /* on diagonal part */
        sdnz[vertices[j]] += 1.0;
      } else {
        sonz[vertices[j]] += 1.0;
      }
      /* 3rd neighbor  - adjacent */
      if (cells[(j+3)% four ] >= 0) wght = .5; else wght = 1.0; /* check if it has an adjacent cell */
      if (cproc == procs[vertices[(j+3) % four]]) { /* on diagonal part */
        sdnz[vertices[j]] += wght;
      } else {
        sonz[vertices[j]] += wght;
      }
   
    }
  }

  /* tell Petsc we nonlonger need access to the array */
  ierr = VecRestoreArray(x_local,&sdnz);CHKERRQ(ierr);
  ierr = VecRestoreArray(z_local,&sonz);CHKERRQ(ierr);
  ierr = VecRestoreArray(w_local,&procs);CHKERRQ(ierr);

  ierr = VecSet(x,zero);CHKERRQ(ierr);
  ierr = VecScatterBegin(x_local,x,ADD_VALUES,SCATTER_REVERSE,gtol);CHKERRQ(ierr);
  ierr = VecScatterEnd(x_local,x,ADD_VALUES,SCATTER_REVERSE,gtol);CHKERRQ(ierr);
  ierr = VecGetArray(x,&sdnz);CHKERRQ(ierr);

  ierr = VecSet(z,zero);CHKERRQ(ierr);
  ierr = VecScatterBegin(z_local,z,ADD_VALUES,SCATTER_REVERSE,gtol);CHKERRQ(ierr);
  ierr = VecScatterEnd(z_local,z,ADD_VALUES,SCATTER_REVERSE,gtol);CHKERRQ(ierr);
  ierr = VecGetArray(z,&sonz);CHKERRQ(ierr);

  /* didn't need to mess this up 
  dnz  = (int *) PetscMalloc((2*vertex_n+1)*sizeof(int));CHKPTRQ(dnz);
  onz  = (int *) PetscMalloc((2*vertex_n+1)*sizeof(int));CHKPTRQ(onz);
  for ( i=0; i<vertex_n; i++ ) {
    dnz[2*i] = 1 + (int) PetscReal(sdnz[i]);
    dnz[2*i+1] = 1 + (int) PetscReal(sdnz[i]);
    onz[2*i] = (int) PetscReal(sonz[i]);
    onz[2*i+1] = (int) PetscReal(sonz[i]);
  }  
  */
  dnz  = (int *) PetscMalloc((vertex_n+1)*sizeof(int));CHKPTRQ(dnz);
  onz  = (int *) PetscMalloc((vertex_n+1)*sizeof(int));CHKPTRQ(onz);
  for ( i=0; i<vertex_n; i++ ) {
    dnz[i] = 1 + (int) PetscReal(sdnz[i]);
    onz[i] = (int) PetscReal(sonz[i]);
  }
  ierr = VecRestoreArray(x,&sdnz);CHKERRQ(ierr);
  ierr = VecRestoreArray(z,&sonz);CHKERRQ(ierr);

  ierr = MatCreateMPIBAIJ(comm, 2, 2*vertex_n,2*vertex_n,PETSC_DETERMINE,PETSC_DETERMINE,0,dnz,0,onz,&A); CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMappingBlocked(A,ltog);CHKERRQ(ierr);

  /* Dupicate the matrix for now.  Later the Jacobian will not ahve the same nonzero structure  */
   ierr = MatCreateMPIBAIJ(comm, 2, 2*vertex_n,2*vertex_n,PETSC_DETERMINE,PETSC_DETERMINE,0,dnz,0,onz,&J); CHKERRQ(ierr);


/*   ierr = ISLocalToGlobalMappingView(ltog,VIEWER_STDOUT_SELF); CHKERRQ(ierr); */

  PetscFree(dnz);
  PetscFree(onz);
  algebra->A = A;
  algebra->J = J;

  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "AppCxtSetMatrix"
int AppCtxSetMatrix(AppCtx* appctx)
{
  AppAlgebra *algebra = &appctx->algebra;
  AppGrid    *grid    = &appctx->grid;
  MPI_Comm   comm = appctx->comm;
  Scalar     srank,*procs,*sdnz,zero = 0.0,wght,*sonz;
  int        ierr, rank,*vert_ptr,vertices[8],cproc,i,j,k,*dnz,vertex_n = grid->vertex_n;
  int        cell_n = grid->cell_n, *cell_vertex = grid->cell_vertex;
const int  four = 4, two = 2;  /* quad element, number of variables */
  int        *cell_cell = grid->cell_cell,*cells,*onz;
  Mat        A = algebra->A;
  IS         vertex_boundary = grid->vertex_boundary;
  Scalar     one = 1.0;
  AppElement *phi = &appctx->element;
  IS vertex_doubled = grid->vertex_doubled;
  IS vertex_boundary_doubled;
  int *vertex_boundary_ptr, *vertex_boundary_array;  
  int vertex_count = grid->vertex_n_ghosted;
  ISLocalToGlobalMapping dltog = grid->dltog;
  int vertex_boundary_n;
  double u_val = appctx->functions.u_val;
  double *vertex_values = grid->vertex_value;
 
  double values[4*4*2*2];  /* the integral of the combination of phi's */
  double coors[4][2]; /* the coordinates of one element */

  PetscFunctionBegin;
  MPI_Comm_rank(comm,&rank); 
  

/* ---------------------------------------------------------------
        loop over local elements, putting values into matrix 
        ---------------------------------------------------------------*/
/*   ierr = ISView(vertex_boundary,VIEWER_STDOUT_SELF); CHKERRQ(ierr); */

  for ( i=0; i<cell_n; i++ ) {
    vert_ptr = cell_vertex + four*i;   
    for ( j=0; j<four; j++) {
      coors[j][0] = vertex_values[2*vert_ptr[j]];
      coors[j][1] = vertex_values[2*vert_ptr[j]+1];
    }
    /*    Compute the element stiffness    */  
    ComputeMatrix( coors, phi, u_val, v,  values );

    ierr     = MatSetValuesBlockedLocal(A,four,vert_ptr,four,vert_ptr,values,ADD_VALUES);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /*  -------------------------------------------------------------
         Apply Dirichlet boundary conditions
      -----------------------------------------------------------*/
  /* need to get indices, then double them and zero those rows. */

  ierr = ISGetSize(vertex_boundary, &vertex_boundary_n);
  ierr = ISGetIndices(vertex_boundary, &vertex_boundary_ptr);CHKERRQ(ierr);
  vertex_boundary_array  = (int*)PetscMalloc(2*(vertex_boundary_n+1)*sizeof(int)); CHKPTRQ(vertex_boundary_array);

  for(i=0;i<vertex_boundary_n;i++){
    vertex_boundary_array[2*i] = 2*vertex_boundary_ptr[i];
    vertex_boundary_array[2*i+1] = 2*vertex_boundary_ptr[i] + 1;
  }
  ierr = ISRestoreIndices(vertex_boundary, &vertex_boundary_ptr);
  ierr = ISCreateGeneral(comm,2*vertex_boundary_n, vertex_boundary_array, &vertex_boundary_doubled);CHKERRQ(ierr);

   printf("\nview vertex_boundary_doubled IS\n"); 
   ierr = ISView(vertex_boundary_doubled, VIEWER_STDOUT_SELF);CHKERRQ(ierr); 

  ierr = MatSetLocalToGlobalMapping(A, dltog);CHKERRQ(ierr);

/*    printf("\nview the doubled localtoglobal mapping\n");  */
/*    ierr = ISLocalToGlobalMappingView(dltog, VIEWER_STDOUT_SELF);CHKERRQ(ierr);  */
 
  ierr = MatZeroRowsLocal(A,vertex_boundary_doubled,&one);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatView(A,  VIEWER_STDOUT_SELF);CHKERRQ(ierr); 

  PetscFunctionReturn(0);
}


/* ComputeMatrix: computes integrals of gradients of local phi_i and phi_j on the given quadrangle by changing variables to the reference quadrangle and reference basis elements phi_i and phi_j.  The formula used is

integral (given element) of <grad phi_j', grad phi_i'> =
integral over (ref element) of 
    <(grad phi_j composed with h)*(grad h)^-1,
     (grad phi_i composed with h)*(grad h)^-1>*det(grad h).
this is evaluated by quadrature:
= sum over gauss points, above evaluated at gauss pts
*/
void ComputeMatrix(double coords[4][2], AppElement *phi, double  u, DFP v, double *result){
  /*
    Dh        Jacobian of h at each of 4 gauss pts
    Dhinv     its inverse
    DphiDhinv the partials (of each basis fn at each point)
    detDh     determinant of Dh
    gauss     the image of each gauss pt
  */
  
  int i,j,k,ii ;
  double Dh[4][2][2]; 
  double Dhinv[4][2][2];
  double DphiDhinv[4][4][2];
  double detDh[4];
  double gauss[4][2];
  int havefluxterms = 0; /* temporary flag */

  /* the image of the reference element is given by sum (coord i)*phi_i */
  for(j=0;j<4;j++){ /* loop over points */
    gauss[j][0] = 0; gauss[j][1] = 0;
    for( k=0;k<4;k++ ){
      gauss[j][0] += coords[k][0]*phi->Values[k][j];
      gauss[j][1] += coords[k][1]*phi->Values[k][j];
    }
/*   printf("\n gauss pt %f  %f \n", gauss[j][0], gauss[j][1]); */
}

  /* Jacobian */
  for(i=0;i<4;i++){ /* loop over Gauss points */
    Dh[i][0][0] = 0; Dh[i][0][1] = 0; Dh[i][1][0] = 0; Dh[i][1][1] = 0;
    for(k=0; k<4; k++ ){
      Dh[i][0][0] += coords[k][0]*phi->DxValues[k][i];
      Dh[i][0][1] += coords[k][0]*phi->DyValues[k][i];
      Dh[i][1][0] += coords[k][1]*phi->DxValues[k][i];
      Dh[i][1][0] += coords[k][1]*phi->DyValues[k][i];
     
    }
   /*  printf("Dh %f\t%f\n%f\t%f\n", Dh[i][0][0], Dh[i][0][1], Dh[i][1][0], Dh[i][1][1]); */
  }

  /* Determinant of the Jacobian */
  for( j=0; j<4; j++){   /* loop over Gauss points */
    detDh[j] = Dh[j][0][0]*Dh[j][1][1] - Dh[j][0][1]*Dh[j][1][0];
  }

  /* Inverse of the Jacobian */
  for( j=0; j<4; j++){   /* loop over Gauss points */
    Dhinv[j][0][0] = Dh[j][1][1]/detDh[j];
    Dhinv[j][0][1] = -Dh[j][0][1]/detDh[j];
    Dhinv[j][1][0] = -Dh[j][1][0]/detDh[j];
    Dhinv[j][1][1] = Dh[j][0][0]/detDh[j];
  }

  /* partial of phi at h(gauss pt) times Dhinv */
  
  /* loop over gauss, the basis fns, then d/dx or d/dy */
  for( i=0;i<4;i++ ){  /* loop over Gauss points */
    for( j=0;j<4;j++ ){ /* loop over basis functions */

      /* make sure of i,j order in the DxValues, DyValues */
      DphiDhinv[i][j][0] = phi->DxValues[j][i]*Dhinv[i][0][0] +
                           phi->DyValues[j][i]*Dhinv[i][1][0];
      DphiDhinv[i][j][1] = phi->DxValues[j][i]*Dhinv[i][0][1] +
	                   phi->DyValues[j][i]*Dhinv[i][1][1];
  }
}

  /* Stiffness Terms */
  /* Now Integral.  term is <DphiDhinv[i],DphiDhinv[j]>*abs(detDh) */


   for( i=0;i<4;i++ ){ /* loop over first basis fn */
     for( j=0; j<4; j++){ /* loop over second */
    
       /* keep in mind we are throwing in a 2x2 block for each 1x1 */

       result[16*i + 2*j] = 0;
       result[16*i + 2*j+1] = 0;
       result[16*i + 8 +2*j] = 0;
       result[16*i + 9 +2*j] = 0;

       /* funny ordering of 2x2 blocks in the 4x4 piece */
       for(k=0;k<4;k++){ /* loop over gauss points */
	 result[16*i + 2*j] +=  (DphiDhinv[k][i][0]*DphiDhinv[k][j][0] + 
                              DphiDhinv[k][i][1]*DphiDhinv[k][j][1])*
	                      PetscAbsDouble(detDh[k]);
       }
       /* the off-diagonals stay zero */
       for(k=0;k<4;k++){ /* loop over gauss points */
	 result[16*i +9 + 2*j] +=  (DphiDhinv[k][i][0]*DphiDhinv[k][j][0] + 
                              DphiDhinv[k][i][1]*DphiDhinv[k][j][1])*
	                      PetscAbsDouble(detDh[k]);
       }

     }
   }

   /* Flux Terms */
   /* Now Integral.  term is <DphiDhinv[i],Beta]>Phi[j]*abs(detDh) */

   if ( havefluxterms ){   
     for( i=0;i<4;i++ ){ /* loop over first basis fn */
       for( j=0; j<4; j++){        /* loop over second */
	 for(k=0;k<4;k++){ 	 /* loop over gauss points */
	   result[4*i + j] +=  (DphiDhinv[k][i][0]*u + /* was u(gauss[k][0],gauss[k][1]) */ 
                              DphiDhinv[k][i][1]*v(gauss[k][0],gauss[k][1])  )*
	   phi->Values[j][k]*PetscAbsDouble(detDh[k]);
	 }
       }
     }
   }

}


/* ----------------------------------------------------------------------- */
/*
   AppCtxViewMatlab - Views solution using Matlab via socket connections.

   Input Parameter:
   appctx - user-defined application context

   Note:
   See the companion Matlab file mscript.m for usage instructions.
*/
#undef __FUNC__
#define __FUNC__ "AppCxtViewMatlab"
int AppCtxViewMatlab(AppCtx* appctx)
{
  int    ierr,*cell_vertex,rstart,rend;
  Viewer viewer = VIEWER_MATLAB_WORLD;
  double *vertex_values;
  IS     isvertex;

  PetscFunctionBegin;

  /* First, send solution vector to Matlab */
  ierr = VecView(appctx->algebra.x,viewer); CHKERRQ(ierr);

  /* Next, send vertices to Matlab */
  ierr = AODataKeyGetOwnershipRange(appctx->aodata,"vertex",&rstart,&rend); CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD,rend-rstart,rstart,1,&isvertex); CHKERRQ(ierr);
  ierr = AODataSegmentGetIS(appctx->aodata,"vertex","values",isvertex,(void **)&vertex_values);
         CHKERRQ(ierr);
  ierr = PetscDoubleView(2*(rend-rstart),vertex_values,viewer); CHKERRQ(ierr);
  ierr = AODataSegmentRestoreIS(appctx->aodata,"vertex","values",PETSC_NULL,(void **)&vertex_values);
         CHKERRQ(ierr);
  ierr = ISDestroy(isvertex); CHKERRQ(ierr);

  /* 
     Send list of vertices for each cell; these MUST be in the global (not local!) numbering); 
     this cannot use appctx->grid->cell_vertex 
  */
  ierr = AODataSegmentGetIS(appctx->aodata,"cell","vertex",appctx->grid.cell_global,
        (void **)&cell_vertex); CHKERRQ(ierr);
  ierr = PetscIntView(4*appctx->grid.cell_n,cell_vertex,viewer); CHKERRQ(ierr);
  ierr = AODataSegmentRestoreIS(appctx->aodata,"cell","vertex",PETSC_NULL,(void **)&cell_vertex); 
         CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}



