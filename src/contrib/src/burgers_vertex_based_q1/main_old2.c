

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
  double zero = 0.0; double onep2 = 1.2345;
  IS                     vertex_global = grid->vertex_global;
  SLES                   sles;
  SNES                   snes;
  PC pc;
  KSP ksp;
  Mat J;  /* Jacobian */
  Vec f;
  Vec x;  /* f is for the nonlinear function evaluation, x is the solution */
  
  PetscFunctionBegin;

  /*        Create vector to contain load and various work vectors  */
  ierr = AppCtxCreateVector(appctx); CHKERRQ(ierr);

  /*      Create the sparse matrix, with correct nonzero pattern  */
  ierr = AppCtxCreateMatrix(appctx); CHKERRQ(ierr);

  /*     Set the quadrature values for the reference square element  */
  ierr = AppCtxSetReferenceElement(appctx);CHKERRQ(ierr);

  /*      Set the right hand side values into the vectors   */
  ierr = AppCtxSetRhs(appctx); CHKERRQ(ierr);

  /*      Set the matrix entries   */
  ierr = AppCtxSetMatrix(appctx); CHKERRQ(ierr);

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
 
 /* Instead call FormInitialGuess  */
  x = algebra->x;
  ierr = VecSet(x,onep2); CHKERRQ(ierr);

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
  printf("(zeros) about to view the vector f\n");
  ierr = VecView(f, VIEWER_STDOUT_WORLD);CHKERRQ(ierr); 
 
  /* add rhs to get constant part */
  ierr = VecAXPY(f,mone,b); CHKERRQ(ierr); /* this says f = f - 1*b */
  printf("(plus b)about to view the vector f\n");
  ierr = VecView(f, VIEWER_STDOUT_WORLD);CHKERRQ(ierr); 

printf("input vector x\n");
  ierr = VecView(x, VIEWER_STDOUT_WORLD);CHKERRQ(ierr); 
  /*apply matrix to the input vector x, to get linear part */
  /* Assuming mattrix doesn't need to be recomputed */
  ierr = MatMultAdd(A, x, f, f); CHKERRQ(ierr);  /* f = A*x + f */

  printf("(plus A*x)about to view the vector f\n");
  ierr = VecView(f, VIEWER_STDOUT_WORLD);CHKERRQ(ierr); 

 /* create nonlinear part */
  /* Need to call SetNonlinear on the input vector  */
  ierr = SetNonlinear(x, appctx, f);CHKERRQ(ierr);
  
  printf("(nonlinear)about to view the vector f\n");
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
  
  /* need to just add in the values *
  ierr = MatConvert(A, MATSAME, MAT_INITIAL_MATRIX, jac);CHKERRQ(ierr);
    ierr = MatConvert(A, MATSAME, MAT_INITIAL_MATRIX,B);CHKERRQ(ierr); 

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
  ISLocalToGlobalMapping ltog = grid->ltog;

  AppElement *phi = &appctx->element;
 VecScatter gtol = algebra->gtol;
  Vec w_local = algebra->w_local;
  int cell_n = grid->cell_n, i,j,k, ierr, vertex_n = grid->vertex_n, vertex_count = grid->vertex_n_ghosted,  *vert_ptr,  *cell_vertex = grid->cell_vertex,  *cell_cell = grid->cell_cell;
  double  *vertex_values = grid->vertex_value, *uvvals, *value_ptr;
  double values[4*4*2*2];  /* the integral of the combination of phi's */
  double coors[8]; /* the coordinates of one element */

  PetscFunctionBegin;
  /* Matrix is set to the linear part already, so just ADD_VALUES the nonlinear part  */

ierr = VecScatterBegin(x, w_local, INSERT_VALUES, SCATTER_FORWARD, gtol); CHKERRQ(ierr);
ierr = VecScatterEnd(x, w_local, INSERT_VALUES, SCATTER_FORWARD, gtol); CHKERRQ(ierr);
ierr = VecGetArray(w_local, &uvvals);

/*   loop over local elements, putting values into matrix -*/
  for ( i=0; i<cell_n; i++ )
  {
    vert_ptr = cell_vertex + 4*i;   
    value_ptr = uvvals + 8*i;
    for ( j=0; j<4; j++) {
      coors[2*j] = vertex_values[2*vert_ptr[j]];
      coors[2*j+1] = vertex_values[2*vert_ptr[j]+1];
    }
    /* compute the values of basis functions on this element */
    ierr = SetLocalElement(phi, coors);CHKERRQ(ierr);
    /*    Compute the partial derivatives of the nonlinear map    */  
    ierr = ComputeJacobian( phi, value_ptr,  values );CHKERRQ(ierr);
    /*  Set the values in the matrix */

/*  ierr = MatSetLocalToGlobalMappingBlocked(*jac,ltog);CHKERRQ(ierr); */

    ierr  = MatSetValuesBlockedLocal(*jac,4,vert_ptr,4,vert_ptr,values,ADD_VALUES);CHKERRQ(ierr);
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
  double result[8], *value_ptr, coors[8], *vertex_values = grid->vertex_value,  *uvvals;
  int ierr, i, j, cell_n = grid->cell_n, *vertex_ptr, *cell_vertex = grid->cell_vertex;
  int k;
  /*  Loop over local elements, extracting the values from x  and add them into f  */
  
  ierr = VecScatterBegin(x, w_local, INSERT_VALUES, SCATTER_FORWARD, gtol); CHKERRQ(ierr);
  ierr = VecScatterEnd(x, w_local, INSERT_VALUES, SCATTER_FORWARD, gtol); CHKERRQ(ierr);

  ierr = VecGetArray(w_local, &uvvals); CHKERRQ(ierr);
  /**** HERE, this thing is half the length it should be  ****/
  printf("the array of extracted local vector\n");
  for(k=0;k<8;k++){ printf("%f\n", uvvals[k]); }

  /* set a flag in computation of local elements */
  phi->dorhs = 0;
  
  for(i=0;i<cell_n;i++){
    vertex_ptr = cell_vertex + 4*i; 
    value_ptr = uvvals + 8*i;

    /* create nonlinear part */
    for ( j=0; j<4; j++) {
      coors[2*j] = vertex_values[2*vertex_ptr[j]];
      coors[2*j+1] = vertex_values[2*vertex_ptr[j]+1];
    }

    /* compute the values of basis functions on this element */
    SetLocalElement(phi, coors);
    /* do the integrals */
    ierr = ComputeNonlinear(phi, value_ptr, result);CHKERRQ(ierr);
    printf("result of compute nonlinear");
    for(k=0;k<8;k++){ printf("%f\n", result[k]); }

    /* put result in */
    ierr = VecSetValuesBlockedLocal(f, 4, vertex_ptr, result, ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(w_local, &uvvals);  CHKERRQ(ierr);

  ierr = VecAssemblyBegin(f);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(f);CHKERRQ(ierr);

}
#undef __FUNC__
#define __FUNC__ "Computeacobian"

/* input is x, output the nonlinear part into f for a particulat element */
/* Much of the code is dublicated from ComputeMatrix; the integral is different */
int ComputeJacobian(AppElement *phi, double *uv, double *result)
{
  /* How can I test this??  */
  int i,j,k,ii ;
  double u[4],v[4];
  double dxint[4][4][4], dyint[4][4][4]; 

  /* copy array into more convenient form */
  for(i=0;i<4;i++){
    u[i] = uv[2*i]; v[i] = uv[2*i+1];
  }
 
  /* INTEGRAL */ 
  /* The nonlinear map takes( u0,v0,u1,v1,u2,v2,u3,v3 ) to 
      ( integral term1 *  phi0, integral term2 * phi0, ..., integral term1*phi3, int term2*phi3)
   Loop first over the phi.  Then integrate two parts of the terms.
Term 1: (ui*uj*phi_i*dx_j + vi*uj*phi_i*dy_j)
Term 2: (ui*vj*phi_i*dx_j + vi*vj*phi_i*dy_j)
*/


/* Make a database of integrals of phi_i*phi_j(dx or dy)*phi_k */
  for(i=0;i<4;i++){
    for(j=0;j<4;j++){
      for(k=0;k <= i;k++){
	 dxint[i][j][k] = 0; dyint[i][j][k] = 0;
	for(ii=0;ii<4;ii++){/* loop over basis points */
	  dxint[i][j][k] += phi->dx[4*i+ii]*phi->Values[j][ii]*phi->Values[k][ii];
	  dyint[i][j][k] += phi->dy[4*i+ii]*phi->Values[j][ii]*phi->Values[k][ii];
	}; 
	dxint[i][k][j] = dxint[i][j][k]; dyint[i][k][j] = dyint[i][j][k];
      }
    }
  }

  /* now loop over the columns of the matrix */
   for( k=0;k<4;k++ ){ 
     /* the terms are u*ux + v*uy and u*vx+v*vy  */
     for(i = 0;i<4;i++){  
       result[8*k + 2*i] = 0; result[8*k + 2*i + 1] = 0;   /* Stuff from Term 1 */
       result[16*k + 2*i]=0; result[16*k + 2*i + 1] = 0;  /* Stuff from Term 2 */
       for(j=0;j<4;j++){
	 result[8*k + 2*i] +=   u[j]*dxint[i][j][k] + u[j]*dxint[j][i][k] + v[i]*dyint[i][j][k];
	 result[8*k+2*i+1] +=   u[j]*dyint[j][i][k];

	 result[16*k +2*i] += v[j]*dxint[j][i][k];
	 result[16*k+2*i+1] += u[j]*dxint[i][j][k] + v[j]*dyint[j][i][k] + v[j]*dyint[i][j][k];
       }     
     }
   }
   PetscFunctionReturn(0);
}



#undef __FUNC__
#define __FUNC__ "ComputeNonlinear"

/* input is x, output the nonlinear part into f for a particulat element */
int ComputeNonlinear(AppElement *phi, double *uvvals, double *result)
{ 
  int i,j,k,ii ;
  double u[4],v[4];

  /* copy array into more convenient form */
  printf("the input to compute nonlinear\n");
  for(i=0;i<4;i++){ 
    u[i] = uvvals[2*i]; v[i] = uvvals[2*i+1]; 
    printf("%f\n%f\n", uvvals[2*i], uvvals[2*i+1]);
  }

  /* INTEGRAL */
 /* terms are u*du/dx + v*du/dy, u*dv/dx + v*dv/dy */
  /* Go element by element.  
Compute 
 u_i * phi_i * u_j * phi_j_x + v_i*phi_i*u_j*phi_j_y * phi_k
and
 u_i * phi_i * v_j * phi_j_x + v_i*phi_i*v_j*phi_j_y * phi_k.
Put the result in index k.  Add all possibilities up to get contribution to k, and loop over k.*/


/* Could exploit a little symetry to cut iterations from 4*4*4 to 2*4*4  */
   for( k=0;k<4;k++ ){ /* loop over first basis fn */
     result[2*k] = 0; result[2*k+1] = 0;
     for( i=0; i<4; i++){ /* loop over second */
       for( j=0; j<4; j++){/* loop over third */
	 for(ii=0;ii<4;ii++){ /* loop over gauss points */
	 result[2*k] += 
	   (u[i]*u[j]*phi->Values[i][ii]*phi->dx[4*j+ii] +
	    v[i]*u[j]*phi->Values[i][ii]*phi->dy[4*j+ii])*phi->Values[k][ii]*phi->detDh[ii]; 
	 result[2*k+1] +=
	   (u[i]*v[j]*phi->Values[i][ii]*phi->dx[4*j+ii] +
	    v[i]*v[j]*phi->Values[i][ii]*phi->dy[4*j+ii])*phi->Values[k][ii]*phi->detDh[ii];
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

  ierr = VecCreateSeq(PETSC_COMM_SELF,vertex_n_ghosted,&w_local);CHKERRQ(ierr);
  ierr = VecDuplicate(w_local,&x_local);CHKERRQ(ierr);
  ierr = VecDuplicate(w_local,&z_local);CHKERRQ(ierr);
  ierr = VecScatterCreate(b,vertex_global,w_local,0,&gtol);CHKERRQ(ierr);
  /********** HEre need a blocked vecScatter  ****************/
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
  int        *cell_vertex = grid->cell_vertex,*vertices, *indices, j;
  double     *vertex_values = grid->vertex_value, *bvs, xval, yval;
  Vec        b = algebra->b;
  int two = 2;
  /*      Room to hold the coordinates of a single cell, plus the 
     RHS generated from a single cell.  */
  double coors[8]; /* quad cell */
  double values[8]; /* number of elements * number of variables */  

  /* set flag for element computation */
    phi->dorhs = 1;
  /*     Loop over elements computing load one element at a time 
        and putting into right-hand-side-*/
  for ( i=0; i<cell_n; i++ )
 {
    vertices = cell_vertex+4*i;
    /*  Load the cell vertex coordinates */
    for ( j=0; j<4; j++) {
      coors[2*j] = vertex_values[2*vertices[j]];
      coors[2*j+1] = vertex_values[2*vertices[j]+1];    }

    /* compute the values of basis functions on this element */
    SetLocalElement(phi, coors);

    /* compute the  element load (integral of f with the 4 basis elements)  */
     ComputeRHS( f, g, phi, values );

    ierr = VecSetValuesBlockedLocal(b,4,vertices,values,ADD_VALUES);CHKERRQ(ierr);
  }

  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);

  /*  -------------------------------------------------------------
         Apply Dirichlet boundary conditions
      -----------------------------------------------------------*/  
  /* need to set the points on RHS corresponding to vertices on the boundary to
     the desired value.  In this case, if the b.c. is zero, then set these
     values to zero */
  
  ierr = ISGetIndices(vertex_boundary, &indices);CHKERRQ(ierr);
  ierr = ISGetSize(vertex_boundary, &nindices); CHKERRQ(ierr);

  bvs = (double*)PetscMalloc(two*(nindices+1)*sizeof(double)); CHKPTRQ(bvs);

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

  PetscFunctionReturn(0);
}


int ComputeRHS( DFP f, DFP g, AppElement *phi, double *integrals){
  int i,j,k;
  /* need to go over each element , then each variable */
 for( i = 0; i < 4; i++ ){ /* loop over basis functions */
   integrals[2*i] = 0.0; 
   integrals[2*i+1] = 0.0; 
   for( j = 0; j < 4; j++ ){ /* loop over Gauss points */
     integrals[2*i] +=  f(phi->x[j], phi->y[j])*(phi->Values[i][j])*phi->detDh[j];
     integrals[2*i+1] +=  f(phi->x[j], phi->y[j])*(phi->Values[i][j])*phi->detDh[j];
   }
 }
PetscFunctionReturn(0);
}

int SetLocalElement(AppElement *phi, double *coords)
{
  int i,j,k,ii ;
  double Dh[4][2][2], Dhinv[4][2][2]; 
  double *dx = phi->dx, *dy = phi->dy;
  double *detDh = phi->detDh;
  double *x = phi->x, *y = phi->y;  /* image of gauss point */

  int do_rhs = phi->dorhs;

  /* Could put in a flag to skip computing this when it isn't needed */

  if (do_rhs){
  /* the image of the reference element is given by sum (coord i)*phi_i */
    for(j=0;j<4;j++){ /* loop over points */
      x[j] = 0; y[j] = 0;
      for( k=0;k<4;k++ ){
	x[j] += coords[2*k]*phi->Values[k][j];
	y[j] += coords[2*k+1]*phi->Values[k][j];
      }
    }
  }
  /* Jacobian */
  for(i=0;i<4;i++){ /* loop over Gauss points */
    Dh[i][0][0] = 0; Dh[i][0][1] = 0; Dh[i][1][0] = 0; Dh[i][1][1] = 0;
    for(k=0; k<4; k++ ){
      Dh[i][0][0] += coords[2*k]*phi->DxValues[k][i];
      Dh[i][0][1] += coords[2*k]*phi->DyValues[k][i];
      Dh[i][1][0] += coords[2*k+1]*phi->DxValues[k][i];
      Dh[i][1][0] += coords[2*k+1]*phi->DyValues[k][i];    
    }
  }

  /* Determinant of the Jacobian */
  for( j=0; j<4; j++){   /* loop over Gauss points */
    detDh[j] = PetscAbsDouble(Dh[j][0][0]*Dh[j][1][1] - Dh[j][0][1]*Dh[j][1][0]);
  }


  /* fix the flag so that we don't compute this in rhs case */
  if( do_rhs != 0 ){
    /* Inverse of the Jacobian */
    for( j=0; j<4; j++){   /* loop over Gauss points */
      Dhinv[j][0][0] = Dh[j][1][1]/detDh[j];
      Dhinv[j][0][1] = -Dh[j][0][1]/detDh[j];
      Dhinv[j][1][0] = -Dh[j][1][0]/detDh[j];
      Dhinv[j][1][1] = Dh[j][0][0]/detDh[j];
    }
    
    /* Notice that phi~ = phi(h), so Dphi~ = Dphi*Dh, so Dphi~ = Dphi*(Dh)inv */       
    /* partial of phi at h(gauss pt) times Dhinv */
    /* loop over gauss, the basis fns, then d/dx or d/dy */
    for( i=0;i<4;i++ ){  /* loop over Gauss points */
      for( j=0;j<4;j++ ){ /* loop over basis functions */
	dx[4*j+i] = phi->DxValues[j][i]*Dhinv[i][0][0] +  phi->DyValues[j][i]*Dhinv[i][1][0];
	dy[4*j+i] = phi->DxValues[j][i]*Dhinv[i][0][1] + phi->DyValues[j][i]*Dhinv[i][1][1];
      }
    }
  }
PetscFunctionReturn(0);
}

int AppCtxSetReferenceElement(AppCtx* appctx){

  AppElement *phi = &appctx->element;
  double psi, psi_m, psi_p, psi_pp, psi_mp, psi_pm, psi_mm;

phi->dorhs = 0;

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
 ierr = MatSetLocalToGlobalMappingBlocked(J,ltog);CHKERRQ(ierr);


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
  double *vertex_values = grid->vertex_value;
 
  double values[4*4*2*2];  /* the integral of the combination of phi's */
  double coors[8]; /* the coordinates of one element */

  PetscFunctionBegin;
  MPI_Comm_rank(comm,&rank); 

  /* set flag for phi computation */
    phi->dorhs = 0;
  
/*    loop over local elements, putting values into matrix  */
  for ( i=0; i<cell_n; i++ ) {
    vert_ptr = cell_vertex + four*i;   
    for ( j=0; j<four; j++) {
      coors[2*j] = vertex_values[2*vert_ptr[j]];
      coors[2*j+1] = vertex_values[2*vert_ptr[j]+1];
    }
    /* compute the values of basis functions on this element */
    ierr = SetLocalElement(phi, coors); CHKERRQ(ierr);

    /*    Compute the element stiffness    */  
    ierr = ComputeMatrix( phi, values ); CHKERRQ(ierr);

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

  ierr = MatSetLocalToGlobalMapping(A, dltog);CHKERRQ(ierr);
  ierr = MatZeroRowsLocalIS(A,vertex_boundary_doubled,one);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

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
int ComputeMatrix( AppElement *phi, double *result){
 
  int i,j,k;
 
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
	 result[16*i + 2*j] +=  (phi->dx[4*k+i]*phi->dx[4*k+j] + 
                                 phi->dy[4*k+i]*phi->dy[4*k+j])*
	                         phi->detDh[k];
       }
       /* the off-diagonals stay zero */
       for(k=0;k<4;k++){ /* loop over gauss points */
	 result[16*i +9 + 2*j] +=  (phi->dx[4*k+i]*phi->dx[4*k+j] + 
                                    phi->dy[4*k+i]*phi->dy[4*k+j])*
	                            phi->detDh[k];
       }
     }
   }
PetscFunctionReturn(0);
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



