#include "appctx.h"

/*
         Sets up the linear system associated with the PDE and solves it
*/
#undef __FUNC__
#define __FUNC__ "AppCxtSolve"
int AppCtxSolve(AppCtx* appctx)
{
  AppAlgebra  *algebra = &appctx->algebra;
  MPI_Comm    comm = appctx->comm;
  SLES        sles;
  int         ierr, its;

  PetscFunctionBegin;

  /*     A) Set the quadrature values for the reference square element  */
  ierr = SetReferenceElement(appctx);CHKERRQ(ierr);

  /*     1) Create vector to contain load and various work vectors  */
  ierr = AppCtxCreateRhs(appctx); CHKERRQ(ierr);

  /*     2)  Create the sparse matrix, with correct nonzero pattern  */
  ierr = AppCtxCreateMatrix(appctx); CHKERRQ(ierr);

  /*     3)  Set the right hand side values into the vectors   */
  ierr = AppCtxSetRhs(appctx); CHKERRQ(ierr);

  /*     4)  Set the matrix entries   */
  ierr = AppCtxSetMatrix(appctx); CHKERRQ(ierr);

  /* view sparsity structure of the matrix */
  if (appctx->view.show_matrix ) {  
    ierr = PetscPrintf(PETSC_COMM_WORLD,"The stiffness matrix, before bc applied\n");CHKERRQ(ierr);
    ierr = MatView(appctx->algebra.A, VIEWER_DRAW_WORLD );CHKERRQ(ierr);
  }

  /*     5) Set the rhs boundary conditions */
  ierr = SetBoundaryConditions(appctx); CHKERRQ(ierr);

  /*     6) Set the matrix boundary conditions */
  ierr = SetMatrixBoundaryConditions(appctx); CHKERRQ(ierr);

  /* view sparsity structure of the matrix */
  if( appctx->view.show_matrix ) {  
    ierr = PetscPrintf(PETSC_COMM_WORLD,"The stiffness matrix, after bc applied\n");CHKERRQ(ierr);
    ierr = MatView(appctx->algebra.A, VIEWER_DRAW_WORLD );CHKERRQ(ierr);
  }
  
  /*      Solve the linear system  */
  ierr = SLESCreate(comm,&sles);CHKERRQ(ierr);
  ierr = SLESSetOperators(sles,algebra->A,algebra->A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = SLESSetFromOptions(sles);CHKERRQ(ierr);
  ierr = SLESSolve(sles,algebra->b,algebra->x,&its);CHKERRQ(ierr);

  /*      Free the solver data structures */
  ierr = SLESDestroy(sles); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*
       1  -  Generates the "global" parallel vector to contain the 
	     right hand side and solution.
*/
#undef __FUNC__
#define __FUNC__ "AppCxtCreateRhs"
int AppCtxCreateRhs(AppCtx *appctx)
{
  AppGrid     *grid = &appctx->grid;
  AppAlgebra  *algebra = &appctx->algebra;
  MPI_Comm    comm = appctx->comm;
  int         ierr;

  PetscFunctionBegin;
  /*  Create vector to contain load,  local size should be number of  vertices  on this proc.  */
  ierr = VecCreateMPI(comm,grid->vertex_local_count,PETSC_DECIDE,&algebra->b);CHKERRQ(ierr);

  /* This allows one to set entries into the vector using the LOCAL numbering: via VecSetValuesLocal() */
  ierr = VecSetLocalToGlobalMapping(algebra->b,grid->ltog);CHKERRQ(ierr);

  /* Generate the vector to contain the solution */
  ierr = VecDuplicate(algebra->b, &algebra->x);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*
      2  - Generates the "global" parallel matrix
*/
#undef __FUNC__
#define __FUNC__ "AppCxtCreateMatrix"
int AppCtxCreateMatrix(AppCtx* appctx)
{

  AppAlgebra  *algebra = &appctx->algebra;
  AppGrid     *grid    = &appctx->grid;
  MPI_Comm    comm = appctx->comm;
  int         ierr; 
  PetscFunctionBegin;

  /* use very rough estimate for nonzeros on and off the diagonal */
  ierr = MatCreateMPIAIJ(comm,grid->vertex_local_count,grid->vertex_local_count,PETSC_DETERMINE,PETSC_DETERMINE,9,0,3,0,&algebra->A);CHKERRQ(ierr);

  /* Allows one to set values into the matrix using the LOCAL numbering, via MatSetValuesLocal() */
  ierr = MatSetLocalToGlobalMapping(algebra->A, grid->ltog);  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*
     3 - Computes the entries in the right hand side and sets them into the parallel vector
         Uses B
*/
#undef __FUNC__
#define __FUNC__ "AppCxtSetRhs"
int AppCtxSetRhs(AppCtx* appctx)
{
  /********* Collect context informatrion ***********/
  AppGrid    *grid = &appctx->grid;
  AppAlgebra *algebra = &appctx->algebra;
  AppElement *phi = &appctx->element;

  /****** Internal Variables ***********/
  int        ierr, i;
  int        *vertex_ptr;
  int        bn =4; /* basis count */
  int        vertexn = 4; /* degree of freedom count */

  PetscFunctionBegin;
  /* loop over local cells */
  for(i=0;i<grid->cell_n;i++){
    /* loop over degrees of freedom and cell coords */

    /* vertex_ptr points to place in the vector to set the values */
    vertex_ptr = grid->cell_vertex + vertexn*i; 

    /* coords_ptr points to the coordinates of the current cell */
    phi->coords = grid->cell_coords + 2*bn*i;  /*number of cell coords */

    /* compute the values of basis functions on this element */
    ierr = SetLocalElement(phi);CHKERRQ(ierr); 

    /* compute the  element load (integral of f with the 4 basis elements)  */
    /* values get put into phi->rhsresult  */
    ierr = ComputeRHSElement(phi);CHKERRQ(ierr);

    /*********  Set Values *************/
    ierr = VecSetValuesLocal(algebra->b, bn, vertex_ptr, phi->rhsresult, ADD_VALUES); CHKERRQ(ierr);
  }
  /********* Assemble Data **************/
  ierr = VecAssemblyBegin(algebra->b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(algebra->b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}  

/*
      4 - Computes the element stiffness matrices and stick into 
   global stiffness matrix. Uses C.
*/
#undef __FUNC__
#define __FUNC__ "AppCxtSetMatrix"
int AppCtxSetMatrix(AppCtx* appctx)
{
  /********* Collect contex informatrion ***********/
  AppAlgebra *algebra = &appctx->algebra;
  AppGrid    *grid    = &appctx->grid;
  AppElement *phi = &appctx->element; 

  /****** Internal Variables ***********/
  int        i, ierr;
  int        *vertex_ptr;
  int        bn =4; /* basis count */
  int        vertexn = 4; /* degree of freedom count */

  PetscFunctionBegin;

  /* loop over cells */
  for(i=0;i<grid->cell_n;i++){

    /* loop over degrees of freedom and cell coords */
    /* vertex_ptr points to place in the vector to set the values */
    vertex_ptr = grid->cell_vertex + vertexn*i;

    /* coords_ptr points to the coordinates of the current cell */
    phi->coords = grid->cell_coords + 2*bn*i;/*number of cell coords */

    /* compute the values of basis functions on this element */
    ierr = SetLocalElement(phi); CHKERRQ(ierr);
   
    /*    Compute the element stiffness  */  
    /* result is returned in phi->stiffnessresult */
    ierr = ComputeStiffnessElement(phi); CHKERRQ(ierr);

    /*********  Set Values *************/
    ierr = MatSetValuesLocal(algebra->A,vertexn,vertex_ptr,vertexn,vertex_ptr,(double*)phi->stiffnessresult,ADD_VALUES);CHKERRQ(ierr);
  }
  /********* Assemble Data **************/
  ierr = MatAssemblyBegin(algebra->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(algebra->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
      5   - Apply the Dirichlet boundary conditions (see 6 also).
     This places the Dirichlet function value on the right hand side
     and 6 sticks a row of the identity matrix on the left side 
     thus forcing the solution at the given points to match the 
     Dirichlet function.
*/
#undef __FUNC__
#define __FUNC__ "SetBoundaryConditions"
int SetBoundaryConditions(AppCtx *appctx)
{
 /********* Collect context informatrion ***********/
  AppAlgebra *algebra = &appctx->algebra;
  AppGrid    *grid = &appctx->grid;

  /****** Internal Variables ***********/
  int        ierr, i;
  double     xval, yval; 
  int        *vertex_ptr; 

  PetscFunctionBegin;
  /* Dirichlet Boundary Conditions */

  /*  -------------------------------------------------------------
         Apply Dirichlet boundary conditions
      -----------------------------------------------------------*/
  
  /* need to set the points on RHS corresponding to vertices on the boundary to
     the desired value. */
  ierr = ISGetIndices(grid->vertex_boundary, &vertex_ptr); CHKERRQ(ierr);
  for(i=0;i<grid->boundary_count;i++){
    xval = grid->boundary_coords[2*i];
    yval = grid->boundary_coords[2*i+1];
    grid->boundary_values[i] = bc(xval, yval);
  }
  ierr = VecSetValuesLocal(algebra->b, grid->boundary_count, vertex_ptr, grid->boundary_values, INSERT_VALUES);CHKERRQ(ierr);
  ierr = ISRestoreIndices(grid->vertex_boundary,&vertex_ptr);CHKERRQ(ierr);
 
  ierr = VecAssemblyBegin(algebra->b); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(algebra->b); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*
     6 - Set the matrix boundary conditions (see also 5). Replace the corresponding 
         rows in the matrix with the identity.
*/
#undef __FUNC__
#define __FUNC__ "SetMatrixBoundaryConditions"
int SetMatrixBoundaryConditions(AppCtx *appctx)
{
  AppAlgebra *algebra = &appctx->algebra;
  AppGrid    *grid = &appctx->grid;

  double     one = 1.0;
  int        ierr;

  PetscFunctionBegin;

  ierr = MatZeroRowsLocal(algebra->A, grid->vertex_boundary,&one);CHKERRQ(ierr); 
  
  PetscFunctionReturn(0);
}

/* -------------The next functions apply to single elements ------------------------------*/
/*
     Returns the value of the shape function or its xi or eta derivative at 
   any point in the REFERENCE element. xi and eta are the coordinates in the reference
   element.
*/
static double InterpolatingFunctionsElement(int partial, int node, double xi, double eta)
{

  /* 4 node bilinear interpolation functions */
  if (partial == 0){
    if( node == 0){return 0.25 *(1-xi)*          (1-eta)         ;}
    if( node == 1){return 0.25 *         (1+xi)*(1-eta)         ;}
    if( node == 2){return 0.25 *         (1+xi)         *(1+eta);}
    if( node == 3){return 0.25 *(1-xi)*                   (1+eta);}
  }  
  /*d/dxi */
  if (partial == 1){
    if( node == 0){return 0.25 *(  -1)*          (1-eta)         ;}
    if( node == 1){return 0.25 *                 1*(1-eta)         ;}
    if( node == 2){return 0.25 *                 1         *(1+eta);}
    if( node == 3){return 0.25 *(  -1)*                   (1+eta);}
  }   
  /*d/deta*/
  if (partial == 2){
    if( node == 0){return 0.25 *(1-xi)*          (-1)         ;}
    if( node == 1){return 0.25 *         (1+xi)*(-1)         ;}
    if( node == 2){return 0.25 *         (1+xi)         *(1);}
    if( node == 3){return 0.25 *(1-xi)*                   (1);}
  }
  return 0.0;  
}

#undef __FUNC__
#define __FUNC__ "SetReferenceElement"
/* 
     A - Computes the numerical integration (Gauss) points and evaluates the basis funtions at
   these points. This is done ONCE for the element, the information is stored in the AppElement
   data structure and then used repeatedly to compute each element load and element stiffness.
   Uses InterpolatingFunctions(). 
*/
int SetReferenceElement(AppCtx* appctx)
{
  int        i,j;
  int        bn = 4; /* basis count*/
  int        qn = 4; /*quadrature count */
  double     t;  /* for quadrature point */
  double     gx[4], gy[4]; /* gauss points: */  
  AppElement *phi = &appctx->element;

  PetscFunctionBegin;
  t =  sqrt(3.0)/3.0;

  /* set Gauss points */
  gx[0] = -t;   gx[1] = t; 
  gx[2] = t;  gx[3] = -t; 

  gy[0] = -t; gy[1] = -t; 
  gy[2] = t;  gy[3] = t; 

  /* set quadrature weights */
  phi->weights[0] = 1; phi->weights[1] = 1; 
  phi->weights[2] = 1; phi->weights[3] = 1; 

  /* Set the reference values  */
  for(i=0;i<bn;i++){  /* loop over functions*/
    for(j=0;j<qn;j++){/* loop over Gauss points */
      appctx->element.RefVal[i][j] =  InterpolatingFunctionsElement(0,i,gx[j], gy[j]);
      appctx->element.RefDx[i][j]  =  InterpolatingFunctionsElement(1,i,gx[j], gy[j]);
      appctx->element.RefDy[i][j]  =  InterpolatingFunctionsElement(2,i,gx[j], gy[j]);
    }
  }
  PetscFunctionReturn(0);
}
			  
/*
    B - Computes derivative information for each element. This data is used
    in C) and D) to compute the element load and stiffness.
*/
#undef __FUNC__
#define __FUNC__ "SetLocalElement"
int SetLocalElement(AppElement *phi )
{
  /* the coords array consists of pairs (x[0],y[0],...,x[7],y[7]) representing 
     the images of the support points for the 4 basis functions */ 
  int    i,j;
  int    bn = 4, qn = 4; /* basis count, quadrature count */
  double Dh[4][2][2], Dhinv[4][2][2];
 
  PetscFunctionBegin;
 /* The function h takes the reference element to the local element.
                  h(x,y) = sum(i) of alpha_i*phi_i(x,y),
   where alpha_i is the image of the support point of the ith basis fn */

  /*Values */
  for(i=0;i<qn;i++){ /* loop over the Gauss points */
    phi->x[i] = 0; phi->y[i] = 0; 
    for(j=0;j<bn;j++){/*loop over the basis functions, and support points */
      phi->x[i] += phi->coords[2*j]*phi->RefVal[j][i];
      phi->y[i] += phi->coords[2*j+1]*phi->RefVal[j][i];
    }
  }

  /* Jacobian */
  for(i=0;i<qn;i++){ /* loop over Gauss points */
    Dh[i][0][0] = 0; Dh[i][0][1] = 0; Dh[i][1][0] = 0; Dh[i][1][1] = 0;
    for(j=0; j<bn; j++ ){/* loop over functions */
      Dh[i][0][0] += phi->coords[2*j]*phi->RefDx[j][i];
      Dh[i][0][1] += phi->coords[2*j]*phi->RefDy[j][i];
      Dh[i][1][0] += phi->coords[2*j+1]*phi->RefDx[j][i];
      Dh[i][1][1] += phi->coords[2*j+1]*phi->RefDy[j][i];    
    }
  }

  /* Determinant of the Jacobian */
  for( i=0; i<qn; i++){   /* loop over Gauss points */
    phi->detDh[i] = Dh[i][0][0]*Dh[i][1][1] - Dh[i][0][1]*Dh[i][1][0];
  }

  /* Inverse of the Jacobian */
  for( i=0; i<qn; i++){   /* loop over Gauss points */
    Dhinv[i][0][0] = Dh[i][1][1]/phi->detDh[i];
    Dhinv[i][0][1] = -Dh[i][0][1]/phi->detDh[i];
    Dhinv[i][1][0] = -Dh[i][1][0]/phi->detDh[i];
    Dhinv[i][1][1] = Dh[i][0][0]/phi->detDh[i];
  }
    

  /* Notice that phi~ = phi(h), so Dphi~ = Dphi*Dh, (chain rule)
     so Dphi~ = Dphi*(Dh)inv    (multiply by (Dh)inv   */       
  /* partial of phi at h(gauss pt) times Dhinv */
  /* loop over Gauss, the basis fns, then d/dx or d/dy */
  for( i=0;i<qn;i++ ){  /* loop over Gauss points */
    for( j=0;j<bn;j++ ){ /* loop over basis functions */
      phi->dx[j][i] = phi->RefDx[j][i]*Dhinv[i][0][0] + phi->RefDy[j][i]*Dhinv[i][1][0];
      phi->dy[j][i] = phi->RefDx[j][i]*Dhinv[i][0][1] + phi->RefDy[j][i]*Dhinv[i][1][1];
    }
  }

  PetscFunctionReturn(0);
}
/*
        B - Computes an element load
*/
#undef __FUNC__
#define __FUNC__ "ComputeRHS"
int ComputeRHSElement( AppElement *phi )
{
  int i,j;
  int bn, qn; /* basis count, quadrature count */

  PetscFunctionBegin;
  bn = 4;
  qn = 4;
  /* need to go over each element , then each variable */
  for( i = 0; i < bn; i++ ){ /* loop over basis functions */
    phi->rhsresult[i] = 0.0; 
    for( j = 0; j < qn; j++ ){ /* loop over Gauss points */
      phi->rhsresult[i] +=  phi->weights[j] *f(phi->x[j], phi->y[j])*(phi->RefVal[i][j])*PetscAbsDouble(phi->detDh[j]); 
   }
 }
 PetscFunctionReturn(0);
}

/* ComputeStiffness: computes integrals of gradients of local phi_i and phi_j on the given quadrangle 
     by changing variables to the reference quadrangle and reference basis elements phi_i and phi_j.  
     The formula used is

     integral (given element) of <grad phi_j', grad phi_i'> =
                                        integral over (ref element) of 
                                      <(grad phi_j composed with h)*(grad h)^-1,
                                      (grad phi_i composed with h)*(grad h)^-1>*det(grad h).
      this is evaluated by quadrature:
      = sum over Gauss points, above evaluated at gauss pts
*/
#undef __FUNC__
#define __FUNC__ "ComputeStiffness"
int ComputeStiffnessElement( AppElement *phi )
{
  int i,j,k;
  int bn, qn; /* basis count, quadrature count */

  PetscFunctionBegin;
  bn = 4;  
  qn = 4;
  /* Stiffness Terms */
  /* could even do half as many by exploiting symmetry  */
  for( i=0;i<bn;i++ ){ /* loop over first basis fn */
    for( j=0; j<bn; j++){ /* loop over second */
      phi->stiffnessresult[i][j] = 0;
    }
  }

  /* Now Integral.  term is <DphiDhinv[i],DphiDhinv[j]>*abs(detDh) */
  for( i=0;i<bn;i++ ){ /* loop over first basis fn */
    for( j=0; j<bn; j++){ /* loop over second */
      for(k=0;k<qn;k++){ /* loop over Gauss points */
        phi->stiffnessresult[i][j] += phi->weights[k]*
                  (phi->dx[i][k]*phi->dx[j][k] + phi->dy[i][k]*phi->dy[j][k])*
	          PetscAbsDouble(phi->detDh[k]);
      }
    }
  }
  PetscFunctionReturn(0);
}




