#include "appctx.h"

#undef __FUNC__
#define __FUNC__ "AppCxtSetRhs"
int AppCtxSetRhs(AppCtx* appctx)
{
  /********* Collect context informatrion ***********/
  AppGrid    *grid = &appctx->grid;
  AppAlgebra *algebra = &appctx->algebra;
  AppElement *phi = &appctx->element;

  int ierr, i;
  int *vertex_ptr;
  int bn =4; /* basis count */
  int vertexn = 4; /* degree of freedom count */

  /* loop over cells */
  for(i=0;i<grid->cell_n;i++){
    /* loop over degrees of freedom and cell coords */

    /* vertex_ptr points to place in the vector to set the values */
    vertex_ptr = grid->cell_vertex + vertexn*i; 
    /* coords_ptr points to the coordinates of the current cell */
    phi->coords = grid->cell_coords + 2*bn*i;  /*number of cell coords */

    /* compute the values of basis functions on this element */
    SetLocalElement(phi); 

    /* compute the  element load (integral of f with the 4 basis elements)  */
    /* values get put into phi->rhsresult  */
    ComputeRHS(phi);

    /*********  Set Values *************/
    ierr = VecSetValuesLocal(algebra->b, bn, vertex_ptr, phi->rhsresult, ADD_VALUES); CHKERRQ(ierr);
  }
  /********* Assemble Data **************/
  ierr = VecAssemblyBegin(algebra->b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(algebra->b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}  

/* need to compute u from w by convolving with kernel, then multiply my grad_perp */
/* once u is computed, construct the nonlinear matrix */
#undef __FUNC__
#define __FUNC__ "ComputeNonlinear"
int ComputeNonlinear(AppCtx* appctx)
{

  PetscFunctionBegin;

  /* STEP ONE, compute the kernel */

  /* STEP TWO apply grad_perp */

  /* STEP THREE construct nonlinear matrix u grad()*/

  PetscFunctionReturn(0);
}


#undef __FUNC__
#define __FUNC__ "AppCxtSetStiffness"
int AppCtxSetStiffness(AppCtx* appctx)
{
/********* Collect contex informatrion ***********/
  AppAlgebra *algebra = &appctx->algebra;
  AppGrid    *grid    = &appctx->grid;
  AppElement *phi = &appctx->element; 

/****** Internal Variables ***********/
  int i, ii, ierr;
  int *vertex_ptr;
  int bn =4; /* basis count */
  int vertexn = 4; /* degree of freedom count */
  double *result;

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
    /* the result is the integral of laplacian, since the problem is (minus) laplacian = f, 
       multiply result by -1 */
    ierr = ComputeStiffness(phi); CHKERRQ(ierr);
    result = (double *)phi->stiffnessresult;
    for(ii=0;ii<16;ii++){result[ii]= -result[ii];}

    /*********  Set Values *************/
    ierr = MatSetValuesLocal(algebra->A, vertexn, vertex_ptr, vertexn, vertex_ptr, result, ADD_VALUES);CHKERRQ(ierr);
  }
  /********* Assemble Data **************/
  ierr = MatAssemblyBegin(algebra->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(algebra->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNC__
#define __FUNC__ "AppCxtSetMassMatrix"
int AppCtxSetMassMatrix(AppCtx* appctx)
{
/********* Collect contex informatrion ***********/
  AppAlgebra *algebra = &appctx->algebra;
  AppGrid    *grid    = &appctx->grid;
  AppElement *phi = &appctx->element; 

/****** Internal Variables ***********/
  int i, ii, ierr;
  int *vertex_ptr;
  int bn =4; /* basis count */
  int vertexn = 4; /* degree of freedom count */
  double *result;

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
    /* the result is the integral of laplacian, since the problem is (minus) laplacian = f, 
       multiply result by -1 */
    ierr = ComputeMassMatrix(phi); CHKERRQ(ierr);
    result = (double *)phi->vmass;
     /*********  Set Values *************/
    ierr = MatSetValuesLocal(algebra->M, vertexn, vertex_ptr, vertexn, vertex_ptr, result, ADD_VALUES);CHKERRQ(ierr);
  }
  /********* Assemble Data **************/
  ierr = MatAssemblyBegin(algebra->M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(algebra->M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNC__
#define __FUNC__ "AppCxtSetGradPerp"
int AppCtxSetGradPerp(AppCtx* appctx)
{
/********* Collect contex informatrion ***********/
  AppAlgebra *algebra = &appctx->algebra;
  AppGrid    *grid    = &appctx->grid;
  AppElement *phi = &appctx->element; 

/****** Internal Variables ***********/
  int i, ii, ierr;
  int *vertex_ptr;
  int bn =4; /* basis count */
  int vertexn = 4; /* degree of freedom count */
  double *result;

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
    /* the result is the integral of laplacian, since the problem is (minus) laplacian = f, 
       multiply result by -1 */
    ierr = ComputeGradPerp(phi); CHKERRQ(ierr);
    result = (double *)phi->gradperp;
  
    /*********  Set Values *************/
    ierr = MatSetValuesLocal(algebra->M, vertexn, vertex_ptr, 2*vertexn, vertex_ptr, result, ADD_VALUES);CHKERRQ(ierr);
  }
  /********* Assemble Data **************/
  ierr = MatAssemblyBegin(algebra->M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(algebra->M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "SetBoundaryConditions"
int SetBoundaryConditions(AppCtx *appctx)
{
 /********* Collect context informatrion ***********/
  AppElement *phi = &appctx->element;
  AppAlgebra *algebra = &appctx->algebra;
  AppGrid *grid = &appctx->grid;

  int ierr, i;
  double   xval, yval; 
  int *vertex_ptr; 

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


#undef __FUNC__
#define __FUNC__ "SetMatrixBoundaryConditions"
int SetMatrixBoundaryConditions(AppCtx *appctx)
{
  double one = 1.0;
  int ierr;
  AppAlgebra *algebra = &appctx->algebra;
  AppGrid *grid = &appctx->grid;

  /**********  boundary conditions ************/
  ierr = MatZeroRowsLocalIS(algebra->A, grid->vertex_boundary,one);CHKERRQ(ierr); 
  
  ierr = MatAssemblyBegin(algebra->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(algebra->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}






