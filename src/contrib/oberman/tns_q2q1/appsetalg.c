
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
  int *df_ptr;
  double *coords_ptr;
  int vbn = phi->vel_basis_count;
  int dfn = phi->df_element_count;

  /* loop over cells */
  for(i=0;i<grid->cell_n;i++){
    /* loop over degrees of freedom and cell coords */
    df_ptr = grid->cell_df + dfn*i;
    coords_ptr = grid->cell_vcoords + 2*vbn*i;  /*number of cell coords */
    /* compute the values of basis functions on this element */
    SetLocalElement(phi, coords_ptr); 
    /* compute the  element load (integral of f with the 4 basis elements)  */
    ComputeRHS( phi );/* f,g are rhs functions */
    /*********  Set Values *************/
    ierr = VecSetValuesLocal(algebra->b, 2*vbn, df_ptr, phi->rhsresult, ADD_VALUES);CHKERRQ(ierr);
  }
  /********* Assemble Data **************/
  ierr = VecAssemblyBegin(algebra->b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(algebra->b);CHKERRQ(ierr);
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
  AppEquations *equations = &appctx->equations;

/****** Internal Variables ***********/
  int i, ierr;
  int *vdf_ptr;
  double *coords_ptr;
  double *vvalues;

  int vbn, pbn, dfn; 
  vbn = phi->vel_basis_count;  
  pbn = phi->p_basis_count;
  dfn = phi->df_element_count;

  PetscFunctionBegin;

   /* loop over cells */
  for(i=0;i<grid->cell_n;i++){
    /* loop over degrees of freedom and cell coords */
    vdf_ptr = grid->cell_df + dfn*i;
    coords_ptr = grid->cell_vcoords + 2*vbn*i;
    /* compute the values of basis functions on this element */
    ierr = SetLocalElement(phi, coords_ptr); CHKERRQ(ierr);
    /*  Now Compute the element mass  */  
    ierr = ComputeMass(phi); CHKERRQ(ierr);
    /* spread the values puts the u, and v mass terms into a 2*vbn x 2*vbn matrix, 
       with zeros  in the in-between points */
    SpreadMassValues(phi);
    vvalues = (double *)phi->vresult;
   
    /*********  Set Values *************/
    ierr = MatSetValuesLocal(algebra->M, 2*vbn, vdf_ptr, 2*vbn, vdf_ptr, vvalues, ADD_VALUES);CHKERRQ(ierr);
  }
  /********* Assemble Data **************/
  ierr = MatAssemblyBegin(algebra->M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(algebra->M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
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
  int i, j,ii, jj, ierr;
  int *vdf_ptr, *pdf_ptr;
  double *coords_ptr;
  double  *pvalues, *tpvalues, *vvalues;
  int vbn, pbn, dfn; 

  vbn = phi->vel_basis_count;  
  pbn = phi->p_basis_count;
  dfn = phi->df_element_count;

  PetscFunctionBegin;

   /* loop over cells */
  for(i=0;i<grid->cell_n;i++){
    /* loop over degrees of freedom and cell coords */
    vdf_ptr = grid->cell_df + dfn*i;
    pdf_ptr = grid->cell_df + 2*vbn + dfn*i; /* starting with #16 */
    coords_ptr = grid->cell_vcoords + 2*vbn*i;
    /* compute the values of basis functions on this element */
    ierr = SetLocalElement(phi, coords_ptr); CHKERRQ(ierr);
    /*  Now Compute the element stiffness  and the divergence.
    ( These routines could be done separately)  */  

    /* STIFFNESS */
    /* ComputeStiffness returns a full vbn*vbn matrix containing the integrals of
       grad_phi_i*grad_phi_j. */
    ierr = ComputeStiffness(phi); CHKERRQ(ierr);
    /* scale stiffness by viscosity, and put the right sign in */
    for(ii =0; ii<vbn;ii++){for(jj=0;jj<vbn;jj++){phi->vstiff[ii][jj] *= -equations->eta;}}
    /* spread the values duplicates the puts the u, and v stiffness terms into a 2*vbn x 2*vbn matrix, with zeros  in the in-between points */
    SpreadValues(phi);
    vvalues = (double *)phi->vresult;
    
    /* D - pressure, divergence */
    /* Now Compute the pressure, the integral of (negative) partial of phi against psi */
    ierr = ComputePressure( phi ); CHKERRQ(ierr);
    pvalues = (double *)phi->presult;
    /* get the transpose values as well */
    TransposeValues( phi );
    tpvalues = (double *)phi->tpresult;

    /*********  Set Values *************/
       /* stiffness term */
    ierr = MatSetValuesLocal(algebra->A, 2*vbn, vdf_ptr, 2*vbn, vdf_ptr, vvalues, ADD_VALUES);CHKERRQ(ierr);
    /* pressure/incompressiblilty term */
    ierr = MatSetValuesLocal(algebra->A, pbn, pdf_ptr, 2*vbn, vdf_ptr, pvalues, ADD_VALUES);CHKERRQ(ierr);
    ierr = MatSetValuesLocal(algebra->A, 2*vbn, vdf_ptr, pbn, pdf_ptr, tpvalues, ADD_VALUES);CHKERRQ(ierr);
  /* TEMPORARY penalty term */
     for(jj=0;jj<4;jj++){ 
     ierr = MatSetValuesLocal(algebra->A, 1, pdf_ptr+jj, 1, pdf_ptr +jj, &equations->penalty, ADD_VALUES);CHKERRQ(ierr);} 

  }

  /********* Assemble Data **************/
  ierr = MatAssemblyBegin(algebra->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(algebra->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

int TransposeValues(AppElement *phi){
  int vbn, pbn; 
  int i,j;

  vbn = phi->vel_basis_count;  
  pbn = phi->p_basis_count;

  for(i=0;i<pbn;i++){
    for(j=0;j<2*vbn;j++){
      phi->tpresult[j][i] = phi->presult[i][j];
    }
  }
PetscFunctionReturn(0);
}


int SpreadMassValues(AppElement *phi){
  int i,j;  
  int vbn, vqn, pbn; /* basis count, quadrature count */
  vbn = phi->vel_basis_count;  vqn = phi->vel_quad_count;
  pbn = phi->p_basis_count;
  
  /*first zero result*/
  for(i=0; i<2*vbn; i++){/* pressure basis fn loop */
    for( j=0; j<2*vbn; j++){ /* velocity basis fn loop  */
      phi->vresult[i][j] = 0;
    }
  }
  
  /* now spread the values: even rows alternate value, zero, 
     odd rows, alternate zero value (i.e no cross terms in the stiffness matrix )*/
  for(i=0;i<vbn;i++){ /* rows */
    /* even numbered rows */
    for(j=0;j<vbn;j++){ /* columns */
      phi->vresult[2*i][2*j] = phi->vmass[i][j];
      phi->vresult[2*i][2*j+1] = 0;
    }
    /* odd numbered rows */
    for(j=0;j<vbn;j++){ /* columns */
      phi->vresult[2*i+1][2*j] = 0;
      phi->vresult[2*i+1][2*j+1] = phi->vmass[i][j];
    }
  }
  PetscFunctionReturn(0);
}

int SpreadValues(AppElement *phi){

  int i,j;  
  int vbn, vqn, pbn; /* basis count, quadrature count */
  vbn = phi->vel_basis_count;  vqn = phi->vel_quad_count;
  pbn = phi->p_basis_count;

  /*first zero result*/
   for(i=0; i<2*vbn; i++){/* pressure basis fn loop */
     for( j=0; j<2*vbn; j++){ /* velocity basis fn loop  */
       phi->vresult[i][j] = 0;
     }
   }
   /* now spread the values: even rows alternate value, zero, 
      odd rows, alternate zero value (i.e no cross terms in the stiffness matrix )*/

  for(i=0;i<vbn;i++){ /* rows */
    /* even numbered rows */
    for(j=0;j<vbn;j++){ /* columns */
      phi->vresult[2*i][2*j] = phi->vstiff[i][j];
      phi->vresult[2*i][2*j+1] = 0;
    }
    /* odd numbered rows */
    for(j=0;j<vbn;j++){ /* columns */
    phi->vresult[2*i+1][2*j] = 0;
    phi->vresult[2*i+1][2*j+1] = phi->vstiff[i][j];
    }
  }
PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "SetPartialDx"
/* input vector is x, output is f.  Loop over elements, getting coords of each vertex and 
computing load vertex by vertex.  Set the values into f.  */
int SetPartialDx(Vec x, AppCtx *appctx, Vec f)
{
/********* Collect context informatrion ***********/
  AppElement *phi = &appctx->element;
  AppAlgebra *algebra = &appctx->algebra;
  AppGrid *grid = &appctx->grid;

/****** Internal Variables ***********/
  double *coords_ptr;
  double  *uvvals;
  int ierr, i, j, ii;
  int *df_ptr;
  int vbn, dfn; 
   vbn = phi->vel_basis_count;    dfn = phi->df_element_count;


 
  /* Scatter the input values from the global vector g, to those on this processor */
   ierr = VecScatterBegin( x, algebra->v1_local, INSERT_VALUES, SCATTER_FORWARD, algebra->dfv1gtol); CHKERRQ(ierr); 
   ierr = VecScatterEnd( x,  algebra->v1_local, INSERT_VALUES, SCATTER_FORWARD, algebra->dfv1gtol); CHKERRQ(ierr); 
  ierr = VecGetArray( algebra->v1_local, &uvvals); CHKERRQ(ierr);

  /* loop over cells */
  for(i=0;i<grid->cell_n;i++){
    /* loop over degrees of freedom and cell coords */
    df_ptr = grid->cell_df + dfn*i;
    coords_ptr = grid->cell_vcoords + 2*vbn*i;
      /* Need to point to the uvvals associated to the velocity degrees of freedom.  */
    for ( j=0; j<vbn; j++){      
      phi->u[j] = uvvals[df_ptr[j]];
 /*      phi->v[j] = uvvals[df_ptr[2*j+1]]; */
    }
    /* compute the values of basis functions on this element */
     ierr = SetLocalElement(phi, coords_ptr);CHKERRQ(ierr);
    /* do the integrals */
    ierr = ComputePartialDx( phi );CHKERRQ(ierr);
    /* put result in */
    /* NOT VECSETVALUESLOCAL */
    ierr = VecSetValuesLocal(f, vbn, df_ptr, phi->result, ADD_VALUES);CHKERRQ(ierr);
    VecView(f,0);
  }
  ierr = VecAssemblyBegin(f);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "SetPartialDy"
/* input vector is x, output is f.  Loop over elements, getting coords of each vertex and 
computing load vertex by vertex.  Set the values into f.  */
int SetPartialDy(Vec x, AppCtx *appctx, Vec f)
{
/********* Collect context informatrion ***********/
  AppElement *phi = &appctx->element;
  AppAlgebra *algebra = &appctx->algebra;
  AppGrid *grid = &appctx->grid;

/****** Internal Variables ***********/
  double *coords_ptr;
  double  *uvvals;
  int ierr, i, j, ii;
  int *df_ptr;
  int vbn, dfn; 
   vbn = phi->vel_basis_count;    dfn = phi->df_element_count;

  /* Scatter the input values from the global vector g, to those on this processor */
  ierr = VecScatterBegin( x, algebra->v1_local, INSERT_VALUES, SCATTER_FORWARD, algebra->dfv1gtol); CHKERRQ(ierr);
  ierr = VecScatterEnd( x,  algebra->v1_local, INSERT_VALUES, SCATTER_FORWARD, algebra->dfv1gtol); CHKERRQ(ierr);
  ierr = VecGetArray( algebra->v1_local, &uvvals); CHKERRQ(ierr);

  /* loop over cells */
  for(i=0;i<grid->cell_n;i++){
    /* loop over degrees of freedom and cell coords */
    df_ptr = grid->cell_df + dfn*i;
    coords_ptr = grid->cell_vcoords + 2*vbn*i;
      /* Need to point to the uvvals associated to the velocity degrees of freedom.  */
    for ( j=0; j<vbn; j++){      
      phi->u[j] = uvvals[df_ptr[j]];
 /*      phi->v[j] = uvvals[df_ptr[2*j+1]]; */
    }
    /* compute the values of basis functions on this element */
     ierr = SetLocalElement(phi, coords_ptr);CHKERRQ(ierr);
    /* do the integrals */
    ierr = ComputePartialDy( phi );CHKERRQ(ierr);
    /* put result in */
    ierr = VecSetValuesLocal(f, vbn, df_ptr, phi->result, ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(f);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



#undef __FUNC__
#define __FUNC__ "SetNonlinearFunction"
/* input vector is x, output is f.  Loop over elements, getting coords of each vertex and 
computing load vertex by vertex.  Set the values into f.  */
int SetNonlinearFunction(Vec x, AppCtx *appctx, Vec f)
{
/********* Collect context informatrion ***********/
  AppElement *phi = &appctx->element;
  AppAlgebra *algebra = &appctx->algebra;
  AppGrid *grid = &appctx->grid;

/****** Internal Variables ***********/
  double *coords_ptr;
  double  *uvvals;
  int ierr, i, j, ii;
  int *df_ptr;
  int vbn, dfn; 
   vbn = phi->vel_basis_count;    dfn = phi->df_element_count;

  /* Scatter the input values from the global vector g, to those on this processor */
  ierr = VecScatterBegin( x, algebra->f_local, INSERT_VALUES, SCATTER_FORWARD, algebra->dfgtol); CHKERRQ(ierr);
  ierr = VecScatterEnd( x,  algebra->f_local, INSERT_VALUES, SCATTER_FORWARD, algebra->dfgtol); CHKERRQ(ierr);
  ierr = VecGetArray( algebra->f_local, &uvvals); CHKERRQ(ierr);

  /* loop over cells */
  for(i=0;i<grid->cell_n;i++){
    /* loop over degrees of freedom and cell coords */
    df_ptr = grid->cell_df + dfn*i;
    coords_ptr = grid->cell_vcoords + 2*vbn*i;
      /* Need to point to the uvvals associated to the velocity degrees of freedom.  */
    for ( j=0; j<vbn; j++){      
      phi->u[j] = uvvals[df_ptr[2*j]];
      phi->v[j] = uvvals[df_ptr[2*j+1]];
    }

    /* compute the values of basis functions on this element */
     ierr = SetLocalElement(phi, coords_ptr);CHKERRQ(ierr);
    /* do the integrals */
    ierr = ComputeNonlinear(phi );CHKERRQ(ierr);
    /* stupid modification: */
    for(ii=0;ii<2*vbn;ii++){phi->nlresult[ii] = -phi->nlresult[ii];}
    /* put result in */
    ierr = VecSetValuesLocal(f, 2*vbn, df_ptr, phi->nlresult, ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(f);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNC__
#define __FUNC__ "SetBoundaryConditions"
int SetBoundaryConditions(Vec x, AppCtx *appctx, Vec f)
{
 /********* Collect context informatrion ***********/
  AppElement *phi = &appctx->element;
  AppAlgebra *algebra = &appctx->algebra;
  AppGrid *grid = &appctx->grid;
  AppEquations *equations = &appctx->equations;
  int ierr, i;
  double   xval, yval; 
  double *inlet_vvals, *outlet_vvals, *wall_vals, *outlet_pvals,  *inlet_pvals; 
  double *vals;
  int *df_ptr; 

  /* Velocity  INLET */
  if (equations->vin_flag){
    ierr = VecScatterBegin( x, algebra->f_vinlet, INSERT_VALUES, SCATTER_FORWARD, algebra->gtol_vinlet);     CHKERRQ(ierr);
    ierr = VecScatterEnd( x,  algebra->f_vinlet, INSERT_VALUES, SCATTER_FORWARD, algebra->gtol_vinlet);   CHKERRQ(ierr);
    ierr = VecGetArray( algebra->f_vinlet, &inlet_vvals); CHKERRQ(ierr);
    ierr = ISGetIndices(grid->isinlet_vdf, &df_ptr); CHKERRQ(ierr);
    /* for each df, have 2 coords */
    for( i=0; i < grid->inlet_vcount; i++){
      equations->xval = grid->inlet_coords[2*i];
      equations->yval = grid->inlet_coords[2*i+1];
      grid->inlet_values[i] = inlet_vvals[i]  - bc1(equations)  ;
      i = i+1;
      equations->xval = grid->inlet_coords[2*i];
      equations->yval = grid->inlet_coords[2*i+1];
      grid->inlet_values[i] = inlet_vvals[i]  - bc2(equations)  ;
    }
    ierr = VecSetValuesLocal(f, grid->inlet_vcount, df_ptr, grid->inlet_values, INSERT_VALUES);  
    ierr = ISRestoreIndices(grid->isinlet_vdf, &df_ptr); CHKERRQ(ierr);
  
  }
   /*VOUTLET*/
  if(equations->vout_flag){
    ierr = VecScatterBegin( x, algebra->f_voutlet, INSERT_VALUES, SCATTER_FORWARD, algebra->gtol_voutlet); CHKERRQ(ierr);
    ierr = VecScatterEnd( x,  algebra->f_voutlet, INSERT_VALUES, SCATTER_FORWARD, algebra->gtol_voutlet);  CHKERRQ(ierr);
    ierr = VecGetArray( algebra->f_voutlet, &outlet_vvals); CHKERRQ(ierr);
    ierr = ISGetIndices(grid->isoutlet_vdf, &df_ptr); CHKERRQ(ierr);
    /* for each df, have 2 coords */
    for( i=0; i < grid->outlet_vcount; i++){
      equations->xval = grid->outlet_coords[2*i];
      equations->yval = grid->outlet_coords[2*i+1];
      grid->outlet_values[i] = outlet_vvals[i]  - bc1(equations)  ;
      i = i+1;
      equations->xval = grid->outlet_coords[2*i];
      equations->yval = grid->outlet_coords[2*i+1];
      grid->outlet_values[i] = outlet_vvals[i]  - bc2(equations)  ;
    }
    ierr = VecSetValuesLocal(f, grid->outlet_vcount, df_ptr, grid->outlet_values, INSERT_VALUES);
    ierr = ISRestoreIndices(grid->isoutlet_vdf, &df_ptr); CHKERRQ(ierr);
 }

 /* Pressure */
   /* POUTLET */
  if(equations->pout_flag){
    ierr = VecScatterBegin( x, algebra->f_poutlet, INSERT_VALUES, SCATTER_FORWARD, algebra->gtol_poutlet);   CHKERRQ(ierr);
    ierr = VecScatterEnd( x,  algebra->f_poutlet, INSERT_VALUES, SCATTER_FORWARD, algebra->gtol_poutlet);   CHKERRQ(ierr);
    ierr = VecGetArray( algebra->f_poutlet, &outlet_pvals); CHKERRQ(ierr);    
    ierr = ISGetIndices(grid->isoutlet_pdf, &df_ptr); CHKERRQ(ierr);
    ierr = VecSetValuesLocal(f, grid->outlet_pcount, df_ptr, outlet_pvals, INSERT_VALUES); CHKERRQ(ierr); 
   ierr = ISRestoreIndices(grid->isoutlet_pdf, &df_ptr); CHKERRQ(ierr);
printf("ERROR: currently just setting pressure outlet to zero\n");

  }

 /* PINLET  */
  if (equations->pin_flag){
    ierr = VecScatterBegin( x, algebra->f_pinlet, INSERT_VALUES, SCATTER_FORWARD, algebra->gtol_pinlet);     CHKERRQ(ierr);
  ierr = VecScatterEnd( x,  algebra->f_pinlet, INSERT_VALUES, SCATTER_FORWARD, algebra->gtol_pinlet);   CHKERRQ(ierr);
  ierr = VecGetArray( algebra->f_pinlet, &inlet_pvals); CHKERRQ(ierr);
  /* Here Setting Pressure to 01 */
  for( i=0; i < grid->inlet_pcount; i++){ 
    grid->inlet_pvalues[i] = inlet_pvals[i] - 10; 
  } 
  printf("ERROR: the pressure boundary conditions need to be corrected\n");
/*   ierr = VecSetValuesLocal(f, grid->inlet_pcount, grid->inlet_pdf, grid->inlet_pvalues, INSERT_VALUES);      CHKERRQ(ierr);  */
  }

  /* WALL */
  if(equations->wall_flag){
    ierr = VecScatterBegin( x, algebra->f_wall, INSERT_VALUES, SCATTER_FORWARD, algebra->gtol_wall); CHKERRQ(ierr);
    ierr = VecScatterEnd( x,  algebra->f_wall, INSERT_VALUES, SCATTER_FORWARD, algebra->gtol_wall); CHKERRQ(ierr);
    ierr = VecGetArray( algebra->f_wall, &wall_vals); CHKERRQ(ierr);
    ierr = ISGetIndices(grid->iswall_vdf, &df_ptr); CHKERRQ(ierr);
  /* just need to set these values to f */
    if(equations->dirichlet_flag){
  /* for each df, have 2 coords */
    for( i=0; i < grid->wall_vcount; i++){
      equations->xval = grid->wall_coords[2*i];
      equations->yval = grid->wall_coords[2*i+1];
      grid->wall_values[i] = wall_vals[i]  - bc1(equations)  ;
      i = i+1;
      equations->xval = grid->wall_coords[2*i];
      equations->yval = grid->wall_coords[2*i+1];
      grid->wall_values[i] = wall_vals[i]  - bc2(equations)  ;
    }
    ierr = VecSetValuesLocal(f, grid->wall_vcount, df_ptr, grid->wall_values, INSERT_VALUES);   CHKERRQ(ierr);
    }
    else{
      ierr = VecSetValuesLocal(f, grid->wall_vcount, df_ptr, wall_vals, INSERT_VALUES);   CHKERRQ(ierr);}
    ierr = ISRestoreIndices(grid->iswall_vdf, &df_ptr); CHKERRQ(ierr);
  }
  /* YWALL */
  if(equations->ywall_flag){
    ierr = VecScatterBegin( x, algebra->f_ywall, INSERT_VALUES, SCATTER_FORWARD, algebra->gtol_ywall); CHKERRQ(ierr);
    ierr = VecScatterEnd( x,  algebra->f_ywall, INSERT_VALUES, SCATTER_FORWARD, algebra->gtol_ywall); CHKERRQ(ierr);
    ierr = VecGetArray( algebra->f_ywall, &vals); CHKERRQ(ierr);
    ierr = ISGetIndices(grid->isywall_vdf, &df_ptr); CHKERRQ(ierr);
  /* just need to set these values to f */
    ierr = VecSetValuesLocal(f, grid->ywall_vcount, df_ptr, vals, INSERT_VALUES);   CHKERRQ(ierr);
    ierr = ISRestoreIndices(grid->isywall_vdf, &df_ptr); CHKERRQ(ierr);
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
  int  i,j, ii, ierr;
  int  *df_ptr; 
  double *coords_ptr;
  double   *uvvals, cell_values[2*9];
  double values[2*9*2*9];  /* the integral of the combination of phi's */
  double one = 1.0;
  int vbn, dfn;
  vbn = phi->vel_basis_count;  
  dfn = phi->df_element_count;

  PetscFunctionBegin;

  /* Matrix is set to the linear part already, so just ADD_VALUES the nonlinear part  */ 
  ierr = VecScatterBegin(g, algebra->f_local, INSERT_VALUES, SCATTER_FORWARD, algebra->dfgtol); CHKERRQ(ierr);
  ierr = VecScatterEnd(g, algebra->f_local, INSERT_VALUES, SCATTER_FORWARD, algebra->dfgtol); CHKERRQ(ierr);
  ierr = VecGetArray(algebra->f_local, &uvvals);
 
  /* loop over cells */
  for(i=0;i<grid->cell_n;i++){
   /* loop over degrees of freedom and cell coords */
    df_ptr = grid->cell_df + dfn*i;
    coords_ptr = grid->cell_vcoords + 2*vbn*i;
    /* Need to point to the uvvals associated to the velocity dfs (can ignore pressure) */
    for ( j=0; j<2*vbn; j++){      cell_values[j] = uvvals[df_ptr[j]];  }
    /* compute the values of basis functions on this element */
    ierr = SetLocalElement(phi, coords_ptr);CHKERRQ(ierr);
    /*    Compute the partial derivatives of the nonlinear map    */  
    ierr = ComputeJacobian( phi, cell_values,  values );CHKERRQ(ierr);
  /* stupid modification: */
    for(ii=0;ii<2*vbn*2*vbn;ii++){values[ii] = -values[ii];}
    /*  Set the values in the matrix */
    ierr  = MatSetValuesLocal(*jac,2*vbn,df_ptr,2*vbn,df_ptr,values,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

 PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "SetJacobianBoundaryConditions"
int SetJacobianBoundaryConditions(AppCtx *appctx, Mat* jac)
{
  double one = 1.0;
  int ierr;
  AppGrid    *grid    = &appctx->grid;
  AppEquations *equations = &appctx->equations;

  /**********  boundary conditions ************/
  /* here should zero rows corresponding to dfs where bc imposed */
  if(equations->wall_flag){      
    ierr = MatZeroRowsLocalIS(*jac, grid->iswall_vdf,one);CHKERRQ(ierr);  } 
  if(equations->ywall_flag){      
    ierr = MatZeroRowsLocalIS(*jac, grid->isywall_vdf,one);CHKERRQ(ierr);  } 
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



