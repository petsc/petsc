
#include "appctx.h"

/*----------------------------------------------------------------------------
         Sets up the non-linear system associated with the PDE and solves it
*/

extern PetscErrorCode FormInitialGuess(AppCtx*);
extern PetscErrorCode SetBoundaryConditions(Vec,AppCtx *,Vec);
extern PetscErrorCode SetJacobian(Vec,AppCtx *,Mat*);

#undef __FUNCT__
#define __FUNCT__ "AppCxtSolve"
PetscErrorCode AppCtxSolve(AppCtx* appctx)
{
  AppAlgebra     *algebra = &appctx->algebra;
  SNES           snes;
  PetscErrorCode ierr;
  PetscInt       its;
 
  PetscFunctionBegin;

  /*  1) Create vector to contain load and various work vectors  */
  ierr = AppCtxCreateVector(appctx);CHKERRQ(ierr);

  /*  2) Create the sparse matrix, with correct nonzero pattern  */
  ierr = AppCtxCreateMatrix(appctx);CHKERRQ(ierr);

  /*  A) Set the quadrature values for the reference square element  */
  ierr = AppCtxSetReferenceElement(appctx);CHKERRQ(ierr);

  /*  3) Set the right hand side values into the vectors   */
  ierr = AppCtxSetRhs(appctx);CHKERRQ(ierr);

  /*  4) Set the stiffness matrix entries   */
  /* The coeff of diffusivity.  LATER call a function set equations */
  appctx->equations.eta =-0.04;  
  ierr = AppCtxSetMatrix(appctx);CHKERRQ(ierr);
  /* MatView(algebra->A,PETSC_VIEWER_STDOUT_SELF); */

  /*  5) Create the nonlinear solver context  */
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);

  /*  6) Set function evaluation rountine and vector */
  ierr = SNESSetFunction(snes,algebra->f,FormStationaryFunction,(void*)appctx);CHKERRQ(ierr);
  
  /*  7) Set Jacobian   */ 
  ierr = SNESSetJacobian(snes,algebra->J,algebra->J,FormStationaryJacobian,(void*)appctx);CHKERRQ(ierr);
  
  /*  8) Set Solver Options,could put internal options here      */
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  
  /*  9) Initial guess */
  ierr = FormInitialGuess(appctx);CHKERRQ(ierr); 
  
  /*  10) Solve the non-linear system  */
  ierr = SNESSolve(snes,PETSC_NULL,algebra->g);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);

  if(0){VecView(algebra->g,PETSC_VIEWER_STDOUT_SELF);}

  printf("the number of its, %d\n",(int)its);

  ierr = SNESDestroy(snes);CHKERRQ(ierr);  
  PetscFunctionReturn(0);
}

/*---------------------------------------------------------------------------- 
  A) Set the quadrature values for the reference square element

The following functions set the reference element, and the local element for the quadrature.  
Set reference element is called only once, at initialization, while set reference element 
must be called over each element.  */
PetscErrorCode AppCtxSetReferenceElement(AppCtx* appctx){

  AppElement *phi = &appctx->element;
  double psi,psi_m,psi_p,psi_pp,psi_mp,psi_pm,psi_mm;

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

/*-------------------------------------------------------------*/ 
/* B) called by 3) and 4) */

PetscErrorCode SetLocalElement(AppElement *phi,double *coords)
{
  PetscInt i,j,k;
  double Dh[4][2][2],Dhinv[4][2][2];
  double *dx = phi->dx,*dy = phi->dy;
  double *detDh = phi->detDh;
  double *x = phi->x,*y = phi->y;  /* image of gauss point */

  /* Could put in a flag to skip computing this when it isn't needed */

  /* the image of the reference element is given by sum (coord i)*phi_i */
    for(j=0;j<4;j++){ /* loop over gauss points */
      x[j] = 0; y[j] = 0;
      for(k=0;k<4;k++){
        x[j] += coords[2*k]*phi->Values[k][j];
        y[j] += coords[2*k+1]*phi->Values[k][j];
      }
    }
  /* Jacobian */
  for(i=0;i<4;i++){ /* loop over Gauss points */
    Dh[i][0][0] = 0; Dh[i][0][1] = 0; Dh[i][1][0] = 0; Dh[i][1][1] = 0;
    for(k=0; k<4; k++){
      Dh[i][0][0] += coords[2*k]*phi->DxValues[k][i];
      Dh[i][0][1] += coords[2*k]*phi->DyValues[k][i];
      Dh[i][1][0] += coords[2*k+1]*phi->DxValues[k][i];
      Dh[i][1][1] += coords[2*k+1]*phi->DyValues[k][i];
    }
  }

  /* Determinant of the Jacobian */
  for(j=0; j<4; j++){   /* loop over Gauss points */
    detDh[j] = PetscAbsReal(Dh[j][0][0]*Dh[j][1][1] - Dh[j][0][1]*Dh[j][1][0]);
  }
  /* Inverse of the Jacobian */
    for(j=0; j<4; j++){   /* loop over Gauss points */
      Dhinv[j][0][0] = Dh[j][1][1]/detDh[j];
      Dhinv[j][0][1] = -Dh[j][0][1]/detDh[j];
      Dhinv[j][1][0] = -Dh[j][1][0]/detDh[j];
      Dhinv[j][1][1] = Dh[j][0][0]/detDh[j];
    }
       
    /* Notice that phi~ = phi(h), so Dphi~ = Dphi*Dh, so Dphi~ = Dphi*(Dh)inv */

    /* partial of phi at h(gauss pt) times Dhinv */
    /* loop over gauss, the basis fns, then d/dx or d/dy */
    for(i=0;i<4;i++){  /* loop over Gauss points */
      for(j=0;j<4;j++){ /* loop over basis functions */
        dx[4*j+i] = phi->DxValues[j][i]*Dhinv[i][0][0] +  phi->DyValues[j][i]*Dhinv[i][1][0];
        dy[4*j+i] = phi->DxValues[j][i]*Dhinv[i][0][1] + phi->DyValues[j][i]*Dhinv
[i][1][1];
      }
    }
PetscFunctionReturn(0);
}  

/*------------------------------------------------------------------------
  1) Create vector to contain load and various work vectors  

         -  Generates the "global" parallel vector to contain the 
	    right hand side and solution.
         -  Generates "ghosted" local vectors for local computations etc.
         -  Generates scatter context for updating ghost points etc.
*/
#undef __FUNCT__
#define __FUNCT__ "AppCxtCreateVector"
PetscErrorCode AppCtxCreateVector(AppCtx* appctx)
{
  /* Want everything here to be DF driven, not vertex driven 

Output of CreateVec:
  Global vectors:
    - b for rhs
    - f for nonlinear function
    - g for solution, initial guess
    - *solnv for time step solutions
  Local Vectors:
    - f_local for local part of nonlinear function
  Scatter:
    - dfgtol
*/
/********* Collect context informatrion ***********/
  AppGrid                *grid = &appctx->grid;
  AppAlgebra             *algebra = &appctx->algebra;
  MPI_Comm               comm = appctx->comm;
 
 /************* Variables to set ************/
/* global to local mapping for vectors */

 /********** Internal Variables **********/
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  Create vector to contain load, nonlinear function, and initial guess  
   This is driven by the DFs: local size should be number of  DFs on this proc.  */
  ierr = VecCreateMPI(comm,grid->df_local_count,PETSC_DECIDE,&algebra->b);CHKERRQ(ierr);
  ierr = VecSetLocalToGlobalMapping(algebra->b,grid->dfltog);CHKERRQ(ierr);
 
  /* For load and solution vectors.  Duplicated vectors inherit the blocking */
  ierr = VecDuplicate(algebra->b,&algebra->f);CHKERRQ(ierr);/*  the nonlinear function */
  ierr = VecDuplicate(algebra->b,&algebra->g);CHKERRQ(ierr);/*  the initial guess  */
  ierr = VecDuplicateVecs(algebra->b,NSTEPS+1,&algebra->solnv);CHKERRQ(ierr);  /* the soln at each time step */
  /* later dynamically make block of size 16 of soltution vectors for dynamic time stepping */

  /* The size of f_local should be the number of dfs including those corresponding to ghosted vertices on this proc 
 (should be equal in burger case to 2*vertex_n_ghosted)*/
  ierr = VecCreateSeq(PETSC_COMM_SELF,grid->df_n_ghosted,&algebra->f_local);CHKERRQ(ierr);
  ierr = VecScatterCreate(algebra->f,grid->df_global,algebra->f_local,0,&algebra->dfgtol);CHKERRQ(ierr);

 /* here create a vector and a scatter for the boundary vertices */
ierr = VecCreateSeq(PETSC_COMM_SELF,2*grid->vertex_boundary_count,&algebra->f_boundary);CHKERRQ(ierr);
  ierr = VecScatterCreate(algebra->f,grid->isboundary_df,algebra->f_boundary,0,&algebra->dfbgtol);CHKERRQ(ierr);
 

  /* for vecscatter,second argument is the IS to scatter.  the fourth argument is the index set to scatter to.  Putting in 0 defaults to {0,..., sizeofvec -1} */  
  
  /************ STOP here, figure out the vectors needed for MatCreate later on ***********/
  /* Check the old version to see what was done */

  PetscFunctionReturn(0);
}

/*-----------------------------------------------------------------------------
  2) Create the sparse matrix, with correct nonzero patter

     Creates the sparse matrix (with the correct nonzero pattern) 
We need a matrix for, e.g the stiffness, and one for the jacobian of the nonlinear map.
Generally, the size of the matrix is df_total * df_total, 
but we need to compute the nonzero structure for efficiency purposes.
DO LATER
*/
#undef __FUNCT__
#define __FUNCT__ "AppCxtCreateMatrix"
PetscErrorCode AppCtxCreateMatrix(AppCtx* appctx)
{
/********* Collect context informatrion ***********/
  AppAlgebra             *algebra = &appctx->algebra;
  AppGrid                *grid    = &appctx->grid;
  MPI_Comm               comm = appctx->comm;
PetscErrorCode ierr; 
  PetscFunctionBegin;
  /* now create the matrix */
  /* using very rough estimate for nonzeros on and off the diagonal */
  ierr = MatCreateMPIAIJ(comm,grid->df_local_count,grid->df_local_count,PETSC_DETERMINE,PETSC_DETERMINE,10,0,5,0,&algebra->A);CHKERRQ(ierr);
  /* Set the local to global mapping */
  ierr = MatSetLocalToGlobalMapping(algebra->A,grid->dfltog);CHKERRQ(ierr);

  /* ditto for jac */
  ierr = MatCreateMPIAIJ(comm,grid->df_local_count,grid->df_local_count,PETSC_DETERMINE,PETSC_DETERMINE,10,0,5,0,&algebra->J);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(algebra->J,grid->dfltog);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*----------------------------------------------------------------*/
/*  3) Set the right hand side values into the vectors.    
       Called B)                                         */

#undef __FUNCT__
#define __FUNCT__ "AppCxtSetRhs"
PetscErrorCode AppCtxSetRhs(AppCtx* appctx)
{
  /********* Collect context informatrion ***********/
  AppGrid    *grid = &appctx->grid;
  AppAlgebra *algebra = &appctx->algebra;
  AppElement *phi = &appctx->element;

  PetscErrorCode ierr,i;
  PetscInt *df_ptr;
  double *coords_ptr;
  double  values[8];

  /* set flag for element computation */
  phi->dorhs = 1;

 /* loop over cells */
  for(i=0;i<grid->cell_n;i++){
    /* loop over degrees of freedom and cell coords */
    df_ptr = grid->cell_df + 8*i;
    coords_ptr = grid->cell_coords + 8*i;

    /* compute the values of basis functions on this element */
    SetLocalElement(phi,coords_ptr);

    /* compute the  element load (integral of f with the 4 basis elements)  */
    ComputeRHS(pde_f,pde_g,phi,values);/* f,g are rhs functions */

    /*********  Set Values *************/
    ierr = VecSetValuesLocal(algebra->b,8,df_ptr,values,ADD_VALUES);CHKERRQ(
ierr);
  }

  /********* Assemble Data **************/
  ierr = VecAssemblyBegin(algebra->b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(algebra->b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}     

PetscErrorCode ComputeRHS(DFP f,DFP g,AppElement *phi,double *integrals){
  PetscInt i,j;
  /* need to go over each element,then each variable */
 for(i = 0; i < 4; i++){ /* loop over basis functions */
   integrals[2*i] = 0.0;
   integrals[2*i+1] = 0.0;
   for(j = 0; j < 4; j++){ /* loop over Gauss points */
     integrals[2*i] +=  f(phi->x[j],phi->y[j])*(phi->Values[i][j])*
       PetscAbsReal(phi->detDh[j]);
     integrals[2*i+1] +=  g(phi->x[j],phi->y[j])*(phi->Values[i][j])*
       PetscAbsReal(phi->detDh[j]);
   }
 }
PetscFunctionReturn(0);
}   

/*---------------------------------------------------------*/
/*  4) Set the stiffness matrix entries   
       Called B)                                           */

#undef __FUNCT__
#define __FUNCT__ "AppCxtSetMatrix"
PetscErrorCode AppCtxSetMatrix(AppCtx* appctx)
{
/********* Collect contex informatrion ***********/
  AppAlgebra *algebra = &appctx->algebra;
  AppGrid    *grid    = &appctx->grid;
  AppElement *phi = &appctx->element;
  AppEquations *equations = &appctx->equations;

/****** Internal Variables ***********/
  PetscInt i,*df_ptr;
  PetscErrorCode ierr;
  double *coords_ptr;
  double values[8*8];

  PetscFunctionBegin;
  /************ Set Up **************/
  /* set flag for phi computation */
  phi->dorhs = 0;
  /* loop over cells */
  for(i=0;i<grid->cell_n;i++){
    /* loop over degrees of freedom and cell coords */
    df_ptr = grid->cell_df + 8*i;
    coords_ptr = grid->cell_coords + 8*i;

    /* compute the values of basis functions on this element */
    ierr = SetLocalElement(phi,coords_ptr);CHKERRQ(ierr);

    /*    Compute the element stiffness    */
    ierr = ComputeMatrix(phi,values);CHKERRQ(ierr);
    /*********  Set Values *************/
    ierr = MatSetValuesLocal(algebra->A,8,df_ptr,8,df_ptr,values,ADD_VALUES);CHKERRQ(ierr);
  }

  /********* Assemble Data **************/
  ierr = MatAssemblyBegin(algebra->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(algebra->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  /********** Multiply by the viscosity coeff ***************/
  ierr = MatScale(algebra->A,equations->eta);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}   

/*
ComputeMatrix:
   computes integrals of gradients of local phi_i and phi_j on the given quadrangle
by changing variables to the reference quadrangle and reference basis elements phi_i
and phi_j.  The formula used is

integral (given element) of <grad phi_j', grad phi_i'> =
integral over (ref element) of
    <(grad phi_j composed with h)*(grad h)^-1,
     (grad phi_i composed with h)*(grad h)^-1>*det(grad h).
this is evaluated by quadrature:
= sum over gauss points, above evaluated at gauss pts
*/
PetscErrorCode ComputeMatrix(AppElement *phi,double *result){
   PetscInt i,j,k;

  /* Stiffness Terms */
  /* Now Integral.  term is <DphiDhinv[i],DphiDhinv[j]>*abs(detDh) */
   for(i=0;i<4;i++){ /* loop over first basis fn */
     for(j=0; j<4; j++){ /* loop over second */
       /* keep in mind we are throwing in a 2x2 block for each 1x1 */
       result[16*i + 2*j] = 0;
       result[16*i + 2*j+1] = 0;
       result[16*i + 8 +2*j] = 0;
       result[16*i + 9 +2*j] = 0;

       /* funny ordering of 2x2 blocks in the 4x4 piece */
       for(k=0;k<4;k++){ /* loop over gauss points */
         result[16*i + 2*j] +=
           (phi->dx[4*i+k]*phi->dx[4*j+k] +
            phi->dy[4*i+k]*phi->dy[4*j+k])*
           PetscAbsReal(phi->detDh[k]);
       }
       /* the off-diagonals stay zero */
       for(k=0;k<4;k++){ /* loop over gauss points */
         result[16*i +9 + 2*j] +=
           (phi->dx[4*i+k]*phi->dx[4*j+k] +
            phi->dy[4*i+k]*phi->dy[4*j+k])*
           PetscAbsReal(phi->detDh[k]);
       }
     }
   }
PetscFunctionReturn(0);
}                       

/*---------------------------------------------------------------------------- 
    6) Set function evaluation rountine and vector 

FormStationaryFunction - Evaluates the nonlinear function, F(x), 
        which is the discretised equations, 
   Input Parameters:
    - the vector x, corresponding to u values at each vertex
    - snes, the SNES context
    - appctx
   Output Parameter:
    - f, the value of the function
*/
#undef __FUNCT__
#define __FUNCT__ "FormStationaryFunction"
PetscErrorCode FormStationaryFunction(SNES snes,Vec x,Vec f,void *dappctx)
{
/********* Collect context informatrion ***********/
  AppCtx *appctx = (AppCtx *)dappctx;
  AppAlgebra *algebra = &appctx->algebra;

/* Later may want to have these computed from here, with a flag passed 
to see if they need to be recomputed */
  /* A is the (already computed) linear part*/
  Mat A = algebra->A;

  /* b is the (already computed) rhs */ 
  Vec  b = algebra->b;
  /* Internal Variables */
  PetscErrorCode ierr;
  double zero = 0.0,mone = -1.0;

/****** Perform computation ***********/
  /* need to zero f */
  ierr = VecSet(f,zero);CHKERRQ(ierr); 

  /* add rhs to get constant part */
  ierr = VecAXPY(f,mone,b);CHKERRQ(ierr); /* this says f = f - 1*b */

  /*apply matrix to the input vector x, to get linear part */
  /* Assuming matrix doesn't need to be recomputed */
  ierr = MatMultAdd(A,x,f,f);CHKERRQ(ierr);  /* f = A*x + f */

 /* create nonlinear part */
  ierr = SetNonlinearFunction(x,appctx,f);CHKERRQ(ierr);

/*  printf("output of nonlinear fun (before bc imposed)\n");    */
/*   ierr = VecView(f,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);     */

  ierr = SetBoundaryConditions(x,appctx,f);CHKERRQ(ierr);
 /* printf("output of nonlinear fun \n");    */
/*    ierr = VecView(f,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);     */
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetNonlinearFunction"
/* input vector is g, output is f.  Loop over elements, getting coords of each vertex and 
computing load vertex by vertex.  Set the values into f.  */
PetscErrorCode SetNonlinearFunction(Vec g,AppCtx *appctx,Vec f)
{
/********* Collect context informatrion ***********/
  AppElement *phi = &appctx->element;
  AppAlgebra *algebra = &appctx->algebra;
  AppGrid *grid = &appctx->grid;
 
/****** Internal Variables ***********/
  double result[8];
  double *coords_ptr;
  double cell_values[8],*uvvals;
  PetscErrorCode ierr,i,j;
  PetscInt *df_ptr;

  /* Scatter the input values from the global vector g, to those on this processor */
  ierr = VecScatterBegin(g,algebra->f_local,INSERT_VALUES,SCATTER_FORWARD,algebra->dfgtol);CHKERRQ(ierr);
  ierr = VecScatterEnd(g, algebra->f_local,INSERT_VALUES,SCATTER_FORWARD,algebra->dfgtol);CHKERRQ(ierr);
  ierr = VecGetArray(algebra->f_local,&uvvals);CHKERRQ(ierr);

  /* set a flag in computation of local elements */
  phi->dorhs = 0;
  /* loop over cells */
  for(i=0;i<grid->cell_n;i++){
    /* loop over degrees of freedom and cell coords */
    df_ptr = grid->cell_df + 8*i;
    coords_ptr = grid->cell_coords + 8*i;
      /* Need to point to the uvvals associated to the vertices */
    for (j=0; j<8; j++){
      cell_values[j] = uvvals[df_ptr[j]];   
    }
    /* compute the values of basis functions on this element */
     ierr = SetLocalElement(phi,coords_ptr);CHKERRQ(ierr);
    /* do the integrals */
    ierr = ComputeNonlinear(phi,cell_values,result);CHKERRQ(ierr);
    /* put result in */
    ierr = VecSetValuesLocal(f,8,df_ptr,result,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(algebra->f_local,&uvvals);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(f);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeNonlinear"
/* input is x,output the nonlinear part into f for a particular element */
PetscErrorCode ComputeNonlinear(AppElement *phi,double *uvvals,double *result)
{
  PetscInt i,j,k,ii ;
  double u[4],v[4];

  /* copy array into more convenient form */
  for(i=0;i<4;i++){  u[i] = uvvals[2*i]; v[i] = uvvals[2*i+1];  }

  /* INTEGRAL */
 /* terms are u*du/dx + v*du/dy, u*dv/dx + v*dv/dy */
  /* Go element by element.
Compute
(u_i * phi_i * u_j * phi_j_x + v_i*phi_i*u_j*phi_j_y) * phi_k
and
(u_i * phi_i * v_j * phi_j_x + v_i*phi_i*v_j*phi_j_y) * phi_k.
Put the result in index k.  Add all possibilities up to get contribution to k, and
 loop over k.*/

/* Could exploit a little symetry to cut iterations from 4*4*4 to 2*4*4  */
   for(k=0;k<4;k++){ /* loop over first basis fn */
     result[2*k] = 0; result[2*k+1] = 0;
     for(i=0; i<4; i++){ /* loop over second */
       for(j=0; j<4; j++){/* loop over third */
         for(ii=0;ii<4;ii++){ /* loop over gauss points */
         result[2*k] +=
           (u[i]*u[j]*phi->Values[i][ii]*phi->dx[4*j+ii] +
            v[i]*u[j]*phi->Values[i][ii]*phi->dy[4*j+ii])*phi->Values[k][ii]*
         PetscAbsReal(phi->detDh[ii]);
         result[2*k+1] +=
           (u[i]*v[j]*phi->Values[i][ii]*phi->dx[4*j+ii] +
            v[i]*v[j]*phi->Values[i][ii]*phi->dy[4*j+ii])*phi->Values[k][ii]*
          PetscAbsReal(phi->detDh[ii]);
         }
       }
     }
   }

   PetscFunctionReturn(0);
}                     

#undef __FUNCT__
#define __FUNCT__ "SetBoundaryConditions"
PetscErrorCode SetBoundaryConditions(Vec g,AppCtx *appctx,Vec f)
{
 /********* Collect context informatrion ***********/
  AppAlgebra *algebra = &appctx->algebra;
  AppGrid *grid = &appctx->grid;

  PetscErrorCode ierr,i;
  double  *bvs,xval,yval,*uvvals; 

  /* Scatter the input values from the global vector g, to those on this processor */
  ierr = VecScatterBegin(g,algebra->f_boundary,INSERT_VALUES,SCATTER_FORWARD,algebra->dfbgtol);CHKERRQ(ierr);
  ierr = VecScatterEnd(g, algebra->f_boundary,INSERT_VALUES,SCATTER_FORWARD,algebra->dfbgtol);CHKERRQ(ierr);
  ierr = VecGetArray(algebra->f_boundary,&uvvals);CHKERRQ(ierr);

  /* create space for the array of boundary values */
  bvs = grid->bvs;

  for(i = 0; i < grid->vertex_boundary_count; i++){
    /* get the vertex_value corresponding to element of vertices
       then evaluate bc(vertex value) and put this in bvs(i) */
    xval = grid->bvc[2*i];
    yval = grid->bvc[2*i+1];
    bvs[2*i] = uvvals[2*i] - pde_bc1(xval,yval);
    bvs[2*i+1] = uvvals[2*i+1] - pde_bc2(xval,yval);    
  }

 /*********  Set Values *************/
  ierr = VecSetValues(f,2*grid->vertex_boundary_count,grid->boundary_df,bvs,INSERT_VALUES);CHKERRQ(ierr);

  /********* Assemble Data **************/
 ierr = VecAssemblyBegin(f);CHKERRQ(ierr);
 ierr = VecAssemblyEnd(f);CHKERRQ(ierr);
 
 /* Destroy stuff */
  ierr = VecRestoreArray(algebra->f_boundary,&uvvals);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*---------------------------------------------------------------------*/
/*  7) Set Jacobian   */

#undef __FUNCT__
#define __FUNCT__ "FormStationaryJacobian"
PetscErrorCode FormStationaryJacobian(SNES snes,Vec g,Mat *jac,Mat *B,MatStructure *flag,void *dappctx)
{
  AppCtx *appctx = (AppCtx *)dappctx;
  AppAlgebra *algebra = &appctx->algebra;
  PetscErrorCode ierr;

  /* copy the linear part into jac.*/
  ierr= MatCopy(algebra->A,*jac,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

  /* the nonlinear part */
  ierr = SetJacobian(g,appctx,jac);CHKERRQ(ierr);

  /* Set flag */
  *flag = DIFFERENT_NONZERO_PATTERN;  /*  is this right? */
  PetscFunctionReturn(0);
} 

/* input is the input vector,output is the jacobian jac */
#undef __FUNCT__
#define __FUNCT__ "SetJacobian"
PetscErrorCode SetJacobian(Vec g,AppCtx *appctx,Mat* jac)
{
/********* Collect context informatrion ***********/
  AppAlgebra *algebra = &appctx->algebra;
  AppGrid    *grid    = &appctx->grid;
  AppElement *phi = &appctx->element;
  
/****** Internal Variables ***********/
  PetscInt  i,j,*df_ptr; 
  PetscErrorCode ierr;
  double *coords_ptr;
  double   *uvvals,cell_values[8];
  double values[8*8];  /* the integral of the combination of phi's */
 double one = 1.0;

  PetscFunctionBegin;
  /* Matrix is set to the linear part already, so just ADD_VALUES the nonlinear part  */ 
  ierr = VecScatterBegin(g,algebra->f_local,INSERT_VALUES,SCATTER_FORWARD,algebra->dfgtol);CHKERRQ(ierr);
  ierr = VecScatterEnd(g,algebra->f_local,INSERT_VALUES,SCATTER_FORWARD,algebra->dfgtol);CHKERRQ(ierr);
  ierr = VecGetArray(algebra->f_local,&uvvals);
 
  /* loop over cells */
  for(i=0;i<grid->cell_n;i++){
   /* loop over degrees of freedom and cell coords */
    df_ptr = grid->cell_df + 8*i;
    coords_ptr = grid->cell_coords + 8*i;
    /* Need to point to the uvvals associated to the vertices */
    for (j=0; j<8; j++){
      cell_values[j] = uvvals[df_ptr[j]];
    }
    /* compute the values of basis functions on this element */
    ierr = SetLocalElement(phi,coords_ptr);CHKERRQ(ierr);

    /*    Compute the partial derivatives of the nonlinear map    */  
    ierr = ComputeJacobian(phi,cell_values,values);CHKERRQ(ierr);

    /*  Set the values in the matrix */
    ierr  = MatSetValuesLocal(*jac,8,df_ptr,8,df_ptr,values,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(algebra->f_local,&uvvals);
  ierr = MatAssemblyBegin(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /********** The process repeats for setting boundary conditions ************/

  ierr = MatZeroRowsIS(*jac,grid->isboundary_df,one);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);

}

#undef __FUNCT__
#define __FUNCT__ "ComputeJacobian"

/* input is x, output the nonlinear part into f for a particulat element */
/* Much of the code is dublicated from ComputeMatrix; the integral is different */
PetscErrorCode ComputeJacobian(AppElement *phi,double *uv,double *result)
{
  /* How can I test this??  */
  PetscInt i,j,k,ii ;
  double u[4],v[4];
  double dxint[4][4][4]; /* This is integral of phi_dx[i]*phi[j]*phi[k] */
  double dyint[4][4][4]; /* This is integral of phi_dy[i]*phi[j]*phi[k] */

  /* copy array into more convenient form */
  for(i=0;i<4;i++){    u[i] = uv[2*i];     v[i] = uv[2*i+1];}
 
  /* INTEGRAL */ 
  /* The nonlinear map takes(u0,v0,u1,v1,u2,v2,u3,v3) to 
      (integral term1 *  phi0, integral term2 * phi0, ..., integral term1*phi3, int term2*phi3)
   Loop first over the phi.  Then integrate two parts of the terms.
Term 1: (ui*uj*phi_i*dx_j + vi*uj*phi_i*dy_j)
Term 2: (ui*vj*phi_i*dx_j + vi*vj*phi_i*dy_j)
*/

  /* could  exploit symmetry to cut down on iterations tohere */
/* Make a database of integrals of phi_i(dx or dy)*phi_j*phi_k */
  for(j=0;j<4;j++){
    for(i=0;i<4;i++){
      for(k=0;k < 4;k++){
	 dxint[i][j][k] = 0; 
	 dyint[i][j][k] = 0;
	for(ii=0;ii<4;ii++){/* loop over basis gauss points */
	  dxint[i][j][k] += 
	    phi->dx[4*i+ii]*phi->Values[j][ii]*phi->Values[k][ii]*
	    PetscAbsReal(phi->detDh[ii]);
	  dyint[i][j][k] += 
	    phi->dy[4*i+ii]*phi->Values[j][ii]*phi->Values[k][ii]*
	    PetscAbsReal(phi->detDh[ii]);
	}
      }
    }
  }

  /* now loop over the columns of the matrix */
  for(k=0;k<4;k++){ 
    /* the terms are u*u_x + v*u_y and u*v_x+v*v_y  */
    for(i = 0;i<4;i++){  
      result[16*k + 2*i] = 0;
      result[16*k + 2*i + 1] = 0;   /* Stuff from Term 1 */
      result[16*k + 8 + 2*i]=0; 
      result[16*k + 8 + 2*i + 1] = 0;  /* Stuff from Term 2 */
      for(j=0;j<4;j++){
	result[16*k + 2*i] +=   u[j]*dxint[i][j][k] + u[j]*dxint[j][i][k] + v[j]*dyint[i][j][k];
	result[16*k+2*i+1] +=   u[j]*dyint[j][i][k];

	result[16*k + 8 + 2*i] += v[j]*dxint[j][i][k];
	result[16*k+ 8 + 2*i+1] += u[j]*dxint[i][j][k] + v[j]*dyint[j][i][k] + v[j]*dyint[i][j][k];
      }     
    }
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------------------*/
/*  9) Initial guess */

#undef __FUNCT__
#define __FUNCT__ "FormInitialGuess"
PetscErrorCode FormInitialGuess(AppCtx* appctx)
{
    AppAlgebra *algebra = &appctx->algebra;
    PetscErrorCode ierr;
    double onep1 = 1.234;
    ierr = VecSet(algebra->g,onep1);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}           
