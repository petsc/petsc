
static char help[] ="Solves advection-diffusion, quasigeostrophic.\n  Generate grid using input.\n  Options:\n  -matlab_graphics\t pipe solution to matlab (visualize with bscript).\n -show_is\t print the local index sets and local to global mappings (for use with >1 processor).\n  -show_ao\t print the contents of the ao database.\n  -show_matrix\t visualize the sparsity structure of the stiffness matrix.\n See README file for more information.\n";

#include "appctx.h"

int main( int argc, char **argv )
{
  int            ierr;
  AppCtx         *appctx;

  /* ---------------------------------------------------------------------
     Initialize PETSc
     --------------------- ---------------------------------------------------*/
  PetscInitialize(&argc,&argv,(char *)0,help);

  /*  Load the grid database*/
  ierr = AppCtxCreate(PETSC_COMM_WORLD,&appctx); CHKERRA(ierr);

  /*        Create vector to contain load and various work vectors  */
  ierr = AppCtxCreateRhs(appctx); CHKERRQ(ierr);
  /*      Create the sparse matrix, with correct nonzero pattern  */
  ierr = AppCtxCreateMatrix(appctx); CHKERRQ(ierr);
  /*     Set the quadrature values for the reference square element  */
  ierr = SetReferenceElement(appctx);CHKERRQ(ierr);

  /* set the diffusion matrix, advection etc */
  ierr = SetOperators(appctx);CHKERRQ(ierr);

 /* setup the graphics routines to view the grid  */
  ierr = AppCtxGraphics(appctx); CHKERRA(ierr);
 
  /*   Setup the linear system and solve it*/
  ierr = AppCtxSolve(appctx);CHKERRQ(ierr);

  /* Send to  matlab viewer */
  if (appctx->view.matlabgraphics) {    AppCtxViewMatlab(appctx);  }

  /*      Destroy all datastructures  */
  ierr = AppCtxDestroy(appctx); CHKERRA(ierr);

  PetscFinalize();
  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "SetOperators"
int SetOperators(AppCtx* appctx)
{
  int ierr;

  /* call set mass matrix, set advection matrix, set grad_perp */
  ierr = AppSetStiffness(appctx) ;CHKERRQ(ierr);
  ierr = AppSetMassMatrix(appctx);CHKERRQ(ierr);
  ierr = AppSetGradPerp(appctx);CHKERRQ(ierr);
PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "SetInitialValue"
int SetInitialValue(AppCtx* appctx)
{
 
/********* Collect contex informatrion ***********/
  AppAlgebra *algebra = &appctx->algebra;
  AppGrid    *grid    = &appctx->grid;
  AppElement *phi = &appctx->element; 
 AppEquations *equations = &appctx->equations;

/****** Internal Variables ***********/
  int i, ii, j,ierr;
  int *vertex_ptr;
  int bn =4; /* basis count */
  int vertexn = 4; /* degree of freedom count */
  double result[4];

  PetscFunctionBegin;

  /* loop over cells */
  for(i=0;i<grid->cell_n;i++){

    /* loop over degrees of freedom and cell coords */
    /* vertex_ptr points to place in the vector to set the values */
    vertex_ptr = grid->cell_vertex + vertexn*i;
    /* coords_ptr points to the coordinates of the current cell */
    phi->coords = grid->cell_coords + 2*bn*i;/*number of cell coords */

    for(j=0;j<4;j++){
      equations->xval = phi->coords[2*j];
      equations->yval = phi->coords[2*j+1];
      result[j] = f_init(equations->xval, equations->yval);
    }
    ierr = VecSetValuesLocal(algebra->g, 4, vertex_ptr, result, INSERT_VALUES);CHKERRQ(ierr);
  }
  /********* Assemble Data **************/
  ierr = VecAssemblyBegin(algebra->g);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(algebra->g);CHKERRQ(ierr);

PetscFunctionReturn(0);
}


/*
         Sets up the linear system associated with the PDE and solves it
*/
#undef __FUNC__
#define __FUNC__ "AppCxtSolve"
int AppCtxSolve(AppCtx* appctx)
{
  AppGrid                *grid = &appctx->grid;
  AppAlgebra             *algebra = &appctx->algebra;
  MPI_Comm               comm = appctx->comm;
  SLES                   sles=appctx->sles;
  int ierr, its;

  PetscFunctionBegin;

  /* create sles context */
  ierr = SLESCreate(comm,&sles);CHKERRQ(ierr);

  /* get initial state */
  ierr = SetInitialValue(appctx); CHKERRQ(ierr);


  /* Call timestepping  */

  ierr = TimeStep(appctx); CHKERRQ(ierr);

  /* destroy sles */
  ierr = SLESDestroy(sles); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}



int TimeStep(AppCtx* appctx)
{

  AppGrid                *grid = &appctx->grid;
  AppAlgebra             *algebra = &appctx->algebra;
  AppElement          *element = &appctx->element;
  AppEquations *equations = &appctx->equations;
  SLES sles= appctx->sles;

 int ierr, its;
  int flag;
  int i;
  int zero = 0, one = 1;
  double dzero = 0.0;
  double done = 1.0;
  double idt;
  double meta = -equations->eta;

   /* Determine time increment*/
  equations->dt = (equations->final_time - equations->initial_time)/equations->Nsteps;
  idt = 1/equations->dt;

    /* now loop and solve */
  for(i=0;i<equations->Nsteps;i++)
    {
      /* need to compute u from w then set the operator u.grad() */
      ierr = ComputeNonlinear(appctx);CHKERRQ(ierr);
      /* now assemble the stuff, then solve */
     
      /* w_old is initially in g, later put into g. */
      /*rhs = 1/dt*Mg */
      ierr = MatMult(algebra->M, algebra->g, algebra->b);CHKERRQ(ierr);
      ierr = VecScale(algebra->b,idt);
 
      /*operator D = 1/dt*M + C - eta*A*/
      MatCopy(algebra->M, algebra->D,DIFFERENT_NONZERO_PATTERN );
      MatScale(algebra->D,idt);
      MatAXPY(algebra->D, algebra->C, done);
      MatAXPY(algebra->D, algebra->A, meta);

      /*       Solve the linear system  */
      ierr = SLESSetOperators(sles,algebra->D,algebra->D,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = SLESSetFromOptions(sles);CHKERRQ(ierr);
      ierr = SLESSolve(sles,algebra->b,algebra->x,&its);CHKERRQ(ierr);

      /* send this step to the matlab viewer */  
      if (appctx->view.matlabgraphics){
	ierr = VecView(algebra->x, VIEWER_MATLAB_WORLD); CHKERRQ(ierr); 
	ierr = PetscIntView(1, &one, VIEWER_MATLAB_WORLD);CHKERRQ(ierr);
      }
      /* put solution into g */
      ierr = VecCopy(algebra->x, algebra->g); CHKERRQ(ierr);
    }

PetscFunctionReturn(0);
}
