
static char help[] ="Solves a simple linear PDE in 2D on an unstructured grid\n";


/*
       Demonstrates how one may write a unstructured grid PDE solver using the 
   PETSc AOData-base infra-structure. One should copy the files 
       appctx.h appload.c appview.c and ex2.c 
   and add and delete data-structures and code as needed for your particular 
   application.

*/

#include "appctx.h"

extern int AppCtxSetRhs(AppCtx*);
extern int AppCtxSetMatrix(AppCtx*);

int main( int argc, char **argv )
{
  int            ierr;
  AppCtx         *appctx;
  AppAlgebra     *algebra;
  AppGrid        *grid;

  /* ---------------------------------------------------------------------
     Initialize PETSc
     ------------------------------------------------------------------------*/

  PetscInitialize(&argc,&argv,(char *)0,help);

  /* 
      Load the grid database and initialize graphics 
  */
  ierr = AppCtxCreate(PETSC_COMM_WORLD,&appctx); CHKERRA(ierr);
  algebra = &appctx->algebra;
  grid    = &appctx->grid;

  /*
      Setup the linear system and solve it
  */
  ierr = AppCtxSolve(appctx);CHKERRQ(ierr);

  /*
      Visualize solution
  */
  if (appctx->view.showsomething) {
    ierr = VecScatterBegin(algebra->x,algebra->w_local,INSERT_VALUES,SCATTER_FORWARD,algebra->gtol);CHKERRQ(ierr);
    ierr = VecScatterEnd(algebra->x,algebra->w_local,INSERT_VALUES,SCATTER_FORWARD,algebra->gtol);CHKERRQ(ierr);
    ierr = DrawZoom(appctx->view.drawglobal,AppCtxViewSolution,appctx); CHKERRA(ierr);
  }

  /*
      Destroy all datastructures
  */
  ierr = AppCtxDestroy(appctx); CHKERRA(ierr);

  PetscFinalize();

  PetscFunctionReturn(0);
}


#undef __FUNC__
#define __FUNC__ "AppCxtSolve"
int AppCtxSolve(AppCtx* appctx)
{
  AppGrid                *grid = &appctx->grid;
  AppAlgebra             *algebra = &appctx->algebra;
  MPI_Comm               comm = appctx->comm;
  Vec                    x,b,x_local,w_local,z,z_local;
  ISLocalToGlobalMapping ltog = grid->ltog;
  int                    vertex_n = grid->vertex_n,vertex_n_ghosted = grid->vertex_n_ghosted,ierr,its;
  VecScatter             gtol;
  IS                     vertex_global = grid->vertex_global;
  Mat                    A;

  PetscFunctionBegin;

  /*
        Create vector to contain load
  */
  ierr = VecCreateMPI(comm,vertex_n,PETSC_DECIDE,&b);CHKERRQ(ierr);
  ierr = VecSetLocalToGlobalMapping(b,ltog);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&x);
  ierr = VecDuplicate(b,&z);

  ierr = VecCreateSeq(PETSC_COMM_SELF,vertex_n_ghosted,&w_local);CHKERRQ(ierr);
  ierr = VecDuplicate(w_local,&x_local);CHKERRQ(ierr);
  ierr = VecDuplicate(w_local,&z_local);CHKERRQ(ierr);
  ierr = VecScatterCreate(b,vertex_global,w_local,0,&gtol);CHKERRQ(ierr);

  algebra->x       = x;
  algebra->b       = b;
  algebra->z       = z;
  algebra->w_local = w_local;
  algebra->x_local = x_local;
  algebra->z_local = z_local;
  algebra->gtol    = gtol;


  ierr = AppCtxSetRhs(appctx); CHKERRQ(ierr);

  ierr = AppCtxSetMatrix(appctx); CHKERRQ(ierr);

  A = algebra->A;

  ierr = SLESCreate(comm,&sles);CHKERRQ(ierr);
  ierr = SLESSetOperators(sles,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = SLESSetFromOptions(sles);CHKERRQ(ierr);
  ierr = SLESSolve(sles,b,x,&its);CHKERRQ(ierr);
  ierr = SLESDestroy(sles); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "AppCxtSetRhs"
int AppCtxSetRhs(AppCtx* appctx)
{
  AppGrid    *grid = &appctx->grid;
  AppAlgebra *algebra = &appctx->algebra;
  Scalar     *values;
  int        ierr, i, cell_n = grid->cell_n, ncell = grid->ncell;
  int        *cell_vertex = grid->cell_vertex,*vertices,j;
  Vec        b = algebra->b;
  double     *coors;

  values = (Scalar *) PetscMalloc(ncell*sizeof(Scalar));CHKPTRQ(values);
  coors  = (double*) PetscMalloc(2*ncell*sizeof(double));CHKPTRQ(coors);

  for ( i=0; i<ncell; i++ ) values[i] = 1.0;

  /*  ---------------------------------------------------------------
        Loop over elements computing load and putting into right-hand-side
        ----------------------------------------------------------------*/
  for ( i=0; i<cell_n; i++ ) {
    vertices = cell_vertex+ncell*i;
    for ( j=0; j<ncell; j++) {
      coors[2*j]   = cell_vertex[vertices[j]];
      coors[2*j+1] = cell_vertex[vertices[j]+1];
    }

    ierr = VecSetValuesLocal(b,ncell,vertices,values,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
  ierr = PetscFree(values);CHKERRQ(ierr);
  ierr = PetscFree(coors);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "AppCxtSetMatrix"
int AppCtxSetMatrix(AppCtx* appctx)
{
  AppAlgebra *algebra = &appctx->algebra;
  AppGrid    *grid    = &appctx->grid;
  Vec        w_local = algebra->w_local, x = algebra->x, x_local = algebra->x_local, z_local = algebra->z_local;
  Vec        z = algebra->z;
  VecScatter gtol = algebra->gtol;
  MPI_Comm   comm = appctx->comm;
  Scalar     srank,*procs,*sdnz,zero = 0.0,*values,wght,*sonz;
  int        ierr, rank,*vertices,cproc,i,j,*dnz,vertex_n = grid->vertex_n;
  int        cell_n = grid->cell_n, *cell_vertex = grid->cell_vertex, ncell = grid->ncell;
  int        *cell_cell = grid->cell_cell,*cells,*onz;
  Mat        A;
  double     *coors;
  ISLocalToGlobalMapping ltog = grid->ltog;
  IS         vertex_boundary = grid->vertex_boundary;
  Scalar     one = 1.0;

  PetscFunctionBegin;
  MPI_Comm_rank(comm,&rank); 

  /* ------------------------------------------------
      Determine non-zero structure of the matrix 
      --------------------------------------------*/
  
  /* 1) make proc[] contain the processor number of each ghosted vertex */
  srank = rank;
  ierr = VecSet(x,srank);CHKERRQ(ierr);
  ierr = VecScatterBegin(x,w_local,INSERT_VALUES,SCATTER_FORWARD,gtol);CHKERRQ(ierr);
  ierr = VecScatterEnd(x,w_local,INSERT_VALUES,SCATTER_FORWARD,gtol);CHKERRQ(ierr);
  ierr = VecGetArray(w_local,&procs);CHKERRQ(ierr);
  ierr = VecSet(x_local,zero);CHKERRQ(ierr);
  ierr = VecGetArray(x_local,&sdnz);CHKERRQ(ierr);
  ierr = VecSet(z_local,zero);CHKERRQ(ierr);
  ierr = VecGetArray(z_local,&sonz);CHKERRQ(ierr);

  /* 2) loop over local elements; count matrix nonzeros */
  if (ncell == 3) {
    /*
          Linear triangular elements
    */
    for ( i=0; i<cell_n; i++ ) {
      vertices = cell_vertex + ncell*i;
      cells    = cell_cell   + ncell*i;
      for ( j=0; j<ncell; j++ ) {
        cproc = procs[vertices[j]];
        /* next neighbor */
        if (cells[j] >= 0) wght = .5; else wght = 1.0;
        if (cproc == procs[vertices[(j+1) % ncell]]) { /* on diagonal part */
          sdnz[vertices[j]] += wght;
        } else {
          sonz[vertices[j]] += wght;
        }
        /* previous neighbor */
        if (cells[(j+2) % ncell] >= 0) wght = .5; else wght = 1.0;
        if (cproc == procs[vertices[(j+2) % ncell]]) { /* on diagonal part */
          sdnz[vertices[j]] += wght;
        } else {
          sonz[vertices[j]] += wght;
        }
      }
    }
  } else if (ncell == 6) {
    /*
        quadratic triangular elements
    */
    for ( i=0; i<cell_n; i++ ) {
      vertices = cell_vertex + ncell*i;
      cells    = cell_cell   + 3*i;
      for ( j=0; j<ncell; j += 2 ) {
        /*----
           First we take care of the corner vertex relations with other nodes
	*/
        cproc = procs[vertices[j]];
        /* 1st neighbor (on middle of edge) */
        if (cells[j/2] >= 0) wght = .5; else wght = 1.0;
        if (cproc == procs[vertices[j+1]]) { /* on diagonal part */
          sdnz[vertices[j]] += wght;
        } else {
          sonz[vertices[j]] += wght;
        }
        /* 2nd neighbor */
        if (cproc == procs[vertices[(j+2) % ncell]]) { /* on diagonal part */
          sdnz[vertices[j]] += wght;
        } else {
          sonz[vertices[j]] += wght;
        }
        /* 3rd neighbor (on middle of opposite edge) */
        if (cproc == procs[vertices[(j+3) % ncell]]) { /* on diagonal part */
          sdnz[vertices[j]] += 1.0;
        } else {
          sonz[vertices[j]] += 1.0;
        }
        /* 4th neighbor */
        if (cells[(j/2 + 2) % 3] >= 0) wght = .5; else wght = 1.0;
        if (cproc == procs[vertices[(j+4) % ncell]]) { /* on diagonal part */
          sdnz[vertices[j]] += wght;
        } else {
          sonz[vertices[j]] += wght;
        }
        /* 5th neighbor */
        if (cproc == procs[vertices[(j+5) % ncell]]) { /* on diagonal part */
          sdnz[vertices[j]] += wght;
        } else {
          sonz[vertices[j]] += wght;
        }
        /*----
           Now we take care of the edge node relations with other nodes
	*/
        cproc = procs[vertices[j+1]];
        /* 1st neighbor (before it on vertex) */
        if (cells[j/2] >= 0) wght = .5; else wght = 1.0;
        if (cproc == procs[vertices[j]]) { /* on diagonal part */
          sdnz[vertices[j+1]] += wght;
        } else {
          sonz[vertices[j+1]] += wght;
        }
        /* 2nd neighbor */
        if (cproc == procs[vertices[(j+2) % ncell]]) { /* on diagonal part */
          sdnz[vertices[j+1]] += wght;
        } else {
          sonz[vertices[j+1]] += wght;
        }
        /* 3rd neighbor (on middle of opposite edge) */
        if (cproc == procs[vertices[(j+3) % ncell]]) { /* on diagonal part */
          sdnz[vertices[j+1]] += 1.0;
        } else {
          sonz[vertices[j+1]] += 1.0;
        }
        /* 4th neighbor */
        if (cproc == procs[vertices[(j+4) % ncell]]) { /* on diagonal part */
          sdnz[vertices[j+1]] += 1.0;
        } else {
          sonz[vertices[j+1]] += 1.0;
        }
        /* 5th neighbor */
        if (cproc == procs[vertices[(j+5) % ncell]]) { /* on diagonal part */
          sdnz[vertices[j+1]] += 1.0;
        } else {
          sonz[vertices[j+1]] += 1.0;
        }
      }
    }
  }

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
  dnz  = (int *) PetscMalloc((vertex_n+1)*sizeof(int));CHKPTRQ(dnz);
  onz  = (int *) PetscMalloc((vertex_n+1)*sizeof(int));CHKPTRQ(onz);
  for ( i=0; i<vertex_n; i++ ) {
    dnz[i] = 1 + (int) sdnz[i];
    onz[i] = (int) sonz[i];
  }  
  ierr = VecRestoreArray(x,&sdnz);CHKERRQ(ierr);
  ierr = VecRestoreArray(z,&sonz);CHKERRQ(ierr);


  ierr = MatCreateMPIAIJ(comm,vertex_n,vertex_n,PETSC_DETERMINE,PETSC_DETERMINE,
                         0,dnz,0,onz,&A); CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(A,ltog);CHKERRQ(ierr);

  /*  ---------------------------------------------------------------
        loop over local elements, putting values into matrix 
        ---------------------------------------------------------------*/
  values = (Scalar *) PetscMalloc(ncell*ncell*sizeof(Scalar));CHKPTRQ(values);
  coors  = (double *) PetscMalloc(2*ncell*sizeof(double));CHKPTRQ(coors);

  for ( i=0; i<ncell*ncell; i++) values[i] = 1.0;

  for ( i=0; i<cell_n; i++ ) {
    vertices = cell_vertex + ncell*i;   
    for ( j=0; j<ncell; j++) {
      coors[2*j]   = cell_vertex[vertices[j]];
      coors[2*j+1] = cell_vertex[vertices[j]+1];
    }
    ierr     = MatSetValuesLocal(A,ncell,vertices,ncell,vertices,values,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscFree(values);CHKERRQ(ierr);

  /*  -------------------------------------------------------------
         Apply Dirichlet boundary conditions
      -----------------------------------------------------------*/
  ierr = MatZeroRowsLocalIS(A,vertex_boundary,one);CHKERRQ(ierr);

  algebra->A = A;

  PetscFunctionReturn(0);
}
