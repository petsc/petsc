
static char help[] ="Solves a simple linear PDE in 2D on an unstructured grid\n\
   consisting of triangles using quadratic finite elements";


/*
       Demonstrates how one may write a unstructured grid PDE solver using the 
   PETSc AOData-base infra-structure. 

       To solve a different (but similar problem) one should copy the files 
       appctx.h appload.c appview.c and main.c and add and delete data-structures 
   and code as needed for your particular application.

*/

#include "appctx.h"

extern int AppCtxSetRhs(AppCtx*);
extern int AppCtxCreateRhs(AppCtx*);
extern int AppCtxSetMatrix(AppCtx*);
extern int AppCtxCreateMatrix(AppCtx*);
extern int AppCtxViewMatlab(AppCtx*);

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

  if (appctx->view.showsomething) {
    /*
       Visualize the grid 
    */
    ierr = DrawZoom(appctx->view.drawglobal,AppCtxViewGrid,appctx); CHKERRA(ierr);
  }

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

  if (appctx->view.matlabgraphics) {
    AppCtxViewMatlab(appctx);
  }

  /*
      Destroy all datastructures
  */
  ierr = AppCtxDestroy(appctx); CHKERRA(ierr);

  PetscFinalize();

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
  ISLocalToGlobalMapping ltog = grid->ltog;
  int                    vertex_n = grid->vertex_n,vertex_n_ghosted = grid->vertex_n_ghosted,ierr,its;
  IS                     vertex_global = grid->vertex_global;
  SLES                   sles;

  PetscFunctionBegin;

  /*
        Create vector to contain load and various work vectors
  */
  ierr = AppCtxCreateRhs(appctx); CHKERRQ(ierr);

  /*
      Create the sparse matrix, with correct nonzero pattern
  */
  ierr = AppCtxCreateMatrix(appctx); CHKERRQ(ierr);

  /*
      Set the right hand side values into the vectors 
  */
  ierr = AppCtxSetRhs(appctx); CHKERRQ(ierr);

  /*
      Set the matrix entries 
  */
  ierr = AppCtxSetMatrix(appctx); CHKERRQ(ierr);

  /*
       Solve the linear system
  */
  ierr = SLESCreate(comm,&sles);CHKERRQ(ierr);
  ierr = SLESSetOperators(sles,algebra->A,algebra->A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = SLESSetFromOptions(sles);CHKERRQ(ierr);
  ierr = SLESSolve(sles,algebra->b,algebra->x,&its);CHKERRQ(ierr);
  ierr = SLESDestroy(sles); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*
         -  Generates the "global" parallel vector to contain the right hand side 
               and solution.
         -  Generates "ghosted" local vectors for local computations etc.
         -  Generates scatter context for updating ghost points etc.
*/
#undef __FUNC__
#define __FUNC__ "AppCxtCreateRhs"
int AppCtxCreateRhs(AppCtx* appctx)
{
  AppGrid                *grid = &appctx->grid;
  AppAlgebra             *algebra = &appctx->algebra;
  MPI_Comm               comm = appctx->comm;
  Vec                    x,b,x_local,w_local,z,z_local;
  ISLocalToGlobalMapping ltog = grid->ltog;
  int                    vertex_n = grid->vertex_n,vertex_n_ghosted = grid->vertex_n_ghosted,ierr,its;
  VecScatter             gtol;
  IS                     vertex_global = grid->vertex_global;
  SLES                   sles;

  PetscFunctionBegin;

  /*
        Create vector to contain load and various work vectors
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

  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "AppCxtSetRhs"
int AppCtxSetRhs(AppCtx* appctx)
{
  AppGrid    *grid = &appctx->grid;
  AppAlgebra *algebra = &appctx->algebra;
  Scalar     *values;
  int        ierr, i, cell_n = grid->cell_n, ncell = 6;
  int        *cell_vertex = grid->cell_vertex,*vertices,j;
  Vec        b = algebra->b;
  double     *coors;

  /*
      Room to hold the coordinates of a single cell, plus the 
     RHS generated from a single cell.
  */
  coors  = (double*) PetscMalloc(2*ncell*sizeof(double));CHKPTRQ(coors);
  values = (Scalar *) PetscMalloc(ncell*sizeof(Scalar));CHKPTRQ(values);


  /*  ---------------------------------------------------------------
        Loop over elements computing load one element at a time 
        and putting into right-hand-side
        ----------------------------------------------------------------*/
  for ( i=0; i<cell_n; i++ ) {
    vertices = cell_vertex+ncell*i;
    /* 
       Load the cell vertex coordinates 
    */
    for ( j=0; j<ncell; j++) {
      coors[2*j]   = cell_vertex[vertices[j]];
      coors[2*j+1] = cell_vertex[vertices[j]+1];
    }
    /*
       Here is where you would call the routine to compute the 
      element load
    */
    for ( j=0; j<ncell; j++ ) values[j] = 1.0;

    ierr = VecSetValuesLocal(b,ncell,vertices,values,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
  PetscFree(values);  
  PetscFree(coors);
  PetscFunctionReturn(0);
}

/*
     Creates the sparse matrix (with the correct nonzero pattern) that will
  be later filled with the stiffness matrix
*/
#undef __FUNC__
#define __FUNC__ "AppCxtSetMatrix"
int AppCtxCreateMatrix(AppCtx* appctx)
{
  AppAlgebra             *algebra = &appctx->algebra;
  AppGrid                *grid    = &appctx->grid;
  Vec                    w_local = algebra->w_local, x = algebra->x, x_local = algebra->x_local;
  Vec                    z_local = algebra->z_local;
  Vec                    z = algebra->z;
  VecScatter             gtol = algebra->gtol;
  MPI_Comm               comm = appctx->comm;
  Scalar                 srank,*procs,*sdnz,zero = 0.0,*values,wght,*sonz;
  int                    ierr, rank,*vertices,cproc,i,j,*dnz,vertex_n = grid->vertex_n;
  int                    cell_n = grid->cell_n, *cell_vertex = grid->cell_vertex, ncell = 6;
  int                    *cell_cell = grid->cell_cell,*cells,*onz;
  Mat                    A;
  double                 *coors;
  ISLocalToGlobalMapping ltog = grid->ltog;
  IS                     vertex_boundary = grid->vertex_boundary;
  Scalar                 one = 1.0;

  PetscFunctionBegin;
  MPI_Comm_rank(comm,&rank); 

  /* ------------------------------------------------
      Determine non-zero structure of the matrix 
      --------------------------------------------*/
  
  /* 1) make proc[] contain the processor number of each ghosted vertex */
  srank = rank;
  ierr = VecSet(&srank,x);CHKERRQ(ierr);
  ierr = VecScatterBegin(x,w_local,INSERT_VALUES,SCATTER_FORWARD,gtol);CHKERRQ(ierr);
  ierr = VecScatterEnd(x,w_local,INSERT_VALUES,SCATTER_FORWARD,gtol);CHKERRQ(ierr);
  ierr = VecGetArray(w_local,&procs);CHKERRQ(ierr);
  ierr = VecSet(&zero,x_local);CHKERRQ(ierr);
  ierr = VecGetArray(x_local,&sdnz);CHKERRQ(ierr);
  ierr = VecSet(&zero,z_local);CHKERRQ(ierr);
  ierr = VecGetArray(z_local,&sonz);CHKERRQ(ierr);

  /* 2) loop over local elements; count matrix nonzeros */

  /*
       quadratic triangular elements
  */
  /* loop over cells */
  for ( i=0; i<cell_n; i++ ) {
    vertices = cell_vertex + ncell*i;
    cells    = cell_cell   + 3*i;
    for ( j=0; j<ncell; j += 2 ) {
      /*----
         First we take care of the corner vertex relations with other nodes
      */
      cproc = PetscReal(procs[vertices[j]]);
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
      cproc = PetscReal(procs[vertices[j+1]]);
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

  ierr = VecRestoreArray(x_local,&sdnz);CHKERRQ(ierr);
  ierr = VecRestoreArray(z_local,&sonz);CHKERRQ(ierr);
  ierr = VecRestoreArray(w_local,&procs);CHKERRQ(ierr);
  ierr = VecSet(&zero,x);CHKERRQ(ierr);
  ierr = VecScatterBegin(x_local,x,ADD_VALUES,SCATTER_REVERSE,gtol);CHKERRQ(ierr);
  ierr = VecScatterEnd(x_local,x,ADD_VALUES,SCATTER_REVERSE,gtol);CHKERRQ(ierr);
  ierr = VecGetArray(x,&sdnz);CHKERRQ(ierr);
  ierr = VecSet(&zero,z);CHKERRQ(ierr);
  ierr = VecScatterBegin(z_local,z,ADD_VALUES,SCATTER_REVERSE,gtol);CHKERRQ(ierr);
  ierr = VecScatterEnd(z_local,z,ADD_VALUES,SCATTER_REVERSE,gtol);CHKERRQ(ierr);
  ierr = VecGetArray(z,&sonz);CHKERRQ(ierr);
  dnz  = (int *) PetscMalloc((vertex_n+1)*sizeof(int));CHKPTRQ(dnz);
  onz  = (int *) PetscMalloc((vertex_n+1)*sizeof(int));CHKPTRQ(onz);
  for ( i=0; i<vertex_n; i++ ) {
    dnz[i] = 1 + (int) PetscReal(sdnz[i]);
    onz[i] = (int) PetscReal(sonz[i]);
  }  
  ierr = VecRestoreArray(x,&sdnz);CHKERRQ(ierr);
  ierr = VecRestoreArray(z,&sonz);CHKERRQ(ierr);


  ierr = MatCreateMPIAIJ(comm,vertex_n,vertex_n,PETSC_DETERMINE,PETSC_DETERMINE,
                         0,dnz,0,onz,&A); CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(A,ltog);CHKERRQ(ierr);
  PetscFree(dnz);
  PetscFree(onz);
  algebra->A = A;

  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "AppCxtSetMatrix"
int AppCtxSetMatrix(AppCtx* appctx)
{
  AppAlgebra *algebra = &appctx->algebra;
  AppGrid    *grid    = &appctx->grid;
  MPI_Comm   comm = appctx->comm;
  Scalar     srank,*procs,*sdnz,zero = 0.0,*values,wght,*sonz;
  int        ierr, rank,*vertices,cproc,i,j,*dnz,vertex_n = grid->vertex_n;
  int        cell_n = grid->cell_n, *cell_vertex = grid->cell_vertex, ncell = 6;
  int        *cell_cell = grid->cell_cell,*cells,*onz;
  Mat        A = algebra->A;
  double     *coors;
  IS         vertex_boundary = grid->vertex_boundary;
  Scalar     one = 1.0;

  PetscFunctionBegin;
  MPI_Comm_rank(comm,&rank); 


  /*  ---------------------------------------------------------------
        loop over local elements, putting values into matrix 
        ---------------------------------------------------------------*/
  /*
        Room for coordinates and element stiffness for one element
  */
  coors  = (double *) PetscMalloc(2*ncell*sizeof(double));CHKPTRQ(coors);
  values = (Scalar *) PetscMalloc(ncell*ncell*sizeof(Scalar));CHKPTRQ(values);


  for ( i=0; i<cell_n; i++ ) {
    vertices = cell_vertex + ncell*i;   
    for ( j=0; j<ncell; j++) {
      coors[2*j]   = cell_vertex[vertices[j]];
      coors[2*j+1] = cell_vertex[vertices[j]+1];
    }
    /* 
        Here one would call the routine that computes the element stiffness
    */
    for ( j=0; j<ncell*ncell; j++) values[j] = 1.0;

    ierr     = MatSetValuesLocal(A,ncell,vertices,ncell,vertices,values,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFree(values);
  PetscFree(coors);

  /*  -------------------------------------------------------------
         Apply Dirichlet boundary conditions
      -----------------------------------------------------------*/
  ierr = MatZeroRowsLocal(A,vertex_boundary,&one);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


#undef __FUNC__
#define __FUNC__ "AppCxtViewMatlab"
int AppCtxViewMatlab(AppCtx* appctx)
{
  int    ierr,*cell_vertex,rstart,rend;
  Viewer viewer = VIEWER_MATLAB_WORLD;
  double *vertex_values;
  IS     isvertex;

  PetscFunctionBegin;

  /* send solution vector first */
  ierr = VecView(appctx->algebra.x,viewer);CHKERRQ(ierr);

  /* send vertices */
  ierr = AODataKeyGetOwnershipRange(appctx->aodata,"vertex",&rstart,&rend);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD,rend-rstart,rstart,1,&isvertex);CHKERRQ(ierr);
  ierr = AODataSegmentGetIS(appctx->aodata,"vertex","values",isvertex,(void **)&vertex_values);CHKERRQ(ierr);
  ierr = PetscDoubleView(2*(rend-rstart),vertex_values,viewer);CHKERRQ(ierr);
  ierr = AODataSegmentRestoreIS(appctx->aodata,"vertex","values",PETSC_NULL,(void **)&vertex_values);CHKERRQ(ierr);
  ierr = ISDestroy(isvertex);CHKERRQ(ierr);

  /* 
     send list of vertices for each cell MUST be in global numbering (not local) this 
     cannot use appctx->grid->cell_vertex 
  */
  ierr = AODataSegmentGetIS(appctx->aodata,"cell","vertex",appctx->grid.cell_global,(void **)&cell_vertex);CHKERRQ(ierr);
  ierr = PetscIntView(6*appctx->grid.cell_n,cell_vertex,viewer);CHKERRQ(ierr);
  ierr = AODataSegmentRestoreIS(appctx->aodata,"cell","vertex",PETSC_NULL,(void **)&cell_vertex);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}
