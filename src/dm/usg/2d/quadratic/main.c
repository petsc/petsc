
static char help[] = 
 "Solves a simple linear PDE in 2D on an unstructured grid consisting of\n\
  triangles discretized by quadratic finite elements.  Input parameters include:\n\
    -f grid_file            - specify input grid file (e.g., grids/R or grids/B)\n\
    -matlab_graphics        - activate Matlab graphics (for use with mscript.m)\n\
    -show_numbers           - show numbers of each proc\n\
    -show_elements          - show elements of each proc\n\
    -show_vertices          - show vertices of each proc\n\
    -show_boundary          - show boundary edges of each proc\n\
    -show_boundary_vertices - show boundary vertices of each proc\n\n";

/*
   Demonstrates how to write a unstructured grid PDE solver using the 
   PETSc AOData-base infrastructure. 

   To solve a different (but similar) problem, one should copy the files 
        appctx.h appload.c appview.c and main.c
   and then add/delete data structures and code as needed for a particular
   application.
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
       Initialize PETSc and load the grid database
     --------------------------------------------------------------------- */

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


  /* ---------------------------------------------------------------------
       Form the linear system and then solve it
     --------------------------------------------------------------------- */

  /*
      Setup the linear system and solve it
  */
  ierr = AppCtxSolve(appctx); CHKERRA(ierr);

  /*
      Visualize solution
  */
  if (appctx->view.showsomething) {
    ierr = VecScatterBegin(algebra->x,algebra->w_local,INSERT_VALUES,SCATTER_FORWARD,algebra->gtol); CHKERRA(ierr);
    ierr = VecScatterEnd(algebra->x,algebra->w_local,INSERT_VALUES,SCATTER_FORWARD,algebra->gtol); CHKERRA(ierr);
    ierr = DrawZoom(appctx->view.drawglobal,AppCtxViewSolution,appctx); CHKERRA(ierr);
  }

  if (appctx->view.matlabgraphics) {
    AppCtxViewMatlab(appctx);
  }


  /* ---------------------------------------------------------------------
       Clean up
     --------------------------------------------------------------------- */

  /*
      Destroy all data structures
  */
  ierr = AppCtxDestroy(appctx); CHKERRA(ierr);

  PetscFinalize();

  PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------------- */
/*
    AppCtxSolve - Sets up the linear system associated with the PDE and
    then solves it.

    Input Parameter:
    appctx - user-defined application context
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
     Create the right-hand-side vector and various work vectors
  */
  ierr = AppCtxCreateRhs(appctx); CHKERRQ(ierr);

  /*
     Create the sparse matrix, with correct nonzero sparsity pattern
  */
  ierr = AppCtxCreateMatrix(appctx); CHKERRQ(ierr);

  /*
     Compute the right-hand-side vector values
  */
  ierr = AppCtxSetRhs(appctx); CHKERRQ(ierr);

  /*
     Compute the matrix entries 
  */
  ierr = AppCtxSetMatrix(appctx); CHKERRQ(ierr);

  /*
     Solve the linear system
  */
  ierr = SLESCreate(comm,&sles); CHKERRQ(ierr);
  ierr = SLESSetOperators(sles,algebra->A,algebra->A,DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);
  ierr = SLESSetFromOptions(sles); CHKERRQ(ierr);
  ierr = SLESSolve(sles,algebra->b,algebra->x,&its); CHKERRQ(ierr);
  ierr = SLESDestroy(sles); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------------- */
/*
   AppCtxCreateRhs - Creates vector data structures and scattering info.
     - Generates the "global" parallel vector to contain the right-hand-side 
       and solution vectors.
     - Generates "ghosted" local vectors for local computations, etc.
     - Generates scatter context for updating ghost points

   Input Parameter:
   appctx - user-defined application context
*/
#undef __FUNC__
#define __FUNC__ "AppCxtCreateRhs"
int AppCtxCreateRhs(AppCtx* appctx)
{
  AppGrid                *grid = &appctx->grid;       /* grid context */
  AppAlgebra             *algebra = &appctx->algebra; /* algebra context */
  MPI_Comm               comm = appctx->comm;         /* communicator */
  Vec                    x, b;                        /* global parallel vectors */
  Vec                    x_local,w_local,z,z_local;   /* ghosted local vectors */
  VecScatter             gtol;                        /* global-to-local scattering context */
  ISLocalToGlobalMapping ltog = grid->ltog;           /* local-to-global mapping */
  SLES                   sles;                        /* linear solver context variable */
  int                    ierr,its;
  int                    vertex_n = grid->vertex_n;   /* number of unique local vertices */
  int                    vertex_n_ghosted = grid->vertex_n_ghosted; /* number of ghosted
                                                                       local vertices */
  IS                     vertex_global = grid->vertex_global;       /* global number of each
                                                                       vertex on this proc */

  PetscFunctionBegin;

  /*
     Create right-hand-side vector and various work vectors
  */
  ierr = VecCreateMPI(comm,vertex_n,PETSC_DECIDE,&b); CHKERRQ(ierr);
  ierr = VecSetLocalToGlobalMapping(b,ltog); CHKERRQ(ierr);
  ierr = VecDuplicate(b,&x);
  ierr = VecDuplicate(b,&z);

  /*
     Create ghosted local vectors
  */
  ierr = VecCreateSeq(PETSC_COMM_SELF,vertex_n_ghosted,&w_local); CHKERRQ(ierr);
  ierr = VecDuplicate(w_local,&x_local); CHKERRQ(ierr);
  ierr = VecDuplicate(w_local,&z_local); CHKERRQ(ierr);
  ierr = VecScatterCreate(b,vertex_global,w_local,0,&gtol); CHKERRQ(ierr);

  /*
     Set vectors in the user-defined structure
  */
  algebra->x       = x;
  algebra->b       = b;
  algebra->z       = z;
  algebra->w_local = w_local;
  algebra->x_local = x_local;
  algebra->z_local = z_local;
  algebra->gtol    = gtol;

  PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------------- */
/*
   AppCtxSetRhs - Computes right-hand-side vector.

   Input Parameter:
   appctx - user-defined application context
*/
#undef __FUNC__
#define __FUNC__ "AppCxtSetRhs"
int AppCtxSetRhs(AppCtx* appctx)
{
  AppGrid    *grid = &appctx->grid;                     /* grid context */
  AppAlgebra *algebra = &appctx->algebra;               /* algebra context */
  Vec        b = algebra->b;                            /* right-hand-side vector */
  Scalar     *values;                                   /* right-hand-side values for an element */
  double     *coors,*vertex_value = grid->vertex_value; /* coordinates for an element */
  int        ncell = 6;                                 /* number of nodes per cell */
  int        cell_n = grid->cell_n;                     /* number of cells on this proc */
  int        *cell_vertex = grid->cell_vertex;          /* vertices of the cells (local ordering) */
  int        *vertices, j, ierr, i;

  /*
     Allocate space to hold the coordinates of a single cell, and the 
     right-hand-side contribution generated from a single cell.
  */
  coors  = (double*) PetscMalloc(2*ncell*sizeof(double));CHKPTRQ(coors);
  values = (Scalar *) PetscMalloc(ncell*sizeof(Scalar));CHKPTRQ(values);


  /* ---------------------------------------------------------------
        Loop over elements, computing load one element at a time 
        and putting into right-hand-side
     ---------------------------------------------------------------- */

  for ( i=0; i<cell_n; i++ ) {
    vertices = cell_vertex+ncell*i;
    /* 
       Load the cell vertex coordinates 
    */
    for ( j=0; j<ncell; j++) {
      coors[2*j]   = vertex_value[2*vertices[j]];
      coors[2*j+1] = vertex_value[2*vertices[j]+1];
    }
    /*
      Here is where you would call the routine to compute the 
      element load
    */
    for ( j=0; j<ncell; j++ ) values[j] = 1.0;

    ierr = VecSetValuesLocal(b,ncell,vertices,values,ADD_VALUES); CHKERRQ(ierr);
  }

  /*  
     Complete matrix assembly
  */
  ierr = VecAssemblyBegin(b); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b); CHKERRQ(ierr);

  PetscFree(values);  
  PetscFree(coors);
  PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------------- */
/*
   AppCtxCreateMatrix - Creates the sparse matrix (with the correct nonzero 
   sparsity pattern) that will be later filled with the stiffness matrix.

   Input Parameter:
   appctx - user-defined application context
*/
#undef __FUNC__
#define __FUNC__ "AppCxtSetMatrix"
int AppCtxCreateMatrix(AppCtx* appctx)
{
  MPI_Comm               comm = appctx->comm;              /* communicator */
  AppAlgebra             *algebra = &appctx->algebra;      /* algebra context */
  AppGrid                *grid    = &appctx->grid;         /* grid context */
  Vec                    w_local = algebra->w_local;       /* ghosted local vectors */
  Vec                    x_local = algebra->x_local;
  Vec                    z_local = algebra->z_local;
  Vec                    z = algebra->z, x = algebra->x;   /* parallel vectors */
  VecScatter             gtol = algebra->gtol;             /* global-to-local scatter context */
  Mat                    A;                                /* matrix to define linear system */
  ISLocalToGlobalMapping ltog = grid->ltog;                /* local-to-global mapping */
  int                    vertex_n = grid->vertex_n;        /* number of unique local vertices */
  int                    cell_n = grid->cell_n;            /* number of local cells */
  int                    *cell_vertex = grid->cell_vertex; /* vertices of cells (local numbering) */
  int                    ncell = 6;                        /* number of nodes per cell */
  int                    *cell_cell = grid->cell_cell;     /* neighbors of each cell */
  IS                     vertex_boundary = grid->vertex_boundary; /* vertices on boundary 
                                                                    (including ghosts) */
  int                    ierr, rank, *vertices, cproc, i, j, *dnz, *cells,*onz;
  Scalar                 srank, *procs, *sdnz, zero = 0.0, *values, wght, *sonz;
  Scalar                 one = 1.0;
  double                 *coors;

  PetscFunctionBegin;
  MPI_Comm_rank(comm,&rank); 

  /* ------------------------------------------------
      Determine nonzero structure of the matrix 
     ------------------------------------------------ */
  
  /* (1) Make proc[] contain the processor number of each ghosted vertex */
  srank = rank;
  ierr = VecSet(x,srank); CHKERRQ(ierr);
  ierr = VecScatterBegin(x,w_local,INSERT_VALUES,SCATTER_FORWARD,gtol); CHKERRQ(ierr);
  ierr = VecScatterEnd(x,w_local,INSERT_VALUES,SCATTER_FORWARD,gtol); CHKERRQ(ierr);
  ierr = VecGetArray(w_local,&procs); CHKERRQ(ierr);
  ierr = VecSet(x_local,zero); CHKERRQ(ierr);
  ierr = VecGetArray(x_local,&sdnz); CHKERRQ(ierr);
  ierr = VecSet(z_local,zero); CHKERRQ(ierr);
  ierr = VecGetArray(z_local,&sonz); CHKERRQ(ierr);

  /* (2) Loop over local elements; count matrix nonzeros */

  /*
     Use quadratic triangular elements
  */
  /* Loop over cells */
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

  /*
     Restore arrays
  */
  ierr = VecRestoreArray(x_local,&sdnz); CHKERRQ(ierr);
  ierr = VecRestoreArray(z_local,&sonz); CHKERRQ(ierr);
  ierr = VecRestoreArray(w_local,&procs); CHKERRQ(ierr);

  /*
     Scatter from local to global
  */
  ierr = VecSet(x,zero); CHKERRQ(ierr);
  ierr = VecScatterBegin(x_local,x,ADD_VALUES,SCATTER_REVERSE,gtol); CHKERRQ(ierr);
  ierr = VecScatterEnd(x_local,x,ADD_VALUES,SCATTER_REVERSE,gtol); CHKERRQ(ierr);
  ierr = VecGetArray(x,&sdnz); CHKERRQ(ierr);
  ierr = VecSet(z,zero); CHKERRQ(ierr);
  ierr = VecScatterBegin(z_local,z,ADD_VALUES,SCATTER_REVERSE,gtol); CHKERRQ(ierr);
  ierr = VecScatterEnd(z_local,z,ADD_VALUES,SCATTER_REVERSE,gtol); CHKERRQ(ierr);
  ierr = VecGetArray(z,&sonz); CHKERRQ(ierr);

  /*
     Allocate space for work arrays to contain sparsity structure info
       dnz (diagonal portion), onz (off-fiagonal portion)
  */

  dnz  = (int *) PetscMalloc((vertex_n+1)*sizeof(int));CHKPTRQ(dnz);
  onz  = (int *) PetscMalloc((vertex_n+1)*sizeof(int));CHKPTRQ(onz);
  for ( i=0; i<vertex_n; i++ ) {
    dnz[i] = 1 + (int) PetscReal(sdnz[i]);
    onz[i] = (int) PetscReal(sonz[i]);
  }  

  /*
     Restore arrays
  */
  ierr = VecRestoreArray(x,&sdnz); CHKERRQ(ierr);
  ierr = VecRestoreArray(z,&sonz); CHKERRQ(ierr);

  /*
     Create parallel matrix with preallocation of nonzeros 
  */
  ierr = MatCreateMPIAIJ(comm,vertex_n,vertex_n,PETSC_DETERMINE,PETSC_DETERMINE,
                         0,dnz,0,onz,&A); CHKERRQ(ierr);

  /*
    Set local-to-global mapping so that we can set matrix extries later
    using the local numbering 
  */
  ierr = MatSetLocalToGlobalMapping(A,ltog); CHKERRQ(ierr);
  algebra->A = A;

  PetscFree(dnz);
  PetscFree(onz);

  PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------------- */
/*
   AppCtxSetRhs - Compute the stiffness matrix entries.

   Input Parameter:
   appctx - user-defined application context
*/
#undef __FUNC__
#define __FUNC__ "AppCxtSetMatrix"
int AppCtxSetMatrix(AppCtx* appctx)
{
  AppAlgebra *algebra = &appctx->algebra;             /* algebra context */
  AppGrid    *grid    = &appctx->grid;                /* grid context */
  MPI_Comm   comm = appctx->comm;                     /* communicator */
  Mat        A = algebra->A;                          /* matrix */
  IS         vertex_boundary = grid->vertex_boundary; /* vertices on boundary 
                                                         (including ghosts) */
  double     *coors,*vertex_value = grid->vertex_value;/* coordinates for an element */
  int        vertex_n = grid->vertex_n;               /* number of unique local vertices */
  int        ncell = 6;                               /* number of nodes per cell */
  int        cell_n = grid->cell_n;                   /* number of cells on this proc */
  int        *cell_vertex = grid->cell_vertex;        /* vertices of the cells (local ordering) */
  int        *cell_cell = grid->cell_cell;            /* neighbors of each cell */
  int        ierr, rank, *vertices, cproc, i, j,*dnz, *cells,*onz;
  Scalar     srank, *procs, *sdnz, zero = 0.0, *values, wght, *sonz, one = 1.0;

  PetscFunctionBegin;
  MPI_Comm_rank(comm,&rank); 

  /* ---------------------------------------------------------------
       Loop over local elements, putting values into matrix 
     --------------------------------------------------------------- */

  /*
        Allocate space for coordinates and element stiffness for one element
  */
  coors  = (double *) PetscMalloc(2*ncell*sizeof(double));CHKPTRQ(coors);
  values = (Scalar *) PetscMalloc(ncell*ncell*sizeof(Scalar));CHKPTRQ(values);

  for ( i=0; i<cell_n; i++ ) {
    vertices = cell_vertex + ncell*i;   
    for ( j=0; j<ncell; j++) {
      coors[2*j]   = vertex_value[2*vertices[j]];
      coors[2*j+1] = vertex_value[2*vertices[j]+1];
    }
    /* 
        Here one would call the routine that computes the element stiffness
    */
    for ( j=0; j<ncell*ncell; j++) values[j] = 1.0;

    ierr = MatSetValuesLocal(A,ncell,vertices,ncell,vertices,values,ADD_VALUES);CHKERRQ(ierr);
  }

  /* 
     Assemble matrix
  */
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  PetscFree(values);
  PetscFree(coors);

  /* -----------------------------------------------------------
       Apply Dirichlet boundary conditions
     ----------------------------------------------------------- */

  ierr = MatZeroRowsLocal(A,vertex_boundary,&one); CHKERRQ(ierr);

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
  ierr = PetscIntView(6*appctx->grid.cell_n,cell_vertex,viewer); CHKERRQ(ierr);
  ierr = AODataSegmentRestoreIS(appctx->aodata,"cell","vertex",PETSC_NULL,(void **)&cell_vertex); 
         CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}
