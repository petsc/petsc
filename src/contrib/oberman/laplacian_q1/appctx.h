/*
       Defines data structures for writing a simple cell (element) based PDE code
    for solving scalar PDE problems like the Laplacian.
*/

#if !defined(__APPCTX_H)
#define __APPCTX_H

#include "petscao.h"           /* allows using the PETSc AOData-base routines for grid information */
#include "petscksp.h"         /* allows using PETSc linear solvers */
#include "petscpf.h"

/*-------------------------------------------------------------------

    The AppGrid data structure:
      contains all the information about the grid cells, vertices boundaries etc. 
      It is created by Appload() (see appload.c) from the AO database.
*/
typedef struct {
  /********* Data structures for cells ************/
  PetscInt cell_n;   /* the number of cells on this processor,*/

  /* ---- coordinates of each of the 4 vertices corresponding to each cell
     cell_coords[0],cell_coords[1] represent x,y of the first cell's first vertice 
     cell_coords[0],cell_coords[1] represent x,y of the first cell's second vertice etc, */
  PetscReal *cell_coords;

  /* ---- index  for each of the 4 vertices of a given cell in the local (per processor) numbering */
  PetscInt *cell_vertex;

  /********* Data structures for vertices ************/
  PetscInt vertex_n;  /* number of distinct vertices on local cells, including ghost vertices*/
  PetscInt vertex_local_n; /* number of distinct vertices on local cells, excluding ghost vertices */
 
  /* ---- local to global mapping for vertices, i.e. if you apply ltog to a list of
     vertices in local (per processor) numbering it will return them in global (parallel) numbering */
  ISLocalToGlobalMapping ltog;

  /********* Data structures for the boundary conditions ************/
  IS      vertex_boundary;  /* local indices of vertices on the boundary */
  PetscInt     boundary_n;   /* number of vertices on boundary (including ghost vertices) */
  PetscReal *boundary_values;  /* work space for the boundary values */
  PetscReal *boundary_coords;  /* the coordinates of the boundary points */

  /********* Data structures for graphics ******************** */
  IS     iscell;                   /* cells owned by this processor in global numbering */
  /* FIX THIS */ PetscInt    *global_cell_vertex;      /* vertices for each local cell in global numbering */
} AppGrid;

/*------------------------------------------------------------

    The AppAlgebra data structure:
      contains all the linear algebra objects needed to solve the linear
      problem. It is created in appalgebra.c
*/
typedef struct {
  Vec b;           /* Global vector for the rhs */
  Vec x;           /* Global vector for the solution */
  Mat A;           /* Global matrix for the stiffness */
} AppAlgebra;

/*------------------------------------------------------------------
    The AppView data structure:
      contains information about what is to be displayed and where
*/
typedef struct {
  PetscTruth show_solution;      /* plots solution in Matlab (using bscript.m) */
  PetscTruth show_matrix;         /* displays sparsity pattern of matrix */
  PetscTruth show_griddata;       /* dumps grid database, orderings etc to screen */
  PetscTruth show_grid;           /* plots the grid with numbering of cells, vertices etc. */

  PetscDraw  drawlocal;           /* graphics window for drawing local per processor part of global grid */
  PetscDraw  drawglobal;          /* graphics window for drawing global (parallel grid) */
} AppView;


/* ---------------------------------------------------------
   The AppElement data structure:
     contains information about the finite element basis functions on the 
     REFERENCE ELEMENT and then work space used to contain results in computing
     the element stiffness and element load.
*/
typedef struct {
  /* ********** same for all elements, i.e. for the reference element********* */
  PetscReal RefVal[4][4];/* values of the reference interpolating (basis) functions at the Gauss pts */
  PetscReal RefDx[4][4]; /* values of the derivatives of the reference basis functions ...*/
  PetscReal RefDy[4][4];

  PetscReal weights[4];  /* quadrature weights */
 
  /* **********computed for each element while computing the stiffness ******** */

  PetscReal dx[4][4], dy[4][4];/* values of the derivatives actual element's interpolating functions at the Gauss pts */
  PetscReal detDh[4];

  PetscReal xy[8];  /* the images (locations) of the Gauss pts in the actual element */

  PetscReal rhsresult[4];  /* results of local integrations */
  PetscReal stiffnessresult[4][4];

  PetscReal *coords;  /* pointer to coords of current cell */
  PF        rhs;
} AppElement;

/*----------------------------------------------------
  AppCtx:
    entire application context; any data in the computation can be access 
    through this.
*/
typedef struct {
  MPI_Comm   comm;
  AOData     aodata;
  AppGrid    grid;
  AppAlgebra algebra;  
  AppView    view;
  AppElement element;
  PF         bc;
} AppCtx;

/*-----------------------------------------------------*/
/* function declarations */

extern PetscErrorCode AppCtxCreate(MPI_Comm,AppCtx **);
extern PetscErrorCode AppCtxDestroy(AppCtx *);

extern PetscErrorCode AppCtxViewGrid(PetscDraw,void*);
extern PetscErrorCode AppCtxViewMatlab(AppCtx*);
extern PetscErrorCode AppCtxGraphics(AppCtx *appctx);

extern PetscErrorCode AppCtxSetLocal(AppCtx *);
extern PetscErrorCode AppCtxSolve(AppCtx*);

extern PetscErrorCode AppCtxCreateRhs(AppCtx*);
extern PetscErrorCode AppCtxCreateMatrix(AppCtx*);
extern PetscErrorCode AppCtxSetMatrix(AppCtx*);
extern PetscErrorCode AppCtxSetRhs(AppCtx*);
extern PetscErrorCode SetBoundaryConditions(AppCtx*);
extern PetscErrorCode SetMatrixBoundaryConditions(AppCtx *appctx);

extern PetscErrorCode ComputeRHSElement( AppElement *phi );
extern PetscErrorCode ComputeStiffnessElement( AppElement *phi );

extern PetscErrorCode SetReferenceElement(AppCtx* appctx);
extern PetscErrorCode SetLocalElement(AppElement *phi );

#endif
