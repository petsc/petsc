






/*
       Defines data structures for writing simple cell (element) based PDE codes.
*/

#if !defined(__APPCTX_H)
#define __APPCTX_H

#include "ao.h"           /* allows using the PETSc AOData-base routines for grid information */
#include "sles.h"         /* allows using PETSc linear solvers */

/*
    The AppGrid data structure contains all the information about the grid cells, vertices 
    boundaries etc. It is created by Appload() (see appload.c) from the AO database.
*/
typedef struct {

  /* the coords of each of the 4 vertices corresponding to each cell */
  double *cell_coords;

  /* the index  for each of the 4 vertices of a given cell 
     in a local (per processor) numbering */
  int *cell_vertex;

  /* the local to global mapping for vertices, i.e. if you apply ltog to a list of
     vertices in local (per processor) numbering it will return them in global (parallel) numbering */
  ISLocalToGlobalMapping ltog;

  /* the number of cells on this processor,*/
  int cell_n; 
  /* number of  vertices on local cells (including those "shared" by other processors, i.e. 
     includes "ghost vertices" ) */
  int vertex_count;
  /* only those vertices on local processor which actually belong to this processor, i.e. does not 
     include "ghost vertices" */
  int vertex_local_count;
 
  /********* Data structures for the boundary conditions ************/
  /* the local indices of vertices on the boundary */
  IS  vertex_boundary;
  int    boundary_count;    /* number of vertices on boundary (including ghost vertices) */
  double *boundary_values;  /* work space for the boundary values */
  double *boundary_coords;  /* the coordinates of the boundary points */

  /********* Data structures for graphics ******************** */
  IS     iscell;                   /* cells owned by this processor in global numbering */
  int    *global_cell_vertex;      /* vertices for each local cell in global numbering */
} AppGrid;

/*
    The AppAlgebra data structure contains all the linear algebra objects needed to solve the linear
    problem. It is created in appsetalg.c
*/
typedef struct {
  Vec b;           /* Global vector for the rhs */
  Vec x;           /* Global vector for the solution */
  Mat A;           /* for the stiffness */
} AppAlgebra;

/*
    The AppView data structure contains information about what is to be displayed and where
*/
typedef struct {

  PetscTruth matlabgraphics;      /* plots solution in Matlab (using bscript.m) */
  PetscTruth show_matrix;         /* displays sparsity pattern of matrix */
  PetscTruth show_griddata;       /* dumps grid database, orderings etc to screen */
  PetscTruth show_grid;           /* plots the grid with numbering of cells, vertices etc. */

  Draw       drawlocal;           /* graphics window for draw local per processor part of global grid */
  Draw       drawglobal;          /* graphics window for drawing global (parallel grid) */

} AppView;


/* 
   This data structure contains information about the finite element basis functions on the 
   REFERENCE ELEMENT and then work space used to contain results in computing the element 
   stiffness and element load.
*/
typedef struct {

  /* the first two sets are the same for all elements, i.e. they are for the reference element */

  /* values of the reference interpolating functions at the Gauss pts */
  double RefVal[4][4];
  double RefDx[4][4];
  double RefDy[4][4];

  /* quadrature weights */
  double  weights[4];

  /* the rest are computed for each element while computing the stiffness matrices */

  /* values of the local interpolating fns at the Gauss pts */
  double dx[4][4], dy[4][4];
  double detDh[4];

  /* the images of the Gauss pts in the local element */
  double x[4], y[4];

  /* results of local integrations */
  double rhsresult[4];
  double stiffnessresult[4][4];

  /* pointer to coords of current cell */
  double *coords;

} AppElement;

/*
        Entire application context; any data in the computation can be access 
    through this.
*/
typedef struct {
  MPI_Comm   comm;
  AOData     aodata;
  AppGrid    grid;
  AppAlgebra algebra;  
  AppView    view;
  AppElement element;
} AppCtx;


/* function declarations */

extern int AppCtxCreate(MPI_Comm,AppCtx **);
extern int AppCtxDestroy(AppCtx *);

extern int AppCtxViewGrid(Draw,void*);
extern int AppCtxViewMatlab(AppCtx*);
extern int AppCtxGraphics(AppCtx *appctx);

extern int AppCtxSetLocal(AppCtx *);
extern int AppCtxSolve(AppCtx*);

double f(double, double);
double bc(double, double);
double u(double, double);
double v(double, double);

extern int AppCtxCreateRhs(AppCtx*);
extern int AppCtxCreateMatrix(AppCtx*);
extern int AppCtxSetMatrix(AppCtx*);
extern int AppCtxSetRhs(AppCtx*);
extern int SetBoundaryConditions(AppCtx*);
extern int SetMatrixBoundaryConditions(AppCtx *appctx);

extern int ComputeRHSElement( AppElement *phi );
extern int ComputeStiffnessElement( AppElement *phi );

extern int SetReferenceElement(AppCtx* appctx);
extern int SetLocalElement(AppElement *phi );

#endif
