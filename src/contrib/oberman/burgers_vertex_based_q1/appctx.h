
/*
       Defines some simple data structures for writing cell (element) based PDE code
    for solving multi-component nonlinear PDE like Burgers equation.

*/
#if !defined(__APPCTX_H)
#define __APPCTX_H

#include "petscao.h"        /* allows using the PETSc AOData-base routines for grid information */
#include "petscsnes.h"      /* allows using PETSc nonlinear solvers */
#include "petscbt.h"       /* allows using PETSc's logical bit arrays */

/*--------------------------------------------------------------------

    The AppGrid data structure:
      contains all the information about the grid cells, vertices boundaries etc. 
      It is created by Appload() (see appload.c) from the AO database.
*/
/*

        cell_vertex          - vertices of the cells (in local numbering)
        cell_global          - global number of each cell on this processor
        cell_cell            - 
        vertex_n             - number of unique vertices on this processor 
        vertex_n_ghosted     - number of vertices including ghost ones
        vertex_global        - global number of each vertex on this processor
        vertex_value         - 
        vertex_boundary      - list of on processor vertices (including ghosts)
                               that are on the boundary
       vertex_global_blocked       - the list of vertices expanded by a facter of 
                                           the number of degrees of freedom in the problem
       vertex_boundary_blocked - the boundary vertices, blocked.		       
        vertex_boundary_flag - bit array indicating for all on processor vertices (including ghosts) 
                               if they are on the boundary
        ltog                 - mapping from local numbering of vertices (including ghosts)
                               to global
       dltog                - the corresponding mapping for the DFs of the vertices
       NVs                  -the number of vertices per cell (4 in the case of billinear elements)

*/

typedef struct {
  /********* Data structures for cells ************/
  PetscInt                    cell_n;          /* number of cells on this process */

  /* ---- index  for each of the 4 vertices of a given cell in the local (per processor) numbering */
  PetscInt                    *cell_vertex;

  IS                     cell_global;

  /* neighbors of the cell.  (-1) indicates no neighbor. ordering cycles clockwise */
  PetscInt                    *cell_neighbors;

  /********* Data structures for vertices ************/
  PetscInt                    vertex_n; /* number of distinct vertices on local cells, including ghost vertices*/
  PetscInt                    vertex_local_n; /* number of distinct vertices on local cells, excluding ghost vertices */

  IS                     vertex_global;

  PetscReal                 *vertex_coords; /* x,y coordinates of vertices on this processor */

  PetscBT                vertex_boundary_flag;
  IS                     vertex_boundary;
  IS                     vertex_global_blocked;  /* for 2 degrees of freedom */
  IS                     vertex_boundary_blocked; /* for 2 deg of freedom */
  ISLocalToGlobalMapping ltog;
  ISLocalToGlobalMapping dltog;    /* for 2 deg of freedom */
  PetscInt                    NVs;
} AppGrid;

/*
    gtol             - global to local vector scatter
                       (used to move data from x to w_local for example
    A                - parallel sparse stiffness matrix
    b                - parallel vector contains right hand side
    x                - parallel vector contains solution
    w_local, x_local - (work) sequential vectors contains local plus ghosted elements
*/
typedef struct {
  Vec   b; /* rhs */
  Vec g; /* initial guess */
  Vec f; /* for the nonlinear function*/
  Vec f_local;
  Vec x,z;
  Vec                    w_local,x_local,z_local;  /* local ghosted work vectors */
  VecScatter             gtol;
  VecScatter dgtol;   /* the scatter for blocked vectors */
  Mat                    A,J;
} AppAlgebra;

/*
    drawlocal    - window where processor local portion is drawn
    drawglobal   - window where entire grid is drawn

*/

typedef struct {
  PetscDraw       drawlocal;
  PetscDraw       drawglobal;
  PetscTruth matlabgraphics;
  PetscTruth show_grid;
  PetscTruth show_solution;
} AppView;

/* 
Additional structure for the discretization.
Values at the gauss points of the bilinear basis functions
*/

typedef struct {
  PetscInt dorhs;     /* flag to see if we are computing rhs */
  PetscReal Values[4][4];  /* values of reference element */
  PetscReal DxValues[4][4]; /* for reference element */
  PetscReal DyValues[4][4]; /* for reference element */
  PetscReal dx[16]; /* for local element */
  PetscReal dy[16]; /*for local element */
  PetscReal detDh[4]; /* determinant of map from reference element to the local element */
  PetscReal x[4];/* x coord of image of gauss point */
  PetscReal y[4];/* y coord of image of gauss point */
} AppElement;

 

 typedef PetscReal (*DFP)(PetscReal,PetscReal); /* pointer to a function of 2 vars */

typedef struct {
  char *rhs_string;
  char *bc_string;
  DFP f; /* rhs for u */
  DFP g; /* rhs for v */
  DFP bc1;/* bc for u */
  DFP bc2;/* bc for v */
  PetscReal eta; /* the viscosity */

  /*  int DFs;   Number of Degrees of Freedom in the Problem */
}AppEquations;


/*
      comm   - MPI communictor where grid etc are stored
      aodata - grid database
*/
typedef struct {
  MPI_Comm   comm;
  AOData     aodata;
  AppGrid    grid;
  AppAlgebra algebra;
  AppView    view;
  AppElement element;
  AppEquations equations;
} AppCtx;


extern PetscErrorCode AppCtxView(PetscDraw,void*);
extern PetscErrorCode AppCtxViewSolution(PetscDraw,void*);
extern PetscErrorCode AppCtxCreate(MPI_Comm,AppCtx **);
extern PetscErrorCode AppCtxDestroy(AppCtx *);
extern PetscErrorCode AppCtxSetLocal(AppCtx *);
extern PetscErrorCode AppCtxSolve(AppCtx*);
extern PetscErrorCode AppCtxGraphics(AppCtx *);
extern PetscErrorCode AppCtxViewMatlab(AppCtx*);


 PetscReal pde_f(PetscReal, PetscReal); 
 PetscReal pde_g(PetscReal, PetscReal); 
 PetscReal pde_bc1(PetscReal, PetscReal); 
 PetscReal pde_bc2(PetscReal, PetscReal); 
PetscReal pde_soln(PetscReal, PetscReal);

PetscErrorCode AppCtxSetRhs(AppCtx*);
PetscErrorCode AppCtxCreateVector(AppCtx*);
PetscErrorCode AppCtxSetMatrix(AppCtx*);
PetscErrorCode AppCtxCreateMatrix(AppCtx*);
PetscErrorCode FormFunction(SNES, Vec, Vec, void *);
PetscErrorCode FormJacobian(SNES, Vec , Mat *, Mat *, MatStructure *, void *);
PetscErrorCode SetNonlinearFunction(Vec, AppCtx *, Vec);

extern PetscErrorCode AppCtxSetReferenceElement(AppCtx*);
extern PetscErrorCode AppCtxSetFunctions(AppCtx*);
extern PetscErrorCode SetLocalElement(AppElement *, PetscReal *);
extern PetscErrorCode ComputeRHS( DFP, DFP, AppElement *, PetscReal *);
extern PetscErrorCode ComputeMatrix( AppElement *, PetscReal *);
extern PetscErrorCode ComputeNonlinear(AppElement *, PetscReal *, PetscReal *);
extern PetscErrorCode ComputeJacobian(AppElement *, PetscReal *, PetscReal *);

extern PetscErrorCode FormInitialGuess(AppCtx *);
extern PetscErrorCode SetJacobian(Vec,AppCtx *,Mat*);
#endif
