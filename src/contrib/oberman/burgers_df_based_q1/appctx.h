/*
    Defines some simple data structures for writing cell (element) based PDE codes.

    Generally one would write a code by starting with the data structures below and 
    then delete unneeded information. 
*/
#if !defined(__APPCTX_H)
#define __APPCTX_H

#include "petscao.h"
#include "petscbt.h"
#include "petscsnes.h"
/*
  cell_n               - number of cells on this processor 
  cell_vertex          - vertices of the cells (in local numbering)
  cell_global          - global numbering of each cell on this processor
  cell_cell            - neighbors of the cell.  (-1) indicates no neighbor.
                               ordering cycles clockwise from left
  vertex_n             - number of unique vertices on this processor 
  vertex_n_ghosted     - number of vertices on this processor, including ghost ones
  vertex_global        - global numbering of vertices on this processor, 
                                          including ghost pts
  vertex_value         - x,y coordinates of vertices on this processor
  vertex_boundary      - list of vertices (including ghosts)
                               that are on the boundary on this processor
  vertex_global_blocked - the list/global numbering of vertices expanded by a factor
                                   of the number of degrees of freedom in the problem
  vertex_boundary_blocked - the boundary vertices, blocked.		       
  vertex_boundary_flag - bit array indicating for all on processor vertices 
                                   (including ghosts) if they are on the boundary
  ltog                 - mapping from local numbering of vertices (including ghosts)
                               to global
  dltog (is never used?) - the corresponding mapping for the DFs of the vertices
  NVs (is never used?)  -the number of vertices per cell (4 in the case of billinear elements)
*/    

/*-----------------------------------------------------------------------*/
typedef struct {
  IS           cell_global,vertex_global,df_global; /* global numbering of cells, vertices, dfs */
  IS   isboundary_df; /* is for df on boundary */
  PetscInt  *boundary_df; /* the corresponding array  */
  IS   isvertex_boundary; /* is for vertices on boundary */
  PetscInt *vertex_boundary;
  double *bvs; /* array of space for boundary values */
  double *bvc;  /* array of boundary vertex coordinates */
  double *cell_coords;   /* coords of the vertices corresponding to each cell */ 
 /* sizes of local df (including ghosted),df on this proc, cells, vertices, ghosted vertices */
  PetscInt df_n_ghosted, vertex_boundary_count, df_local_count, cell_n, vertex_n, vertex_n_ghosted; 

  PetscInt *vertex_df;   /* info on DF associated to vertices */
  PetscInt  *cell_df,*cell_vertex,*cell_cell; /* info on df, vertices, neighbours, indexed by cells */

  double                 *vertex_value; /* numerical coords of the vertices */ 
  ISLocalToGlobalMapping ltog,dfltog;  /* ltog associated with vertices, degrees of freedom */

} AppGrid;

/*--------------------------------------------------------------------------------------
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
  Vec *solnv; /* array of solution vectors at each time step */
  Vec f_local; /* local values of nonlinear function */
  Vec f_boundary; /* for local boundary values */
 
  Vec x,z;
  VecScatter dfgtol;   /* the scatter for degrees of freedom  vectors */
  VecScatter dfbgtol;   /* the scatter for degrees of freedom  vectors on the boundary */

  Mat A,J;
} AppAlgebra;

/*-------------------------------------------------------------------------------
    drawlocal    - window where processor local portion is drawn
    drawglobal   - window where entire grid is drawn

*/

typedef struct {
  PetscDraw       drawlocal;
  PetscDraw       drawglobal;
  PetscTruth matlabgraphics;
  PetscTruth show_grid;
  PetscTruth show_solution;
  PetscTruth show_griddata;
} AppView;

/* --------------------------------------------------------------------------
for j= 1:tsteps,
figure(1)
fill3(cellx,celly,cellz1(:,:,j),cellz1(:,:,j))
%
figure(2)
fill3(cellx,celly,cellz2(:,:,j),cellz2(:,:,j))
end
*/

/*--------------------------------------------------------------------------
Additional structure for the discretization.
Values at the gauss points of the bilinear basis functions
*/

typedef struct {
  PetscInt dorhs;     /* flag to see if we are computing rhs */
  double Values[4][4];  /* values of reference element */
  double DxValues[4][4]; /* for reference element */
  double DyValues[4][4]; /* for reference element */
  double dx[16]; /* for local element */
  double dy[16]; /*for local element */
  double detDh[4]; /* determinant of map from reference element to the local element */
  double x[4];/* x coord of image of gauss point */
  double y[4];/* y coord of image of gauss point */
} AppElement;

 
/*--------------------------------------------------------------------------*/

 typedef double (*DFP)(double,double); /* pointer to a function of 2 vars */

typedef struct {
  char *rhs_string;
  char *bc_string;
  DFP f; /* rhs for u */
  DFP g; /* rhs for v */
  DFP bc1;/* bc for u */
  DFP bc2;/* bc for v */
  double eta; /* the viscosity */

  /*  int DFs;   Number of Degrees of Freedom in the Problem */
}AppEquations;


/*------------------------------------------------------------------
      comm   - MPI communictor where grid etc are stored
      aodata - grid database
*/
typedef struct {
  double dt;
  MPI_Comm   comm;
  AOData     aodata;
  AppGrid    grid;
  AppAlgebra algebra;
  AppView    view;
  AppElement element;
  AppEquations equations;
} AppCtx;

/*-------------------------------------------------------------*/

extern PetscErrorCode AppCtxView(PetscDraw,void*);
extern PetscErrorCode AppCtxViewSolution(PetscDraw,void*);
extern PetscErrorCode AppCtxCreate(MPI_Comm,AppCtx **);
extern PetscErrorCode AppCtxDestroy(AppCtx *);
extern PetscErrorCode AppCtxSetLocal(AppCtx *);
extern PetscErrorCode AppCtxSolve(AppCtx*);
extern PetscErrorCode AppCtxGraphics(AppCtx *);
extern PetscErrorCode AppCtxViewMatlab(AppCtx*);


double pde_f(double,double); 
double pde_g(double,double); 
double pde_bc1(double,double); 
double pde_bc2(double,double); 
double pde_soln(double,double);

PetscErrorCode AppCtxSetRhs(AppCtx*);
PetscErrorCode AppCtxCreateVector(AppCtx*);
PetscErrorCode AppCtxSetMatrix(AppCtx*);
PetscErrorCode AppCtxCreateMatrix(AppCtx*);
PetscErrorCode FormStationaryFunction(SNES,Vec,Vec,void *);
PetscErrorCode FormStationaryJacobian(SNES,Vec,Mat *,Mat *,MatStructure *,void *);
PetscErrorCode FormDynamicFunction(SNES,Vec,Vec,void *);
PetscErrorCode FormDynamicJacobian(SNES,Vec,Mat *,Mat *,MatStructure *,void *);

PetscErrorCode SetNonlinearFunction(Vec,AppCtx *,Vec);

extern PetscErrorCode AppCtxSetReferenceElement(AppCtx*);
extern PetscErrorCode AppCtxSetFunctions(AppCtx*);
extern PetscErrorCode SetLocalElement(AppElement *,double *);
extern PetscErrorCode ComputeRHS(DFP,DFP,AppElement *,double *);
extern PetscErrorCode ComputeMatrix(AppElement *,double *);
extern PetscErrorCode ComputeNonlinear(AppElement *,double *,double*);
extern PetscErrorCode ComputeJacobian(AppElement *,double *,double *);

#define NSTEPS 4
#define DF 2

#endif


