
/*
       Defines some simple data structures for writing cell (element) based PDE codes.

    Generally one would write a code by starting with the data structures below and 
    and to them and deleting from them unneeded information. 
*/
#if !defined(__APPCTX_H)
#define __APPCTX_H

#include <string.h>
#include "ao.h"
#include "math.h"
#include "sles.h"
#include "snes.h"
/*
        cell_n               - number of cells on this processor 
        cell_vertex          - vertices of the cells (in local numbering)
        cell_global          - global number of each cell on this processor
        cell_cell            - neighbors of the cell.  (-1) indicates no neighbor.
                               ordering cycles clockwise from left
        vertex_n             - number of unique vertices on this processor 
        vertex_n_ghosted     - number of vertices including ghost ones
        vertex_global        - global number of each vertex on this processor
        vertex_value         - x,y coordinates of vertices on this processor
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


/******* some of these are never used.... first line? *********/  
typedef struct {
  IS  cell_global, vertex_global, df_global; /* global numbering of cells, vertices, dfs */

  /* not needed */
  IS   isboundary_df; /* is for df on boundary */
  int  *boundary_df; /* the corresponding array  */
  IS   isvertex_boundary; /* is for vertices on boundary */
  int *vertex_boundary;

  /* is and indices for boundary dfs */
IS iswall_vdf, isywall_vdf, isinlet_vdf, isoutlet_vdf, isoutlet_pdf, isinlet_pdf;
int  *wall_vdf, *ywall_vdf, *inlet_vdf, *outlet_vdf, *outlet_pdf, *inlet_pdf;
double *df_coords; /*  x-y  coords for each df in global numbering */
int inlet_vcount, wall_vcount, ywall_vcount, outlet_vcount, outlet_pcount, inlet_pcount;
double *inlet_coords; /* coords of vertices on the inlet */ 
double *inlet_values; /* space for the boundary values  */
double *outlet_values, *outlet_coords;
double *inlet_pvalues;
  double *cell_coords;   /* coords of the vertices corresponding to each cell */ 
 /* sizes of local df (including ghosted), df on this proc, cells, vertices, ghosted vertices */
  int df_count, vertex_boundary_count, df_local_count, cell_n, vertex_n, vertex_n_ghosted; 

  int *vertex_df;   /* info on DF associated to vertices */
  int  *cell_df, *cell_vertex, *cell_cell; /* info on df, vertices, neighbours, indexed by cells */

  double                 *vertex_value; /* numerical coords of the vertices */ 
  ISLocalToGlobalMapping ltog, dfltog;  /* ltog associated with vertices, degrees of freedom */

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

  Vec *solnv; /* array of solution vectors at each time step */

  Vec f_local; /* local values of nonlinear function */
  Vec f_vinlet, f_voutlet, f_poutlet, f_wall, f_ywall, f_pinlet; /* for local boundary values */
 
  VecScatter dfgtol;   /* the scatter for degrees of freedom  vectors */
  VecScatter gtol_vinlet, gtol_voutlet, gtol_wall, gtol_ywall,  gtol_pinlet, gtol_poutlet;   /* the scatter for degrees of freedom  vectors on the boundary */

  Mat A,J;
} AppAlgebra;

/*
    drawlocal    - window where processor local portion is drawn
    drawglobal   - window where entire grid is drawn

    shownumbers  - print the vertex and cell numbers 
    showvertices - draw the vertices as points
    showelements - draw the elements 
    showboundary - draw boundary of domain
    showboundaryvertices - draw points on boundary of domain

    showsomething - flag indicating that some graphic is being used
*/

typedef struct {
  Draw       drawlocal;
  Draw       drawglobal;
  int        matlabgraphics;
  int        shownumbers;
  int        showvertices;
  int        showelements;
  int        showboundary;
  int        showboundaryvertices;
 
  int        showsomething;            
} AppView;

/* 
for j= 1:tsteps,
figure(1)
fill3(cellx,celly,cellz1(:,:,j),cellz1(:,:,j))
%
figure(2)
fill3(cellx,celly,cellz2(:,:,j),cellz2(:,:,j))
end

Additional structure for the discretization.
Values at the gauss points of the bilinear basis functions
*/

/* Good NS */
typedef struct {
  int dorhs;     /* flag to see if we are computing rhs */
  double Values[4][4];  /* values of reference element */
  double DxValues[4][4]; /* for reference element */
  double DyValues[4][4]; /* for reference element */
  double dx[16]; /* for local element */
  double dy[16]; /*for local element */
  double detDh[4]; /* determinant of map from reference element to the local element */
  double x[4];/* x coord of image of gauss point */
  double y[4];/* y coord of image of gauss point */

  /* Stuff for the Centre Element */
  double Centr[4];
  double DxCentr[4];
  double DyCentr[4];
  double dxcentr[4];
  double dycentr[4];
  double detDhCentr;
} AppElement;

 

 typedef double (*DFP)(double,double); /* pointer to a function of 2 vars */

typedef struct {
  char *rhs_string;
  char *bc_string;
  DFP f; /* rhs for u */
  DFP g; /* rhs for v */
  DFP bc1;/* bc for u */
  DFP bc2;/* bc for v */
  double eta; /* the viscosity */
  double tweak; /* mess up initial guess */
int vin_flag, vout_flag, pin_flag, pout_flag, wall_flag, ywall_flag;/* flags for the boundary conditions */
  double penalty; /* penalty parameter */
int penalty_flag;
  int DF;
  /*  int DFs;   Number of Degrees of Freedom in the Problem */
}AppEquations;


/*
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


extern int AppCtxView(Draw,void*);
extern int AppCtxViewSolution(Draw,void*);
extern int AppCtxCreate(MPI_Comm,AppCtx **);
extern int AppCtxDestroy(AppCtx *);
extern int AppCtxSetLocal(AppCtx *);
extern int AppCtxSolve(AppCtx*);
extern int AppCtxGraphics(AppCtx *);
extern int AppCtxViewMatlab(AppCtx*);


 double f(double, double); 
 double g(double, double); 
 double bc1(double, double); 
 double bc2(double, double); 
 double bc3(double, double); 

double soln(double, double);

int AppCtxSetRhs(AppCtx*);
int AppCtxCreateVector(AppCtx*);
int AppCtxSetMatrix(AppCtx*);
int AppCtxCreateMatrix(AppCtx*);
int FormStationaryFunction(SNES snes, Vec x, Vec f, void *appctx);
int FormStationaryJacobian(SNES snes, Vec x, Mat *jac, Mat *B, MatStructure *flag, void *dappctx);
int FormDynamicFunction(SNES snes, Vec x, Vec f, void *appctx);
int FormDynamicJacobian(SNES snes, Vec x, Mat *jac, Mat *B, MatStructure *flag, void *dappctx);

int SetNonlinearFunction(Vec x, AppCtx *appctx, Vec f);
int MonitorFunction(SNES snes, int its, double norm, void *mctx);
extern int SetCentrElement(AppElement *phi, double coords[8]);  
extern int ComputePressure( AppElement *phi, double *result);
extern int AppCtxSetReferenceElement(AppCtx*);
extern int AppCtxSetFunctions(AppCtx*);
extern int SetLocalElement(AppElement *phi, double *coords);
extern int ComputeRHS( DFP f, DFP g, AppElement *phi, double *integrals);
extern int ComputeMatrix( AppElement *phi, double *result);
extern int ComputeNonlinear(AppElement *phi, double *uvvals, double *result);
extern int ComputeJacobian(AppElement *phi, double *uvvals, double *result);

#define NSTEPS 4
#endif


