
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
       vertex_wall_blocked - the boundary vertices blocked so that 1, 2, 4 -> 3,4, ., 6, 7, 12, 13 (i.e tripled, with third entry killed)  This is because u,v are hard at wall, p soft at wall.

        vertex_boundary_flag - bit array indicating for all on processor vertices (including ghosts) 
                               if they are on the boundary
       wall_boundary_flag, open_boundary_flag  - BA designating wall or open vertices in the pipe
        ltog                 - mapping from local numbering of vertices (including ghosts)
                               to global
       dltog                - the corresponding mapping for the DFs of the vertices
       NVs                  -the number of vertices per cell (4 in the case of billinear elements)

*/
        
typedef struct {
/* the number of cells on this processor */
  int                    cell_n;
/* The array of vertices in the local numbering for each cell */
  int                    *cell_vertex;
/* neighbors of the cell. */
  int                    *cell_cell;
/*  number of unique vertices on this processor  */
  int                    vertex_n;
/* number of vertices including ghost ones */
  int                    vertex_n_ghosted;
/* x,y coordinates of vertices on this processor */
  double                 *vertex_value;
  /* Bit array for on proc vertices if they are on the boundary */
  BT                     vertex_boundary_flag;
 BT                     inlet_boundary_flag;
BT                     outlet_boundary_flag;
 BT                     wall_boundary_flag;
/*   global number of each cell on this processor */
  IS                     cell_global;
/* global number of each vertex on this processor */
  IS                     vertex_global;
  /* The same list, expanded into blocks */
 IS                     vertex_global_blocked; 
 /* The index set of the vertices  on this proc on the boundary, including ghosted */
  IS                     vertex_boundary;
  IS                     boundary_wall;
  IS                     boundary_inlet;
 IS                     boundary_outlet;

  /* The same list, expanded into blocks */
  IS                     vertex_boundary_blocked; 
  IS                     vertex_wall_blocked;
  IS                     vertex_inlet_blocked;
  IS                     vertex_outlet_blocked;
  /* map from local numbering of vertices to the global one */
  ISLocalToGlobalMapping ltog;
  /* The same mapping, expanded into blocks */
  ISLocalToGlobalMapping dltog;  

  int NV; /* this is the number of vertices */
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
Additional structure for the discretization.
Values at the gauss points of the bilinear basis functions
*/

typedef struct {
  int dorhs;     /* flag to see if we are computing rhs */
  double Values[4][4];  /* values of reference element */
  double DxValues[4][4]; /* for reference element */
  double DyValues[4][4]; /* for reference element */
  double Centr[4];
  double DxCentr[4];
  double DyCentr[4];
  double dxcentr[4];
  double dycentr[4];
  double detDhCentr;
  double dx[16]; /* for local element */
  double dy[16]; /*for local element */
  double detDh[4]; /* determinant of map from reference element to the local element */
  double x[4];/* x coord of image of gauss point */
  double y[4];/* y coord of image of gauss point */
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

extern int ComputePressure( AppElement *phi, double *result);
extern int SetCentrElement(AppElement *phi, double coords[8]);  
extern int AppCtxSetReferenceElement(AppCtx*);
extern int AppCtxSetFunctions(AppCtx*);
extern int SetLocalElement(AppElement *phi, double *coords);
extern int ComputeRHS( DFP f, DFP g, AppElement *phi, double *integrals);
extern int ComputeMatrix( AppElement *phi, double *result);
extern int ComputeNonlinear(AppElement *phi, double uvvals[8], double result[8]);
extern int ComputeJacobian(AppElement *phi, double *uvvals, double *result);

#define NSTEPS 4
#define DF 3

#endif


