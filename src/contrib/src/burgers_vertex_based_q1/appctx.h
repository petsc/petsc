
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
        
typedef struct {
  int                    cell_n;
  int                    *cell_vertex;
  IS                     cell_global;
  int                    *cell_cell;
  int                    vertex_n,vertex_n_ghosted;
  IS                     vertex_global;

  double                 *vertex_value;
  BT                     vertex_boundary_flag;
  IS                     vertex_boundary;
  IS                     vertex_global_blocked;  /* for 2 degrees of freedom */
  IS                     vertex_boundary_blocked; /* for 2 deg of freedom */
  ISLocalToGlobalMapping ltog;
  ISLocalToGlobalMapping dltog;    /* for 2 deg of freedom */
  int                    NVs;
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
double soln(double, double);

int AppCtxSetRhs(AppCtx*);
int AppCtxCreateVector(AppCtx*);
int AppCtxSetMatrix(AppCtx*);
int AppCtxCreateMatrix(AppCtx*);
int FormFunction(SNES snes, Vec x, Vec f, void *appctx);
int FormJacobian(SNES snes, Vec x, Mat *jac, Mat *B, MatStructure *flag, void *dappctx);
int SetNonlinearFunction(Vec x, AppCtx *appctx, Vec f);

extern int AppCtxSetReferenceElement(AppCtx*);
extern int AppCtxSetFunctions(AppCtx*);
extern int SetLocalElement(AppElement *phi, double *coords);
extern int ComputeRHS( DFP f, DFP g, AppElement *phi, double *integrals);
extern int ComputeMatrix( AppElement *phi, double *result);
extern int ComputeNonlinear(AppElement *phi, double *uvvals, double *result);
extern int ComputeJacobian(AppElement *phi, double *uvvals, double *result);

#endif
