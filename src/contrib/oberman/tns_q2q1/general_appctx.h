
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

typedef struct {
  IS  cell_global, df_global; /* global numbering of cells, dfs */
  ISLocalToGlobalMapping  dfltog;  /* ltog for degrees of freedom */

  /* Cell-based Data */
  double *cell_vcoords, *cell_pcoords; /* coords for pressure and velocity */
  int *cell_vdf, *cell_pdf;  /* local numbers of the v,p, dfs */

  /* Sizes */
  /* the number of local cells */
  int cell_n; 
  /* (size of df_global) - all dfs of local cells(includes ghosted) */
  int df_count; 
  /* number of dfs actually partitionned onto this processor*/
  int df_local_count; 

  /* is and indices for boundary dfs */
  IS iswall_vdf, isinlet_vdf, isoutlet_vdf, isoutlet_pdf, isinlet_pdf;
  int inlet_vcount, wall_vcount, outlet_vcount, outlet_pcount, inlet_pcount;
 /* coords of boundary vertices */ 
  double *inlet_coords. *outlet_coords;
  /* space for the boundary values  */
  double *outlet_values,*inlet_values; 
  double *inlet_pvalues;

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
  Vec f_vinlet, f_voutlet, f_poutlet, f_wall, f_pinlet; /* for local boundary values */
 
  VecScatter dfgtol;   /* the scatter for degrees of freedom  vectors */
  VecScatter gtol_vinlet, gtol_voutlet, gtol_wall, gtol_pinlet, gtol_poutlet;   /* the scatter for degrees of freedom  vectors on the boundary */

  Mat A,J;
} AppAlgebra;
typedef struct { 
  int monitor;
  int show_matrix;
  int show_vector;
  int matlabgraphics;
} AppView;


typedef struct {
  /* pointer to the appropriate arrays */
  double *RefVal;
  double *RefDx, *RefDy;

  /* these are the local values */
  double *dx, *dy;
  double *detDh;
  double *x, *y;

  /* number of interpolation functions, quadrature points  */
  int vel_basis_count;
  int vel_quad_count;
  int p_basis_count;

  /* pointer to quadrature weights */
  double *vweights;
  /* weights */
  double BiquadWeights[9];

} AppElement;

typedef struct {

  double eta; /* the viscosity */
  double tweak; /* mess up initial guess */
  /* flags for the boundary conditions */
  int vin_flag, vout_flag, pin_flag, pout_flag;
  double penalty; /* penalty parameter */
  int penalty_flag;
  int cavity_flag;
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

int AppCtxCreateVector(AppCtx*);
int AppCtxCreateMatrix(AppCtx*);

int FormStationaryFunction(SNES snes, Vec x, Vec f, void *appctx);
int FormStationaryJacobian(SNES snes, Vec x, Mat *jac, Mat *B, MatStructure *flag, void *dappctx);
int FormDynamicFunction(SNES snes, Vec x, Vec f, void *appctx);
int FormDynamicJacobian(SNES snes, Vec x, Mat *jac, Mat *B, MatStructure *flag, void *dappctx);


int AppCtxSetRhs(AppCtx*);
int AppCtxSetMatrix(AppCtx*);
int SetNonlinearFunction(Vec x, AppCtx *appctx, Vec f);


int MonitorFunction(SNES snes, int its, double norm, void *mctx);

extern int AppCtxSetBiLinReferenceElement(AppCtx*);
extern int AppCtxSetBiQuadReferenceElement(AppCtx*);

extern int SetLocalBiLinElement(AppElement *phi, double *coords);
extern int SetLocalBiQuadElement(AppElement *phi, double *coords);



extern int AppCtxSetFunctions(AppCtx*);


extern int ComputeRHS( AppElement *phi, double *integrals);
extern int ComputeMatrix( AppElement *phi, double *result);
extern int ComputeNonlinear(AppElement *phi, double *uvvals, double *result);
extern int ComputeJacobian(AppElement *phi, double *uvvals, double *result);

#define NSTEPS 4
#endif


