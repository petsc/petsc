
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

  /* don't need this stuff in the grid stucture */
  IS  cell_global, df_global; /* global numbering of cells, dfs */
  /* may need this... */
  ISLocalToGlobalMapping  dfltog;  /* ltog for degrees of freedom */


  /* Cell-based Data */
  double *cell_vcoords; /* coords  */
  int *cell_df;

  /* Sizes */
  /* the number of local cells */
  int cell_n; 
  /* (size of df_global) - all dfs of local cells(includes ghosted) */
  int df_count; 
  /* number of dfs actually partitionned onto this processor*/
  int df_local_count; 

  /* Boundary Stuff */
  /* is and indices for boundary dfs */
  IS iswall_vdf, isinlet_vdf, isoutlet_vdf, isoutlet_pdf, isinlet_pdf, isywall_vdf;
  int inlet_vcount, wall_vcount, outlet_vcount, outlet_pcount, inlet_pcount, ywall_vcount;
 /* coords of boundary vertices */ 
  double *inlet_coords, *outlet_coords;
  double *wall_coords;
  /* space for the boundary values  */
  double *outlet_values,*inlet_values; 
  double *wall_values;
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
  Vec f_vinlet, f_voutlet, f_poutlet, f_wall, f_pinlet, f_ywall; /* for local boundary values */
 
  VecScatter dfgtol;   /* the scatter for degrees of freedom  vectors */
  VecScatter gtol_vinlet, gtol_voutlet, gtol_wall, gtol_pinlet, gtol_poutlet, gtol_ywall;   /* the scatter for degrees of freedom  vectors on the boundary */

  Mat A,J;
} AppAlgebra;

typedef struct { 
  PetscTruth monitor;
  PetscTruth show_matrix;
  PetscTruth show_vector;
  PetscTruth matlabgraphics;
  PetscTruth show_griddata;
} AppView;


typedef struct {
  /* pointer to the appropriate arrays */
  double RefVal[9][9];
  double RefDx[9][9], RefDy[9][9];
 /* pressure RefVal */
  double PRefVal[4][9];
  /* these are the local values */
  double dx[9][9], dy[9][9];
  double detDh[9];
  double x[9], y[9];

  /* number of interpolation functions, quadrature points  */
  int vel_basis_count;
  int vel_quad_count;
  int p_basis_count;
  int p_quad_count;
  int df_element_count;
  int dim;

  /* results of local integrations */
  double rhsresult[2*9];
  double presult[4][2*9];
  double tpresult[2*9][4];
  double vstiff[9][9];
  double vresult[2*9][2*9];
  double nlresult[2*9];
 
  /* cell_values */
  double u[9], v[9];

  /* pointer to quadrature weights */
  double *vweights;
  /* weights */
  double BiquadWeights[9];

} AppElement;

typedef struct {

  double eta; /* the viscosity */
  double tweak; /* mess up initial guess */
  /* flags for the boundary conditions */
  PetscTruth vin_flag, vout_flag, pin_flag, pout_flag, wall_flag, ywall_flag;
  PetscTruth dirichlet_flag;
  double penalty; /* penalty parameter */
  PetscTruth penalty_flag;

  PetscTruth nopartition_flag; /* mess with aodatapartition */

  PetscTruth cylinder_flag;
  PetscTruth parabolic_flag;
  PetscTruth stokes_flag;
  PetscTruth cavity_flag;
  double xval, yval;
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


extern int AppCtxCreate(MPI_Comm,AppCtx **);
extern int AppCtxDestroy(AppCtx *);
extern int AppCtxSetLocal(AppCtx *);
extern int AppCtxSolve(AppCtx*);
extern int AppCtxViewMatlab(AppCtx*);

extern  double f(double, double); 
extern  double g(double, double); 
extern  double bc1(AppEquations*); 
extern  double bc2(AppEquations*); 
extern  double bc3(AppEquations*); 

extern int AppCtxCreateVector(AppCtx*);
extern int AppCtxCreateMatrix(AppCtx*);

extern int FormStationaryFunction(SNES snes, Vec x, Vec f, void *appctx);
extern int FormStationaryJacobian(SNES snes, Vec x, Mat *jac, Mat *B, MatStructure *flag, void *dappctx);
extern int FormDynamicFunction(SNES snes, Vec x, Vec f, void *appctx);
extern int FormDynamicJacobian(SNES snes, Vec x, Mat *jac, Mat *B, MatStructure *flag, void *dappctx);
extern int SetJacobianBoundaryConditions(AppCtx *appctx, Mat* jac);
extern int FormInitialGuess(AppCtx* appctx);
extern int AppCtxSetRhs(AppCtx*);
extern int AppCtxSetMatrix(AppCtx*);
extern int SetNonlinearFunction(Vec x, AppCtx *appctx, Vec f);
extern int SetJacobian(Vec g, AppCtx *appctx, Mat* jac);
extern int ComputePressure( AppElement *phi );
extern  int TransposeValues( AppElement *phi );
extern int MonitorFunction(SNES snes, int its, double norm, void *mctx);
extern int SpreadValues( AppElement *phi );
extern int SetReferenceElement(AppCtx*);
extern int SetLocalElement(AppElement *phi, double *coords);
extern int SetQuadrature(AppElement *element);
extern int AppCtxSetFunctions(AppCtx*);
extern int SetBoundaryConditions(Vec g, AppCtx *appctx, Vec f);
extern int ComputeRHS( AppElement *phi );
extern int ComputeStiffness( AppElement *phi );
extern int ComputeNonlinear(AppElement *phi );
extern int ComputeJacobian(AppElement *phi, double *uvvals, double *result);

#define NSTEPS 4
#endif


