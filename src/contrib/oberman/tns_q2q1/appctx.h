
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
  /* number of dfs actually partitioned onto this processor*/
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

  /* time stepping stuff */
  IS df_v;

  /* data analysis  stuff */
  IS df_v1, df_v2;


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
  Vec b; /* rhs */
  Vec g; /* initial guess */
  Vec f; /* for the nonlinear function*/
  Vec conv, convl, convll; /* for the convection terms (in explicit convection calc) */

  Vec vortictiy; /* for the vorticity */
  Vec v1,v2,v1a,v1b,v2a,v2b; /* for the components of v1,v2 */
Vec v1_local, v2_local;
  Vec soln, soln1, soln2;
/*   Vec *solnv;  */
/* array of solution vectors at each time step */

  VecScatter dfvtov1, dfvtov2; /* the scatter from the global solution vector, to the global velocity components*/
  VecScatter dfv1gtol, dfv2gtol; /* the scatter from the global velocity components to the local ones*/


  Vec f_local; /* local values of nonlinear function */
  Vec f_vinlet, f_voutlet, f_poutlet, f_wall, f_pinlet, f_ywall; /* for local boundary values */
 
  VecScatter dfgtol;   /* the scatter for degrees of freedom  vectors */
  VecScatter gtol_vinlet, gtol_voutlet, gtol_wall, gtol_pinlet, gtol_poutlet, gtol_ywall;   /* the scatter for degrees of freedom  vectors on the boundary */
 Vec dtvec;  /* for jacobian in dynamic */
  Mat A,J,M;
} AppAlgebra;



typedef struct { 
  int monitor;
  int show_matrix;
  int show_vector;
  int matlabgraphics;
  int show_is;
  int show_ao;
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
 double result[2*9];
  double rhsresult[2*9];
  double presult[4][2*9];
  double tpresult[2*9][4];
  double vstiff[9][9];
  double vresult[2*9][2*9];
  double nlresult[2*9];
  double vmass[9][9];
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
  int vin_flag, vout_flag, pin_flag, pout_flag, wall_flag, ywall_flag;
  int dirichlet_flag;
  double penalty; /* penalty parameter */
  int penalty_flag;

  int nopartition_flag; /* mess with aodatapartition */

  int shear_flag;
  int cylinder_flag;
  int parabolic_flag;
  int stokes_flag;
  int convection_flag;
  int preconconv_flag;
  int cavity_flag;
  double xval, yval;
  int inner_steps;

  /* how many iterations to go before plotting */
  int Nplot;

  /* timestepping stuff */
  double dt;
  int Nsteps;
  double initial_time; 
  double final_time;
  double current_time;
  double amp;
  double frequency;
  double offset;

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
extern int AppCtxSetMassMatrix(AppCtx* appctx);
extern int SetNonlinearFunction(Vec x, AppCtx *appctx, Vec f);
extern int SetJacobian(Vec g, AppCtx *appctx, Mat* jac);
extern int ComputePressure( AppElement *phi );
extern  int TransposeValues( AppElement *phi );
extern int MonitorFunction(SNES snes, int its, double norm, void *mctx);
extern int SpreadValues( AppElement *phi );
extern int SpreadMassValues(AppElement *phi);
extern int SetReferenceElement(AppCtx*);
extern int SetLocalElement(AppElement *phi, double *coords);
extern int SetQuadrature(AppElement *element);
extern int AppCtxSetFunctions(AppCtx*);
extern int SetBoundaryConditions(Vec g, AppCtx *appctx, Vec f);
extern int ComputePartialDy(AppElement *phi );
extern int ComputePartialDx(AppElement *phi );

extern int ComputeRHS( AppElement *phi );
extern int ComputeStiffness( AppElement *phi );
extern int ComputeNonlinear(AppElement *phi );
extern int ComputeJacobian(AppElement *phi, double *uvvals, double *result);
extern int ComputeMass(AppElement *phi);
#endif


