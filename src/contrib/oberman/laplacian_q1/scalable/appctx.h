
/*
       Defines data structures for writing a simple cell (element) based PDE code
    for solving scalar PDE problems like the Laplacian.
*/

#if !defined(__APPCTX_H)
#define __APPCTX_H

#include "petscao.h"           /* allows using the PETSc AOData-base routines for grid information */
#include "petscksp.h"         /* allows using PETSc linear solvers */
#include "petscpf.h"

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

/* ---------------------------------------------------------
   The AppElement data structure:
     contains information about the finite element basis functions on the 
     REFERENCE ELEMENT and then work space used to contain results in computing
     the element stiffness and element load.
*/
typedef struct {
  /* ********** same for all elements, i.e. for the reference element ********* */
  double RefVal[4][4];/* values of the reference interpolating functions at the Gauss pts */
  double RefDx[4][4];
  double RefDy[4][4];

  double weights[4];  /* quadrature weights */
 
  /* **********computed for each element while computing the stiffness ******** */

  double dx[4][4], dy[4][4];/* values of the local interpolating functions at the Gauss pts */
  double detDh[4];

  double xy[8];  /* the images of the Gauss pts in the local element */

  double rhsresult[4];  /* results of local integrations */
  double stiffnessresult[4][4];

  double *coords;  /* pointer to coords of current cell */
  PF     rhs;
} AppElement;

/*----------------------------------------------------
  AppPartition:
    contains all grid and partition information
    admits only rectangular grids
*/
typedef struct {
  MPI_Comm comm;
  int m;
  int rank;
  int size;
  int nelx;
  int nely;
  int nsdx;
  int nsdy;
  double xmin;
  double ymin;
  double delx;
  double dely;
  int local_nodes[4];
  double coords[8];
}AppPartition;

/*----------------------------------------------------
  AppCtx:
    entire application context; any data in the computation can be access 
    through this.
*/
typedef struct {
  MPI_Comm     comm;
  AppPartition part;
  AppAlgebra   algebra;  
  AppElement   element;
  PF           bc;
} AppCtx;

/*-----------------------------------------------------*/
/* function declarations */

extern int AppCtxCreate(MPI_Comm,AppCtx **);
extern int AppCtxDestroy(AppCtx *);

extern int AppCtxSolve(AppCtx*);

extern int AppCtxCreateRhsAndMatrix(AppCtx*);
extern int AppCtxSetMatrix(AppCtx*);
extern int AppCtxSetRhs(AppCtx*);
extern int SetBoundaryConditions(AppCtx*);
extern int SetMatrixBoundaryConditions(AppCtx *appctx);

extern int ComputeRHSElement( AppElement *phi );
extern int ComputeStiffnessElement( AppElement *phi );

extern int SetReferenceElement(AppCtx* appctx);
extern int SetLocalElement(AppElement *phi );

extern int AppPartitionSetUp(AppPartition*,MPI_Comm,int,int,int,int,double,double,double,double);
extern int AppPartitionGetNodes(AppPartition *part, int el, int **nodes);
extern int AppPartitionGetCoords(AppPartition *part, int el, double **coords);
extern int AppPartitionCreateLocalToGlobalMapping(AppPartition *part, ISLocalToGlobalMapping *mapping);
extern int AppPartitionGetBoundaryNodesAndCoords(AppPartition *part, int *n, int **boundary, double **coords);

#endif

















