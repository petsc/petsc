
/*
    Defines some simple data structures for writing cell-based (element-based) PDE codes.

    Generally one would write a code by starting with the data structures below and 
    adding to them and deleting from them unneeded information. 
*/
#if !defined(__APPCTX_H)
#define __APPCTX_H

#include "ao.h"
#include "mat.h"
#include "sles.h"
#include "snes.h"

/*
  AppGrid: grid data

  cell_n               - number of local cells (elements)
  cell_vertex          - vertices of the cells (in local numbering)
  cell_global          - global number of each local cell 
  vertex_n             - number of unique local vertices
  vertex_n_ghosted     - number of vertices (including ghost vertices)
  vertex_global        - global number of each vertex on this processor (including ghosts)
  vertex_value         - x,y coordinates of vertices on this processor (including ghosts)
  vertex_boundary      - list of on processor vertices (including ghosts) that are on the boundary
  vertex_boundary_flag - bit array indicating for all on processor vertices (including ghosts) 
                         if the are on the boundary
  ltog                 - mapping from local numbering of vertices (including ghosts)
                         to global
  cell_cell            - neighbors of each cell
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
  ISLocalToGlobalMapping ltog;
} AppGrid;

/*
  AppAlgebra: linear algebra data

  gtol             - global-to-local vector scatter
                     (e.g., used to move data from z to z_local)
  A                - parallel sparse stiffness matrix
  b                - parallel vector, containing right-hand side
  x                - parallel vector, containing solution
  z                - parallel work vector
  w_local, x_local - sequential work vectors (containing local plus ghosted entries)
  z_local          
*/

typedef struct {
  Vec                    b,x,z;
  Vec                    w_local,x_local,z_local;  /* local ghosted work vectors */
  VecScatter             gtol;
  Mat                    A;
} AppAlgebra;

/*
  AppView: viewer data

  drawlocal            - window where processor local portion is drawn
  drawglobal           - window where entire grid is drawn
  shownumbers          - print the vertex and cell numbers 
  showvertices         - draw the vertices as points
  showelements         - draw the elements 
  showboundary         - draw boundary of domain
  showboundaryvertices - draw points on boundary of domain
  showsomething        - flag indicating that at least one of the flags above is true
*/

typedef struct {
  Draw       drawlocal;
  Draw       drawglobal;
  int        shownumbers;
  int        showvertices;
  int        showelements;
  int        showboundary;
  int        showboundaryvertices;
  int        showsomething;            
  int        matlabgraphics;
} AppView;

/*
   AppCtx: application context

   comm   - MPI communictor where grid etc are stored
   aodata - grid database
*/
typedef struct {
  MPI_Comm   comm;
  AOData     aodata;
  AppGrid    grid;
  AppAlgebra algebra;
  AppView    view;
} AppCtx;

/* 
   Declare application-defined routines 
*/
extern int AppCtxViewGrid(Draw,void*);
extern int AppCtxViewSolution(Draw,void*);
extern int AppCtxCreate(MPI_Comm,AppCtx **);
extern int AppCtxDestroy(AppCtx *);
extern int AppCtxSetLocal(AppCtx *);
extern int AppCtxSolve(AppCtx*);

#endif
