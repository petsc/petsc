
static char help[] ="Allows generation of a 2d quadrilateral grid.\n  Command line parameters n m indicate grid dimensions\n.  input is -xintervals n, -yintervals n, -xmin n, -xmax n, -ymin n -ymax n.\n";

#include <stdlib.h>
#include "ao.h"
#include <string.h>

typedef struct {

  /* stuff to go into the database */
  double  *cell_coords;
  /* quad_quads is the list of neighbout info, important for partitioning */
  int   *quad_quads;

  /* quad_vertices is a pointer to the number of each vertex */
   int    *quad_vertices;
  /* verrtices is the array of vertices */
   double *vertices;
 
  /* total number of cells and vertices */
  int celltotal, vertextotal;

  /* current count, for inputQuads */
   int    n_quads, n_vertices;

  /* edge stuff for neighbour info */
  int n_edges;
   int     max_edges;
   int   *quad_edges;
   int    *edge_vertices,*edge_quads;

  /* grid input info */
  double xmin,xmax,ymin,ymax;
  int xintervals, yintervals;

  /* boundary flags */
   BT     vertex_boundary;
  /* flags for viewing */
  int show_ao;

} AGrid;

int xintervals, yintervals;
double     xmin, xmax, ymin, ymax;
extern int AddNodeToList(AGrid *, double, double, int *);
extern int InputGrid(AGrid *);
extern int ComputeNeighbors(AGrid *);
extern int ComputeVertexBoundary(AGrid *);
extern int AGridDestroy(AGrid *);

int main( int argc, char **argv )
{
  int        size, ierr;
  AGrid      agrid;
  int        flag, four = 4, *keys,nmax,i;
  int        geo[4] = {100,0,600,400};        /* size and coordinates of window */
  AOData     aodata;
  Viewer     binary;
  
  /* ---------------------------------------------------------------------
     Initialize PETSc
     ------------------------------------------------------------------------*/

  PetscInitialize(&argc,&argv,(char *)0,help);
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  if (size > 1) {
    SETERRQ(1,1,"Must run input program with exactly one processor");
  }

 /* Set the grid options */
 agrid.xintervals = 1;agrid.yintervals = 1;
 agrid.xmin = 0; agrid.xmax = 1;
 agrid.ymin = 0; agrid.ymax = 1;
 ierr = OptionsGetInt(0,"-xintervals",&agrid.xintervals,&flag);CHKERRQ(ierr);
 ierr = OptionsGetInt(0,"-yintervals",&agrid.yintervals,&flag);CHKERRQ(ierr);
 ierr = OptionsGetDouble(0,"-xmin",&agrid.xmin,&flag);CHKERRQ(ierr);
 ierr = OptionsGetDouble(0,"-xmax",&agrid.xmax,&flag);CHKERRQ(ierr);
 ierr = OptionsGetDouble(0,"-ymin",&agrid.ymin,&flag);CHKERRQ(ierr);
 ierr = OptionsGetDouble(0,"-ymax",&agrid.ymax,&flag);CHKERRQ(ierr);
 
 ierr = OptionsHasName(PETSC_NULL,"-show_ao",&agrid.show_ao); CHKERRQ(ierr);

 /* input the quads  */
  ierr = InputGrid(&agrid); CHKERRA(ierr);

  /*     Generate edge and neighor information  */
  ierr = ComputeNeighbors(&agrid); CHKERRA(ierr);
  /* generate boundary info */
  ierr = ComputeVertexBoundary(&agrid); CHKERRA(ierr);

  /*      Create the database   */
  nmax = PetscMax(agrid.n_quads,agrid.n_vertices);
  nmax = PetscMax(nmax,agrid.n_edges);
  keys = (int*) PetscMalloc(nmax*sizeof(int));CHKPTRA(keys);
  for ( i=0; i<nmax; i++ ) {
    keys[i] = i;
  }


  ierr = AODataCreateBasic(PETSC_COMM_WORLD,&aodata); CHKERRA(ierr);

  /* really just need the cell info.  Need  vertex key to do the partitioning */
  ierr = AODataKeyAdd(aodata,"cell",PETSC_DECIDE,agrid.n_quads);
  ierr = AODataSegmentAdd(aodata,"cell","cell",4,agrid.n_quads,keys,agrid.quad_quads,PETSC_INT);  CHKERRA(ierr);
  ierr = AODataSegmentAdd(aodata,"cell","coords",8,agrid.n_quads,keys,agrid.cell_coords,PETSC_DOUBLE);  CHKERRA(ierr);
  ierr = AODataSegmentAdd(aodata,"cell","vertex",4,agrid.n_quads,keys,agrid.quad_vertices,PETSC_INT);  CHKERRA(ierr);

/* need vertex key, and vertex boundary segment */
  ierr = AODataKeyAdd(aodata,"vertex",PETSC_DECIDE,agrid.n_vertices);CHKERRA(ierr); 
  ierr = AODataSegmentAdd(aodata,"vertex","boundary",1,agrid.n_vertices,keys,agrid.vertex_boundary, PETSC_LOGICAL);CHKERRA(ierr); 

  /* need vertex values for the boundary conditions */
  ierr = AODataSegmentAdd(aodata,"vertex","coords",2,agrid.n_vertices,keys,agrid.vertices, PETSC_DOUBLE);CHKERRA(ierr);


 /* don't need the rest */
  ierr = AODataSegmentAdd(aodata,"cell","edge",4,agrid.n_quads,keys,agrid.quad_edges,PETSC_INT);
  CHKERRA(ierr);
  ierr = AODataKeyAdd(aodata,"edge",PETSC_DECIDE,agrid.n_edges);
  ierr = AODataSegmentAdd(aodata,"edge","vertex",2,agrid.n_edges,keys,agrid.edge_vertices,PETSC_INT);
  CHKERRA(ierr);
  ierr = AODataSegmentAdd(aodata,"edge","cell",2,agrid.n_edges,keys,agrid.edge_quads,PETSC_INT);
  CHKERRA(ierr);
   PetscFree(keys);
  
  /* View */
  if (agrid.show_ao){
    ierr = AODataView(aodata, VIEWER_STDOUT_SELF );CHKERRA(ierr); }
  
  /*    Save the grid database to a file  */
  ierr = ViewerFileOpenBinary(PETSC_COMM_WORLD,"gridfile",BINARY_CREATE,&binary);CHKERRA(ierr);
  ierr = AODataView(aodata,binary); CHKERRA(ierr);
  ierr = ViewerDestroy(binary); CHKERRA(ierr);

  /*      cleanup    */
  ierr = AODataDestroy(aodata); CHKERRA(ierr);
  ierr = AGridDestroy(&agrid);CHKERRA(ierr); 

  PetscFinalize();
  return 0;
}

/*
       input the quads in a rectilinear grid
*/
int InputGrid(AGrid *agrid)
{
  int        cn, i,j,ierr,*quads;
  double     *vertices, *cell_coords;
  double cx,cy;

  double deltax = (agrid->xmax - agrid->xmin)/(double)agrid->xintervals;
  double deltay = (agrid->ymax - agrid->ymin)/(double)agrid->yintervals;

  int ycnt = (agrid->ymax - agrid->ymin)/deltay;
  int xcnt = (agrid->xmax - agrid->xmin)/deltax;

  agrid->celltotal = agrid->xintervals*agrid->yintervals;
  agrid->vertextotal = (agrid->xintervals+1)*(agrid->yintervals+1);

  agrid->n_quads   = 0;
  agrid->n_vertices      = 0;


  /*     Allocate large arrays to hold the nodes and quadrilateral lists   */

  /*quads used to compute neighbours */
  vertices = agrid->vertices = (double *) PetscMalloc(2*agrid->vertextotal*sizeof(double)); CHKPTRQ(vertices);
 
 quads = agrid->quad_vertices = (int *) PetscMalloc(4*agrid->celltotal*sizeof(int)); CHKPTRQ(quads);

  cell_coords = agrid->cell_coords = (double *)PetscMalloc(2*4*agrid->celltotal*sizeof(double)); CHKPTRQ(cell_coords);

  /* each cell is oriented COUNTER-CLOCKWISE */
  for(i=0;i<xcnt;i++){
    for(j=0;j<ycnt;j++){
      cx = xmin + i*deltax; cy = ymin + j*deltay;
      ierr = AddNodeToList(agrid,cx,cy,&cn); CHKERRQ(ierr);
      quads[4*agrid->n_quads] = cn;
      cell_coords[8*agrid->n_quads] = cx;
      cell_coords[8*agrid->n_quads + 1] = cy;
      cx += deltax;
      ierr = AddNodeToList(agrid,cx,cy,&cn); CHKERRQ(ierr);
      quads[4*agrid->n_quads+1] = cn;
      cell_coords[8*agrid->n_quads + 2] = cx;
      cell_coords[8*agrid->n_quads + 2 + 1] = cy;
 
      cy += deltay;
      ierr = AddNodeToList(agrid,cx,cy,&cn); CHKERRQ(ierr);
      quads[4*agrid->n_quads+2] = cn; 
      cell_coords[8*agrid->n_quads + 4] = cx;
      cell_coords[8*agrid->n_quads + 4 + 1] = cy;

      cx -= deltax;
      ierr = AddNodeToList(agrid,cx,cy,&cn); CHKERRQ(ierr);
      quads[4*agrid->n_quads+3] = cn; 
      cell_coords[8*agrid->n_quads + 6] = cx;
      cell_coords[8*agrid->n_quads + 6 + 1] = cy;
 
      agrid->n_quads++;

    }
  }

  PetscFunctionReturn(0);
}


/*
     AddNodeToList - Maintains a list of nodes given so far
*/
int AddNodeToList(AGrid *agrid, double cx, double cy, int *cn)
{
  int i;

  for ( i=0; i<agrid->n_vertices; i++ ) {
    if ((PetscAbsDouble(agrid->vertices[2*i] - cx) < 1.e-8) && (PetscAbsDouble(agrid->vertices[1+2*i] - cy) < 1.e-8)) {
      *cn = i;
      PetscFunctionReturn(0);
    }
  }
  agrid->vertices[2*agrid->n_vertices] = cx;
  agrid->vertices[1+2*agrid->n_vertices] = cy;
  *cn     = (agrid->n_vertices)++;

  PetscFunctionReturn(0);
}

int ComputeNeighbors(AGrid *agrid)
{
  int  i,j,*quad_edges,*edge_quads,*edges,*quads,*neighbors,e;

  agrid->max_edges = 2*agrid->n_vertices;
  agrid->n_edges   = 0;
  agrid->edge_vertices = (int *) PetscMalloc(2*agrid->max_edges*sizeof(int));   CHKPTRA(agrid->edge_vertices);
  edges  = agrid->edge_vertices;
  quad_edges = agrid->quad_edges = (int *) PetscMalloc(4*agrid->celltotal*sizeof(int));CHKPTRA(quad_edges);
  edge_quads = agrid->edge_quads = (int *) PetscMalloc(2*agrid->max_edges*sizeof(int));  CHKPTRA(edge_quads);
  quads = agrid->quad_vertices;

  /*
       Mark all neighbors (to start) with -1 to indicate missing neighbor
  */
  for ( i=0; i<2*agrid->max_edges; i++ ) {
    edge_quads[i] = -1;
  }

  for ( i=0; i<agrid->n_quads; i++ ) {
    for ( j=0; j<agrid->n_edges; j++ ) {
      if (quads[4*i] == edges[2*j+1] && quads[4*i+1] == edges[2*j]) {
        quad_edges[4*i]   = j;
        edge_quads[2*j+1] = i;
        goto found0;
      }
    }
    /*
       Add a new edge to the list 
    */
    edge_quads[2*agrid->n_edges]   = i;
    edges[2*agrid->n_edges]        = quads[4*i];
    edges[2*agrid->n_edges+1]      = quads[4*i+1];
    quad_edges[4*i]                = agrid->n_edges;
    agrid->n_edges++;
    found0:;
    for ( j=0; j<agrid->n_edges; j++ ) {
      if (quads[4*i+1] == edges[2*j+1] && quads[4*i+2] == edges[2*j]) {
        quad_edges[4*i+1] = j;
        edge_quads[2*j+1] = i;
        goto found1;
      } 
    }
    /*
       Add a new edge to the list 
    */
    edge_quads[2*agrid->n_edges]   = i;
    edges[2*agrid->n_edges]        = quads[4*i+1];
    edges[2*agrid->n_edges+1]      = quads[4*i+2];
    quad_edges[4*i+1]              = agrid->n_edges;
    agrid->n_edges++;
    found1:;
    for ( j=0; j<agrid->n_edges; j++ ) {
      if (quads[4*i+2] == edges[2*j+1] && quads[4*i+3] == edges[2*j]) {
        quad_edges[4*i+2] = j;
        edge_quads[2*j+1] = i;
        goto found2;
      } 
    }
    /*
       Add a new edge to the list 
    */
    edge_quads[2*agrid->n_edges]   = i;
    edges[2*agrid->n_edges]        = quads[4*i+2];
    edges[2*agrid->n_edges+1]      = quads[4*i+3];
    quad_edges[4*i+2]              = agrid->n_edges;
    agrid->n_edges++;
    found2:;
    for ( j=0; j<agrid->n_edges; j++ ) {
      if (quads[4*i+3] == edges[2*j+1] && quads[4*i] == edges[2*j]) {
        quad_edges[4*i+3] = j;
        edge_quads[2*j+1] = i;
        goto found3;
      }
    }
    /*
       Add a new edge to the list 
    */
    edge_quads[2*agrid->n_edges]   = i;
    edges[2*agrid->n_edges]        = quads[4*i+3];
    edges[2*agrid->n_edges+1]      = quads[4*i];
    quad_edges[4*i+3]              = agrid->n_edges;
    agrid->n_edges++;
    found3:;

  }

  neighbors = agrid->quad_quads = (int *) PetscMalloc( 4*agrid->n_quads*sizeof(int) );CHKPTRQ(neighbors);
  for ( i=0; i<agrid->n_quads; i++ ) {
    for ( j=0; j<4; j++ ) {
      e = 2*agrid->quad_edges[4*i+j]; 

      /* get the edge neighbor that is not the current quad */
      if ( i == agrid->edge_quads[e] ) e++;
      neighbors[4*i+j] = agrid->edge_quads[e];
    }
  }

  PetscFunctionReturn(0);
}

int ComputeVertexBoundary(AGrid *agrid)
{
  int  i,j,*count,*quad_vertex = agrid->quad_vertices;

  /* 
      allocate bitarray for boundary info
  */
  BTCreate(agrid->n_vertices,agrid->vertex_boundary);

  /*
      count contains number of cells that contain the given vertex 
  */
  count = (int *) PetscMalloc(agrid->n_vertices*sizeof(int));CHKPTRQ(count);
  PetscMemzero(count,agrid->n_vertices*sizeof(int));

  for ( i=0; i<agrid->n_quads; i++ ) {
    for ( j=0; j<4; j++ ) {
      count[quad_vertex[4*i+j]]++;
    }
  }
  for ( i=0; i<agrid->n_vertices; i++ ) {
    if (count[i] < 4) { BTSet(agrid->vertex_boundary,i);}
  }

  PetscFunctionReturn(0);
}


/*
    Frees all the memory space allocated in AGrid
*/
int AGridDestroy(AGrid *agrid)
{
   PetscFree(agrid->vertices);
   PetscFree(agrid->quad_vertices);
   PetscFree(agrid->quad_edges);
   PetscFree(agrid->cell_coords);
   PetscFree(agrid->edge_vertices);
   PetscFree(agrid->edge_quads);
   PetscFree(agrid->quad_quads);
   PetscFunctionReturn(0);
}

