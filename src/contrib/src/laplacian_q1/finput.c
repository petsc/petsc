
static char help[] ="Allows generation of a 2d quadrilateral grid.\n  Command line parameters n m indicate grid dimensions\n.  input is xintervals, y intervals, x min, x max, y min y max.\n";

#include <stdlib.h>
#include "ao.h"
#include "draw.h"
#include <string.h>

/*
    n_quads        - number of quadralaterials
    max_quads      - maximum space allocated for quads
    quad_vertices  - quad vertices; quads[0], quads[1], quads[2], quads[3] is first
    quad_edges     - edges of the quads
    quad_quads     - neighbors of quad
    n_vertices     - number of vertices
    max_vertices   - maximum space allocated for vertices
    x,y            - vertex coordinates

    xmin,ymin,xmax,ymax - bounding box of grid

    n_edges        - total edges in the grid
    edge_vertices  - vertices of all edges 
    max_edges      - maximum space allocated for edges
    edge_quads     - two neighbors who share edge

    vertex_boundary - indicates for each vertex if it is a boundary

*/

typedef struct {
   int    n_quads, n_vertices, n_edges;
   int    max_quads, max_vertices, max_edges;
   int    *quad_vertices,*quad_edges,*quad_quads;
   double *vertices;
   double xmin,xmax,ymin,ymax;
   int    *edge_vertices,*edge_quads;
   BT     vertex_boundary;
   Draw   draw,popup;
} AGrid;

int xintervals, yintervals;
double     xmin, xmax, ymin, ymax;
extern int AddNodeToList(AGrid *, double, double, int *);
extern int InputGrid(AGrid *);
extern int FlipQuads(AGrid *);
extern int ComputeNeighbors(AGrid *);
extern int ComputeVertexBoundary(AGrid *);
extern int ShowNumbering(AGrid *);
extern int AGridDestroy(AGrid *);

int main( int argc, char **argv )
{
  int        size, ierr;
  AGrid      agrid;
  int        flag, four = 4, *keys,nmax,i;
  int        geo[4] = {100,0,600,400};        /* size and coordinates of window */
  AOData     aodata;
  Viewer     binary;
  
  xintervals = atoi(argv[1]);
  yintervals = atoi(argv[2]);
  xmin = atof(argv[3]); xmax = atof(argv[4]);
  ymin = atof(argv[5]); ymax = atof(argv[6]);

  /* ---------------------------------------------------------------------
     Initialize PETSc
     ------------------------------------------------------------------------*/

  PetscInitialize(&argc,&argv,(char *)0,help);
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  if (size > 1) {
    SETERRQ(1,1,"Must run input program with exactly one processor");
  }

  /*
    Get user to input the quads 
  */
  ierr = InputGrid(&agrid); CHKERRA(ierr);

  /* 
     Flip vertices in quads to make sure they are all clockwise
  */
  ierr = FlipQuads(&agrid); CHKERRA(ierr);
  
  /*
     Generate edge and neighor information
  */
  ierr = ComputeNeighbors(&agrid); CHKERRA(ierr);

  ierr = ComputeVertexBoundary(&agrid); CHKERRA(ierr);

  /*
     Show the numbering of the vertices, quads and edges
  */
 /*  ierr = ShowNumbering(&agrid); CHKERRA(ierr); */

/*   ierr = DrawPause(agrid.draw); CHKERRA(ierr); */

  /*
      Create the database 
  */
  nmax = PetscMax(agrid.n_quads,agrid.n_vertices);
  nmax = PetscMax(nmax,agrid.n_edges);
  keys = (int*) PetscMalloc(nmax*sizeof(int));CHKPTRA(keys);
  for ( i=0; i<nmax; i++ ) {
    keys[i] = i;
  }
  ierr = AODataCreateBasic(PETSC_COMM_WORLD,&aodata); CHKERRA(ierr);
    ierr = AODataKeyAdd(aodata,"cell",PETSC_DECIDE,agrid.n_quads);
      ierr = AODataSegmentAdd(aodata,"cell","cell",4,agrid.n_quads,keys,agrid.quad_quads,PETSC_INT);
             CHKERRA(ierr);
      ierr = AODataSegmentAdd(aodata,"cell","vertex",4,agrid.n_quads,keys,agrid.quad_vertices,PETSC_INT);
             CHKERRA(ierr);
      ierr = AODataSegmentAdd(aodata,"cell","edge",4,agrid.n_quads,keys,agrid.quad_edges,PETSC_INT);
             CHKERRA(ierr);
    ierr = AODataKeyAdd(aodata,"edge",PETSC_DECIDE,agrid.n_edges);
      ierr = AODataSegmentAdd(aodata,"edge","vertex",2,agrid.n_edges,keys,agrid.edge_vertices,PETSC_INT);
             CHKERRA(ierr);
      ierr = AODataSegmentAdd(aodata,"edge","cell",2,agrid.n_edges,keys,agrid.edge_quads,PETSC_INT);
             CHKERRA(ierr);
    ierr = AODataKeyAdd(aodata,"vertex",PETSC_DECIDE,agrid.n_vertices);
      ierr = AODataSegmentAdd(aodata,"vertex","values",2,agrid.n_vertices,keys,agrid.vertices,
                              PETSC_DOUBLE);CHKERRA(ierr);
      ierr = AODataSegmentAdd(aodata,"vertex","boundary",1,agrid.n_vertices,keys,agrid.vertex_boundary,
                              PETSC_LOGICAL);CHKERRA(ierr);
  PetscFree(keys);
  /*
      Save the grid database to a file
  */
  ierr = ViewerFileOpenBinary(PETSC_COMM_WORLD,"gridfile",BINARY_CREATE,&binary);CHKERRA(ierr);
  ierr = AODataView(aodata,binary); CHKERRA(ierr);
  ierr = ViewerDestroy(binary); CHKERRA(ierr);


  /*
      cleanup
  */

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
  double     *vertices,cx,cy;
  char       title[120];
extern int xintervals, yintervals;
extern double xmin, xmax, ymin, ymax;

  /*
  double xmin = 0.0, xmax = 1.0;
  double ymin = 0.0, ymax = 1.0;
  */

  double deltax = (xmax - xmin)/(xintervals);
  double deltay = (ymax - ymin)/(yintervals);

  int ycnt = (ymax - ymin)/deltay;
  int xcnt = (xmax - xmin)/deltax;

  agrid->max_quads = 1e5;
  agrid->n_quads   = 0;
  agrid->max_vertices    = 1e5;
  agrid->n_vertices      = 0;
  agrid->xmin      = PETSC_MAX;
  agrid->xmax      = PETSC_MIN;
  agrid->ymin      = PETSC_MAX;
  agrid->ymax      = PETSC_MIN;


  /*
     Allocate large arrays to hold the nodes and quadrilateral lists 
  */
  vertices = agrid->vertices = (double *) PetscMalloc(2*agrid->max_vertices*sizeof(double)); CHKPTRQ(vertices);
  quads = agrid->quad_vertices = (int *) PetscMalloc(4*agrid->max_quads*sizeof(int)); CHKPTRQ(quads);

  for(i=0;i<xcnt;i++){
    for(j=0;j<ycnt;j++){
      cx = xmin + i*deltax; cy = ymin + j*deltay;
      ierr = AddNodeToList(agrid,cx,cy,&cn); CHKERRQ(ierr);
      quads[4*agrid->n_quads] = cn;
      cx += deltax;
      ierr = AddNodeToList(agrid,cx,cy,&cn); CHKERRQ(ierr);
       quads[4*agrid->n_quads+1] = cn;
      
      cy += deltay;
      ierr = AddNodeToList(agrid,cx,cy,&cn); CHKERRQ(ierr);
      quads[4*agrid->n_quads+2] = cn; 

      cx -= deltax;
      ierr = AddNodeToList(agrid,cx,cy,&cn); CHKERRQ(ierr);
      quads[4*agrid->n_quads+3] = cn; 

      agrid->n_quads++;
    }
  }

  PetscFunctionReturn(0);
}

/*
   Changes the node numbering for the quads to make sure they are all in 
   clockwise ordering
*/
int FlipQuads(AGrid *agrid)
{
  int    i,*quads = agrid->quad_vertices, n_quads = agrid->n_quads;
  double *vertices = agrid->vertices, sign;

  for ( i=0; i<n_quads; i++ ) {
    /*
       compute the quantity

            x0      x1    x2      x3
            y0      y1    y2      y3
     */

     sign = vertices[2*quads[4*i]]*vertices[1+2*quads[4*i+1]]   + vertices[2*quads[4*i+1]]*vertices[1+2*quads[4*i+2]] + 
            vertices[2*quads[4*i+2]]*vertices[1+2*quads[4*i+3]] + vertices[2*quads[4*i+3]]*vertices[1+2*quads[4*i]]   -
            vertices[1+2*quads[4*i]]*vertices[2*quads[4*i+1]]   - vertices[1+2*quads[4*i+1]]*vertices[2*quads[4*i+2]] -
            vertices[1+2*quads[4*i+2]]*vertices[2*quads[4*i+3]] - vertices[1+2*quads[4*i+3]]*vertices[2*quads[4*i]];

     if (sign == 0.0) {
       SETERRQ(1,1,"Bad quad");
     } else if (sign > 0) {
       int q1tmp = quads[4*i+1];
       quads[4*i+1] = quads[4*i+3];
       quads[4*i+3] = q1tmp;
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

  if (cx < agrid->xmin)      agrid->xmin = cx;
  else if (cx > agrid->xmax) agrid->xmax = cx;
  if (cy < agrid->ymin)      agrid->ymin = cy;
  else if (cy > agrid->ymax) agrid->ymax = cy;
  PetscFunctionReturn(0);
}

int ComputeNeighbors(AGrid *agrid)
{
  int  i,j,*quad_edges,*edge_quads,*edges,*quads,*neighbors,e;

  agrid->max_edges = 2*agrid->n_vertices;
  agrid->n_edges   = 0;
  edges            = agrid->edge_vertices = (int *) PetscMalloc(2*agrid->max_edges*sizeof(int)); 
                    CHKPTRA(edges);
  quad_edges       = agrid->quad_edges    = (int *) PetscMalloc(4*agrid->max_quads*sizeof(int));
                    CHKPTRA(quad_edges);
  edge_quads       = agrid->edge_quads    = (int *) PetscMalloc(2*agrid->max_edges*sizeof(int));
                    CHKPTRA(edge_quads);
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
     Show the numbering of the vertices, quads and edges
*/
int ShowNumbering(AGrid *agrid)
{
  Draw   draw = agrid->draw;
  int    i, *quads = agrid->quad_vertices, *edges = agrid->edge_vertices;
  char   str[5];
  double *vertices = agrid->vertices,xx,yy;

  /*
     Number the vertices
  */
  for ( i=0; i<agrid->n_vertices; i++ ) {
    sprintf(str,"%d",i);
    DrawString(draw,vertices[2*i],vertices[1+2*i],DRAW_BLUE,str);
  }

  /*
     Number the quads
  */
  for ( i=0; i<agrid->n_quads; i++ ) {
    sprintf(str,"%d",i);
    xx = .25*(vertices[2*quads[4*i]] + vertices[2*quads[4*i+1]] + vertices[2*quads[4*i+2]] + vertices[2*quads[4*i+3]]);
    yy = .25*(vertices[1+2*quads[4*i]] + vertices[1+2*quads[4*i+1]] + vertices[1+2*quads[4*i+2]] + vertices[1+2*quads[4*i+3]]);
    DrawString(draw,xx,yy,DRAW_GREEN,str);
  }

  /*
     Number the edges
  */
  for ( i=0; i<agrid->n_edges; i++ ) {
    sprintf(str,"%d",i);
    xx = .5*(vertices[2*edges[2*i]] + vertices[2*edges[2*i+1]]);
    yy = .5*(vertices[1+2*edges[2*i]] + vertices[1+2*edges[2*i+1]]);
    DrawString(draw,xx,yy,DRAW_VIOLET,str);
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
   PetscFree(agrid->edge_vertices);
   PetscFree(agrid->edge_quads);
   PetscFree(agrid->quad_quads);
   PetscFunctionReturn(0);
}

