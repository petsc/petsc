#include <stdlib.h>
#include "ao.h"
#include <string.h>
#include "draw.h"


static char help[] ="Allows generation of a 2d quadrilateral grid.\n  The data generated is the degrees of freedom needed for the bilinear-constant elements\n";


/* Just need to change to get rid of pressure degrees of freedom */

/* input is xintervals, y intervals, x min, x max, y min y max */


/*
    n_quads        - number of quadralaterials
    max_quads      - maximum space allocated for quads
    quad_vertices  - quad vertices; quads[0], quads[1], quads[2], quads[3] is first
    quad_edges     - edges of the quads
    quad_quads     - neighbors of quad
    n_vertices     - number of vertices
    max_vertices   - maximum space allocated for vertices
    vertices            - vertex coordinates
    cell_coords      - the coordinates of the nodes of each cell
    df_coords        - coords for each df, 
                              (coords of the  vertex df came from for velocity df, 
	              -0 -0 for pressure df)
    nodes - the nodes used for the Q1 C0 elements

    xmin,ymin,xmax,ymax - bounding box of grid

    n_edges        - total edges in the grid
    edge_vertices  - vertices of all edges 
    max_edges      - maximum space allocated for edges
    edge_quads     - two neighbors who share edge
    vertex_boundary - indicates for each vertex if it is a boundary
    boundary_wall - indicates if the vertex on the boundary is a wall 
    boundary_inlet, boundary_outlet - indicates if the vertex on the boundary is open
*/

typedef struct {
  int    n_quads, n_vertices, n_edges;
  int    max_quads, max_vertices, max_edges;
  int    *quad_vertices,*quad_edges,*quad_quads;
  double *cell_coords;
  double *df_coords;
  int *vert_df, *cell_df;
  int *df_list, df_count;
  double *vertices;
  double xmin,xmax,ymin,ymax;
  int    *edge_vertices,*edge_quads;
  int xintervals, yintervals;
  int penalty_flag;

  BT vertex_boundary, boundary_wall, boundary_inlet, boundary_outlet;
  BT wall_vdf, ywall_vdf, inlet_vdf, outlet_pdf, inlet_pdf, outlet_vdf;
  BT df_v;

  int wall_n, outlet_n, inlet_n;
  double *inlet_coords;

   Draw   draw,popup;
} AGrid;

extern int AddNodeToList(AGrid *, double, double, int *);
extern int InputGrid(AGrid *);
extern int FlipQuads(AGrid *);
extern int ComputeNeighbors(AGrid *);
extern int ComputeBoundary(AGrid *);
extern int ShowNumbering(AGrid *);
extern int AGridDestroy(AGrid *);
extern int GenerateNodes(AGrid *);
extern int GenerateBoundaryNodes(AGrid *);

int main( int argc, char **argv )
{
  int        size, ierr;
  AGrid      agrid;
  int        flag, four = 4, *keys, nmax,i;
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

  ierr = OptionsGetInt(0,"-xintervals",&agrid.xintervals,&flag);CHKERRQ(ierr);
  if (!flag) agrid.xintervals = 1;
  ierr = OptionsGetInt(0,"-yintervals",&agrid.yintervals,&flag);CHKERRQ(ierr);
  if (!flag) agrid.yintervals = 1;
  ierr = OptionsGetDouble(0,"-xmin",&agrid.xmin,&flag);CHKERRQ(ierr);
  if (!flag) agrid.xmin = 0;
  ierr = OptionsGetDouble(0,"-xmax",&agrid.xmax,&flag);CHKERRQ(ierr);
  if (!flag) agrid.xmax = 1;
  ierr = OptionsGetDouble(0,"-ymin",&agrid.ymin,&flag);CHKERRQ(ierr);
  if (!flag) agrid.ymin = 0;
  ierr = OptionsGetDouble(0,"-ymax",&agrid.ymax,&flag);CHKERRQ(ierr);
  if (!flag) agrid.ymax = 1;
  
 ierr = OptionsHasName(0,"-penalty",&agrid.penalty_flag);CHKERRQ(ierr);

  /* Get user to input the quads   */
  ierr = InputGrid(&agrid); CHKERRA(ierr);  

    /*      Flip vertices in quads to make sure they are all clockwise  */
  ierr = FlipQuads(&agrid); CHKERRA(ierr); 
  
  /* Generate edge and neighor information*/
  ierr = ComputeNeighbors(&agrid); CHKERRA(ierr);
  ierr = GenerateNodes(&agrid); CHKERRA(ierr);
  ierr = ComputeBoundary(&agrid); CHKERRA(ierr);

  /*  Create the database */
  nmax = PetscMax(agrid.n_quads,agrid.n_vertices);
  nmax = PetscMax(nmax,agrid.n_edges);
  nmax = PetscMax(nmax,agrid.df_count);
  keys = (int*) PetscMalloc(nmax*sizeof(int));CHKPTRA(keys);
  for ( i=0; i<nmax; i++ ) {  keys[i] = i; }



  ierr = AODataCreateBasic(PETSC_COMM_WORLD,&aodata); CHKERRA(ierr);
  /* Original Cell Segments */
  ierr = AODataKeyAdd(aodata,"cell",PETSC_DECIDE,agrid.n_quads);
  ierr = AODataSegmentAdd(aodata,"cell","cell",4,agrid.n_quads,keys,agrid.quad_quads,PETSC_INT);CHKERRA(ierr);
   ierr = AODataSegmentAdd(aodata,"cell","vertex",4,agrid.n_quads,keys,agrid.quad_vertices,PETSC_INT);CHKERRA(ierr); 
/*   ierr = AODataSegmentAdd(aodata,"cell","edge",4,agrid.n_quads,keys,agrid.quad_edges,PETSC_INT);CHKERRA(ierr); */

  /* New Cell Segments */
  /* TRICK to get the cell_df right: subtract the penalty flag */
  ierr = AODataSegmentAdd(aodata,"cell","df",9-agrid.penalty_flag,agrid.n_quads,keys,agrid.cell_df,PETSC_INT);CHKERRA(ierr);
  ierr = AODataSegmentAdd(aodata,"cell","coords",8,agrid.n_quads,keys,agrid.cell_coords,PETSC_DOUBLE);CHKERRA(ierr);


  ierr = AODataKeyAdd(aodata,"edge",PETSC_DECIDE,agrid.n_edges);
  ierr = AODataSegmentAdd(aodata,"edge","vertex",2,agrid.n_edges,keys,agrid.edge_vertices,PETSC_INT);  CHKERRA(ierr);
  ierr = AODataSegmentAdd(aodata,"edge","cell",2,agrid.n_edges,keys,agrid.edge_quads,PETSC_INT);   CHKERRA(ierr);


  /* Original Vertex Segments */
  ierr = AODataKeyAdd(aodata,"vertex",PETSC_DECIDE,agrid.n_vertices);
  ierr = AODataSegmentAdd(aodata,"vertex","values",2,agrid.n_vertices,keys,agrid.vertices,PETSC_DOUBLE);CHKERRA(ierr);
  ierr = AODataSegmentAdd(aodata,"vertex","boundary",1,agrid.n_vertices,keys,agrid.vertex_boundary,PETSC_LOGICAL);  CHKERRA(ierr); 

  /* Original Edge Segments */
  /* Not Used */
/*   ierr = AODataKeyAdd(aodata,"edge",PETSC_DECIDE,agrid.n_edges); */
/*   ierr = AODataSegmentAdd(aodata,"edge","vertex",2,agrid.n_edges,keys,agrid.edge_vertices,PETSC_INT);  CHKERRA(ierr); */
/*   ierr = AODataSegmentAdd(aodata,"edge","cell",2,agrid.n_edges,keys,agrid.edge_quads,PETSC_INT);   CHKERRA(ierr); */


  /* New Df Key - needed for creating scatter, gtol */
  ierr = AODataKeyAdd(aodata,"df",PETSC_DECIDE,agrid.df_count);
  ierr = AODataSegmentAdd(aodata, "df", "coords", 2, agrid.df_count, keys, agrid.df_coords, PETSC_DOUBLE);CHKERRA(ierr);
  ierr = AODataSegmentAdd(aodata,"df","v",1,agrid.df_count,keys,agrid.df_v,PETSC_LOGICAL);  CHKERRA(ierr); 

  /* Boundary - Related Segments */

  ierr = AODataSegmentAdd(aodata,"df","vinlet",1,agrid.df_count,keys,agrid.inlet_vdf,PETSC_LOGICAL);  CHKERRA(ierr); 
  ierr = AODataSegmentAdd(aodata,"df","vwall",1,agrid.df_count,keys,agrid.wall_vdf,PETSC_LOGICAL);  CHKERRA(ierr); 
  ierr = AODataSegmentAdd(aodata,"df","ywall",1,agrid.df_count,keys,agrid.ywall_vdf,PETSC_LOGICAL);  CHKERRA(ierr); 
   ierr = AODataSegmentAdd(aodata,"df","voutlet",1,agrid.df_count,keys,agrid.outlet_vdf,PETSC_LOGICAL);  CHKERRA(ierr); 
  ierr = AODataSegmentAdd(aodata,"df","poutlet",1,agrid.df_count,keys,agrid.outlet_pdf,PETSC_LOGICAL);  CHKERRA(ierr); 
ierr = AODataSegmentAdd(aodata,"df","pinlet",1,agrid.df_count,keys,agrid.inlet_pdf,PETSC_LOGICAL);  CHKERRA(ierr); 
  /* New Vertex Keys, vertex_df, and boundary keys */
  ierr = AODataSegmentAdd(aodata,"vertex","df",2,agrid.n_vertices,keys,agrid.vert_df,PETSC_INT);  CHKERRA(ierr); 
/*   ierr = AODataSegmentAdd(aodata,"vertex","boundary",1,agrid.n_vertices,keys,agrid.vertex_boundary,PETSC_LOGICAL);  CHKERRA(ierr); */
/*   ierr = AODataSegmentAdd(aodata,"vertex","boundary_wall",1,agrid.n_vertices,keys,agrid.boundary_wall,PETSC_LOGICAL);  CHKERRA(ierr); */
/*   ierr = AODataSegmentAdd(aodata,"vertex","boundary_inlet",1,agrid.n_vertices,keys,agrid.boundary_inlet,PETSC_LOGICAL);  CHKERRA(ierr); */
/*   ierr = AODataSegmentAdd(aodata,"vertex","boundary_outlet",1,agrid.n_vertices,keys,agrid.boundary_outlet,PETSC_LOGICAL);  CHKERRA(ierr); */

  ierr = PetscFree(keys);CHKERRQ(ierr);

  /* View */
  if(0){
    ierr = AODataView(aodata, VIEWER_STDOUT_SELF );CHKERRA(ierr); }

  /*      Save the grid database to a file */
  ierr = ViewerFileOpenBinary(PETSC_COMM_WORLD,"gridfile",BINARY_CREATE,&binary);CHKERRA(ierr);
  ierr = AODataView(aodata,binary); CHKERRA(ierr);
  ierr = ViewerDestroy(binary); CHKERRA(ierr);

  /* cleanup  */
  ierr = AODataDestroy(aodata); CHKERRA(ierr);
  ierr = AGridDestroy(&agrid);CHKERRA(ierr); 
  PetscFinalize();
  return 0;
}

/*       input the quads in a rectilinear grid       */
int InputGrid(AGrid *agrid)
{
  int        cn, i,j,ierr,*quads;
  double     *vertices,cx,cy;
  char       title[120];
 
  double deltax = (agrid->xmax - agrid->xmin)/(agrid->xintervals);
  double deltay = (agrid->ymax - agrid->ymin)/(agrid->yintervals);
 
  int ycnt = (agrid->ymax - agrid->ymin)/deltay;
  int xcnt = (agrid->xmax - agrid->xmin)/deltax;

  agrid->max_quads = 1e5;
  agrid->n_quads   = 0;
  agrid->max_vertices    = 1e5;
  agrid->n_vertices      = 0;
  
  /*
     Allocate large arrays to hold the nodes and quadrilateral lists 
  */

 vertices = agrid->vertices = (double *) PetscMalloc(2*(agrid->xintervals+2)*(agrid->yintervals+2)*sizeof(double)); CHKPTRQ(vertices);
  quads = agrid->quad_vertices = (int *) PetscMalloc(4*(agrid->xintervals+2)*(agrid->yintervals+2)*sizeof(int)); CHKPTRQ(quads);

  /* go clockwise instead of counter-clockwise */
  for(i=0;i<xcnt;i++){
    for(j=0;j<ycnt;j++){
      cx = agrid->xmin + i*deltax; cy = agrid->ymin + j*deltay;
      ierr = AddNodeToList(agrid,cx,cy,&cn); CHKERRQ(ierr);
      quads[4*agrid->n_quads] = cn;
      cy += deltay;
      ierr = AddNodeToList(agrid,cx,cy,&cn); CHKERRQ(ierr);
       quads[4*agrid->n_quads+1] = cn;
      cx += deltax;
      ierr = AddNodeToList(agrid,cx,cy,&cn); CHKERRQ(ierr);
      quads[4*agrid->n_quads+2] = cn;
      cy -= deltay;
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
  PetscFunctionReturn(0);
}

int ComputeNeighbors(AGrid *agrid)
{
  int  i,j,*quad_edges,*edge_quads,*edges,*quads,*neighbors,e;

  agrid->max_edges = 2*agrid->n_vertices;
  agrid->n_edges   = 0;
  edges            = agrid->edge_vertices = (int *) PetscMalloc((2*agrid->max_edges+1)*sizeof(int)); 
                    CHKPTRA(edges);
  quad_edges       = agrid->quad_edges    = (int *) PetscMalloc((4*agrid->max_quads+1)*sizeof(int));
                    CHKPTRA(quad_edges);
  edge_quads       = agrid->edge_quads    = (int *) PetscMalloc((2*agrid->max_edges+1)*sizeof(int));
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

int ComputeBoundary(AGrid *agrid)
{
  int  i,j,*vertex_inclusion_count, *vert_ptr;
 
  /*      allocate bitarray for boundary info   */
  /* old guy */
  BTCreate(agrid->n_vertices,agrid->vertex_boundary);
  /* new ones */
  BTCreate(agrid->df_count,agrid->wall_vdf);
  BTCreate(agrid->df_count,agrid->ywall_vdf);
  BTCreate(agrid->df_count,agrid->inlet_vdf);
  BTCreate(agrid->df_count,agrid->outlet_vdf);
  BTCreate(agrid->df_count,agrid->outlet_pdf);
  BTCreate(agrid->df_count,agrid->inlet_pdf);
  /* could eliminate this stuff, since really we are only concerned with the type of boundary, which we determine another way. But this is convenient so that we can use this generator for e.g. laplace problem.  I.e reverse compatibilty.  */
  /*  vertex_inclusion_count contains number of cells that contain the given vertex   */
  vertex_inclusion_count = (int *) PetscMalloc(agrid->n_vertices*sizeof(int));CHKPTRQ(vertex_inclusion_count);
  PetscMemzero(vertex_inclusion_count,agrid->n_vertices*sizeof(int));

  /* Figure out vertex_inclusion_count. loop over quads, figure out which vertices are in less than 4 quads */
  for ( i=0; i<agrid->n_quads; i++ ) {
    for ( j=0; j<4; j++ ) {vertex_inclusion_count[agrid->quad_vertices[4*i+j]]++; }
  }

  /* Now create the bit array which determines the boundary type of a df */
  /* there is some ambiguity on the corner vertices */
  for ( i=0; i<agrid->n_quads; i++ ) {
    vert_ptr = agrid->quad_vertices + 4*i;
    for(j=0;j<4;j++){
      if (vertex_inclusion_count[vert_ptr[j]] < 4){ 
	BTSet(agrid->vertex_boundary,vert_ptr[j]);
	if (PetscAbsDouble(agrid->vertices[2*vert_ptr[j]] -  agrid->xmin) < 1.e-8){
	  BTSet(agrid->inlet_vdf, agrid->vert_df[2*vert_ptr[j]]);
	  BTSet(agrid->inlet_vdf, agrid->vert_df[2*vert_ptr[j]+1]);
	  if( !agrid->penalty_flag ){
	    BTSet(agrid->inlet_pdf, agrid->cell_df[9*i+8]);
	  }
	}
	if (PetscAbsDouble(agrid->vertices[2*vert_ptr[j]] - agrid->xmax) < 1.e-8){
	  if( !agrid->penalty_flag ){
	  BTSet(agrid->outlet_pdf, agrid->cell_df[9*i + 8]);
	  }
	  BTSet(agrid->outlet_vdf, agrid->vert_df[2*vert_ptr[j]]);
	  BTSet(agrid->outlet_vdf, agrid->vert_df[2*vert_ptr[j]+1]);
	}
	if (PetscAbsDouble(agrid->vertices[2*vert_ptr[j]+1] - agrid->ymax) < 1.e-8 ||
	     PetscAbsDouble(agrid->vertices[2*vert_ptr[j]+1] - agrid->ymin) < 1.e-8 ){
	  BTSet(agrid->wall_vdf, agrid->vert_df[2*vert_ptr[j]]);
	  BTSet(agrid->wall_vdf, agrid->vert_df[2*vert_ptr[j]+1]);
	  BTSet(agrid->ywall_vdf, agrid->vert_df[2*vert_ptr[j]+1]);


	}
      }
    }
  }
  PetscFunctionReturn(0);
}

/* Generates the global listing of nodes from the cell list and vertex list */
/* Include a pressure degree of Freedom  */
int GenerateNodes(AGrid *agrid)
{
  int *vert_df, *cell_df;
  int i, j, *vert_ptr, df_count;
  double *cell_coords, *df_coords;

  vert_df = (int *) PetscMalloc( 2*agrid->n_vertices*sizeof(int) );CHKPTRQ(vert_df);
  cell_df = (int *) PetscMalloc( 8*agrid->n_quads*sizeof(int) );CHKPTRQ(cell_df);
  cell_coords = (double *)PetscMalloc( 8*agrid->n_quads*sizeof(double) ); CHKPTRQ(cell_coords);
  df_coords = (double *)PetscMalloc( 2*(2*agrid->n_vertices + agrid->n_quads)*sizeof(double) ); CHKPTRQ(df_coords);

 BTCreate(2*agrid->n_vertices + agrid->n_quads, agrid->df_v);

  /* Figure out cell_df, df_v, df_coords */
  for(i=0;i<2*agrid->n_vertices;i++) {vert_df[i] = -1;}
  df_count = 0;

  for(i=0;i<agrid->n_quads;i++){
     vert_ptr = agrid->quad_vertices + 4*i;
     for( j=0; j<4;j++){
       if( vert_df[2*vert_ptr[j]] == -1 ){ 
	 vert_df[2*vert_ptr[j]] =  df_count;
	 cell_df[8*i + 2*j] = df_count; 
	 BTSet(agrid->df_v, df_count);
	 df_coords[2*df_count] = agrid->vertices[2*vert_ptr[j]]; /*  make df_coords  x*/
	 df_coords[2*df_count+1] = agrid->vertices[2*vert_ptr[j]+1]; /*  make df_coords y */
	 df_count++;
	 vert_df[2*vert_ptr[j]+1] = df_count;
	 cell_df[8*i + 2*j + 1] = df_count;
	 BTSet(agrid->df_v, df_count);
	 df_coords[2*df_count] = agrid->vertices[2*vert_ptr[j]]; /*  make df_coords x*/
	 df_coords[2*df_count+1] = agrid->vertices[2*vert_ptr[j]+1]; /* make df_coords y */
	 df_count++;
       }
       else {
	 cell_df[8*i + 2*j] = vert_df[2*vert_ptr[j]];
	 cell_df[8*i + 2*j + 1] =  vert_df[2*vert_ptr[j]+1];
       }
       /* Now make the cell coords */
       cell_coords[8*i + 2*j] = agrid->vertices[2*vert_ptr[j]];
       cell_coords[8*i + 2*j+1] = agrid->vertices[2*vert_ptr[j]+1];     
     }
  }

  agrid->df_count = df_count;
  agrid->vert_df = vert_df;
  agrid->cell_df = cell_df;
  agrid->cell_coords = cell_coords;
  agrid->df_coords = df_coords;
  PetscFunctionReturn(0);
}

/*    Frees all the memory space allocated in AGrid*/
int AGridDestroy(AGrid *agrid)
{
   ierr = PetscFree(agrid->vertices);CHKERRQ(ierr);
   ierr = PetscFree(agrid->quad_vertices);CHKERRQ(ierr);
   ierr = PetscFree(agrid->quad_edges);CHKERRQ(ierr);
   ierr = PetscFree(agrid->edge_vertices);CHKERRQ(ierr);
   ierr = PetscFree(agrid->edge_quads);CHKERRQ(ierr);
   ierr = PetscFree(agrid->quad_quads);CHKERRQ(ierr);
   ierr = PetscFree(agrid->vert_df);CHKERRQ(ierr);
   ierr = PetscFree(agrid->cell_df);CHKERRQ(ierr);
   PetscFunctionReturn(0);
}

