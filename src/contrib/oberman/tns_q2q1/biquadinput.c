#include <stdlib.h>
#include "ao.h"
#include <string.h>
#include "draw.h"
#include "bitarray.h"

static char help[] ="Allows generation of a 2d quadrilateral grid.\n  The data generated is the degrees of freedom needed for the biquadratic-bilinear elements\n";

typedef struct {

  int *cell_vdf, *cell_pdf;
  int *cell_df;
  double *cell_vcoords, *cell_pcoords;

  double *df_coords;
  int *vert_vdf, *vert_pdf;
  int df_count;

  double *vertices;
  double xmin,  xmax,ymin,ymax;

  int xintervals, yintervals;
  int n_quads, n_vertices;
  int max_quads, max_vertices;
  int *quad_vertices;

  /* neighbour related stuff */
int max_edges, n_edges, *edge_vertices, *quad_edges;
int *edge_quads, *quad_quads;
int *quad_short_vertices;

  BT wall_vdf, inlet_vdf, outlet_pdf, inlet_pdf, wall_pdf, outlet_vdf;
  BT ywall_vdf;

  BT df_v;  /* bit array of all the velocity degrees of freedom */
 BT df_v1, df_v2;  /* bit array for each of the velocity degrees of freedom */


  int show_ao;


  int nv, np, nt;

} AGrid;

extern int AddNodeToList(AGrid *, double, double, int *);
extern int InputGrid(AGrid *);
extern int ComputeBoundary(AGrid *);
extern int ShowNumbering(AGrid *);
extern int AGridDestroy(AGrid *);
extern int GenerateNodes(AGrid *);
extern int GenerateBoundaryNodes(AGrid *);
extern int ComputeNeighbors(AGrid *);

int main( int argc, char **argv )
{
  AGrid      agrid;
  AOData     aodata;
  Viewer     binary;
  int flag,  *keys, nmax,i;
  int size, ierr;
BT junk;
  /* ---------------------------------------------------------------------
     Initialize PETSc
     ------------------------------------------------------------------------*/
  PetscInitialize(&argc,&argv,(char *)0,help);
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  if (size > 1) {
    SETERRQ(1,1,"Must run input program with exactly one processor");
  }

  /* set the sizes */
  agrid.nv = 9;
  agrid.np = 4;
  agrid.nt = 2*agrid.nv + agrid.np;

 /* Set the grid options */
 agrid.xintervals = 1;agrid.yintervals = 1;agrid.xmin = 0;
 agrid.xmax = 1;agrid.ymin = 0; agrid.ymax = 1;
  ierr = OptionsGetInt(0,"-xintervals",&agrid.xintervals,&flag);CHKERRQ(ierr);
  ierr = OptionsGetInt(0,"-yintervals",&agrid.yintervals,&flag);CHKERRQ(ierr);
  ierr = OptionsGetDouble(0,"-xmin",&agrid.xmin,&flag);CHKERRQ(ierr);
  ierr = OptionsGetDouble(0,"-xmax",&agrid.xmax,&flag);CHKERRQ(ierr);
  ierr = OptionsGetDouble(0,"-ymin",&agrid.ymin,&flag);CHKERRQ(ierr);
  ierr = OptionsGetDouble(0,"-ymax",&agrid.ymax,&flag);CHKERRQ(ierr);

  ierr = OptionsHasName(PETSC_NULL,"-show_ao",&agrid.show_ao); CHKERRQ(ierr);

  /* input the quads   */
  ierr = InputGrid(&agrid); CHKERRA(ierr);  

 /* Generate edge and neighor information*/
  ierr = ComputeNeighbors(&agrid); CHKERRA(ierr);

  /* Generate the degree of freedom and boundary information */
  ierr = GenerateNodes(&agrid); CHKERRA(ierr); 
  ierr = ComputeBoundary(&agrid); CHKERRA(ierr);

  /*  Create the database */
  nmax = PetscMax(agrid.n_quads,agrid.n_vertices);
  nmax = PetscMax(nmax,agrid.df_count);
  keys = (int*) PetscMalloc(nmax*sizeof(int));CHKPTRA(keys);
  for ( i=0; i<nmax; i++ ) {  keys[i] = i; }

  ierr = AODataCreateBasic(PETSC_COMM_WORLD,&aodata); CHKERRA(ierr);
  
/*  Cell Segments */
  ierr = AODataKeyAdd(aodata,"cell",PETSC_DECIDE,agrid.n_quads);  
 ierr = AODataSegmentAdd(aodata,"cell","df",agrid.nt,agrid.n_quads,keys,agrid.cell_df,PETSC_INT);CHKERRA(ierr);
  ierr = AODataSegmentAdd(aodata,"cell","vcoords",2*agrid.nv,agrid.n_quads,keys,agrid.cell_vcoords,PETSC_DOUBLE);CHKERRA(ierr);
  /* cell neighbors, needed for aodata partition */
 ierr = AODataSegmentAdd(aodata,"cell","cell",4,agrid.n_quads,keys,agrid.quad_quads,PETSC_INT);CHKERRA(ierr);

  /* DF Segments  */
  ierr = AODataKeyAdd(aodata,"df",PETSC_DECIDE,agrid.df_count);
  ierr = AODataSegmentAdd(aodata, "df", "coords", 2,  agrid.df_count, keys, agrid.df_coords, PETSC_DOUBLE);CHKERRA(ierr);
  ierr = AODataSegmentAdd(aodata,"df","v",1,agrid.df_count,keys,agrid.df_v,PETSC_LOGICAL);  CHKERRA(ierr); 
 ierr = AODataSegmentAdd(aodata,"df","v1",1,agrid.df_count,keys,agrid.df_v1,PETSC_LOGICAL);  CHKERRA(ierr); 
ierr = AODataSegmentAdd(aodata,"df","v2",1,agrid.df_count,keys,agrid.df_v2,PETSC_LOGICAL);  CHKERRA(ierr); 



  /* Boundary - Related Segments */
  ierr = AODataSegmentAdd(aodata,"df","vinlet",1,agrid.df_count,keys,agrid.inlet_vdf,PETSC_LOGICAL);  CHKERRA(ierr); 
  ierr = AODataSegmentAdd(aodata,"df","vwall",1,agrid.df_count,keys,agrid.wall_vdf,PETSC_LOGICAL);  CHKERRA(ierr); 
  ierr = AODataSegmentAdd(aodata,"df","voutlet",1,agrid.df_count,keys,agrid.outlet_vdf,PETSC_LOGICAL);  CHKERRA(ierr); 
   ierr = AODataSegmentAdd(aodata,"df","poutlet",1,agrid.df_count,keys,agrid.outlet_pdf,PETSC_LOGICAL);  CHKERRA(ierr);  
 ierr = AODataSegmentAdd(aodata,"df","pinlet",1,agrid.df_count,keys,agrid.inlet_pdf,PETSC_LOGICAL);  CHKERRA(ierr);  
 ierr = AODataSegmentAdd(aodata,"df","pwall",1,agrid.df_count,keys,agrid.wall_pdf,PETSC_LOGICAL);  CHKERRA(ierr);  
 ierr = AODataSegmentAdd(aodata,"df","ywall",1,agrid.df_count,keys,agrid.ywall_vdf,PETSC_LOGICAL);  CHKERRA(ierr);  

  ierr = PetscFree(keys);CHKERRQ(ierr);

  /* View */
  if (agrid.show_ao){
  ierr = AODataView(aodata, VIEWER_STDOUT_SELF );CHKERRA(ierr); }

  /*      Save the grid database to a file */
  ierr = ViewerFileOpenBinary(PETSC_COMM_WORLD,"gridfile",BINARY_CREATE,&binary);CHKERRA(ierr);
  ierr = AODataView(aodata,binary); CHKERRA(ierr);
  ierr = ViewerDestroy(binary); CHKERRA(ierr);

  /* cleanup  */
  ierr = AODataDestroy(aodata); CHKERRA(ierr);
  ierr = AGridDestroy(&agrid);CHKERRA(ierr); 
  PetscFinalize();
  PetscFunctionReturn(0);
}

/*       input the quads in a rectilinear grid       */
int InputGrid(AGrid *agrid)
{
  int        cn, i,j,ierr,*quads, *short_quads;
  double     *vertices,cx,cy;

  double deltax = (agrid->xmax - agrid->xmin)/((double)agrid->xintervals);
  double deltay = (agrid->ymax - agrid->ymin)/((double)agrid->yintervals);
 
  int ycnt = (agrid->ymax - agrid->ymin)/deltay;
  int xcnt = (agrid->xmax - agrid->xmin)/deltax;

  agrid->max_quads = 1e5;
  agrid->n_quads   = 0;
  agrid->max_vertices    = 1e5;
  agrid->n_vertices      = 0;
  
  /*     Allocate large arrays to hold the nodes and quadrilateral lists   */

 vertices = agrid->vertices = (double *) PetscMalloc(2*agrid->nv*(agrid->xintervals+2)*(agrid->yintervals+2)*sizeof(double)); CHKPTRQ(vertices);
  quads = agrid->quad_vertices = (int *) PetscMalloc(2*agrid->nv*(agrid->xintervals+2)*(agrid->yintervals+2)*sizeof(int)); CHKPTRQ(quads);
 short_quads = agrid->quad_short_vertices = (int *) PetscMalloc(agrid->nv*(agrid->xintervals+2)*(agrid->yintervals+2)*sizeof(int)); CHKPTRQ(quads);
  /* go counter-clockwise */
  for(i=0;i<xcnt;i++){
    for(j=0;j<ycnt;j++){

      cx = agrid->xmin + i*deltax; cy = agrid->ymin + j*deltay;
      ierr = AddNodeToList(agrid,cx,cy,&cn); CHKERRQ(ierr);
      quads[agrid->nv*agrid->n_quads] = cn;
      short_quads[4*agrid->n_quads] = cn;

      cx += deltax/2;
      ierr = AddNodeToList(agrid,cx,cy,&cn); CHKERRQ(ierr);
      quads[agrid->nv*agrid->n_quads+1] = cn;
      cx += deltax/2;
      ierr = AddNodeToList(agrid,cx,cy,&cn); CHKERRQ(ierr);
      quads[agrid->nv*agrid->n_quads+2] = cn;
      short_quads[4*agrid->n_quads+1] = cn;

      cy += deltay/2;
      ierr = AddNodeToList(agrid,cx,cy,&cn); CHKERRQ(ierr);
      quads[agrid->nv*agrid->n_quads+3] = cn;
      cy += deltay/2;
      ierr = AddNodeToList(agrid,cx,cy,&cn); CHKERRQ(ierr);
      quads[agrid->nv*agrid->n_quads+4] = cn;
      short_quads[4*agrid->n_quads+2] = cn;

      cx -= deltax/2;
      ierr = AddNodeToList(agrid,cx,cy,&cn); CHKERRQ(ierr);
      quads[agrid->nv*agrid->n_quads+5] = cn;
      cx -= deltax/2;
      ierr = AddNodeToList(agrid,cx,cy,&cn); CHKERRQ(ierr);
      quads[agrid->nv*agrid->n_quads+6] = cn;
      short_quads[4*agrid->n_quads+3] = cn;

      cy -= deltay/2;
      ierr = AddNodeToList(agrid,cx,cy,&cn); CHKERRQ(ierr);
      quads[agrid->nv*agrid->n_quads+7] = cn; 
      cx += deltax/2;
      ierr = AddNodeToList(agrid,cx,cy,&cn); CHKERRQ(ierr);
      quads[agrid->nv*agrid->n_quads+8] = cn; 

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
  edges  = agrid->edge_vertices = (int *) PetscMalloc((2*agrid->max_edges+1)*sizeof(int)); CHKPTRA(edges);
  quad_edges  = agrid->quad_edges  = (int *) PetscMalloc((4*agrid->max_quads+1)*sizeof(int));CHKPTRA(quad_edges);
  edge_quads= agrid->edge_quads = (int *) PetscMalloc((2*agrid->max_edges+1)*sizeof(int)); CHKPTRA(edge_quads);
  quads = agrid->quad_short_vertices;

  /*       Mark all neighbors (to start) with -1 to indicate missing neighbor  */
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


/* Generates the global listing of nodes from the cell list and vertex list */
/* Include a pressure degree of Freedom  */
/* This version is bilinear */
int GenerateNodes(AGrid *agrid)
{
  int *vert_vdf, *vert_pdf,  *cell_vdf, *cell_pdf, *cell_df;
  int i, j, k,  *vert_ptr, df_count;
  double *cell_vcoords, *cell_pcoords;

  /* vert_vdf  */
  vert_vdf = (int *) PetscMalloc( 2*agrid->n_vertices*sizeof(int) );CHKPTRQ(vert_vdf);
  vert_pdf = (int *) PetscMalloc( agrid->n_vertices*sizeof(int) );CHKPTRQ(vert_vdf);

  /* degrees of freedom per cell: agrid->nv*2 velocity  & 4 pressure*/
  cell_vdf = (int *) PetscMalloc( 2*agrid->nv*agrid->n_quads*sizeof(int) );CHKPTRQ(cell_vdf);
  cell_pdf = (int *) PetscMalloc( agrid->np*agrid->n_quads*sizeof(int) );CHKPTRQ(cell_pdf);
  cell_df = (int *) PetscMalloc( agrid->nt*agrid->n_quads*sizeof(int) );CHKPTRQ(cell_df);
 /* total coords: agrid->nv*2 = 2*agrid->nv */
  cell_vcoords = (double *)PetscMalloc( 2*agrid->nv*agrid->n_quads*sizeof(double) ); CHKPTRQ(cell_vcoords);
  cell_pcoords = (double *)PetscMalloc( 2*agrid->np*agrid->n_quads*sizeof(double) ); CHKPTRQ(cell_pcoords);

  for(i=0;i<2*agrid->n_vertices;i++) {vert_vdf[i] = -1; }
  for(i=0;i<agrid->n_vertices;i++) {vert_pdf[i] = -1; }

  df_count = 0;

  for(i=0;i<agrid->n_quads;i++){/* loop over the quads */
     vert_ptr = agrid->quad_vertices + agrid->nv*i;
     for( j=0; j<4; j++){/* j runs through 4 pairs of vertices */
       /* first case: corner. (even numbered vertices)  */
       k = 2*j; /* k is the current vertex */
       if( vert_vdf[2*vert_ptr[k]] == -1 ){ /* if this df has not been previously assigned */ 
	 vert_vdf[2*vert_ptr[k]] =  df_count;
	 cell_vdf[2*agrid->nv*i +2*k] = df_count; 
     	 df_count++;
	 /* second vdf */
	 vert_vdf[2*vert_ptr[k]+1] = df_count;
	 cell_vdf[2*agrid->nv*i +2*k +1] = df_count;
	 df_count++;
	 /* the  p df */
	 vert_pdf[vert_ptr[k]] = df_count;
	 cell_pdf[agrid->np*i+j] = df_count;
	 df_count++;
       }
       else {  /* this vertex has already got df's assigned */
	 cell_vdf[2*agrid->nv*i+2*k] = vert_vdf[2*vert_ptr[k]];
	 cell_vdf[2*agrid->nv*i+2*k+1] =  vert_vdf[2*vert_ptr[k]+1];
	 cell_pdf[agrid->np*i+j] = vert_pdf[vert_ptr[k]];
       }
       /* Now make the cell coords */
       cell_vcoords[2*agrid->nv*i + 2*k] = agrid->vertices[2*vert_ptr[k]];
       cell_vcoords[2*agrid->nv*i + 2*k+1] = agrid->vertices[2*vert_ptr[k]+1];     
       cell_pcoords[2*agrid->np*i + k] = agrid->vertices[2*vert_ptr[k]];
       cell_pcoords[2*agrid->np*i + k + 1] = agrid->vertices[2*vert_ptr[k]+1];     
       
       /*second case: edge (odd numbered vertex) no need for pressure here */
       k = 2*j + 1; /* k is the current vertex */
       if(vert_vdf[2*vert_ptr[k]] == -1 ){ /* if this df has not been previously assigned */ 
	 /* first vdf */
	 vert_vdf[2*vert_ptr[k]] =  df_count;
	 cell_vdf[2*agrid->nv*i + 2*k] = df_count; 
	 df_count++;
	 /* second  vdf */
	 vert_vdf[2*vert_ptr[k]+1] = df_count;
	 cell_vdf[2*agrid->nv*i + 2*k + 1] = df_count;
	 df_count++;
       }
       else {  /* this vertex has already got df's assigned */
	 cell_vdf[2*agrid->nv*i+2*k] = vert_vdf[2*vert_ptr[k]];	
	 cell_vdf[2*agrid->nv*i+2*k+1] =  vert_vdf[2*vert_ptr[k]+1];
       }
       /* now make the cell coords */
       cell_vcoords[2*agrid->nv*i + 2*k] = agrid->vertices[2*vert_ptr[k]];
       cell_vcoords[2*agrid->nv*i + 2*k + 1] = agrid->vertices[2*vert_ptr[k]+1];     
     }
     /* the last case: the inner vertex */
     k = 8; /* k is the current vertex */
     if(vert_vdf[2*vert_ptr[k]] == -1 ){ /* if this df has not been previously assigned */ 
	 /* first vdf */
	 vert_vdf[2*vert_ptr[k]] =  df_count;
	 cell_vdf[2*agrid->nv*i + 2*k] = df_count; 
	 df_count++;
	 /* second  vdf */
	 vert_vdf[2*vert_ptr[k]+1] = df_count;
	 cell_vdf[2*agrid->nv*i + 2*k + 1] = df_count;
	 df_count++;
       }
       else {  /* this vertex has already got df's assigned */
	 cell_vdf[2*agrid->nv*i+2*k] = vert_vdf[2*vert_ptr[k]];	
	 cell_vdf[2*agrid->nv*i+2*k+1] =  vert_vdf[2*vert_ptr[k]+1];
       }
       /* now make the cell coords */
       cell_vcoords[2*agrid->nv*i + 2*k] = agrid->vertices[2*vert_ptr[k]];
       cell_vcoords[2*agrid->nv*i + 2*k + 1] = agrid->vertices[2*vert_ptr[k]+1];    
  }

  /* now combine cell_pdf and cell_vdf to get cell_df - for the purposes of partitioning */
 for(i=0;i<agrid->n_quads;i++){
   for(j=0;j<2*agrid->nv;j++){ 
     cell_df[agrid->nt*i + j] = cell_vdf[2*agrid->nv*i + j];}
   for(j=0;j<agrid->np;j++){
     cell_df[agrid->nt*i +2*agrid->nv+ j] = cell_pdf[agrid->np*i + j];}
 }

 /* set the df_v */
 BTCreate(df_count, agrid->df_v);
 
 for(i=0;i<agrid->n_quads;i++){
    for(j=0;j<2*agrid->nv;j++){
 	 BTSet(agrid->df_v, cell_vdf[2*agrid->nv*i + j]);
    }
 }

 /* set df_v1, and df_v2 */
 /* the degrees of freedom corresponding to v1 and v2 */
 BTCreate(df_count, agrid->df_v1);
  BTCreate(df_count, agrid->df_v2);
 
 for(i=0;i<agrid->n_quads;i++){
    for(j=0;j<agrid->nv;j++){
      BTSet(agrid->df_v1, cell_vdf[2*agrid->nv*i + 2*j]);
      BTSet(agrid->df_v2, cell_vdf[2*agrid->nv*i + 2*j+1]);

    }
 }

  agrid->df_count = df_count;
  agrid->vert_vdf = vert_vdf;
  agrid->vert_pdf = vert_pdf;
  agrid->cell_vdf = cell_vdf;
  agrid->cell_pdf = cell_pdf;
  agrid->cell_df = cell_df;
  agrid->cell_vcoords = cell_vcoords;
  agrid->cell_pcoords = cell_pcoords;
  
  PetscFunctionReturn(0);
}

int ComputeBoundary(AGrid *agrid)
{

  int  i,j;
  int ierr;  
  double *df_coords;

  /* new ones */
  BTCreate(agrid->df_count, agrid->wall_vdf);
  BTCreate(agrid->df_count, agrid->inlet_vdf);
  BTCreate(agrid->df_count, agrid->outlet_vdf);
  BTCreate(agrid->df_count, agrid->outlet_pdf);
  BTCreate(agrid->df_count, agrid->inlet_pdf);
  BTCreate(agrid->df_count, agrid->wall_pdf);
  BTCreate(agrid->df_count, agrid->ywall_vdf);
  /* need to create space for the coords - inlet and outlet velocity coords, possibly pressure later */
  /* for now make global array of coords, but only fill in the coords of inlet and outlet.  */
  df_coords = (double *)PetscMalloc( 2*agrid->df_count*sizeof(double) ); CHKPTRQ(df_coords);
  for( i=0;i<2*agrid->df_count;i++ ){ df_coords[i] = -1; }

  /* Now create the bit array which determines the boundary type of a df */

	/* do the pressure separately */
  /* just do the same things, but with cell_pcoords */
  
  for ( i=0; i<agrid->n_vertices; i++ ) {/* i is the current vertex */

 /* if we are on a wall */
    if (PetscAbsDouble(agrid->vertices[2*i+1] - agrid->ymax) < 1.e-8 ||
	PetscAbsDouble(agrid->vertices[2*i+1] - agrid->ymin) < 1.e-8 ){
      BTSet(agrid->wall_vdf,  agrid->vert_vdf[2*i]);
      BTSet(agrid->wall_vdf, agrid->vert_vdf[2*i+1]);
      BTSet(agrid->ywall_vdf, agrid->vert_vdf[2*i+1]);
     /* pressure bc */
      BTSet(agrid->wall_pdf, agrid->vert_pdf[i]);

      /* put in the coords (two coords for each df, to be safe)*/
      df_coords[2*agrid->vert_vdf[2*i]] = agrid->vertices[2*i];
      df_coords[2*agrid->vert_vdf[2*i]+1] = agrid->vertices[2*i+1]; 
      
      df_coords[2*agrid->vert_vdf[2*i+1]] = agrid->vertices[2*i];
      df_coords[2*agrid->vert_vdf[2*i+1]+1] = agrid->vertices[2*i+1]; 
    }
    /* make sure the BT are disjoint */
    else{
    /* if we are on the left : */
    if (PetscAbsDouble(agrid->vertices[2*i] -  agrid->xmin) < 1.e-8){
      /*velocity bc */
      BTSet(agrid->inlet_vdf, agrid->vert_vdf[2*i]);
      BTSet(agrid->inlet_vdf, agrid->vert_vdf[2*i+1]);
      /* pressure bc */
      BTSet(agrid->inlet_pdf, agrid->vert_pdf[i]);
      /* put in the coords (two coords for each df, to be safe)*/
      df_coords[2*agrid->vert_vdf[2*i]] = agrid->vertices[2*i];
      df_coords[2*agrid->vert_vdf[2*i]+1] = agrid->vertices[2*i+1]; 
      
      df_coords[2*agrid->vert_vdf[2*i+1]] = agrid->vertices[2*i];
      df_coords[2*agrid->vert_vdf[2*i+1]+1] = agrid->vertices[2*i+1]; 
    }
    /* if we are on the right : */
    if (PetscAbsDouble(agrid->vertices[2*i] - agrid->xmax) < 1.e-8){
      BTSet(agrid->outlet_vdf, agrid->vert_vdf[2*i]);
      BTSet(agrid->outlet_vdf, agrid->vert_vdf[2*i+1]);
      /* pressure bc */
     BTSet(agrid->outlet_pdf, agrid->vert_pdf[i]);
 
      /* put in the coords (two coords for each df, to be safe)*/
      df_coords[2*agrid->vert_vdf[2*i]] = agrid->vertices[2*i];
      df_coords[2*agrid->vert_vdf[2*i]+1] = agrid->vertices[2*i+1]; 
      
      df_coords[2*agrid->vert_vdf[2*i+1]] = agrid->vertices[2*i];
      df_coords[2*agrid->vert_vdf[2*i+1]+1] = agrid->vertices[2*i+1]; 
    }
    }
  }
  /* set df coords */
  agrid->df_coords = df_coords;
  PetscFunctionReturn(0);
}

/*    Frees all the memory space allocated in AGrid*/
int AGridDestroy(AGrid *agrid)
{
   PetscFree(agrid->vertices);
   PetscFree(agrid->quad_vertices);
   PetscFree(agrid->vert_vdf);
   PetscFree(agrid->cell_vdf);
   PetscFunctionReturn(0);
}


