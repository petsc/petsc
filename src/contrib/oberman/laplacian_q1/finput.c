
static char help[] ="Generation of a rectangular 2d quadrilateral grid.\n\
  Command line parameters -m m -n n indicate number grid lines\n\
  -xmin xmin, -xmax xmax, -ymin ymin -ymax ymax.\n";

#include "petscao.h"
#include "src/dm/ao/aoimpl.h"  /* need direct access to AOData2dGrid structure */
#include "petscdraw.h"

double xmin,xmax,ymin,ymax;
int    xintervals,yintervals; 
extern int InputGrid(AOData2dGrid);

int main( int argc, char **argv )
{
  int          size, ierr;
  AOData2dGrid agrid;
  AOData       aodata;
  Viewer       binary;
  Draw         draw;
  /* ---------------------------------------------------------------------
     Initialize PETSc
     ------------------------------------------------------------------------*/

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);
  if (size > 1) {
    SETERRQ(1,1,"Must run input program with exactly one processor");
  }
  xmin = 0.0; xmax = 1.0;
  ymin = 0.0; ymax = 1.0;
  ierr = OptionsGetDouble(0,"-xmin",&xmin,0);CHKERRA(ierr);
  ierr = OptionsGetDouble(0,"-xmax",&xmax,0);CHKERRA(ierr);
  ierr = OptionsGetDouble(0,"-ymin",&ymin,0);CHKERRA(ierr);
  ierr = OptionsGetDouble(0,"-ymax",&ymax,0);CHKERRA(ierr);
  xintervals = 5; yintervals = 5;
  ierr = OptionsGetInt(0,"-xintervals",&xintervals ,0);CHKERRA(ierr);
  ierr = OptionsGetInt(0,"-yintervals",&yintervals,0);CHKERRA(ierr);


 /*---------------------------------------------------------------------
     Open the graphics window
     ------------------------------------------------------------------------*/
  ierr = DrawOpenX(PETSC_COMM_WORLD,PETSC_NULL,"Grid",PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,&draw);CHKERRQ(ierr);

  ierr = AOData2dGridCreate(&agrid);CHKERRA(ierr);

  /*
    Create grid
  */
  ierr = InputGrid(agrid);CHKERRA(ierr);

  /* 
     Flip vertex in cell to make sure they are all clockwise
  */
  ierr = AOData2dGridFlipCells(agrid);CHKERRA(ierr);
  
  /*
     Generate edge and neighor information
  */
  ierr = AOData2dGridComputeNeighbors(agrid);CHKERRA(ierr);

  ierr = AOData2dGridComputeVertexBoundary(agrid);CHKERRA(ierr);

  /*
     Show the numbering of the vertex, cell and edge
  */
  ierr = AOData2dGridDraw(agrid,draw);CHKERRA(ierr);

  ierr = DrawPause(draw);CHKERRA(ierr);

  /*
      Create the database 
  */
  ierr = AOData2dGridToAOData(agrid,&aodata);CHKERRA(ierr);

  /*
      Save the grid database to a file
  */
  ierr = ViewerBinaryOpen(PETSC_COMM_WORLD,"gridfile",BINARY_CREATE,&binary);CHKERRA(ierr);
  ierr = AODataView(aodata,binary);CHKERRA(ierr);
  ierr = ViewerDestroy(binary);CHKERRA(ierr);

  /*
     Close the graphics window and cleanup
  */
  ierr = DrawDestroy(draw);CHKERRA(ierr);

  ierr = AODataDestroy(aodata);CHKERRA(ierr);

  ierr = AOData2dGridDestroy(agrid);CHKERRA(ierr); 

  PetscFinalize();

  return 0;
}

#undef __FUNC__
#define __FUNC__ "InputGrid"
/*
       input the cells in a rectilinear grid
*/
int InputGrid(AOData2dGrid agrid)
{
  int        cn, i,j,ierr,*cells;
  double     *vertex,cx,cy;
  char       title[120];

  PetscFunctionBegin;
  /*
  double xmin = 0.0, xmax = 1.0;
  double ymin = 0.0, ymax = 1.0;
  */

  double deltax = (xmax - xmin)/(xintervals);
  double deltay = (ymax - ymin)/(yintervals);

  int ycnt = (ymax - ymin)/deltay;
  int xcnt = (xmax - xmin)/deltax;

  agrid->cell_max = 1e5;
  agrid->cell_n   = 0;
  agrid->vertex_max    = 1e5;
  agrid->vertex_n      = 0;
  agrid->xmin      = PETSC_MAX;
  agrid->xmax      = PETSC_MIN;
  agrid->ymin      = PETSC_MAX;
  agrid->ymax      = PETSC_MIN;


  /*
     Allocate large arrays to hold the nodes and cellrilateral lists 
  */
  vertex = agrid->vertex = (double *) PetscMalloc(2*agrid->vertex_max*sizeof(double)); CHKPTRQ(vertex);
  cells = agrid->cell_vertex = (int *) PetscMalloc(4*agrid->cell_max*sizeof(int)); CHKPTRQ(cells);

  for(i=0;i<xcnt;i++){
    for(j=0;j<ycnt;j++){
      cx = xmin + i*deltax; cy = ymin + j*deltay;
      ierr = AOData2dGridAddNode(agrid,cx,cy,&cn); CHKERRQ(ierr);
      cells[4*agrid->cell_n] = cn;
      cx += deltax;
      ierr = AOData2dGridAddNode(agrid,cx,cy,&cn); CHKERRQ(ierr);
       cells[4*agrid->cell_n+1] = cn;
      
      cy += deltay;
      ierr = AOData2dGridAddNode(agrid,cx,cy,&cn); CHKERRQ(ierr);
      cells[4*agrid->cell_n+2] = cn; 

      cx -= deltax;
      ierr = AOData2dGridAddNode(agrid,cx,cy,&cn); CHKERRQ(ierr);
      cells[4*agrid->cell_n+3] = cn; 

      agrid->cell_n++;
    }
  }

  PetscFunctionReturn(0);
}

