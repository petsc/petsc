static char help[] ="Generation of a rectangular 2d quadrilateral grid.\n\
  Command line parameters -nx nx -ny ny indicate number of elements\n\
  -xmin xmin, -xmax xmax, -ymin ymin -ymax ymax.\n";

#include "petscao.h"
#include "src/dm/ao/aoimpl.h"  /* need direct access to AOData2dGrid structure */
#include "petscdraw.h"
#include <string.h>
#include <stdlib.h>

double x_min,x_max,y_min,y_max;
int    n_x, n_y; 
static PetscErrorCode InputGrid(AOData *aodata);

#undef  __FUNCT__
#define __FUNCT__ "main"
int main( int argc, char **argv )
{
  PetscErrorCode ierr;
  PetscMPIInt    size;
  char           filename[513] = "gridfile";
  AOData         aodata;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size > 1) {
    SETERRQ(PETSC_ERR_USER,"Must run input program with exactly one processor");
  }
  x_min = 0.0; x_max = 1.0;
  y_min = 0.0; y_max = 1.0;
  ierr = PetscOptionsGetReal(0,"-xmin",&x_min,0);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(0,"-xmax",&x_max,0);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(0,"-ymin",&y_min,0);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(0,"-ymax",&y_max,0);CHKERRQ(ierr);
  n_x = 5; n_y = 5;
  ierr = PetscOptionsGetInt(0,"-nx",&n_x,0);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(0,"-ny",&n_y,0);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(0,"-f",filename,512,0);CHKERRQ(ierr)

  /*
     Create the grid database.
  */
  ierr = InputGrid(&aodata);CHKERRQ(ierr);

  /*
     Save the grid database to a file.
  */
  {
    PetscViewer binary;
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,PETSC_BINARY_CREATE,&binary);CHKERRQ(ierr);
    ierr = AODataView(aodata,binary);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(binary);CHKERRQ(ierr);
  }

  ierr = AODataDestroy(aodata);CHKERRQ(ierr);

  {
    char command[530] = "\\rm -f ";
    strcat(command,filename);
    strcat(command,".info");
    system(command);
  }

  PetscFinalize();
  return 0;
}
#undef  __FUNCT__
#define __FUNCT__ "InputGrid"
int InputGrid (AOData *aodata)
{
  PetscErrorCode ierr;
  int i, j;
  int *indices;
  const int n_cells    = n_x * n_y;
  const int n_vertices = (n_x + 1) * (n_y + 1);
  const int n_edges    = 2 * n_x * n_y + n_x + n_y;

  PetscFunctionBegin;

  ierr = PetscMalloc (n_edges * sizeof(int),&indices);CHKERRQ(ierr);
  for (i=0; i<n_edges; i++) { indices[i] = i; }

  ierr = AODataCreateBasic(PETSC_COMM_SELF,aodata);CHKERRQ(ierr);
  ierr = AODataKeyAdd(*aodata,"cell",  n_cells,   n_cells);CHKERRQ(ierr);
  ierr = AODataKeyAdd(*aodata,"edge",  n_edges,   n_edges);CHKERRQ(ierr);
  ierr = AODataKeyAdd(*aodata,"vertex",n_vertices,n_vertices);CHKERRQ(ierr);

  /* Create list of vertices and mark the boundary ones. */
  {
    PetscTruth flg;
    const double del_x = (x_max - x_min) / n_x;
    const double del_y = (y_max - y_min) / n_y;
    PetscBT boundary;
    double *coords, *p;
    ierr = PetscOptionsHasName(PETSC_NULL,"-dirichlet_on_left",&flg);CHKERRQ(ierr);
    ierr = PetscBTCreate(n_vertices,boundary);CHKERRQ(ierr);
    ierr = PetscMalloc (2 * n_vertices * sizeof(PetscReal),&coors);CHKERRQ(ierr);
    p = coords;
    if (!flg) { /* All the boundary is Dirichlet */
      for (i=0; i<=n_x; i++) {
        for (j=0; j<=n_y; j++) {
	  *(p++) = x_min + i * del_x;
          *(p++) = y_min + j * del_y;
	  if ( (!(i%n_x)) || (!(j%n_y)) ) { PetscBTSet(boundary,i*(n_y+1)+j); }
        }
      }
    } else { /* Only left boundary is Dirichlet. */
      for (i=0; i<=n_x; i++) {
        for (j=0; j<=n_y; j++) {
	  *(p++) = x_min + i * del_x;
          *(p++) = y_min + j * del_y;
        }
      }
      for (j=0; j<=n_y; j++) { ierr = PetscBTSet(boundary,j);CHKERRQ(ierr); }
    }
    ierr = AODataSegmentAdd(*aodata,"vertex","values"  ,2,n_vertices,indices,coords  ,PETSC_DOUBLE);CHKERRQ(ierr);
    ierr = AODataSegmentAdd(*aodata,"vertex","boundary",1,n_vertices,indices,boundary,PETSC_LOGICAL);CHKERRQ(ierr);
    ierr = PetscFree(coords);CHKERRQ(ierr);
    ierr = PetscFree(boundary);CHKERRQ(ierr);
  }

  /* Create list of edges. Each edge contains 2 vertices. Each non-boundary edge is shared by 2 cells. */
  {
    int *edge_vertices, *edge_cells, *p, *q;
    ierr = PetscMalloc (2 * n_edges * sizeof(int),&edge_vertices);CHKERRQ(ierr);
    p    = edge_vertices;
    ierr = PetscMalloc (2 * n_edges * sizeof(int),&edge_cells);CHKERRQ(ierr);
    q    = edge_cells;

    {/* i = 0 */
      for (j=0; j<n_y; j++) {
        *(p++) =                 (j);  *(q++) =             (j-1); /* when j==0, boundary */
        *(p++) =       (n_y+1) + (j);  *(q++) =             (j);
        *(p++) =                 (j);  *(q++) =                -1; /* boundary */
        *(p++) =                 (j+1);  *(q++) =             (j);
      }
    }
    for (i=1; i<n_x; i++) {
      {/* j = 0 */
        *(p++) = (i  )*(n_y+1)        ;  *(q++) =                -1; /* boundary */ 
        *(p++) = (i+1)*(n_y+1)        ;  *(q++) = (i  )*n_y        ;
        *(p++) = (i  )*(n_y+1)        ;  *(q++) = (i-1)*n_y        ;
        *(p++) = (i  )*(n_y+1) + (  1);  *(q++) = (i  )*n_y        ;
      }
      for (j=1; j<n_y; j++) {
        *(p++) = (i  )*(n_y+1) + (j);  *(q++) = (i  )*n_y + (j-1); 
        *(p++) = (i+1)*(n_y+1) + (j);  *(q++) = (i  )*n_y + (j);
        *(p++) = (i  )*(n_y+1) + (j);  *(q++) = (i-1)*n_y + (j);
        *(p++) = (i  )*(n_y+1) + (j+1);  *(q++) = (i  )*n_y + (j);
      }
    }
    for (i=0; i<n_x; i++) {
      {/* j = n_y; */
        *(p++) = (i  )*(n_y+1) + (n_y);  *(q++) = (i  )*n_y + n_y-1;
        *(p++) = (i+1)*(n_y+1) + (n_y);  *(q++) = -1; /* boundary */
      }
    }
    {/* i = n_x; */
      for (i=n_x, j=0; j<n_y; j++) {
        *(p++) = (n_x)*(n_y+1) + (j);  *(q++) = (n_x-1)*n_y + (j);
        *(p++) = (n_x)*(n_y+1) + (j+1);  *(q++) = -1; /* boundary */
      }
    }
    ierr = AODataSegmentAdd(*aodata,"edge","vertex",2,n_edges,indices,edge_vertices,PETSC_INT);CHKERRQ(ierr);
    ierr = AODataSegmentAdd(*aodata,"edge","cell"  ,2,n_edges,indices,edge_cells   ,PETSC_INT);CHKERRQ(ierr);
    ierr = PetscFree(edge_cells);CHKERRQ(ierr);
    ierr = PetscFree(edge_vertices);CHKERRQ(ierr);
  }

  /* Create list of cells. */
  /* First, each non-boundary cell has 4 neighbours: west, north, east and south. */
  {
    int *cell_cells, *p;
    ierr = PetscMalloc (4 * n_cells * sizeof(int),&cell_cells);CHKERRQ(ierr);
    p = cell_cells;

    {/* i = 0 */
      {/* j = 0; */
        *(p++) =                -1; /* boundary */
        *(p++) =             (  1);
        *(p++) = (  1)*n_y        ;
        *(p++) =                -1; /* boundary */
      }
      for (j=1; j<(n_y-1); j++) {
        *(p++) =                -1; /* boundary */
        *(p++) =             (j+1);
        *(p++) = (  1)*n_y + (  j);
        *(p++) =             (j-1);
      }
      {/* j = n_y-1; */
        *(p++) =                -1; /* boundary */
        *(p++) =                -1; /* boundary */
        *(p++) = (  1)*n_y + n_y-1;
        *(p++) =             n_y-2;
      }
    }
    for (i=1; i<(n_x-1); i++) {
      {/* j = 0 */
        *(p++) = (i-1)*n_y        ;
        *(p++) = (i  )*n_y + (  1);
        *(p++) = (i+1)*n_y        ;
        *(p++) =               -1; /* boundary */
      }
      for (j=1; j<(n_y-1); j++) {
        *(p++) = (i-1)*n_y + (  j);
        *(p++) = (i  )*n_y + (j+1);
        *(p++) = (i+1)*n_y + (  j);
        *(p++) = (i  )*n_y + (j-1);
      }
      {/* j = n_y-1 */
        *(p++) = (i-1)*n_y + n_y-1;
        *(p++) =               -1; /* boundary */
        *(p++) = (i+1)*n_y + n_y-1;
        *(p++) = (i  )*n_y + n_y-2;
      }
    }
    {/* i = n_x-1 */
      {/* j = 0 */
        *(p++) = (n_x-2)*n_y        ;
        *(p++) = (n_x-1)*n_y + (  1);
        *(p++) =                 -1; /* boundary */
        *(p++) =                 -1; /* boundary */
      }
      for (j=1; j<(n_y-1); j++) {
        *(p++) = (n_x-2)*n_y + (  j);
        *(p++) = (n_x-1)*n_y + (j+1);
        *(p++) =                 -1; /* boundary */
        *(p++) = (n_x-1)*n_y + (j-1);
      }
      {/* j = n_y-1 */
        *(p++) = (n_x-2)*n_y + n_y-1;
        *(p++) =                 -1; /* boundary */
        *(p++) =                 -1; /* boundary */
        *(p++) = (n_x-1)*n_y + n_y-2;
      }
    }

    ierr = AODataSegmentAdd(*aodata,"cell","cell",4,n_cells,indices,cell_cells,PETSC_INT);CHKERRQ(ierr);
    ierr = PetscFree(cell_cells);CHKERRQ(ierr);
  }
  /* Then, each cell has 4 vertices: SW, NW, NE, SE. */
  {
    int *cell_vertices, *p;
    ierr = PetscMalloc (4 * n_cells * sizeof(int),&cell_vertices);CHKERRQ(ierr);
    p = cell_vertices;
    for (i=0; i<n_x; i++) {
      for (j=0; j<n_y; j++) {
        *(p++) = (  i)*(n_y+1) + (  j);
        *(p++) = (  i)*(n_y+1) + (j+1);
        *(p++) = (i+1)*(n_y+1) + (j+1);
        *(p++) = (i+1)*(n_y+1) + (  j);
      }
    }
    ierr = AODataSegmentAdd(*aodata,"cell","vertex",4,n_cells,indices,cell_vertices,PETSC_INT);CHKERRQ(ierr);
    ierr = PetscFree(cell_vertices);CHKERRQ(ierr);
  }
  /* Finally, each cell has 4 edges: west, north, east, south. */
  {
    int *cell_edges, *p;
    ierr = PetscMalloc (4 * n_cells * sizeof(int),&cell_edges);CHKERRQ(ierr);
    p = cell_edges;

    for (i=0; i<(n_x-1); i++) {
      for (j=0; j<(n_y-1); j++) {
        *(p++) = 2*((i  )*n_y + (j  )) + 1;
        *(p++) = 2*((i  )*n_y + (j+1))    ;
        *(p++) = 2*((i+1)*n_y + (j  )) + 1;
        *(p++) = 2*((i  )*n_y + (j  ))    ;
      }
      {/* j = n_y-1 */
        *(p++) = 2*((i  )*n_y + n_y-1) + 1;
        *(p++) = 2*n_x*n_y + i            ; /* north boundary */          
        *(p++) = 2*((i+1)*n_y + n_y-1) + 1;
        *(p++) = 2*((i  )*n_y + n_y-1)    ;
      }
    }
    {/* i = n_x-1 */
      for (j=0; j<(n_y-1); j++) {
        *(p++) = 2*((n_x-1)*n_y + (j  )) + 1;
        *(p++) = 2*((n_x-1)*n_y + (j+1))    ;
        *(p++) = 2*n_x*n_y + n_x + j        ; /* east  boundary */
        *(p++) = 2*((n_x-1)*n_y + (j  ))    ;
      }
      {/* j = n_y-1 */
        *(p++) = 2*((n_x-1)*n_y + n_y-1) + 1;
        *(p++) = 2*n_x*n_y + n_x       - 1; /* north boundary */
        *(p++) = 2*n_x*n_y + n_x + n_y - 1; /* east  boundary */
        *(p++) = 2*((n_x-1)*n_y + n_y-1)    ;
      }
    }
    ierr = AODataSegmentAdd(*aodata,"cell","edge",4,n_cells,indices,cell_edges,PETSC_INT);CHKERRQ(ierr);
    ierr = PetscFree(cell_edges);CHKERRQ(ierr);
  }
  ierr = PetscFree(indices);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
