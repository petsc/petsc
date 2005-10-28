
static char help[] ="Allows inputing a 2d  grid into a AO database.\n";

/*

*/

#include "petscao.h"
#include "petscbt.h"

int main(int argc, char **argv)
{
  int          size, ierr;
  AOData2dGrid agrid;
  AOData       aodata;
  PetscViewer       binary;
  PetscDraw         draw;

  /* ---------------------------------------------------------------------
     Initialize PETSc
     ------------------------------------------------------------------------*/

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size > 1) {
    SETERRQ(1,"Must run input program with exactly one processor");
  }

  /*---------------------------------------------------------------------
     Open the graphics window
     ------------------------------------------------------------------------*/
  ierr = PetscDrawCreate(PETSC_COMM_WORLD,PETSC_NULL,"Input grid",PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetFromOptions(draw);CHKERRQ(ierr);

  ierr = AOData2dGridCreate(&agrid);CHKERRQ(ierr);

  /*
    Get user to input the cell 
  */
  ierr = AOData2dGridInput(agrid,draw);CHKERRQ(ierr);

  /* 
     Flip vertex in cell to make sure they are all clockwise
  */
  ierr = AOData2dGridFlipCells(agrid);CHKERRQ(ierr);
  
  /*
     Generate edge and neighor information
  */
  ierr = AOData2dGridComputeNeighbors(agrid);CHKERRQ(ierr);

  ierr = AOData2dGridComputeVertexBoundary(agrid);CHKERRQ(ierr);

  /*
     Show the numbering of the vertex, cell and edge
  */
  ierr = AOData2dGridDraw(agrid,draw);CHKERRQ(ierr);

  ierr = PetscDrawPause(draw);CHKERRQ(ierr);

  /*
      Create the database 
  */
  ierr = AOData2dGridToAOData(agrid,&aodata);CHKERRQ(ierr);

  /*
      Save the grid database to a file
  */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"gridfile",FILE_MODE_WRITE,&binary);CHKERRQ(ierr);
  ierr = AODataView(aodata,binary);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(binary);CHKERRQ(ierr);


  /*
     Close the graphics window and cleanup
  */
  ierr = PetscDrawDestroy(draw);CHKERRQ(ierr);

  ierr = AODataDestroy(aodata);CHKERRQ(ierr);

  ierr = AOData2dGridDestroy(agrid);CHKERRQ(ierr); 

  ierr = PetscFinalize();CHKERRQ(ierr);

  return 0;
}

