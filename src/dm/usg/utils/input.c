/* $Id: makefile,v 1.5 1999/07/17 16:54:04 balay Exp bsmith $ */
static char help[] ="Allows inputing a 2d  grid into a AO database.\n";

/*

*/

#include "ao.h"
#include "bitarray.h"

int main(int argc, char **argv)
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

  /*---------------------------------------------------------------------
     Open the graphics window
     ------------------------------------------------------------------------*/
  ierr = DrawCreate(PETSC_COMM_WORLD,PETSC_NULL,"Input grid",PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,&draw);CHKERRQ(ierr);
  ierr = DrawSetFromOptions(draw);CHKERRA(ierr);

  ierr = AOData2dGridCreate(&agrid);CHKERRA(ierr);

  /*
    Get user to input the cell 
  */
  ierr = AOData2dGridInput(agrid,draw);CHKERRA(ierr);

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

