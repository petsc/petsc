#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: triconvert.c,v 1.4 2000/05/08 15:09:28 balay Exp bsmith $";
#endif
/*
      Converts triangulated grid data file.node and file.ele generated
   by the Triangle code of Shewchuk to a PETSc AOData base.

*/

#include "petscao.h"
#include "petscbt.h"

int main(int argc,char **args)
{
  int        ierr,nvertex,dim,nstuff,nbound,dummy,bound,i,ncell,*cell,*cell_edge,*edge_vertex;
  int        nedge,j,k,*edge_cell,*cell_cell,nc,ncp;
  int        i0,i1;
  int        *shift0,*shift1;
  char       filebase[256],filename[256];
  FILE       *file;
  AOData     ao;
  double     *vertex,ddummy;
  PetscBT    vertex_boundary;
  PetscTruth flag;

  PetscInitialize(&argc,&args,0,0);
  ierr = PetscMemzero(filename,256*sizeof(char));CHKERRA(ierr);

  ierr = OptionsGetString(0,"-f",filebase,246,&flag);CHKERRA(ierr);
  if (!flag) {
    SETERRA(1,"Must provide filebase name with -f");
  }

  /*
     Create empty database 
  */
  ierr = AODataCreateBasic(PETSC_COMM_SELF,&ao);CHKERRA(ierr);

  /* -----------------------------------------------------------------------------------
       Read in vertex information 
  */
  ierr = PetscStrcpy(filename,filebase);CHKERRA(ierr);
  ierr = PetscStrcat(filename,".node");CHKERRA(ierr);
  file = fopen(filename,"r"); 
  if (!file) {  
    SETERRA1(1,"Unable to open node file: %s",filename);
  }
  fscanf(file,"%d %d %d %d\n",&nvertex,&dim,&nstuff,&nbound);  
  if (dim != 2) {
    SETERRA(1,"Triangulation is not in two dimensions");
  }
  vertex = (double *)PetscMalloc(2*nvertex*sizeof(double));CHKPTRA(vertex);
  ierr   = PetscBTCreate(nvertex,vertex_boundary);CHKERRA(ierr);

  if (nstuff == 1) {
    for (i=0; i<nvertex; i++) {
      fscanf(file,"%d %le %le %le %d\n",&dummy,vertex+2*i,vertex+2*i+1,&ddummy,&bound);
      if (bound) PetscBTSet(vertex_boundary,i);
    }
  } else  if (nstuff == 0) {
    for (i=0; i<nvertex; i++) {
      fscanf(file,"%d %le %le %d\n",&dummy,vertex+2*i,vertex+2*i+1,&bound);
      if (bound) PetscBTSet(vertex_boundary,i);
    }
  } else SETERRA(1,"No support yet for that number of vertex quantities");
  fclose(file);

  /*  
     Put vertex into database 
  */
  ierr = AODataKeyAdd(ao,"vertex",nvertex,nvertex);CHKERRA(ierr);
  ierr = AODataSegmentAdd(ao,"vertex","values",2,nvertex,0,vertex,PETSC_DOUBLE);CHKERRA(ierr);
  ierr = AODataSegmentAdd(ao,"vertex","boundary",1,nvertex,0,vertex_boundary,PETSC_LOGICAL);CHKERRA(ierr);
  ierr = PetscBTDestroy(vertex_boundary);CHKERRA(ierr);

  /* -----------------------------------------------------------------------------------
      Read in triangle information 
  */
  ierr = PetscStrcpy(filename,filebase);CHKERRA(ierr);
  ierr = PetscStrcat(filename,".ele");CHKERRA(ierr);
  file = fopen(filename,"r"); 
  if (!file) {  
    SETERRA(1,"Unable to open element file");
  }
  fscanf(file,"%d %d %d\n",&ncell,&nc,&nstuff);ncp = nc;

  cell = (int *)PetscMalloc(nc*ncell*sizeof(int));CHKPTRA(cell);
  if (nstuff == 0) {
    if (nc == 3) {
      for (i=0; i<ncell; i++) {
        fscanf(file,"%d %d %d %d\n",&dummy,cell+3*i,cell+3*i+1,cell+3*i+2);
      }
    } else if (nc == 6) {
      for (i=0; i<ncell; i++) {
        fscanf(file,"%d %d %d %d %d %d %d\n",&dummy,cell+6*i,cell+6*i+2,cell+6*i+4,
               cell+6*i+3,cell+6*i+5,cell+6*i+1);
      }
    }
  } else {
    SETERRA(1,"No support yet for that number of element quantities");
  }
  fclose(file);
  for (i=0; i<nc*ncell; i++) {
    cell[i]--;    /* shift because triangle starts at 1, not 0 */
  }

  /*  
     Put cell into database 
  */
  ierr = AODataKeyAdd(ao,"cell",ncell,ncell);CHKERRA(ierr);
  ierr = AODataSegmentAdd(ao,"cell","vertex",nc,ncell,0,cell,PETSC_INT);CHKERRA(ierr);

  cell_edge    = (int *)PetscMalloc(nc*ncell*sizeof(int));CHKPTRA(cell_edge);
  edge_cell    = (int *)PetscMalloc(2*nc*ncell*sizeof(int));CHKPTRA(edge_cell);
  edge_vertex  = (int *)PetscMalloc(2*nc*ncell*sizeof(int));CHKPTRA(edge_vertex);
  cell_cell    = (int *)PetscMalloc(3*ncell*sizeof(int));CHKPTRA(cell_cell);

  /*
      Determine edges 
  */
  shift0 = (int *)PetscMalloc(nc*sizeof(int));CHKPTRQ(shift0);
  shift1 = (int *)PetscMalloc(nc*sizeof(int));CHKPTRQ(shift1);
  for (i=0; i<nc; i++) {
    shift0[i] = i; 
    shift1[i] = (i + 1) % nc;
  }


  nedge = 0;
  for (i=0; i<ncell; i++) {
    for (k=0; k<nc; k++) {
      i0 = cell[nc*i+shift0[k]]; 
      i1 = cell[nc*i+shift1[k]]; 
      for (j=0; j<nedge; j++) {
        if ((i0 == edge_vertex[2*j+1] && i1 == edge_vertex[2*j]) || 
            (i1 == edge_vertex[2*j+1] && i0 == edge_vertex[2*j])) {
          cell_edge[nc*i+k]   = j;
          edge_cell[2*j+1]   = i;
          goto found;
        }
      }
      /*
           Add a new edge to the list 
      */
      edge_cell[2*nedge]          = i;
      edge_cell[2*nedge+1]        = -1;
      edge_vertex[2*nedge]        = i0;
      edge_vertex[2*nedge+1]      = i1;
      cell_edge[nc*i+k]           = nedge;
      nedge++;
     
      found:;
    }
  }

  /*
       Determine cell neighbors 
  */
  ierr = PetscStrcpy(filename,filebase);CHKERRA(ierr);
  ierr = PetscStrcat(filename,".neigh");CHKERRA(ierr);
  file = fopen(filename,"r"); 
  if (file) {  
    fscanf(file,"%d %d\n",&ncell,&nc);
    if (nc != 3) SETERRQ(PETSC_ERR_SUP,"Can only handle three neighbors");
    for (i=0; i<ncell; i++) {
      fscanf(file,"%d %d %d %d\n",&dummy,cell_cell+3*i+1,cell_cell+3*i+2,cell_cell+3*i);
    }
    for (i=0; i<3*ncell; i++) {
      cell_cell[i]--;    /* shift because triangle starts at 1, not 0 */
    }
    fclose(file);

  } else { /* no neighbor list file given, generate manually only works for nc == 3 */
    if (nc != 3) SETERRQ(PETSC_ERR_SUP,"No neighbor file given and cannot determine neighbors");

    for (i=0; i<ncell; i++) {
      for (k=0; k<3; k++) {
        i0 = cell_edge[3*i+k];
        if (edge_cell[2*i0] != i) cell_cell[3*i+k] = edge_cell[2*i0];
        else                      cell_cell[3*i+k] = edge_cell[2*i0+1];
      }
    }
  }

  ierr = AODataSegmentAdd(ao,"cell","cell",3,ncell,0,cell_cell,PETSC_INT);CHKERRA(ierr);
  ierr = AODataSegmentAdd(ao,"cell","edge",nc,ncell,0,cell_edge,PETSC_INT);CHKERRA(ierr);

  ierr = AODataKeyAdd(ao,"edge",nedge,nedge);CHKERRA(ierr);
  ierr = AODataSegmentAdd(ao,"edge","vertex",2,nedge,0,edge_vertex,PETSC_INT);CHKERRA(ierr);

  ierr = PetscFree(vertex);CHKERRA(ierr);
  ierr = PetscFree(cell);CHKERRA(ierr);
  ierr = PetscFree(shift0);CHKERRA(ierr);
  ierr = PetscFree(shift1);CHKERRA(ierr);

  /*
      Add information about cell shape and element type to database
  */
  ierr = AODataKeyAdd(ao,"info",1,1);CHKERRA(ierr);
  ierr = AODataSegmentAdd(ao,"info","shape",10,1,0,(void*)"triangular",PETSC_CHAR);CHKERRA(ierr);
  if (ncp == 3) {
    ierr = AODataSegmentAdd(ao,"info","element",6,1,0,(void*)"linear",PETSC_CHAR);CHKERRA(ierr);
  } else if (ncp == 6) {
    ierr = AODataSegmentAdd(ao,"info","element",13,1,0,(void*)"quadratic",PETSC_CHAR);CHKERRA(ierr);
  }

  /*  ierr = AODataView(ao,0); */

  { Viewer binary;
  ierr = ViewerBinaryOpen(PETSC_COMM_WORLD,filebase,BINARY_CREATE,&binary);CHKERRA(ierr);
  ierr = AODataView(ao,binary);CHKERRA(ierr);
  ierr = ViewerDestroy(binary);CHKERRA(ierr);
  }
  
  return 0;
}
