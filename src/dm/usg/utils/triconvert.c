
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
  char       filebase[PETSC_MAX_PATH_LEN],filename[PETSC_MAX_PATH_LEN];
  FILE       *file;
  AOData     ao;
  PetscReal  *vertex;
  PetscBT    vertex_boundary;
  PetscTruth flag;

  PetscInitialize(&argc,&args,0,0);
  ierr = PetscMemzero(filename,PETSC_MAX_PATH_LEN*sizeof(char));CHKERRQ(ierr);

  ierr = PetscOptionsGetString(0,"-f",filebase,PETSC_MAX_PATH_LEN-1,&flag);CHKERRQ(ierr);
  if (!flag) {
    SETERRQ(1,"Must provide filebase name with -f");
  }

  /*
     Create empty database 
  */
  ierr = AODataCreateBasic(PETSC_COMM_SELF,&ao);CHKERRQ(ierr);

  /* -----------------------------------------------------------------------------------
       Read in vertex information 
  */
  ierr = PetscStrcpy(filename,filebase);CHKERRQ(ierr);
  ierr = PetscStrcat(filename,".node");CHKERRQ(ierr);
  file = fopen(filename,"r"); 
  if (!file) {  
    SETERRQ1(1,"Unable to open node file: %s",filename);
  }
  fscanf(file,"%d %d %d %d\n",&nvertex,&dim,&nstuff,&nbound);  
  if (dim != 2) {
    SETERRQ(1,"Triangulation is not in two dimensions");
  }
  ierr = PetscMalloc(2*nvertex*sizeof(PetscReal),&vertex);CHKERRQ(ierr);
  ierr   = PetscBTCreate(nvertex,vertex_boundary);CHKERRQ(ierr);

  if (nstuff == 1) {
    double v0,v1,ddummy;
    for (i=0; i<nvertex; i++) {
      fscanf(file,"%d %le %le %le %d\n",&dummy,&v0,&v1,&ddummy,&bound);
      vertex[2*i]   = v0;
      vertex[2*i+1] = v1;
      if (bound) {ierr = PetscBTSet(vertex_boundary,i);CHKERRQ(ierr);}
    }
  } else  if (nstuff == 0) {
    double v0,v1;
    for (i=0; i<nvertex; i++) {
      fscanf(file,"%d %le %le %d\n",&dummy,&v0,&v1,&bound);
      vertex[2*i]   = v0;
      vertex[2*i+1] = v1;
      if (bound) {ierr = PetscBTSet(vertex_boundary,i);CHKERRQ(ierr);}
    }
  } else SETERRQ(1,"No support yet for that number of vertex quantities");
  fclose(file);

  /*  
     Put vertex into database 
  */
  ierr = AODataKeyAdd(ao,"vertex",nvertex,nvertex);CHKERRQ(ierr);
  ierr = AODataSegmentAdd(ao,"vertex","values",2,nvertex,0,vertex,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = AODataSegmentAdd(ao,"vertex","boundary",1,nvertex,0,vertex_boundary,PETSC_LOGICAL);CHKERRQ(ierr);
  ierr = PetscBTDestroy(vertex_boundary);CHKERRQ(ierr);

  /* -----------------------------------------------------------------------------------
      Read in triangle information 
  */
  ierr = PetscStrcpy(filename,filebase);CHKERRQ(ierr);
  ierr = PetscStrcat(filename,".ele");CHKERRQ(ierr);
  file = fopen(filename,"r"); 
  if (!file) {  
    SETERRQ(1,"Unable to open element file");
  }
  fscanf(file,"%d %d %d\n",&ncell,&nc,&nstuff);ncp = nc;

  ierr = PetscMalloc(nc*ncell*sizeof(int),&cell);CHKERRQ(ierr);
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
    SETERRQ(1,"No support yet for that number of element quantities");
  }
  fclose(file);
  for (i=0; i<nc*ncell; i++) {
    cell[i]--;    /* shift because triangle starts at 1, not 0 */
  }

  /*  
     Put cell into database 
  */
  ierr = AODataKeyAdd(ao,"cell",ncell,ncell);CHKERRQ(ierr);
  ierr = AODataSegmentAdd(ao,"cell","vertex",nc,ncell,0,cell,PETSC_INT);CHKERRQ(ierr);

  ierr = PetscMalloc(nc*ncell*sizeof(int),&cell_edge);CHKERRQ(ierr);
  ierr = PetscMalloc(2*nc*ncell*sizeof(int),&edge_cell);CHKERRQ(ierr);
  ierr = PetscMalloc(2*nc*ncell*sizeof(int),&edge_vertex);CHKERRQ(ierr);
  ierr = PetscMalloc(3*ncell*sizeof(int),&cell_cell);CHKERRQ(ierr);

  /*
      Determine edges 
  */
  ierr = PetscMalloc(nc*sizeof(int),&shift0);CHKERRQ(ierr);
  ierr = PetscMalloc(nc*sizeof(int),&shift1);CHKERRQ(ierr);
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
  ierr = PetscStrcpy(filename,filebase);CHKERRQ(ierr);
  ierr = PetscStrcat(filename,".neigh");CHKERRQ(ierr);
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

  ierr = AODataSegmentAdd(ao,"cell","cell",3,ncell,0,cell_cell,PETSC_INT);CHKERRQ(ierr);
  ierr = AODataSegmentAdd(ao,"cell","edge",nc,ncell,0,cell_edge,PETSC_INT);CHKERRQ(ierr);

  ierr = AODataKeyAdd(ao,"edge",nedge,nedge);CHKERRQ(ierr);
  ierr = AODataSegmentAdd(ao,"edge","vertex",2,nedge,0,edge_vertex,PETSC_INT);CHKERRQ(ierr);

  ierr = PetscFree(vertex);CHKERRQ(ierr);
  ierr = PetscFree(cell);CHKERRQ(ierr);
  ierr = PetscFree(shift0);CHKERRQ(ierr);
  ierr = PetscFree(shift1);CHKERRQ(ierr);

  /*
      Add information about cell shape and element type to database
  */
  ierr = AODataKeyAdd(ao,"info",1,1);CHKERRQ(ierr);
  ierr = AODataSegmentAdd(ao,"info","shape",10,1,0,(void*)"triangular",PETSC_CHAR);CHKERRQ(ierr);
  if (ncp == 3) {
    ierr = AODataSegmentAdd(ao,"info","element",6,1,0,(void*)"linear",PETSC_CHAR);CHKERRQ(ierr);
  } else if (ncp == 6) {
    ierr = AODataSegmentAdd(ao,"info","element",13,1,0,(void*)"quadratic",PETSC_CHAR);CHKERRQ(ierr);
  }

  /*  ierr = AODataView(ao,0); */

  { PetscViewer binary;
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filebase,PETSC_FILE_CREATE,&binary);CHKERRQ(ierr);
  ierr = AODataView(ao,binary);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(binary);CHKERRQ(ierr);
  }
  
  return 0;
}
