static char help[] = "test1.c: test of the new generation ExodusII readers: loads and displays a mesh\n\n";

#include <petscsys.h>
#include <petscdmmesh.hh>

#undef __FUNCT__
#define __FUNCT__ "main"
int main (int argc,char ** argv) {
  DM              dmBody,dmFS;
  PetscBool       inflag;
  char            infilename[PETSC_MAX_PATH_LEN+1];
  PetscErrorCode  ierr;
  PetscInt        my_num_cells,my_num_faces,my_num_vertices;
  int             rank;

  ierr = PetscInitialize(&argc,&argv,(char *)0,help);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  ierr = PetscOptionsGetString(PETSC_NULL,"-i",infilename,PETSC_MAX_PATH_LEN,&inflag);CHKERRQ(ierr);
  if (!inflag) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"No file name given\n");CHKERRQ(ierr);
    PetscFinalize();
    return 0;
  }

  /*
    Reads a mesh
  */  
  ierr = DMMeshCreateExodusNG(PETSC_COMM_WORLD,infilename,&dmBody,&dmFS);CHKERRQ(ierr);

  /*
    Displays the body mesh
  */
  {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\ndmBody:\n");CHKERRQ(ierr);
    ierr = DMView(dmBody,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ALE::Obj<PETSC_MESH_TYPE> meshBody;
    ierr = DMMeshGetMesh(dmBody,meshBody);CHKERRQ(ierr);
    my_num_cells    = meshBody->heightStratum(0)->size();
    my_num_vertices = meshBody->depthStratum(0)->size();
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%i]: Number of vertices in meshBody: \t%i\n",rank,my_num_vertices);CHKERRQ(ierr);
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%i]: Number of cells in meshBody: \t%i\n",rank,my_num_cells);CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRQ(ierr);
    
    meshBody->view("meshBody");
    const ALE::Obj<PETSC_MESH_TYPE::label_type>& cellSets = meshBody->getLabel("Cell Sets");
    const ALE::Obj<PETSC_MESH_TYPE::label_type>& vertexSets = meshBody->getLabel("Vertex Sets");
    cellSets->view("Cell sets");
    vertexSets->view("Vertex sets");
  }
  ierr = DMDestroy(&dmBody);CHKERRQ(ierr);  

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\ndmFS:\n");CHKERRQ(ierr);
  ierr = DMView(dmFS,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  {
    ALE::Obj<PETSC_MESH_TYPE> meshFS;
    ierr = DMMeshGetMesh(dmFS,meshFS);CHKERRQ(ierr);
    my_num_faces    = meshFS->heightStratum(0)->size();
    my_num_vertices = meshFS->depthStratum(0)->size();
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%i]: Number of vertices in meshFS: \t%i\n",rank,my_num_vertices);CHKERRQ(ierr);
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%i]: Number of faces in meshFS: \t%i\n",rank,my_num_faces);CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRQ(ierr);

    meshFS->view("meshFS");
    const ALE::Obj<PETSC_MESH_TYPE::label_type>& faceSets = meshFS->getLabel("Face Sets");
    faceSets->view("Face sets");  
  }
  ierr = DMDestroy(&dmFS);CHKERRQ(ierr);  
  ierr = PetscFinalize();
  return 0;
}

