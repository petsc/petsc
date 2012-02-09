static char help[] = "test2.c: test of the new generation ExodusII readers: loads a mesh and creates matrices\n\n";

#include <petscsys.h>
#include <petscdmmesh.hh>

#undef __FUNCT__
#define __FUNCT__ "main"
int main (int argc,char ** argv) {
  DM              dmBody,dmFS;
  PetscBool       inflag;
  char            infilename[PETSC_MAX_PATH_LEN+1];
  PetscErrorCode  ierr;
  SectionReal     sec1;
  Mat             mat1;
  PetscInt        dof=1;
  int             rank;
  PetscInt        cell,face,conesize,my_num_cells,my_num_faces,my_num_vertices;
  PetscInt        i,j,c,d,k;
  PetscReal      *m_array;


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

  ierr = PetscOptionsGetInt(PETSC_NULL,"-dof",&dof,PETSC_NULL);CHKERRQ(ierr);
  ierr = DMMeshSetMaxDof(dmBody,dof);CHKERRQ(ierr);
  ierr = DMMeshSetMaxDof(dmFS,dof);CHKERRQ(ierr);

  /*
    Testing matrices
  */
  {
    ALE::Obj<PETSC_MESH_TYPE> meshBody,meshFS;
    ierr = DMMeshGetMesh(dmBody,meshBody);CHKERRQ(ierr);
    ierr = DMMeshGetMesh(dmFS,meshFS);CHKERRQ(ierr);
    my_num_cells    = meshBody->heightStratum(0)->size();
    my_num_vertices = meshBody->depthStratum(0)->size();
    my_num_faces    = meshFS->heightStratum(0)->size();
  }


  ierr = DMMeshGetVertexSectionReal(dmBody,"sec1",dof,&sec1);CHKERRQ(ierr);
  ierr = DMMeshCreateMatrix(dmBody,sec1,MATMPIAIJ,&mat1);CHKERRQ(ierr);  
  for (cell = 0; cell < my_num_cells; cell++) {
    ierr = DMMeshGetConeSize(dmBody,cell,&conesize);CHKERRQ(ierr);
    ierr = PetscMalloc(conesize*dof*conesize*dof*sizeof(PetscReal),&m_array);CHKERRQ(ierr);
    for (k = 0,i = 0; i < conesize; i++) {
      for (c = 0; c < dof; c++) {
        for (j = 0; j < conesize; j++) {
          for (d = 0; d < dof; d++,k++) {
            m_array[k] = 1.;
          }
        }
      }
    }
    ierr = DMMeshAssembleMatrix(mat1,dmBody,sec1,cell,m_array,ADD_VALUES);CHKERRQ(ierr);        
    ierr = PetscFree(m_array);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(mat1,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat1,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n==================== Matrix on body mesh\n");
  ierr = MatView(mat1,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  
  ierr = MatZeroEntries(mat1);CHKERRQ(ierr);
  for (face = 0; face < my_num_faces; face++) {
    ierr = DMMeshGetConeSize(dmFS,face,&conesize);CHKERRQ(ierr);
    ierr = PetscMalloc(conesize*dof*conesize*dof*sizeof(PetscReal),&m_array);CHKERRQ(ierr);
    for (k = 0,i = 0; i < conesize; i++) {
      for (c = 0; c < dof; c++) {
        for (j = 0; j < conesize; j++) {
          for (d = 0; d < dof; d++,k++) {
            m_array[k] = 1.;
          }
        }
      }
    }
    ierr = DMMeshAssembleMatrix(mat1,dmFS,sec1,face,m_array,ADD_VALUES);CHKERRQ(ierr);        
    ierr = PetscFree(m_array);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(mat1,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(mat1,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n==================== Matrix on FS mesh\n");
    ierr = MatView(mat1,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }    

  ierr = SectionRealDestroy(&sec1);CHKERRQ(ierr);
  ierr = MatDestroy(&mat1);CHKERRQ(ierr);
  ierr = DMDestroy(&dmBody);CHKERRQ(ierr);  
  ierr = DMDestroy(&dmFS);CHKERRQ(ierr);  

  ierr = PetscFinalize();
  return 0;
}

