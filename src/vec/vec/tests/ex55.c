static char help[] = "Tests I/O of vector and string attribute for HDF5 format\n\n";

#include <petscvec.h>
#include <petscviewerhdf5.h>

int main(int argc,char **args)
{
  PetscErrorCode    ierr;
  Vec               u;
  PetscViewer       viewer;
  char             *attrReadVal, attrWriteVal[20]={"Hello World!!"};

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;

  /* PART 1:  Generate vector, then write it in the given data format */
  ierr = VecCreate(PETSC_COMM_WORLD,&u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)u, "Test_Vec");CHKERRQ(ierr);
  ierr = VecSetSizes(u,PETSC_DECIDE,10);CHKERRQ(ierr);
  ierr = VecSetFromOptions(u);CHKERRQ(ierr);
  ierr = VecSet(u,0.);CHKERRQ(ierr);

  /* write vector and attribute*/
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"vector.dat",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(u,viewer);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Attribute value written: '%s'\n\n",attrWriteVal);CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(viewer,"Test_Vec","Test_Attr",PETSC_STRING,attrWriteVal);CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);

  /* PART 2:  Read in attribute */
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"vector.dat",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/");CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer,"Test_Vec","Test_Attr",PETSC_STRING,&attrReadVal);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Attribute value read: '%s'\n\n",attrReadVal);CHKERRQ(ierr);
  ierr = PetscFree(attrReadVal);CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

     build:
       requires: hdf5

     test:


TEST*/
