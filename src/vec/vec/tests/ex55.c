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
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&u));
  CHKERRQ(PetscObjectSetName((PetscObject)u, "Test_Vec"));
  CHKERRQ(VecSetSizes(u,PETSC_DECIDE,10));
  CHKERRQ(VecSetFromOptions(u));
  CHKERRQ(VecSet(u,0.));

  /* write vector and attribute*/
  CHKERRQ(PetscViewerHDF5Open(PETSC_COMM_WORLD,"vector.dat",FILE_MODE_WRITE,&viewer));
  CHKERRQ(VecView(u,viewer));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Attribute value written: '%s'\n\n",attrWriteVal));
  CHKERRQ(PetscViewerHDF5WriteAttribute(viewer,"Test_Vec","Test_Attr",PETSC_STRING,attrWriteVal));

  CHKERRQ(PetscViewerDestroy(&viewer));
  CHKERRQ(VecDestroy(&u));

  /* PART 2:  Read in attribute */
  CHKERRQ(PetscViewerHDF5Open(PETSC_COMM_WORLD,"vector.dat",FILE_MODE_READ,&viewer));
  CHKERRQ(PetscViewerHDF5ReadAttribute(viewer,"Test_Vec","Test_Attr",PETSC_STRING,NULL,&attrReadVal));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Attribute value read: '%s'\n\n",attrReadVal));
  CHKERRQ(PetscFree(attrReadVal));

  CHKERRQ(PetscViewerDestroy(&viewer));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

     build:
       requires: hdf5

     test:

TEST*/
