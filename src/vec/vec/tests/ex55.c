static char help[] = "Tests I/O of vector and string attribute for HDF5 format\n\n";

#include <petscvec.h>
#include <petscviewerhdf5.h>

int main(int argc,char **args)
{
  Vec               u;
  PetscViewer       viewer;
  char             *attrReadVal, attrWriteVal[20]={"Hello World!!"};

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));

  /* PART 1:  Generate vector, then write it in the given data format */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&u));
  PetscCall(PetscObjectSetName((PetscObject)u, "Test_Vec"));
  PetscCall(VecSetSizes(u,PETSC_DECIDE,10));
  PetscCall(VecSetFromOptions(u));
  PetscCall(VecSet(u,0.));

  /* write vector and attribute*/
  PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD,"vector.dat",FILE_MODE_WRITE,&viewer));
  PetscCall(VecView(u,viewer));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Attribute value written: '%s'\n\n",attrWriteVal));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer,"Test_Vec","Test_Attr",PETSC_STRING,attrWriteVal));

  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(VecDestroy(&u));

  /* PART 2:  Read in attribute */
  PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD,"vector.dat",FILE_MODE_READ,&viewer));
  PetscCall(PetscViewerHDF5ReadAttribute(viewer,"Test_Vec","Test_Attr",PETSC_STRING,NULL,&attrReadVal));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Attribute value read: '%s'\n\n",attrReadVal));
  PetscCall(PetscFree(attrReadVal));

  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

     build:
       requires: hdf5

     test:

TEST*/
