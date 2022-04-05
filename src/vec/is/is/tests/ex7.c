static char help[] = "Tests ISLocate().\n\n";

#include <petscis.h>

static PetscErrorCode TestGeneral(void)
{
  MPI_Comm       comm = PETSC_COMM_SELF;
  const PetscInt idx[] = { 8, 6, 7, -5, 3, 0, 9 };
  PetscInt       n = 7, key = 3, nonkey = 1, keylocation = 4, sortedlocation = 2, location;
  IS             is;

  PetscFunctionBegin;
  PetscCall(ISCreateGeneral(comm,n,idx,PETSC_COPY_VALUES,&is));
  PetscCall(ISLocate(is,key,&location));
  PetscCheck(location == keylocation,comm,PETSC_ERR_PLIB,"Key %" PetscInt_FMT " not at %" PetscInt_FMT ": %" PetscInt_FMT,key,keylocation,location);
  PetscCall(ISLocate(is,nonkey,&location));
  PetscCheck(location < 0,comm,PETSC_ERR_PLIB,"Nonkey %" PetscInt_FMT " found at %" PetscInt_FMT,nonkey,location);
  PetscCall(ISSort(is));
  PetscCall(ISLocate(is,key,&location));
  PetscCheck(location == sortedlocation,comm,PETSC_ERR_PLIB,"Key %" PetscInt_FMT " not at %" PetscInt_FMT ": %" PetscInt_FMT,key,sortedlocation,location);
  PetscCall(ISLocate(is,nonkey,&location));
  PetscCheck(location < 0,comm,PETSC_ERR_PLIB,"Nonkey %" PetscInt_FMT " found at %" PetscInt_FMT,nonkey,location);
  PetscCall(ISDestroy(&is));
  PetscFunctionReturn(0);
}

static PetscErrorCode TestBlock(void)
{
  MPI_Comm       comm = PETSC_COMM_SELF;
  const PetscInt idx[] = { 8, 6, 7, -5, 3, 0, 9, };
  PetscInt       bs = 5, n = 7, key = 16, nonkey = 7, keylocation = 21, sortedlocation = 11, location;
  IS             is;

  PetscFunctionBegin;
  PetscCall(ISCreateBlock(comm,bs,n,idx,PETSC_COPY_VALUES,&is));
  PetscCall(ISLocate(is,key,&location));
  PetscCheck(location == keylocation,comm,PETSC_ERR_PLIB,"Key %" PetscInt_FMT " not at %" PetscInt_FMT ": %" PetscInt_FMT,key,keylocation,location);
  PetscCall(ISLocate(is,nonkey,&location));
  PetscCheck(location < 0,comm,PETSC_ERR_PLIB,"Nonkey %" PetscInt_FMT " found at %" PetscInt_FMT,nonkey,location);
  PetscCall(ISSort(is));
  PetscCall(ISLocate(is,key,&location));
  PetscCheck(location == sortedlocation,comm,PETSC_ERR_PLIB,"Key %" PetscInt_FMT " not at %" PetscInt_FMT ": %" PetscInt_FMT,key,sortedlocation,location);
  PetscCall(ISLocate(is,nonkey,&location));
  PetscCheck(location < 0,comm,PETSC_ERR_PLIB,"Nonkey %" PetscInt_FMT " found at %" PetscInt_FMT,nonkey,location);
  PetscCall(ISDestroy(&is));
  PetscFunctionReturn(0);
}

static PetscErrorCode TestStride(void)
{
  MPI_Comm       comm = PETSC_COMM_SELF;
  PetscInt       stride = 7, first = -3, n = 18, key = 39, keylocation = 6;
  PetscInt       nonkey[] = {-2,123}, i, location;
  IS             is;

  PetscFunctionBegin;
  PetscCall(ISCreateStride(comm,n,first,stride,&is));
  PetscCall(ISLocate(is,key,&location));
  PetscCheck(location == keylocation,comm,PETSC_ERR_PLIB,"Key %" PetscInt_FMT " not at %" PetscInt_FMT ": %" PetscInt_FMT,key,keylocation,location);
  for (i = 0; i < 2; i++) {
    PetscCall(ISLocate(is,nonkey[i],&location));
    PetscCheck(location < 0,comm,PETSC_ERR_PLIB,"Nonkey %" PetscInt_FMT " found at %" PetscInt_FMT,nonkey[i],location);
  }
  PetscCall(ISDestroy(&is));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{

  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  PetscCall(TestGeneral());
  PetscCall(TestBlock());
  PetscCall(TestStride());
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      output_file: output/ex1_1.out

TEST*/
