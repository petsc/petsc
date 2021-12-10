static char help[] = "Tests ISLocate().\n\n";

#include <petscis.h>

static PetscErrorCode TestGeneral(void)
{
  MPI_Comm       comm = PETSC_COMM_SELF;
  const PetscInt idx[] = { 8, 6, 7, -5, 3, 0, 9 };
  PetscInt       n = 7, key = 3, nonkey = 1, keylocation = 4, sortedlocation = 2, location;
  IS             is;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ISCreateGeneral(comm,n,idx,PETSC_COPY_VALUES,&is);CHKERRQ(ierr);
  ierr = ISLocate(is,key,&location);CHKERRQ(ierr);
  if (location != keylocation) SETERRQ3(comm,PETSC_ERR_PLIB,"Key %" PetscInt_FMT " not at %" PetscInt_FMT ": %" PetscInt_FMT,key,keylocation,location);
  ierr = ISLocate(is,nonkey,&location);CHKERRQ(ierr);
  if (location >= 0) SETERRQ2(comm,PETSC_ERR_PLIB,"Nonkey %" PetscInt_FMT " found at %" PetscInt_FMT,nonkey,location);
  ierr = ISSort(is);CHKERRQ(ierr);
  ierr = ISLocate(is,key,&location);CHKERRQ(ierr);
  if (location != sortedlocation) SETERRQ3(comm,PETSC_ERR_PLIB,"Key %" PetscInt_FMT " not at %" PetscInt_FMT ": %" PetscInt_FMT,key,sortedlocation,location);
  ierr = ISLocate(is,nonkey,&location);CHKERRQ(ierr);
  if (location >= 0) SETERRQ2(comm,PETSC_ERR_PLIB,"Nonkey %" PetscInt_FMT " found at %" PetscInt_FMT,nonkey,location);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestBlock(void)
{
  MPI_Comm       comm = PETSC_COMM_SELF;
  const PetscInt idx[] = { 8, 6, 7, -5, 3, 0, 9, };
  PetscInt       bs = 5, n = 7, key = 16, nonkey = 7, keylocation = 21, sortedlocation = 11, location;
  IS             is;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ISCreateBlock(comm,bs,n,idx,PETSC_COPY_VALUES,&is);CHKERRQ(ierr);
  ierr = ISLocate(is,key,&location);CHKERRQ(ierr);
  if (location != keylocation) SETERRQ3(comm,PETSC_ERR_PLIB,"Key %" PetscInt_FMT " not at %" PetscInt_FMT ": %" PetscInt_FMT,key,keylocation,location);
  ierr = ISLocate(is,nonkey,&location);CHKERRQ(ierr);
  if (location >= 0) SETERRQ2(comm,PETSC_ERR_PLIB,"Nonkey %" PetscInt_FMT " found at %" PetscInt_FMT,nonkey,location);
  ierr = ISSort(is);CHKERRQ(ierr);
  ierr = ISLocate(is,key,&location);CHKERRQ(ierr);
  if (location != sortedlocation) SETERRQ3(comm,PETSC_ERR_PLIB,"Key %" PetscInt_FMT " not at %" PetscInt_FMT ": %" PetscInt_FMT,key,sortedlocation,location);
  ierr = ISLocate(is,nonkey,&location);CHKERRQ(ierr);
  if (location >= 0) SETERRQ2(comm,PETSC_ERR_PLIB,"Nonkey %" PetscInt_FMT " found at %" PetscInt_FMT,nonkey,location);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestStride(void)
{
  MPI_Comm       comm = PETSC_COMM_SELF;
  PetscInt       stride = 7, first = -3, n = 18, key = 39, keylocation = 6;
  PetscInt       nonkey[] = {-2,123}, i, location;
  IS             is;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ISCreateStride(comm,n,first,stride,&is);CHKERRQ(ierr);
  ierr = ISLocate(is,key,&location);CHKERRQ(ierr);
  if (location != keylocation) SETERRQ3(comm,PETSC_ERR_PLIB,"Key %" PetscInt_FMT " not at %" PetscInt_FMT ": %" PetscInt_FMT,key,keylocation,location);
  for (i = 0; i < 2; i++) {
    ierr = ISLocate(is,nonkey[i],&location);CHKERRQ(ierr);
    if (location >= 0) SETERRQ2(comm,PETSC_ERR_PLIB,"Nonkey %" PetscInt_FMT " found at %" PetscInt_FMT,nonkey[i],location);
  }
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  ierr = TestGeneral();CHKERRQ(ierr);
  ierr = TestBlock();CHKERRQ(ierr);
  ierr = TestStride();CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      output_file: output/ex1_1.out

TEST*/
