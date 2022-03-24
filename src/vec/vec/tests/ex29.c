
static char help[] = "Tests VecSetValues() and VecSetValuesBlocked() on MPI vectors.\n\
Where at least a couple of mallocs will occur in the stash code.\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscMPIInt    size;
  PetscInt       i,j,r,n = 50,repeat = 1,bs;
  PetscScalar    val,*vals,zero=0.0;
  PetscBool      inv = PETSC_FALSE, subset = PETSC_FALSE,flg;
  Vec            x,y;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  bs   = size;

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-repeat",&repeat,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-subset",&subset,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-invert",&inv,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-bs",&bs,NULL));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&x));
  CHKERRQ(VecSetSizes(x,PETSC_DECIDE,n*bs));
  CHKERRQ(VecSetBlockSize(x,bs));
  CHKERRQ(VecSetFromOptions(x));
  CHKERRQ(VecDuplicate(x,&y));

  if (subset) CHKERRQ(VecSetOption(x,VEC_SUBSET_OFF_PROC_ENTRIES,PETSC_TRUE));

  for (r=0; r<repeat; r++) {
    /* Assemble the full vector on the first and last iteration, otherwise don't set any values */
    for (i=0; i<n*bs*(!r || !(repeat-1-r)); i++) {
      val  = i*1.0;
      CHKERRQ(VecSetValues(x,1,&i,&val,INSERT_VALUES));
    }
    CHKERRQ(VecAssemblyBegin(x));
    CHKERRQ(VecAssemblyEnd(x));
    if (!r) CHKERRQ(VecCopy(x,y)); /* Save result of first assembly */
  }

  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecEqual(x,y,&flg));
  if (!flg) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Vectors from repeat assembly do not match."));

  /* Create a new vector because the old stash is a subset. */
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDuplicate(y,&x));
  if (subset) CHKERRQ(VecSetOption(x,VEC_SUBSET_OFF_PROC_ENTRIES,PETSC_TRUE));

  /* Now do the blocksetvalues */
  CHKERRQ(VecSet(x,zero));
  CHKERRQ(PetscMalloc1(bs,&vals));
  for (r=0; r<repeat; r++) {
    PetscInt up = n*(!r || !(repeat-1-r));
    /* Assemble the full vector on the first and last iteration, otherwise don't set any values */
    for (i=0; i<up; i++) {
      PetscInt ii = inv ? up - i - 1 : i;
      for (j=0; j<bs; j++) vals[j] = (ii*bs+j)*1.0;
      CHKERRQ(VecSetValuesBlocked(x,1,&ii,vals,INSERT_VALUES));
    }
    CHKERRQ(VecAssemblyBegin(x));
    CHKERRQ(VecAssemblyEnd(x));
    if (!r) CHKERRQ(VecCopy(x,y)); /* Save result of first assembly */
  }

  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecEqual(x,y,&flg));
  if (!flg) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Vectors from repeat block assembly do not match."));

  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(PetscFree(vals));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 3
      args: -n 126

   test:
      suffix: bts_test_inv_error
      nsize: 3
      args: -n 4 -invert -bs 2
      output_file: output/ex29_test_inv_error.out

   test:
      suffix: bts
      nsize: 3
      args: -n 126 -vec_assembly_legacy
      output_file: output/ex29_1.out

   test:
      suffix: bts_2
      nsize: 3
      args: -n 126 -vec_assembly_legacy -repeat 2
      output_file: output/ex29_1.out

   test:
      suffix: bts_2_subset
      nsize: 3
      args: -n 126 -vec_assembly_legacy -repeat 2 -subset
      output_file: output/ex29_1.out

   test:
      suffix: bts_2_subset_proper
      nsize: 3
      args: -n 126 -vec_assembly_legacy -repeat 5 -subset
      output_file: output/ex29_1.out

TEST*/
