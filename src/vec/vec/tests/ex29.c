
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

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  bs   = size;

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-repeat",&repeat,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-subset",&subset,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-invert",&inv,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-bs",&bs,NULL));
  PetscCall(VecCreate(PETSC_COMM_WORLD,&x));
  PetscCall(VecSetSizes(x,PETSC_DECIDE,n*bs));
  PetscCall(VecSetBlockSize(x,bs));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x,&y));

  if (subset) PetscCall(VecSetOption(x,VEC_SUBSET_OFF_PROC_ENTRIES,PETSC_TRUE));

  for (r=0; r<repeat; r++) {
    /* Assemble the full vector on the first and last iteration, otherwise don't set any values */
    for (i=0; i<n*bs*(!r || !(repeat-1-r)); i++) {
      val  = i*1.0;
      PetscCall(VecSetValues(x,1,&i,&val,INSERT_VALUES));
    }
    PetscCall(VecAssemblyBegin(x));
    PetscCall(VecAssemblyEnd(x));
    if (!r) PetscCall(VecCopy(x,y)); /* Save result of first assembly */
  }

  PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecEqual(x,y,&flg));
  if (!flg) PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Vectors from repeat assembly do not match."));

  /* Create a new vector because the old stash is a subset. */
  PetscCall(VecDestroy(&x));
  PetscCall(VecDuplicate(y,&x));
  if (subset) PetscCall(VecSetOption(x,VEC_SUBSET_OFF_PROC_ENTRIES,PETSC_TRUE));

  /* Now do the blocksetvalues */
  PetscCall(VecSet(x,zero));
  PetscCall(PetscMalloc1(bs,&vals));
  for (r=0; r<repeat; r++) {
    PetscInt up = n*(!r || !(repeat-1-r));
    /* Assemble the full vector on the first and last iteration, otherwise don't set any values */
    for (i=0; i<up; i++) {
      PetscInt ii = inv ? up - i - 1 : i;
      for (j=0; j<bs; j++) vals[j] = (ii*bs+j)*1.0;
      PetscCall(VecSetValuesBlocked(x,1,&ii,vals,INSERT_VALUES));
    }
    PetscCall(VecAssemblyBegin(x));
    PetscCall(VecAssemblyEnd(x));
    if (!r) PetscCall(VecCopy(x,y)); /* Save result of first assembly */
  }

  PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecEqual(x,y,&flg));
  if (!flg) PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Vectors from repeat block assembly do not match."));

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(PetscFree(vals));
  PetscCall(PetscFinalize());
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
