
static char help[] = "Tests VecSetValues() and VecSetValuesBlocked() on MPI vectors.\n\
Where at least a couple of mallocs will occur in the stash code.\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscInt       i,j,r,n = 50,repeat = 1,bs;
  PetscScalar    val,*vals,zero=0.0;
  PetscBool      inv = PETSC_FALSE, subset = PETSC_FALSE,flg;
  Vec            x,y;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  bs   = size;

  ierr = PetscOptionsGetInt(NULL,NULL,"-repeat",&repeat,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-subset",&subset,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-invert",&inv,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-bs",&bs,NULL);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,n*bs);CHKERRQ(ierr);
  ierr = VecSetBlockSize(x,bs);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&y);CHKERRQ(ierr);

  if (subset) {ierr = VecSetOption(x,VEC_SUBSET_OFF_PROC_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);}

  for (r=0; r<repeat; r++) {
    /* Assemble the full vector on the first and last iteration, otherwise don't set any values */
    for (i=0; i<n*bs*(!r || !(repeat-1-r)); i++) {
      val  = i*1.0;
      ierr = VecSetValues(x,1,&i,&val,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(x);CHKERRQ(ierr);
    if (!r) {ierr = VecCopy(x,y);CHKERRQ(ierr);} /* Save result of first assembly */
  }

  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecEqual(x,y,&flg);CHKERRQ(ierr);
  if (!flg) {ierr = PetscPrintf(PETSC_COMM_WORLD,"Vectors from repeat assembly do not match.");CHKERRQ(ierr);}

  /* Create a new vector because the old stash is a subset. */
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDuplicate(y,&x);CHKERRQ(ierr);
  if (subset) {ierr = VecSetOption(x,VEC_SUBSET_OFF_PROC_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);}

  /* Now do the blocksetvalues */
  ierr = VecSet(x,zero);CHKERRQ(ierr);
  ierr = PetscMalloc1(bs,&vals);CHKERRQ(ierr);
  for (r=0; r<repeat; r++) {
    PetscInt up = n*(!r || !(repeat-1-r));
    /* Assemble the full vector on the first and last iteration, otherwise don't set any values */
    for (i=0; i<up; i++) {
      PetscInt ii = inv ? up - i - 1 : i;
      for (j=0; j<bs; j++) vals[j] = (ii*bs+j)*1.0;
      ierr = VecSetValuesBlocked(x,1,&ii,vals,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(x);CHKERRQ(ierr);
    if (!r) {ierr = VecCopy(x,y);CHKERRQ(ierr);} /* Save result of first assembly */
  }

  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecEqual(x,y,&flg);CHKERRQ(ierr);
  if (!flg) {ierr = PetscPrintf(PETSC_COMM_WORLD,"Vectors from repeat block assembly do not match.");CHKERRQ(ierr);}

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = PetscFree(vals);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
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
