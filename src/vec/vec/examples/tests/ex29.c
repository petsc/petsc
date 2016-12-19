
static char help[] = "Tests VecSetValues() and VecSetValuesBlocked() on MPI vectors.\n\
Where atleast a couple of mallocs will occur in the stash code.\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscInt       i,j,r,n = 50,repeat = 1,bs;
  PetscScalar    val,*vals,zero=0.0;
  PetscBool      subset = PETSC_FALSE,flg;
  Vec            x,y;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  bs   = size;

  ierr = PetscOptionsGetInt(NULL,NULL,"-repeat",&repeat,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-subset",&subset,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
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
    /* Assemble the full vector on the first and last iteration, otherwise don't set any values */
    for (i=0; i<n*(!r || !(repeat-1-r)); i++) {
      for (j=0; j<bs; j++) vals[j] = (i*bs+j)*1.0;
      ierr = VecSetValuesBlocked(x,1,&i,vals,INSERT_VALUES);CHKERRQ(ierr);
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

