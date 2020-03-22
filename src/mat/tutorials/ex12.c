
static char help[] = "Reads a PETSc matrix and vector from a file; expands the matrix with the vector\n\n";

/*T
   Concepts: Mat^ordering a matrix - loading a binary matrix and vector;
   Concepts: Mat^loading a binary matrix and vector;
   Concepts: Vectors^loading a binary vector;
   Concepts: PetscLog^preloading executable
   Processors: 1
T*/



/*
  Include "petscmat.h" so that we can use matrices.
  automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h    - vectors
     petscmat.h    - matrices
     petscis.h     - index sets            petscviewer.h - viewers
*/
#include <petscmat.h>

/*

   Adds a new column and row to the vector (the last) containing the vector
*/
PetscErrorCode PadMatrix(Mat A,Vec v,PetscScalar c,Mat *B)
{
  PetscErrorCode    ierr;
  PetscInt          n,i,*cnt,*indices,nc;
  const PetscInt    *aj;
  const PetscScalar *vv,*aa;

  PetscFunctionBegin;
  ierr = MatGetSize(A,&n,NULL);CHKERRQ(ierr);
  ierr = VecGetArrayRead(v,&vv);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&indices);CHKERRQ(ierr);
  for (i=0; i<n; i++) indices[i] = i;

  /* determine number of nonzeros per row in the new matrix */
  ierr = PetscMalloc1(n+1,&cnt);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ierr = MatGetRow(A,i,&nc,NULL,NULL);CHKERRQ(ierr);
    cnt[i] = nc + (vv[i] != 0.0);
    ierr = MatRestoreRow(A,i,&nc,NULL,NULL);CHKERRQ(ierr);
  }
  cnt[n] = 1;
  for (i=0; i<n; i++) {
    cnt[n] += (vv[i] != 0.0);
  }
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,n+1,n+1,0,cnt,B);CHKERRQ(ierr);
  ierr = MatSetOption(*B,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);

  /* copy over the matrix entries from the matrix and then the vector */
  for (i=0; i<n; i++) {
    ierr = MatGetRow(A,i,&nc,&aj,&aa);CHKERRQ(ierr);
    ierr = MatSetValues(*B,1,&i,nc,aj,aa,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(A,i,&nc,&aj,&aa);CHKERRQ(ierr);
  }
  ierr = MatSetValues(*B,1,&n,n,indices,vv,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatSetValues(*B,n,indices,1,&n,vv,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatSetValues(*B,1,&n,1,&n,&c,INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(v,&vv);CHKERRQ(ierr);
  ierr = PetscFree(cnt);CHKERRQ(ierr);
  ierr = PetscFree(indices);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


int main(int argc,char **args)
{
  Mat            A,B;
  PetscViewer    fd;                        /* viewer */
  char           file[PETSC_MAX_PATH_LEN];  /* input file name */
  PetscErrorCode ierr;
  PetscBool      flg;
  Vec            v;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  /*
     Determine files from which we read the two linear systems
     (matrix and right-hand-side vector).
  */
  ierr = PetscOptionsGetString(NULL,NULL,"-f0",file,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate binary file with the -f0 option");

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&v);CHKERRQ(ierr);
  ierr = VecLoad(v,fd);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = PadMatrix(A,v,3.0,&B);CHKERRQ(ierr);
  ierr = MatView(B,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}



/*TEST

   test:
      args: -f0 ${wPETSC_DIR}/share/petsc/datafiles/matrices/ns-real-int32-float64
      requires: double !complex !define(PETSC_USE_64BIT_INDICES)

TEST*/
