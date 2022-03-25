
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
  PetscInt          n,i,*cnt,*indices,nc;
  const PetscInt    *aj;
  const PetscScalar *vv,*aa;

  PetscFunctionBegin;
  PetscCall(MatGetSize(A,&n,NULL));
  PetscCall(VecGetArrayRead(v,&vv));
  PetscCall(PetscMalloc1(n,&indices));
  for (i=0; i<n; i++) indices[i] = i;

  /* determine number of nonzeros per row in the new matrix */
  PetscCall(PetscMalloc1(n+1,&cnt));
  for (i=0; i<n; i++) {
    PetscCall(MatGetRow(A,i,&nc,NULL,NULL));
    cnt[i] = nc + (vv[i] != 0.0);
    PetscCall(MatRestoreRow(A,i,&nc,NULL,NULL));
  }
  cnt[n] = 1;
  for (i=0; i<n; i++) {
    cnt[n] += (vv[i] != 0.0);
  }
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF,n+1,n+1,0,cnt,B));
  PetscCall(MatSetOption(*B,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE));

  /* copy over the matrix entries from the matrix and then the vector */
  for (i=0; i<n; i++) {
    PetscCall(MatGetRow(A,i,&nc,&aj,&aa));
    PetscCall(MatSetValues(*B,1,&i,nc,aj,aa,INSERT_VALUES));
    PetscCall(MatRestoreRow(A,i,&nc,&aj,&aa));
  }
  PetscCall(MatSetValues(*B,1,&n,n,indices,vv,INSERT_VALUES));
  PetscCall(MatSetValues(*B,n,indices,1,&n,vv,INSERT_VALUES));
  PetscCall(MatSetValues(*B,1,&n,1,&n,&c,INSERT_VALUES));

  PetscCall(MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY));
  PetscCall(VecRestoreArrayRead(v,&vv));
  PetscCall(PetscFree(cnt));
  PetscCall(PetscFree(indices));
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  Mat            A,B;
  PetscViewer    fd;                        /* viewer */
  char           file[PETSC_MAX_PATH_LEN];  /* input file name */
  PetscBool      flg;
  Vec            v;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  /*
     Determine files from which we read the two linear systems
     (matrix and right-hand-side vector).
  */
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f0",file,sizeof(file),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate binary file with the -f0 option");

  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetType(A,MATSEQAIJ));
  PetscCall(MatLoad(A,fd));
  PetscCall(VecCreate(PETSC_COMM_WORLD,&v));
  PetscCall(VecLoad(v,fd));
  PetscCall(MatView(A,PETSC_VIEWER_STDOUT_SELF));
  PetscCall(PadMatrix(A,v,3.0,&B));
  PetscCall(MatView(B,PETSC_VIEWER_STDOUT_SELF));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&v));
  PetscCall(PetscViewerDestroy(&fd));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -f0 ${wPETSC_DIR}/share/petsc/datafiles/matrices/ns-real-int32-float64
      requires: double !complex !defined(PETSC_USE_64BIT_INDICES)

TEST*/
