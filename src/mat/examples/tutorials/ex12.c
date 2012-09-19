
static char help[] = "Reads a PETSc matrix and vector from a file appends the vector the matrix\n\n";

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
#include <../src/mat/impls/aij/seq/aij.h>

#undef __FUNCT__
#define __FUNCT__ "PadMatrix"
PetscErrorCode PadMatrix(Mat A,Vec v,PetscScalar c,Mat *B)
{
  PetscErrorCode ierr;
  PetscInt       n = A->rmap->n,i,*cnt,*indices;
  Mat_SeqAIJ     *aij = (Mat_SeqAIJ*)A->data;
  PetscScalar    *vv;

  PetscFunctionBegin;
  ierr = VecGetArray(v,&vv);CHKERRQ(ierr);
  ierr = PetscMalloc(n*sizeof(PetscInt),&indices);CHKERRQ(ierr);
  for (i=0; i<n; i++) indices[i] = i;

  /* determine number of nonzeros per row in the new matrix */
  ierr = PetscMalloc((n+1)*sizeof(PetscInt),&cnt);CHKERRQ(ierr);
  for (i=0; i<n; i++){
    cnt[i] = aij->i[i+1] - aij->i[i] + (vv[i] != 0.0);
  }
  cnt[n] = 1;
  for (i=0; i<n; i++){
    cnt[n] += (vv[i] != 0.0);
  }
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,n+1,n+1,0,cnt,B);CHKERRQ(ierr);
  ierr = MatSetOption(*B,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);

  /* copy over the matrix entries from the matrix and then the vector */
  for (i=0; i<n; i++){
    ierr = MatSetValues(*B,1,&i,aij->i[i+1] - aij->i[i],aij->j + aij->i[i],aij->a + aij->i[i],INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatSetValues(*B,1,&n,n,indices,vv,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatSetValues(*B,n,indices,1,&n,vv,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatSetValues(*B,1,&n,1,&n,&c,INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecRestoreArray(v,&vv);CHKERRQ(ierr);
  ierr = PetscFree(cnt);CHKERRQ(ierr);
  ierr = PetscFree(indices);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat                   A,B;
  PetscViewer           fd;               /* viewer */
  char                  file[PETSC_MAX_PATH_LEN];     /* input file name */
  PetscErrorCode        ierr;
  PetscBool             flg;
  Vec                   v;

  PetscInitialize(&argc,&args,(char *)0,help);


  /*
     Determine files from which we read the two linear systems
     (matrix and right-hand-side vector).
  */
  ierr = PetscOptionsGetString(PETSC_NULL,"-f0",file,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate binary file with the -f0 option");

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

  ierr = PetscFinalize();
  return 0;
}

