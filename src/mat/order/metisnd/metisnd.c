
#include <petscmat.h>
#include <petsc/private/matorderimpl.h>
#include <metis.h>

/*
    MatGetOrdering_METISND - Find the nested dissection ordering of a given matrix.
*/
PETSC_INTERN PetscErrorCode MatGetOrdering_METISND(Mat mat,MatOrderingType type,IS *row,IS *col)
{
  PetscErrorCode ierr;
  PetscInt       i, nrow,*perm,*iperm;
  const PetscInt *ia,*ja;
  int            status, ival;
  PetscBool      done;
  Mat            B = NULL;
  int            options[METIS_NOPTIONS];

  PetscFunctionBegin;
  ierr = MatGetRowIJ(mat,1,PETSC_TRUE,PETSC_TRUE,&nrow,&ia,&ja,&done);CHKERRQ(ierr);
  if (!done) {
    ierr = MatConvert(mat,MATSEQAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
    ierr = MatGetRowIJ(B,1,PETSC_TRUE,PETSC_TRUE,&nrow,&ia,&ja,&done);CHKERRQ(ierr);
  }
  METIS_SetDefaultOptions(options);
  //options[METIS_OPTION_NUMBERING] = 0;
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)mat),((PetscObject)mat)->prefix,"METISND Options","Mat");CHKERRQ(ierr);

  ival = options[METIS_OPTION_UFACTOR];
  ierr = PetscOptionsInt("-mat_ordering_metisnd_ufactor","maximum unbalance factor","None",ival,&ival,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = PetscMalloc2(nrow,&perm,nrow,&iperm);CHKERRQ(ierr);
  status = METIS_NodeND(&nrow,(idx_t*)ia,(idx_t*)ja,NULL,options,perm,iperm);
  if (B) {
    ierr = MatRestoreRowIJ(B,1,PETSC_TRUE,PETSC_TRUE,NULL,&ia,&ja,&done);CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
  } else {
    ierr = MatRestoreRowIJ(mat,1,PETSC_TRUE,PETSC_TRUE,NULL,&ia,&ja,&done);CHKERRQ(ierr);
  }

  ierr = ISCreateGeneral(PETSC_COMM_SELF,nrow,perm,PETSC_COPY_VALUES,row);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,nrow,perm,PETSC_COPY_VALUES,col);CHKERRQ(ierr);
  ierr = PetscFree2(perm,iperm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

