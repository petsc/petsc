
#include <petscmat.h>
#include <petsc/private/matorderimpl.h>
#include <metis.h>

/*
    MatGetOrdering_METISND - Find the nested dissection ordering of a given matrix.
*/
PETSC_INTERN PetscErrorCode MatGetOrdering_METISND(Mat mat,MatOrderingType type,IS *row,IS *col)
{
  PetscErrorCode ierr;
  PetscInt       i,j,iptr,ival,nrow,*xadj,*adjncy,*perm,*iperm;
  const PetscInt *ia,*ja;
  int            status;
  Mat            B = NULL;
  idx_t          options[METIS_NOPTIONS];
  PetscBool      done;

  PetscFunctionBegin;
  ierr = MatGetRowIJ(mat,0,PETSC_TRUE,PETSC_TRUE,&nrow,&ia,&ja,&done);CHKERRQ(ierr);
  if (!done) {
    ierr = MatConvert(mat,MATSEQAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
    ierr = MatGetRowIJ(B,0,PETSC_TRUE,PETSC_TRUE,&nrow,&ia,&ja,&done);CHKERRQ(ierr);
  }
  METIS_SetDefaultOptions(options);
  options[METIS_OPTION_NUMBERING] = 0;
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)mat),((PetscObject)mat)->prefix,"METISND Options","Mat");CHKERRQ(ierr);

  ival = (PetscInt)options[METIS_OPTION_NSEPS];
  ierr = PetscOptionsInt("-mat_ordering_metisnd_nseps","number of different separators per level","None",ival,&ival,NULL);CHKERRQ(ierr);
  options[METIS_OPTION_NSEPS] = (idx_t)ival;

  ival = (PetscInt)options[METIS_OPTION_NITER];
  ierr = PetscOptionsInt("-mat_ordering_metisnd_niter","number of refinement iterations","None",ival,&ival,NULL);CHKERRQ(ierr);
  options[METIS_OPTION_NITER] = (idx_t)ival;

  ival = (PetscInt)options[METIS_OPTION_UFACTOR];
  ierr = PetscOptionsInt("-mat_ordering_metisnd_ufactor","maximum allowed imbalance","None",ival,&ival,NULL);CHKERRQ(ierr);
  options[METIS_OPTION_UFACTOR] = (idx_t)ival;

  ival = (PetscInt)options[METIS_OPTION_PFACTOR];
  ierr = PetscOptionsInt("-mat_ordering_metisnd_pfactor","minimum degree of vertices that will be ordered last","None",ival,&ival,NULL);CHKERRQ(ierr);
  options[METIS_OPTION_PFACTOR] = (idx_t)ival;

  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = PetscMalloc4(nrow+1,&xadj,ia[nrow],&adjncy,nrow,&perm,nrow,&iperm);CHKERRQ(ierr);
  /* The adjacency list of a vertex should not contain the vertex itself.
  */
  iptr = 0;
  xadj[iptr] = 0;
  for(j=0; j<nrow; j++){
    for(i=ia[j]; i<ia[j+1]; i++){
       if(ja[i] != j)
          adjncy[iptr++] = ja[i];
    }
    xadj[j+1] = iptr;
  }

  status = METIS_NodeND(&nrow,(idx_t*)xadj,(idx_t*)adjncy,NULL,options,(idx_t*)perm,(idx_t*)iperm);
  switch (status) {
  case METIS_OK: break;
  case METIS_ERROR:
    SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_PLIB,"METIS returned with an unspecified error");
  case METIS_ERROR_INPUT:
    SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_PLIB,"METIS received an invalid input");
  case METIS_ERROR_MEMORY:
    SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_MEM,"METIS could not compute ordering");
  default:
    SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_LIB,"Unexpected return value");
  }

  if (B) {
    ierr = MatRestoreRowIJ(B,0,PETSC_TRUE,PETSC_TRUE,NULL,&ia,&ja,&done);CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
  } else {
    ierr = MatRestoreRowIJ(mat,0,PETSC_TRUE,PETSC_TRUE,NULL,&ia,&ja,&done);CHKERRQ(ierr);
  }

  ierr = ISCreateGeneral(PETSC_COMM_SELF,nrow,perm,PETSC_COPY_VALUES,row);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,nrow,perm,PETSC_COPY_VALUES,col);CHKERRQ(ierr);
  ierr = PetscFree4(xadj,adjncy,perm,iperm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

