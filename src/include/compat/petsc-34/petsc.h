#define PETSCVIEWERSAWS "saws"

#undef  __FUNCT__
#define __FUNCT__ "PetscSynchronizedFlush_Compat"
PetscErrorCode PetscSynchronizedFlush_Compat(MPI_Comm comm,FILE *fd)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (fd != PETSC_STDOUT) goto bad;
  ierr = PetscSynchronizedFlush(comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
 bad:
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  PetscFunctionReturn(PETSC_ERR_SUP);
}
#define PetscSynchronizedFlush PetscSynchronizedFlush_Compat
