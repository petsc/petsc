#define VECMPIPTHREAD VECPTHREAD

#undef  __FUNCT__
#define __FUNCT__ "VecCreateSeqWithArray_Compat"
PetscErrorCode VecCreateSeqWithArray_Compat(MPI_Comm comm,PetscInt bs,PetscInt n,const PetscScalar array[],Vec *V)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(V,5);
  ierr = VecCreateSeqWithArray(comm,n,array,V);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*V,bs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define VecCreateSeqWithArray VecCreateSeqWithArray_Compat

#undef  __FUNCT__
#define __FUNCT__ "VecCreateMPIWithArray_Compat"
PetscErrorCode VecCreateMPIWithArray_Compat(MPI_Comm comm,PetscInt bs,PetscInt n,PetscInt N,const PetscScalar array[],Vec *V)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(V,5);
  ierr = VecCreateMPIWithArray(comm,n,N,array,V);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*V,bs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define VecCreateMPIWithArray VecCreateMPIWithArray_Compat
