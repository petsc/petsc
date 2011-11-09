#define MatSetNullSpace MatNullSpaceAttach
#define MatTransposeMatMult MatMatMultTranspose

#undef __FUNCT__
#define __FUNCT__ "MatMatTransposeMult"
static PetscErrorCode MatMatTransposeMult(Mat A,Mat B,MatReuse scall,PetscReal fill,Mat *C)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidHeaderSpecific(B,MAT_CLASSID,2);
  PetscValidType(B,2);
  PetscValidPointer(C,3);
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  PetscFunctionReturn(PETSC_ERR_SUP);
}

