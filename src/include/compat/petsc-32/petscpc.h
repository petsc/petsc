#define PCBDDC "bddc"

#undef __FUNCT__
#define __FUNCT__ "PCFieldSplitSetFields_Compat"
static PetscErrorCode PCFieldSplitSetFields_Compat(PC pc,const char splitname[],PetscInt n,const PetscInt *fields,const PetscInt *fields_col)
{
  PetscInt i;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidCharPointer(splitname,2);
  if (n > 0) PetscValidIntPointer(fields,4);
  if (n > 0) PetscValidIntPointer(fields_col,4);
  for (i=0; i<n; i++) 
    if (fields[i] != fields_col[i]) 
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,__FUNCT__"() columns not supported in this PETSc version");
  ierr = PCFieldSplitSetFields(pc,splitname,n,fields);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define PCFieldSplitSetFields PCFieldSplitSetFields_Compat
