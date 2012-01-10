#define PetscShell                        PetscFwk
#define PETSC_SHELL_CLASSID               PETSC_FWK_CLASSID
#define PetscShellInitializePackage       PetscFwkInitializePackage
#define PetscShellView                    PetscFwkView
#define PetscShellDestroy                 PetscFwkDestroy
#define PetscShellCreate                  PetscFwkCreate
#define PetscShellGetURL                  PetscFwkGetURL
#define PetscShellSetURL                  PetscFwkSetURL
#define PetscShellGetComponent            PetscFwkGetComponent
#define PetscShellRegisterDependence      PetscFwkRegisterDependence
#define PetscShellCall                    PetscFwkCall
#define PetscShellGetVisitor              PetscFwkGetParent
#define PetscShellVisit                   PetscFwkVisit
#define PETSC_SHELL_DEFAULT_              PETSC_FWK_DEFAULT_

#undef __FUNCT__
#define __FUNCT__ "PetscShellRegisterComponentShell"
static PetscErrorCode
PetscShellRegisterComponentShell(PetscShell shell,const char *key,PetscShell component)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(shell,PETSC_SHELL_CLASSID,1);
  PetscValidCharPointer(key,2);
  if (component) {
    PetscValidHeaderSpecific(component,PETSC_SHELL_CLASSID,3);
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
    PetscFunctionReturn(PETSC_ERR_SUP);
  }
  ierr = PetscFwkRegisterComponent(shell,key);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
