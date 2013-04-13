/* ------------------------------------------------------------------------- */

#if PETSC_VERSION_LE(3,3,0)
#define PetscSysInitializePackage()    PetscSysInitializePackage(0)
#define PetscViewerInitializePackage() PetscViewerInitializePackage(0)
#define PetscRandomInitializePackage() PetscRandomInitializePackage(0)
#define ISInitializePackage()          ISInitializePackage(0)
#define VecInitializePackage()         VecInitializePackage(0)
#define PFInitializePackage()          PFInitializePackage(0)
#define MatInitializePackage()         MatInitializePackage(0)
#define PCInitializePackage()          PCInitializePackage(0)
#define KSPInitializePackage()         KSPInitializePackage(0)
#define SNESInitializePackage()        SNESInitializePackage(0)
#define TSInitializePackage()          TSInitializePackage(0)
#define AOInitializePackage()          AOInitializePackage(0)
#define DMInitializePackage()          DMInitializePackage(0)
#endif

#undef  __FUNCT__
#define __FUNCT__ "PetscInitializePackageAll"
static PetscErrorCode PetscInitializePackageAll(void)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscSysInitializePackage();CHKERRQ(ierr);
  ierr = PetscViewerInitializePackage();CHKERRQ(ierr);
  ierr = PetscRandomInitializePackage();CHKERRQ(ierr);
  ierr = ISInitializePackage();CHKERRQ(ierr);
  ierr = VecInitializePackage();CHKERRQ(ierr);
  ierr = PFInitializePackage();CHKERRQ(ierr);
  ierr = MatInitializePackage();CHKERRQ(ierr);
  ierr = PCInitializePackage();CHKERRQ(ierr);
  ierr = KSPInitializePackage();CHKERRQ(ierr);
  ierr = SNESInitializePackage();CHKERRQ(ierr);
  ierr = TSInitializePackage();CHKERRQ(ierr);
  ierr = AOInitializePackage();CHKERRQ(ierr);
  ierr = DMInitializePackage();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef  __FUNCT__
#define __FUNCT__ "<petsc4py.PETSc>"

/* ------------------------------------------------------------------------- */

/*
  Local variables:
  c-basic-offset: 2
  indent-tabs-mode: nil
  End:
*/
