#include <petsc-private/fortranimpl.h>
#include <petsc-private/daimpl.h>
#include <petscsnes.h>
#include <petscdmmg.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmmgsetsnes_                     DMMGSETSNES
#define snesgetsolutionupdate_           SNESGETSOLUTIONUPDATE
#define dmmggetsnes_                     DMMGGETSNES
#define dmmgsetfromoptions_              DMMGSETFROMOPTIONS
#define dmmgsetsneslocal_                DMMGSETSNESLOCAL
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmmgsetsnes_                     dmmgsetsnes
#define snesgetsolutionupdate_           snesgetsolutionupdate
#define dmmggetsnes_                     dmmggetsnes
#define dmmgsetfromoptions_              dmmgsetfromoptions
#define dmmgsetsneslocal_                dmmgsetsneslocal
#endif

EXTERN_C_BEGIN

static PetscErrorCode ourrhs(SNES snes,Vec vec,Vec vec2,void*ctx)
{
  PetscErrorCode ierr = 0;
  DMMG dmmg = (DMMG)ctx;
  (*(void (PETSC_STDCALL *)(SNES*,Vec*,Vec*,void *,PetscErrorCode*))(((PetscObject)(dmmg)->dm)->fortran_func_pointers[0]))(&snes,&vec,&vec2,&ctx,&ierr);
  return ierr;
}

void PETSC_STDCALL dmmgsetsnes_(DMMG **dmmg,void (PETSC_STDCALL *rhs)(SNES*,Vec*,Vec*,void*,PetscErrorCode*),void (PETSC_STDCALL *mat)(DMMG*,Mat*,PetscErrorCode*),PetscErrorCode *ierr)
{
  PetscInt i;
  *ierr = DMMGSetSNES(*dmmg,ourrhs,PETSC_NULL); if (*ierr) return;
  /*
    Save the fortran rhs function in the DM on each level; ourrhs() pulls it out when needed
  */
  for (i=0; i<(**dmmg)->nlevels; i++) {
    ((PetscObject)(*dmmg)[i]->dm)->fortran_func_pointers[0] = (PetscVoidFunction)rhs;
    ((PetscObject)(*dmmg)[i]->dm)->fortran_func_pointers[1] = (PetscVoidFunction)mat;
  }
}

#undef __FUNCT__
#define __FUNCT__ "DMMGFormFunctionFortran"
PetscErrorCode DMMGFormFunctionFortran(SNES snes,Vec X,Vec F,void *ptr)
{
  DMMG           dmmg = (DMMG)ptr;
  PetscErrorCode ierr;
  Vec            localX;
  DM             da = dmmg->dm;
  DMDALocalInfo    info;
  PetscScalar     *xx,*ff;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  ierr = DMGetLocalVector(da,&localX);CHKERRQ(ierr);
  /*
     Scatter ghost points to local vector, using the 2-step process
        DMGlobalToLocalBegin(), DMGlobalToLocalEnd().
  */
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = VecGetArray(localX,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(F,&ff);CHKERRQ(ierr);
  CHKMEMQ;
  (*(void (PETSC_STDCALL *)(DMDALocalInfo*,void*,void*,void*,PetscErrorCode*))(dd->lf))(&info,xx,ff,dmmg->user,&ierr);CHKERRQ(ierr);
  CHKMEMQ;
  ierr = VecRestoreArray(localX,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&ff);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localX);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
} 

#undef __FUNCT__
#define __FUNCT__ "DMMGComputeJacobianFortran"
PetscErrorCode DMMGComputeJacobianFortran(SNES snes,Vec X,Mat *J,Mat *B,MatStructure *flag,void *ptr)
{
  DMMG           dmmg = (DMMG) ptr;
  PetscErrorCode ierr;
  Vec            localX;
  DM             da =  dmmg->dm;
  PetscScalar    *xx;
  DMDALocalInfo    info;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  ierr = DMGetLocalVector(da,&localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = VecGetArray(localX,&xx);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  CHKMEMQ;
  (*(void (PETSC_STDCALL *)(DMDALocalInfo*,void*,Mat*,void*,PetscErrorCode*))(dd->lj))(&info,xx,B,dmmg->user,&ierr);CHKERRQ(ierr);
  CHKMEMQ;
  ierr = VecRestoreArray(localX,&xx);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localX);CHKERRQ(ierr);
  /* Assemble true Jacobian; if it is different */
  if (*J != *B) {
    ierr  = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr  = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr  = MatSetOption(*B,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  *flag = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}


void PETSC_STDCALL dmmgsetsneslocal_(DMMG **dmmg,void (PETSC_STDCALL *rhs)(SNES*,Vec*,Vec*,void*,PetscErrorCode*),void (PETSC_STDCALL *mat)(DMMG*,Mat*,PetscErrorCode*),void* dummy1,void* dummy2,PetscErrorCode *ierr)
{
  PetscInt       i,nlevels = (*dmmg)[0]->nlevels;
  PetscErrorCode (*computejacobian)(SNES,Vec,Mat*,Mat*,MatStructure*,void*) = 0;

  if (!FORTRANNULLFUNCTION(mat)) computejacobian = DMMGComputeJacobianFortran;

  *ierr = DMMGSetSNES(*dmmg,DMMGFormFunctionFortran,computejacobian);if (*ierr) return;

  for (i=0; i<nlevels; i++) {
    *ierr = DMDASetLocalFunction((*dmmg)[i]->dm,(DMDALocalFunction1) (void*)rhs);if (*ierr) return;
    *ierr = DMDASetLocalJacobian((*dmmg)[i]->dm,(DMDALocalFunction1) (void*)mat);if (*ierr) return;
  }
}

void PETSC_STDCALL dmmggetsnes_(DMMG **dmmg,SNES *snes,PetscErrorCode *ierr)
{
  *snes = DMMGGetSNES(*dmmg);
}

void PETSC_STDCALL dmmgsetfromoptions_(DMMG **dmmg,PetscErrorCode *ierr)
{
  *ierr = DMMGSetFromOptions(*dmmg);
}

EXTERN_C_END
