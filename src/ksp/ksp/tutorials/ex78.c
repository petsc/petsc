#include <petsc.h>

static char help[] = "Exercises switching back and forth between different KSP and KSPHPDDM types.\n\n";

int main(int argc,char **args)
{
  KSP            ksp;
#if defined(PETSC_HAVE_HPDDM)
  KSPHPDDMType   type;
  PetscBool      flg;
#endif
  PetscInt       i;
  const char     *common[] = {KSPGMRES,KSPCG,KSPPREONLY};
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&args,NULL,help);if (ierr) return ierr;
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  for (i=0; i<3; i++) {
    ierr = KSPSetType(ksp,common[i]);CHKERRQ(ierr);
    ierr = KSPSetType(ksp,KSPHPDDM);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HPDDM)
    ierr = KSPHPDDMGetType(ksp,&type);CHKERRQ(ierr);
    ierr = PetscStrcmp(KSPHPDDMTypes[type],common[i],&flg);CHKERRQ(ierr);
    PetscCheckFalse(!flg,PetscObjectComm((PetscObject)ksp),PETSC_ERR_PLIB,"KSPType and KSPHPDDMType do not match: %s != %s", common[i], type);
    ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
    ierr = KSPHPDDMGetType(ksp,&type);CHKERRQ(ierr);
    PetscCheckFalse(type != KSP_HPDDM_TYPE_GCRODR,PetscObjectComm((PetscObject)ksp),PETSC_ERR_PLIB,"-ksp_hpddm_type gcrodr and KSPHPDDMType do not match: gcrodr != %s", KSPHPDDMTypes[type]);
    ierr = KSPHPDDMSetType(ksp,KSP_HPDDM_TYPE_BGMRES);CHKERRQ(ierr);
#endif
  }
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      requires: hpddm
      nsize: 1
      suffix: 1
      output_file: output/ex77_preonly.out
      args: -ksp_type hpddm -ksp_hpddm_type gcrodr

TEST*/
