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

  PetscCall(PetscInitialize(&argc,&args,NULL,help));
  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  for (i=0; i<3; i++) {
    PetscCall(KSPSetType(ksp,common[i]));
    PetscCall(KSPSetType(ksp,KSPHPDDM));
#if defined(PETSC_HAVE_HPDDM)
    PetscCall(KSPHPDDMGetType(ksp,&type));
    PetscCall(PetscStrcmp(KSPHPDDMTypes[type],common[i],&flg));
    PetscCheck(flg,PetscObjectComm((PetscObject)ksp),PETSC_ERR_PLIB,"KSPType and KSPHPDDMType do not match: %s != %s", common[i], type);
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(KSPHPDDMGetType(ksp,&type));
    PetscCheckFalse(type != KSP_HPDDM_TYPE_GCRODR,PetscObjectComm((PetscObject)ksp),PETSC_ERR_PLIB,"-ksp_hpddm_type gcrodr and KSPHPDDMType do not match: gcrodr != %s", KSPHPDDMTypes[type]);
    PetscCall(KSPHPDDMSetType(ksp,KSP_HPDDM_TYPE_BGMRES));
#endif
  }
  PetscCall(KSPDestroy(&ksp));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      requires: hpddm
      nsize: 1
      suffix: 1
      output_file: output/ex77_preonly.out
      args: -ksp_type hpddm -ksp_hpddm_type gcrodr

TEST*/
