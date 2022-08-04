static const char help[] = "Uses KSPComputeRitz() on a matrix loaded from disk\n";

#include <petscksp.h>

int main(int argc,char **argv)
{
  Mat         A;
  KSP         ksp;
  char        file[PETSC_MAX_PATH_LEN];
  PetscReal   *tetar,*tetai;
  Vec         b,x,*S;
  PetscInt    i,N = 10,Na = N;
  PetscViewer fd;
  PC          pc;
  PetscBool   harmonic = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,0,help));

  PetscCall(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),NULL));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatLoad(A,fd));
  PetscCall(PetscViewerDestroy(&fd));
  PetscCall(MatView(A,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatCreateVecs(A,&b,&x));
  PetscCall(VecSetRandom(b,NULL));
  PetscCall(VecDuplicateVecs(b,N,&S));
  PetscCall(PetscMalloc2(N,&tetar,N,&tetai));

  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  PetscCall(KSPSetType(ksp,KSPGMRES));
  PetscCall(KSPSetTolerances(ksp,10000*PETSC_MACHINE_EPSILON,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT));
  PetscCall(KSPGetPC(ksp,&pc));
  PetscCall(PCSetType(pc,PCNONE));
  PetscCall(KSPSetComputeRitz(ksp,PETSC_TRUE));
  PetscCall(KSPSetOperators(ksp,A,A));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp,b,x));
  PetscCall(PetscOptionsHasName(NULL,NULL,"-harmonic",&harmonic));
  PetscCall(KSPComputeRitz(ksp,harmonic ? PETSC_FALSE : PETSC_TRUE,PETSC_TRUE,&Na,S,tetar,tetai));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"%% Number of Ritz pairs %" PetscInt_FMT "\n",Na));
  for (i=0; i<Na; i++) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"%% Eigenvalue(s)  %g ",(double)tetar[i]));
    if (tetai[i]) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"%+gi",(double)tetai[i]));
#if !defined(PETSC_USE_COMPLEX)
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  %g ",(double)tetar[i]));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"%+gi",(double)-tetai[i]));
#endif
    }
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"%% Eigenvector\n"));
    PetscCall(VecView(S[i],PETSC_VIEWER_STDOUT_WORLD));
#if !defined(PETSC_USE_COMPLEX)
    if (tetai[i]) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"%% Imaginary part of Eigenvector\n"));
      PetscCall(VecView(S[i+1],PETSC_VIEWER_STDOUT_WORLD));
      i++;
    }
#endif
  }

  PetscCall(PetscFree2(tetar,tetai));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroyVecs(N,&S));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

    test:
      requires: !defined(PETSC_USE_64BIT_INDICES) !complex double
      args: -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/ritz2 -ksp_monitor

    test:
      suffix: 2
      requires: !defined(PETSC_USE_64BIT_INDICES) !complex double
      args: -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/ritz5 -ksp_monitor

    test:
      suffix: harmonic
      requires: !defined(PETSC_USE_64BIT_INDICES) !complex double
      args: -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/ritz5 -ksp_monitor -harmonic

TEST*/
