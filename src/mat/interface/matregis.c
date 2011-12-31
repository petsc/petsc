
#include <petscmat.h>  /*I "petscmat.h" I*/

EXTERN_C_BEGIN
extern PetscErrorCode  MatCreate_MFFD(Mat);
extern PetscErrorCode  MatCreate_MAIJ(Mat);
extern PetscErrorCode  MatCreate_IS(Mat);

extern PetscErrorCode  MatCreate_SeqAIJ(Mat);
extern PetscErrorCode  MatCreate_MPIAIJ(Mat);

#if defined(PETSC_HAVE_PTHREADCLASSES)
extern PetscErrorCode  MatCreate_SeqAIJPThread(Mat);
extern PetscErrorCode  MatCreate_AIJPThread(Mat);
#endif

extern PetscErrorCode  MatCreate_SeqBAIJ(Mat);
extern PetscErrorCode  MatCreate_MPIBAIJ(Mat);

extern PetscErrorCode  MatCreate_SeqSBAIJ(Mat);
extern PetscErrorCode  MatCreate_MPISBAIJ(Mat);

extern PetscErrorCode  MatCreate_SeqBSTRM(Mat);
extern PetscErrorCode  MatCreate_MPIBSTRM(Mat);

extern PetscErrorCode  MatCreate_SeqSBSTRM(Mat);
extern PetscErrorCode  MatCreate_MPISBSTRM(Mat);

extern PetscErrorCode  MatCreate_SeqDense(Mat);
extern PetscErrorCode  MatCreate_MPIDense(Mat);

extern PetscErrorCode  MatCreate_MPIAdj(Mat);
extern PetscErrorCode  MatCreate_Shell(Mat);
extern PetscErrorCode  MatCreate_Composite(Mat);

extern PetscErrorCode  MatCreate_SeqAIJPERM(Mat);
extern PetscErrorCode  MatCreate_MPIAIJPERM(Mat);

extern PetscErrorCode  MatCreate_SeqAIJCRL(Mat);
extern PetscErrorCode  MatCreate_MPIAIJCRL(Mat);

extern PetscErrorCode  MatCreate_Scatter(Mat);
extern PetscErrorCode  MatCreate_BlockMat(Mat);
extern PetscErrorCode  MatCreate_Nest(Mat);
extern PetscErrorCode  MatCreate_IJ(Mat);

#if defined PETSC_HAVE_CUSP
extern PetscErrorCode  MatCreate_SeqAIJCUSP(Mat);
extern PetscErrorCode  MatCreate_MPIAIJCUSP(Mat);
#endif

#if defined PETSC_HAVE_FFTW
extern PetscErrorCode  MatCreate_FFTW(Mat);
#endif
EXTERN_C_END
  
/*
    This is used by MatSetType() to make sure that at least one 
    MatRegisterAll() is called. In general, if there is more than one
    DLL, then MatRegisterAll() may be called several times.
*/
extern PetscBool  MatRegisterAllCalled;

#undef __FUNCT__  
#define __FUNCT__ "MatRegisterAll"
/*@C
  MatRegisterAll - Registers all of the matrix types in PETSc

  Not Collective

  Level: advanced

.keywords: KSP, register, all

.seealso:  MatRegisterDestroy()
@*/
PetscErrorCode  MatRegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  MatRegisterAllCalled = PETSC_TRUE;

  ierr = MatRegisterDynamic(MATMFFD,           path,"MatCreate_MFFD",    MatCreate_MFFD);CHKERRQ(ierr);

  ierr = MatRegisterDynamic(MATMPIMAIJ,        path,"MatCreate_MAIJ",    MatCreate_MAIJ);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSEQMAIJ,        path,"MatCreate_MAIJ",    MatCreate_MAIJ);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATMAIJ,           path,"MatCreate_MAIJ",    MatCreate_MAIJ);CHKERRQ(ierr);

  ierr = MatRegisterDynamic(MATIS,             path,"MatCreate_IS",      MatCreate_IS);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSHELL,          path,"MatCreate_Shell",   MatCreate_Shell);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATCOMPOSITE,      path,"MatCreate_Composite",   MatCreate_Composite);CHKERRQ(ierr);

  ierr = MatRegisterBaseName(MATAIJ,MATSEQAIJ,MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATMPIAIJ,         path,"MatCreate_MPIAIJ",      MatCreate_MPIAIJ);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSEQAIJ,         path,"MatCreate_SeqAIJ",      MatCreate_SeqAIJ);CHKERRQ(ierr);
#if defined(PETSC_HAVE_PTHREADCLASSES)
  ierr = MatRegisterBaseName(MATAIJPTHREAD,MATSEQAIJPTHREAD,0);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSEQAIJPTHREAD,  path,"MatCreate_SeqAIJPThread", MatCreate_SeqAIJPThread);CHKERRQ(ierr);
#endif

  ierr = MatRegisterBaseName(MATAIJPERM,MATSEQAIJPERM,MATMPIAIJPERM);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATMPIAIJPERM,     path,"MatCreate_MPIAIJPERM", MatCreate_MPIAIJPERM);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSEQAIJPERM,     path,"MatCreate_SeqAIJPERM", MatCreate_SeqAIJPERM);CHKERRQ(ierr);

  ierr = MatRegisterBaseName(MATAIJCRL,MATSEQAIJCRL,MATMPIAIJCRL);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSEQAIJCRL,      path,"MatCreate_SeqAIJCRL",  MatCreate_SeqAIJCRL);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATMPIAIJCRL,      path,"MatCreate_MPIAIJCRL",  MatCreate_MPIAIJCRL);CHKERRQ(ierr);

  ierr = MatRegisterBaseName(MATBAIJ,MATSEQBAIJ,MATMPIBAIJ);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATMPIBAIJ,        path,"MatCreate_MPIBAIJ",    MatCreate_MPIBAIJ);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSEQBAIJ,        path,"MatCreate_SeqBAIJ",    MatCreate_SeqBAIJ);CHKERRQ(ierr);

  ierr = MatRegisterBaseName(MATSBAIJ,MATSEQSBAIJ,MATMPISBAIJ);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATMPISBAIJ,       path,"MatCreate_MPISBAIJ",  MatCreate_MPISBAIJ);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSEQSBAIJ,       path,"MatCreate_SeqSBAIJ",  MatCreate_SeqSBAIJ);CHKERRQ(ierr);

  ierr = MatRegisterBaseName(MATBSTRM,MATSEQBSTRM,MATMPIBSTRM);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATMPIBSTRM,        path,"MatCreate_MPIBSTRM",    MatCreate_MPIBSTRM);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSEQBSTRM,        path,"MatCreate_SeqBSTRM",    MatCreate_SeqBSTRM);CHKERRQ(ierr);

  ierr = MatRegisterBaseName(MATSBSTRM,MATSEQSBSTRM,MATMPISBSTRM);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATMPISBSTRM,       path,"MatCreate_MPISBSTRM",  MatCreate_MPISBSTRM);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSEQSBSTRM,       path,"MatCreate_SeqSBSTRM",  MatCreate_SeqSBSTRM);CHKERRQ(ierr);

  ierr = MatRegisterBaseName(MATDENSE,MATSEQDENSE,MATMPIDENSE);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATMPIDENSE,       path,"MatCreate_MPIDense",  MatCreate_MPIDense);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSEQDENSE,       path,"MatCreate_SeqDense",  MatCreate_SeqDense);CHKERRQ(ierr);

  ierr = MatRegisterDynamic(MATMPIADJ,         path,"MatCreate_MPIAdj",    MatCreate_MPIAdj);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSCATTER,        path,"MatCreate_Scatter",   MatCreate_Scatter);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATBLOCKMAT,       path,"MatCreate_BlockMat",  MatCreate_BlockMat);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATNEST,           path,"MatCreate_Nest",      MatCreate_Nest);CHKERRQ(ierr);

  ierr = MatRegisterDynamic(MATIJ,             path,"MatCreate_IJ",   MatCreate_IJ);CHKERRQ(ierr);

#if defined PETSC_HAVE_CUSP
  ierr = MatRegisterBaseName(MATAIJCUSP,MATSEQAIJCUSP,MATMPIAIJCUSP);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATSEQAIJCUSP,     path,"MatCreate_SeqAIJCUSP",  MatCreate_SeqAIJCUSP);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATMPIAIJCUSP,     path,"MatCreate_MPIAIJCUSP",  MatCreate_MPIAIJCUSP);CHKERRQ(ierr);
#endif

#if defined PETSC_HAVE_FFTW
  ierr = MatRegisterDynamic(MATFFTW,           path,"MatCreate_FFTW",        MatCreate_FFTW);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}


