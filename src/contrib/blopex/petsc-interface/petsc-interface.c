/* This code was developed by Merico Argentati, Andrew Knyazev, Ilya Lashuk and Evgueni Ovtchinnikov */

#include "petscsys.h"
#include "petscvec.h"
#include "petscmat.h"
#include <assert.h>
#include "petscblaslapack.h"
#include "interpreter.h"
#include "temp_multivector.h"

static PetscRandom LOBPCG_RandomContext = PETSC_NULL;

int PETSC_dpotrf_interface (char *uplo, int *n, double *a, int * lda, int *info)
{
   PetscBLASInt n_, lda_, info_;
   
/* we assume that "PetscScalar" is just double; we must abort if PETSc has been
   compiled for complex */   

   #ifdef PETSC_USE_COMPLEX
     SETERRQ(1,"dpotrf_interface: PETSC must be compiled without support for complex numbers");
   #endif

   /* type conversion */
   n_ = *n;
   lda_ = *lda;
   info_ = *info;
   
   LAPACKpotrf_(uplo, &n_, (PetscScalar*)a, &lda_, &info_);
   
   *info = info_;
   return 0;
}


int PETSC_dsygv_interface (int *itype, char *jobz, char *uplo, int *
                    n, double *a, int *lda, double *b, int *ldb,
                    double *w, double *work, int *lwork, int *info)
{
   PetscBLASInt itype_, n_, lda_, ldb_, lwork_, info_;

   #ifdef PETSC_USE_COMPLEX
     SETERRQ(1,"dsygv_interface: PETSC must be compiled without support for complex numbers");
   #endif

   itype_ = *itype;
   n_ = *n;
   lda_ = *lda;
   ldb_ = *ldb;
   lwork_ = *lwork;
   info_ = *info;
   

   LAPACKsygv_(&itype_, jobz, uplo, &n_, (PetscScalar*)a, &lda_,
      (PetscScalar*)b, &ldb_, (PetscScalar*)w, (PetscScalar*)work, &lwork_, &info_);
      
   *info = info_;
   return 0;

}
void *
PETSC_MimicVector( void *vvector )
{
	PetscErrorCode 	ierr;
	Vec temp;	

	ierr=VecDuplicate((Vec) vvector, &temp );
        assert (ierr==0);
	return ((void *)temp);
}

int
PETSC_DestroyVector( void *vvector )
{
   PetscErrorCode ierr;
   
   ierr=VecDestroy((Vec) vvector); CHKERRQ(ierr);
   return(0);
}

double
PETSC_InnerProd( void *x, void *y )
{
	PetscErrorCode     ierr;
	double             result;

	ierr=VecDot( (Vec)x, (Vec)y, &result);
        assert(ierr==0);
	return (result);
}

int
PETSC_CopyVector( void *x, void *y )
{
	PetscErrorCode	ierr;
	
	ierr = VecCopy( (Vec)x, (Vec)y ); CHKERRQ(ierr);
	return(0);
}

int
PETSC_ClearVector( void *x )
{
	PetscErrorCode	ierr;	
		
	ierr = VecSet((Vec)x, 0.0); CHKERRQ(ierr);
	return(0);
}
 
int
PETSC_SetRandomValues( void* v, int seed )
{
	PetscErrorCode ierr;

/* note: without previous call to LOBPCG_InitRandomContext LOBPCG_RandomContext will be null,
	and VecSetRandom will use internal petsc random context */
	
        ierr = VecSetRandom((Vec)v, LOBPCG_RandomContext); CHKERRQ(ierr);

	return(0);
}

int
PETSC_ScaleVector( double alpha, void *x)
{
	PetscErrorCode ierr;
	
	ierr = VecScale ((Vec)x, alpha); CHKERRQ(ierr);
	return(0);
}

int
PETSC_Axpy( double alpha,
                void   *x,
                void   *y )
{
	PetscErrorCode ierr;
	
	ierr = VecAXPY( (Vec)y, alpha, (Vec)x ); CHKERRQ(ierr);
	return(0);
}

int
LOBPCG_InitRandomContext(void)
{
	PetscErrorCode ierr;
  /* PetscScalar rnd_bound = 1.0; */
  
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&LOBPCG_RandomContext);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(LOBPCG_RandomContext);CHKERRQ(ierr); 
  
  ierr = PetscRandomSetInterval(LOBPCG_RandomContext,(PetscScalar)-1.0,(PetscScalar)1.0);
	CHKERRQ(ierr);
	return 0;
}

int 
LOBPCG_DestroyRandomContext(void)
{
	PetscErrorCode ierr;
  
	ierr = PetscRandomDestroy(LOBPCG_RandomContext); 
	CHKERRQ(ierr);
	return 0;
}

int
PETSCSetupInterpreter( mv_InterfaceInterpreter *i )
{

  i->CreateVector = PETSC_MimicVector;
  i->DestroyVector = PETSC_DestroyVector;
  i->InnerProd = PETSC_InnerProd;
  i->CopyVector = PETSC_CopyVector;
  i->ClearVector = PETSC_ClearVector;
  i->SetRandomValues = PETSC_SetRandomValues;
  i->ScaleVector = PETSC_ScaleVector;
  i->Axpy = PETSC_Axpy;

  /* Multivector part */

  i->CreateMultiVector = mv_TempMultiVectorCreateFromSampleVector;
  i->CopyCreateMultiVector = mv_TempMultiVectorCreateCopy;
  i->DestroyMultiVector = mv_TempMultiVectorDestroy;

  i->Width = mv_TempMultiVectorWidth;
  i->Height = mv_TempMultiVectorHeight;
  i->SetMask = mv_TempMultiVectorSetMask;
  i->CopyMultiVector = mv_TempMultiVectorCopy;
  i->ClearMultiVector = mv_TempMultiVectorClear;
  i->SetRandomVectors = mv_TempMultiVectorSetRandom;
  i->MultiInnerProd = mv_TempMultiVectorByMultiVector;
  i->MultiInnerProdDiag = mv_TempMultiVectorByMultiVectorDiag;
  i->MultiVecMat = mv_TempMultiVectorByMatrix;
  i->MultiVecMatDiag = mv_TempMultiVectorByDiagonal;
  i->MultiAxpy = mv_TempMultiVectorAxpy;
  i->MultiXapy = mv_TempMultiVectorXapy;
  i->Eval = mv_TempMultiVectorEval;

  return 0;
}
