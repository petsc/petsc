

#ifndef lint
static char vcid[] = "$Id: zmat.c,v 1.21 1996/03/07 00:17:50 balay Exp balay $";
#endif

#include "zpetsc.h"
#include "mat.h"
#include "pinclude/petscfix.h"

#ifdef HAVE_FORTRAN_CAPS
#define matgetreorderingtypefromoptions_ MATGETREORDERINGTYPEFROMOPTIONS
#define matgetformatfromoptions_         MATGETFORMATFROMOPTIONS
#define matgetreordering_                MATGETREORDERING
#define matreorderingregisterall_        MATREORDERINGREGISTERALL
#define matdestroy_                      MATDESTROY
#define matcreatempiaij_                 MATCREATEMPIAIJ
#define matcreateseqaij_                 MATCREATESEQAIJ
#define matcreate_                       MATCREATE
#define matcreateshell_                  MATCREATESHELL
#define matshellsetmulttransadd_         MATSHELLSETMULTTRANSADD
#define matshellsetdestroy_              MATSHELLSETDESTROY
#define matshellsetmult_                 MATSHELLSETMULT
#define matreorderingregisterdestroy_    MATREORDERINGREGISTERDESTROY
#define matcreatempirowbs_               MATCREATEMPIROWBS
#define matcreateseqbdiag_               MATCREATESEQBDIAG
#define matcreatempibdiag_               MATCREATEMPIBDIAG
#define matcreateseqdense_               MATCREATESEQDENSE
#define matcreatempidense_               MATCREATEMPIDENSE
#define matconvert_                      MATCONVERT
#define matreorderingregister_           MATREORDERINGREGISTER
#define matgetsubmatrix_                 MATGETSUBMATRIX
#define matload_                         MATLOAD
#define mattranspose_                    MATTRANSPOSE
#define matgetarray_                     MATGETARRAY
#define matrestorearray_                 MATRESTOREARRAY
#define matgettype_                      MATGETTYPE
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
#define matgettype_                      matgettype
#define matgetformatfromoptions_         matgetformatfromoptions
#define matreorderingregisterall_        matreorderingregisterall
#define matdestroy_                      matdestroy
#define matcreatempiaij_                 matcreatempiaij
#define matcreateseqaij_                 matcreateseqaij
#define matcreate_                       matcreate
#define matshellsetmult_                 matshellsetmult
#define matcreateshell_                  matcreateshell
#define matshellsetmulttransadd_         matshellsetmulttransadd
#define matshellsetdestroy_              matshellsetdestroy
#define matreorderingregisterdestroy_    matreorderingregisterdestroy
#define matgetreordering_                matgetreordering
#define matcreatempirowbs_               matcreatempirowbs
#define matcreateseqbdiag_               matcreateseqbdiag
#define matcreatempibdiag_               matcreatempibdiag
#define matcreateseqdense_               matcreateseqdense
#define matcreatempidense_               matcreatempidense
#define matconvert_                      matconvert
#define matreorderingregister_           matreorderingregister
#define matgetsubmatrix_                 matgetsubmatrix
#define matload_                         matload
#define mattranspose_                    mattranspose
#define matgetarray_                     matgetarray
#define matrestorearray_                 matrestorearray
#endif

#if defined(__cplusplus)
extern "C" {
#endif


  /*
     this next one is TOTALLY wrong 
  */
void matgetreorderingtypefromoptions_(CHAR prefix,MatOrdering *type, 
                                      int *__ierr,int len )
{
  char *t;

  FIXCHAR(prefix,len,t);
  *__ierr = MatGetReorderingTypeFromOptions(t,type);
  FREECHAR(prefix,t);
}

void matgetformatfromoptions_(MPI_Comm comm,CHAR prefix,MatType *type,
                              int *set,int *__ierr,int len)
{
  char *t;
  FIXCHAR(prefix,len,t);
  *__ierr = MatGetFormatFromOptions(
	(MPI_Comm)MPIR_ToPointer_Comm( *(int*)(comm) ),t,type,set);
  FREECHAR(prefix,t);
}

void matgetarray_(Mat mat,Scalar *fa,int *ia, int *__ierr)
{
  Scalar *mm;
  *__ierr = MatGetArray((Mat)MPIR_ToPointer( *(int*)(mat) ),&mm);
  *ia = PetscScalarAddressToFortran(fa,mm);
}

void matrestorearray_(Mat mat,Scalar *fa,int *ia,int *__ierr)
{
  Mat    min = (Mat)MPIR_ToPointer( *(int*)(mat) );
  Scalar *lx = PetscScalarAddressFromFortran(fa,*ia);

  *__ierr = MatRestoreArray(min,&lx);
}

void mattranspose_(Mat mat,Mat *B, int *__ierr )
{
  Mat mm;
  if (FORTRANNULL(B)) B = PETSC_NULL;
  *__ierr = MatTranspose((Mat)MPIR_ToPointer( *(int*)(mat) ),&mm);
  *(int*) B = MPIR_FromPointer(mm);
}

void matload_(Viewer bview,MatType *outtype,Mat *newmat, int *__ierr )
{
  Mat mm;
  *__ierr = MatLoad((Viewer)MPIR_ToPointer( *(int*)(bview) ),*outtype,&mm);
  *(int*) newmat = MPIR_FromPointer(mm);
}

void matgetsubmatrix_(Mat mat,IS irow,IS icol,MatGetSubMatrixCall *scall,
                      Mat *submat, int *__ierr )
{
  Mat mm;
  *__ierr = MatGetSubMatrix(
	(Mat)MPIR_ToPointer( *(int*)(mat) ),
	(IS)MPIR_ToPointer( *(int*)(irow) ),
	(IS)MPIR_ToPointer( *(int*)(icol) ),*scall,&mm);
  *(int*) submat = MPIR_FromPointer(mm);
}

static void (*f5)(int*,int*,int*,int*,int*,int*);
static int ourorder(int* a,int* b,int* c,int *d, int* e)
{
  int ierr;
  (*f5)(a,b,c,d,e,&ierr); CHKERRQ(ierr);
  return 0;
}

void matreorderingregister_(MatOrdering *name,CHAR sname,PetscTruth *sym,int *shift,
            void (*order)(int*,int*,int*,int*,int*,int*),int *__ierr,int len)
{
  char *t;
  
  FIXCHAR(sname,len,t);
  f5 = order;
  *__ierr = MatReorderingRegister(*name,t,*sym,*shift,ourorder);
  FREECHAR(sname,t);
}			       


void matconvert_(Mat mat,MatType *newtype,Mat *M, int *__ierr )
{
  Mat mm;
  *__ierr = MatConvert((Mat)MPIR_ToPointer( *(int*)(mat) ),*newtype,&mm);
  *(int*) M = MPIR_FromPointer(mm);
}

void matcreateseqdense_(MPI_Comm comm,int *m,int *n,Scalar *data,Mat *newmat,int *__ierr )
{
  Mat mm;
  if (FORTRANNULL(data)) data = PETSC_NULL;
  *__ierr = MatCreateSeqDense((MPI_Comm)MPIR_ToPointer_Comm(*(int*)(comm)),
                              *m,*n,data,&mm);
  *(int*) newmat = MPIR_FromPointer(mm);
}

void matcreatempidense_(MPI_Comm comm,int *m,int *n,int *M,int *N,Scalar *data,Mat *newmat, int *__ierr ){
  Mat mm;
  if (FORTRANNULL(data)) data = PETSC_NULL;
*__ierr = MatCreateMPIDense(
	(MPI_Comm)MPIR_ToPointer_Comm( *(int*)(comm) ),*m,*n,*M,*N,data,&mm);
  *(int*) newmat = MPIR_FromPointer(mm);
}

/* Fortran ignores diagv */
void matcreatempibdiag_(MPI_Comm comm,int *m,int *M,int *N,int *nd,int *nb,
                        int *diag,Scalar **diagv,Mat *newmat, int *__ierr )
{
  Mat mm;
  *__ierr = MatCreateMPIBDiag((MPI_Comm)MPIR_ToPointer_Comm( *(int*)(comm) ),
                              *m,*M,*N,*nd,*nb,diag,PETSC_NULL,&mm);
  *(int*) newmat = MPIR_FromPointer(mm);
}

/* Fortran ignores diagv */
void matcreateseqbdiag_(MPI_Comm comm,int *m,int *n,int *nd,int *nb,
                        int *diag,Scalar **diagv,Mat *newmat, int *__ierr )
{
  Mat mm;
  *__ierr = MatCreateSeqBDiag(
    (MPI_Comm)MPIR_ToPointer_Comm( *(int*)(comm) ),*m,*n,*nd,*nb,diag,
    PETSC_NULL,&mm);
  *(int*) newmat = MPIR_FromPointer(mm);
}

/*  Fortran cannot pass in procinfo, hence ignored */
void matcreatempirowbs_(MPI_Comm comm,int *m,int *M,int *nz,int *nnz,
                       void *procinfo,Mat *newmat, int *__ierr )
{
  Mat mm;
  if (FORTRANNULL(nnz)) nnz = PETSC_NULL;
  *__ierr = MatCreateMPIRowbs((MPI_Comm)MPIR_ToPointer_Comm( *(int*)(comm) ),
                               *m,*M,*nz,nnz,PETSC_NULL,&mm);
  *(int*) newmat = MPIR_FromPointer(mm);
}

void matgetreordering_(Mat mat,MatOrdering *type,IS *rperm,IS *cperm, 
                       int *__ierr )
{
  IS i1,i2;
  *__ierr = MatGetReordering((Mat)MPIR_ToPointer(*(int*)(mat)),*type,&i1,&i2);
  *(int*) rperm = MPIR_FromPointer(i1);
  *(int*) cperm = MPIR_FromPointer(i2);
}

void matreorderingregisterdestroy_(int *__ierr)
{
  *__ierr = MatReorderingRegisterDestroy();
}


void matcreateshell_(MPI_Comm comm,int *m,int *n,void *ctx,Mat *mat, int *__ierr )
{
  Mat mm;
  *__ierr = MatCreateShell((MPI_Comm)MPIR_ToPointer_Comm(*(int*)(comm)),
                           *m,*n,ctx,&mm);
  *(int*) mat = MPIR_FromPointer(mm);
}

static int (*f2)(void*,int*,int*,int*);
static int ourmatshellmult(void *ctx,Vec x,Vec f)
{
  int ierr = 0, s2, s3;
  s2 = MPIR_FromPointer(x);
  s3 = MPIR_FromPointer(f);
  (*f2)(ctx,&s2,&s3,&ierr); CHKERRQ(ierr);
  MPIR_RmPointer(s2);
  MPIR_RmPointer(s3);
  return 0;
}
void matshellsetmult_(Mat mat,int (*mult)(void*,int*,int*,int*), int *__ierr )
{
  f2 = mult;
  *__ierr = MatShellSetMult(
	(Mat)MPIR_ToPointer( *(int*)(mat) ),ourmatshellmult);
}
static int (*f1)(void*,int*,int*,int*,int*);
static int ourmatshellmulttransadd(void *ctx,Vec x,Vec f,Vec y)
{
  int ierr = 0, s2, s3,s4;
  s2 = MPIR_FromPointer(x);
  s3 = MPIR_FromPointer(f);
  s4 = MPIR_FromPointer(y);
  (*f1)(ctx,&s2,&s3,&s4,&ierr); CHKERRQ(ierr);
  MPIR_RmPointer(s2);
  MPIR_RmPointer(s3);
  MPIR_RmPointer(s4);
  return 0;
}

void matshellsetmulttransadd_(Mat mat,int (*mult)(void*,int*,int*,int*,int*), 
                              int *__ierr )
{
  f1 = mult;
  *__ierr = MatShellSetMultTransAdd(
	(Mat)MPIR_ToPointer( *(int*)(mat) ),ourmatshellmulttransadd);
}

static int (*f3)(void*,int*);
static int ourmatshelldestroy(void *ctx)
{
  int ierr = 0;
  (*f3)(ctx,&ierr); CHKERRQ(ierr);
  return 0;
}
void matshellsetdestroy_(Mat mat,int (*destroy)(void*,int*), int *__ierr )
{
  f3 = destroy;
  *__ierr = MatShellSetDestroy((Mat)MPIR_ToPointer( *(int*)(mat) ),ourmatshelldestroy);
}

void matgettype_(Mat mm,MatType *type,CHAR name,int *__ierr,int len)
{
  char *tname;

  if (FORTRANNULL(type)) type = PETSC_NULL;
  *__ierr = MatGetType((Mat)MPIR_ToPointer(*(int*)mm),type,&tname);
#if defined(PARCH_t3d)
  {
  char *t = _fcdtocp(name); int len1 = _fcdlen(name);
  if (t != PETSC_NULL_CHAR_Fortran) PetscStrncpy(t,tname,len1);
  }
#else
  if (name != PETSC_NULL_CHAR_Fortran) PetscStrncpy(name,tname,len);
#endif
}

void matcreate_(MPI_Comm comm,int *m,int *n,Mat *V, int *__ierr )
{
  Mat mm;
  *__ierr = MatCreate((MPI_Comm)MPIR_ToPointer_Comm( *(int*)(comm)),*m,*n,&mm);
  *(int*) V = MPIR_FromPointer(mm);
}

void matcreateseqaij_(MPI_Comm comm,int *m,int *n,int *nz,
                           int *nnz,Mat *newmat, int *__ierr )
{
  Mat mm;
  if (FORTRANNULL(nnz)) nnz = PETSC_NULL;
  *__ierr = MatCreateSeqAIJ((MPI_Comm)MPIR_ToPointer_Comm(*(int*)(comm)),
                            *m,*n,*nz,nnz,&mm);
  *(int*) newmat = MPIR_FromPointer(mm);
}

void matdestroy_(Mat mat, int *__ierr )
{
  *__ierr = MatDestroy((Mat)MPIR_ToPointer( *(int*)(mat) ));
   MPIR_RmPointer(*(int*)(mat)); 
}

void matreorderingregisterall_(int *__ierr)
{
  *__ierr = MatReorderingRegisterAll();
}

void matcreatempiaij_(MPI_Comm comm,int *m,int *n,int *M,int *N,
         int *d_nz,int *d_nnz,int *o_nz,int *o_nnz,Mat *newmat, int *__ierr )
{
  Mat mm;
  if (FORTRANNULL(d_nnz)) d_nnz = PETSC_NULL;
  if (FORTRANNULL(o_nnz)) o_nnz = PETSC_NULL;
  *__ierr = MatCreateMPIAIJ((MPI_Comm)MPIR_ToPointer_Comm(*(int*)(comm)),
      *m,*n,*M,*N,*d_nz,d_nnz,*o_nz,o_nnz,&mm);
  *(int*)newmat = MPIR_FromPointer(mm);
}

#if defined(__cplusplus)
}
#endif
