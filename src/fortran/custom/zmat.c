

#ifndef lint
static char vcid[] = "$Id: zmat.c,v 1.27 1996/08/22 22:44:03 curfman Exp bsmith $";
#endif

#include "src/fortran/custom/zpetsc.h"
#include "mat.h"
#include "pinclude/petscfix.h"

#ifdef HAVE_FORTRAN_CAPS
#define matgetreorderingtypefromoptions_ MATGETREORDERINGTYPEFROMOPTIONS
#define matgettypefromoptions_           MATGETTYPEFROMOPTIONS
#define matgetreordering_                MATGETREORDERING
#define matreorderingregisterall_        MATREORDERINGREGISTERALL
#define matdestroy_                      MATDESTROY
#define matcreatempiaij_                 MATCREATEMPIAIJ
#define matcreateseqaij_                 MATCREATESEQAIJ
#define matcreate_                       MATCREATE
#define matcreateshell_                  MATCREATESHELL
#define matreorderingregisterdestroy_    MATREORDERINGREGISTERDESTROY
#define matcreatempirowbs_               MATCREATEMPIROWBS
#define matcreateseqbdiag_               MATCREATESEQBDIAG
#define matcreatempibdiag_               MATCREATEMPIBDIAG
#define matcreateseqdense_               MATCREATESEQDENSE
#define matcreatempidense_               MATCREATEMPIDENSE
#define matconvert_                      MATCONVERT
#define matload_                         MATLOAD
#define mattranspose_                    MATTRANSPOSE
#define matgetarray_                     MATGETARRAY
#define matrestorearray_                 MATRESTOREARRAY
#define matgettype_                      MATGETTYPE
#define matgetinfo_                      MATGETINFO
#define matshellsetoperation_            MATSHELLSETOPERATION
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
#define matgetinfo_                      matgetinfo
#define matgettype_                      matgettype
#define matgettypefromoptions_           matgettypefromoptions
#define matreorderingregisterall_        matreorderingregisterall
#define matdestroy_                      matdestroy
#define matcreatempiaij_                 matcreatempiaij
#define matcreateseqaij_                 matcreateseqaij
#define matcreate_                       matcreate
#define matcreateshell_                  matcreateshell
#define matreorderingregisterdestroy_    matreorderingregisterdestroy
#define matgetreordering_                matgetreordering
#define matcreatempirowbs_               matcreatempirowbs
#define matcreateseqbdiag_               matcreateseqbdiag
#define matcreatempibdiag_               matcreatempibdiag
#define matcreateseqdense_               matcreateseqdense
#define matcreatempidense_               matcreatempidense
#define matconvert_                      matconvert
#define matload_                         matload
#define mattranspose_                    mattranspose
#define matgetarray_                     matgetarray
#define matrestorearray_                 matrestorearray
#define matshellsetoperation_            matshellsetoperation
#endif

#if defined(__cplusplus)
extern "C" {
#endif

void matgetinfo_(Mat mat,MatInfoType *flag,double *finfo,int *__ierr ){
  MatInfo info;
  *__ierr = MatGetInfo(
	    (Mat)PetscToPointer( *(int*)(mat) ),*flag,&info);
  finfo[0]  = info.rows_global;
  finfo[1]  = info.columns_global;
  finfo[2]  = info.rows_local;
  finfo[3]  = info.columns_global;
  finfo[4]  = info.block_size;
  finfo[5]  = info.nz_allocated;
  finfo[6]  = info.nz_used;
  finfo[7]  = info.nz_unneeded;
  finfo[8]  = info.memory;
  finfo[9]  = info.assemblies;
  finfo[10] = info.mallocs;
  finfo[11] = info.fill_ratio_given;
  finfo[12] = info.fill_ratio_needed;
  finfo[13] = info.factor_mallocs;
}

  /*
     this next one is TOTALLY wrong 
  */
void matgetreorderingtypefromoptions_(CHAR prefix,MatReordering *type, 
                                      int *__ierr,int len )
{
  char *t;

  FIXCHAR(prefix,len,t);
  *__ierr = MatGetReorderingTypeFromOptions(t,type);
  FREECHAR(prefix,t);
}

void matgettypefromoptions_(MPI_Comm comm,CHAR prefix,MatType *type,
                              int *set,int *__ierr,int len)
{
  char *t;
  FIXCHAR(prefix,len,t);
  *__ierr = MatGetTypeFromOptions(
	(MPI_Comm)PetscToPointerComm( *(int*)(comm) ),t,type,set);
  FREECHAR(prefix,t);
}

void matgetarray_(Mat mat,Scalar *fa,int *ia, int *__ierr)
{
  Scalar *mm;
  *__ierr = MatGetArray((Mat)PetscToPointer( *(int*)(mat) ),&mm);
  *ia = PetscScalarAddressToFortran(fa,mm);
}

void matrestorearray_(Mat mat,Scalar *fa,int *ia,int *__ierr)
{
  Mat    min = (Mat)PetscToPointer( *(int*)(mat) );
  Scalar *lx = PetscScalarAddressFromFortran(fa,*ia);

  *__ierr = MatRestoreArray(min,&lx);
}

void mattranspose_(Mat mat,Mat *B, int *__ierr )
{
  Mat mm;
  if (FORTRANNULL(B)) B = PETSC_NULL;
  *__ierr = MatTranspose((Mat)PetscToPointer( *(int*)(mat) ),&mm);
  *(int*) B = PetscFromPointer(mm);
}

void matload_(Viewer viewer,MatType *outtype,Mat *newmat, int *__ierr )
{
  Mat mm;
  *__ierr = MatLoad((Viewer)PetscToPointer( *(int*)(viewer) ),*outtype,&mm);
  *(int*) newmat = PetscFromPointer(mm);
}

void matconvert_(Mat mat,MatType *newtype,Mat *M, int *__ierr )
{
  Mat mm;
  *__ierr = MatConvert((Mat)PetscToPointer( *(int*)(mat) ),*newtype,&mm);
  *(int*) M = PetscFromPointer(mm);
}

void matcreateseqdense_(MPI_Comm comm,int *m,int *n,Scalar *data,Mat *newmat,int *__ierr )
{
  Mat mm;
  if (FORTRANNULL(data)) data = PETSC_NULL;
  *__ierr = MatCreateSeqDense((MPI_Comm)PetscToPointerComm(*(int*)(comm)),
                              *m,*n,data,&mm);
  *(int*) newmat = PetscFromPointer(mm);
}

void matcreatempidense_(MPI_Comm comm,int *m,int *n,int *M,int *N,Scalar *data,Mat *newmat, int *__ierr ){
  Mat mm;
  if (FORTRANNULL(data)) data = PETSC_NULL;
*__ierr = MatCreateMPIDense(
	(MPI_Comm)PetscToPointerComm( *(int*)(comm) ),*m,*n,*M,*N,data,&mm);
  *(int*) newmat = PetscFromPointer(mm);
}

/* Fortran ignores diagv */
void matcreatempibdiag_(MPI_Comm comm,int *m,int *M,int *N,int *nd,int *bs,
                        int *diag,Scalar **diagv,Mat *newmat, int *__ierr )
{
  Mat mm;
  *__ierr = MatCreateMPIBDiag((MPI_Comm)PetscToPointerComm( *(int*)(comm) ),
                              *m,*M,*N,*nd,*bs,diag,PETSC_NULL,&mm);
  *(int*) newmat = PetscFromPointer(mm);
}

/* Fortran ignores diagv */
void matcreateseqbdiag_(MPI_Comm comm,int *m,int *n,int *nd,int *bs,
                        int *diag,Scalar **diagv,Mat *newmat, int *__ierr )
{
  Mat mm;
  *__ierr = MatCreateSeqBDiag(
    (MPI_Comm)PetscToPointerComm( *(int*)(comm) ),*m,*n,*nd,*bs,diag,
    PETSC_NULL,&mm);
  *(int*) newmat = PetscFromPointer(mm);
}

/*  Fortran cannot pass in procinfo, hence ignored */
void matcreatempirowbs_(MPI_Comm comm,int *m,int *M,int *nz,int *nnz,
                       void *procinfo,Mat *newmat, int *__ierr )
{
  Mat mm;
  if (FORTRANNULL(nnz)) nnz = PETSC_NULL;
  *__ierr = MatCreateMPIRowbs((MPI_Comm)PetscToPointerComm( *(int*)(comm) ),
                               *m,*M,*nz,nnz,PETSC_NULL,&mm);
  *(int*) newmat = PetscFromPointer(mm);
}

void matgetreordering_(Mat mat,MatReordering *type,IS *rperm,IS *cperm, 
                       int *__ierr )
{
  IS i1,i2;
  *__ierr = MatGetReordering((Mat)PetscToPointer(*(int*)(mat)),*type,&i1,&i2);
  *(int*) rperm = PetscFromPointer(i1);
  *(int*) cperm = PetscFromPointer(i2);
}

void matreorderingregisterdestroy_(int *__ierr)
{
  *__ierr = MatReorderingRegisterDestroy();
}


void matcreateshell_(MPI_Comm comm,int *m,int *n,int *M,int *N,void *ctx,Mat *mat, int *__ierr )
{
  Mat mm;
  *__ierr = MatCreateShell((MPI_Comm)PetscToPointerComm(*(int*)(comm)),
                           *m,*n,*M,*N,ctx,&mm);
  *(int*) mat = PetscFromPointer(mm);
}

void matgettype_(Mat mm,MatType *type,CHAR name,int *__ierr,int len)
{
  char *tname;

  if (FORTRANNULL(type)) type = PETSC_NULL;
  *__ierr = MatGetType((Mat)PetscToPointer(*(int*)mm),type,&tname);
#if defined(USES_CPTOFCD)
  {
  char *t = _fcdtocp(name); int len1 = _fcdlen(name);
  if (t != PETSC_NULL_CHARACTER_Fortran) PetscStrncpy(t,tname,len1);
  }
#else
  if (name != PETSC_NULL_CHARACTER_Fortran) PetscStrncpy(name,tname,len);
#endif
}

void matcreate_(MPI_Comm comm,int *m,int *n,Mat *V, int *__ierr )
{
  Mat mm;
  *__ierr = MatCreate((MPI_Comm)PetscToPointerComm( *(int*)(comm)),*m,*n,&mm);
  *(int*) V = PetscFromPointer(mm);
}

void matcreateseqaij_(MPI_Comm comm,int *m,int *n,int *nz,
                           int *nnz,Mat *newmat, int *__ierr )
{
  Mat mm;
  if (FORTRANNULL(nnz)) nnz = PETSC_NULL;
  *__ierr = MatCreateSeqAIJ((MPI_Comm)PetscToPointerComm(*(int*)(comm)),
                            *m,*n,*nz,nnz,&mm);
  *(int*) newmat = PetscFromPointer(mm);
}

void matdestroy_(Mat mat, int *__ierr )
{
  *__ierr = MatDestroy((Mat)PetscToPointer( *(int*)(mat) ));
   PetscRmPointer(*(int*)(mat)); 
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
  *__ierr = MatCreateMPIAIJ((MPI_Comm)PetscToPointerComm(*(int*)(comm)),
      *m,*n,*M,*N,*d_nz,d_nnz,*o_nz,o_nnz,&mm);
  *(int*)newmat = PetscFromPointer(mm);
}

static int (*theirmult)(int *,int *,int *,int*);

  /* call Fortran multiply function */
static int ourmult(Mat mat, Vec x, Vec y)
{
  int ierr,s1,s2,s3;
  s1 = PetscFromPointer(mat);
  s2 = PetscFromPointer(x);
  s3 = PetscFromPointer(y);
  (*theirmult)(&s1,&s2,&s3,&ierr);
  return ierr;
}

void matshellsetoperation_(Mat mat,MatOperation *op,int (*f)(int*,int*,int*,int*), int *__ierr )
{
  if (*op == MATOP_MULT) {
    *__ierr = MatShellSetOperation((Mat)PetscToPointer(*(int*)(mat)),*op,
                                   (void*) ourmult);
    theirmult = f;
  } else {
    PetscError(__LINE__,__DIR__,__FILE__,1,"Cannot set that matrix operation");
    *__ierr = 0;
  }
}

#if defined(__cplusplus)
}
#endif
