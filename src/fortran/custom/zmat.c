/*$Id: zmat.c,v 1.74 1999/10/24 14:04:19 bsmith Exp bsmith $*/

#include "src/fortran/custom/zpetsc.h"
#include "mat.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define matsetvalue_                     MATSETVALUE
#define matgetrow_                       MATGETROW
#define matrestorerow_                   MATRESTOREROW
#define matgettypefromoptions_           MATGETTYPEFROMOPTIONS
#define matgetordering_                  MATGETORDERING
#define matdestroy_                      MATDESTROY
#define matcreatempiaij_                 MATCREATEMPIAIJ
#define matcreateseqaij_                 MATCREATESEQAIJ
#define matcreatempibaij_                MATCREATEMPIBAIJ
#define matcreateseqbaij_                MATCREATESEQBAIJ
#define matcreate_                       MATCREATE
#define matcreateshell_                  MATCREATESHELL
#define matorderingregisterdestroy_      MATORDERINGREGISTERDESTROY
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
#define matview_                         MATVIEW
#define matfdcoloringcreate_             MATFDCOLORINGCREATE
#define matfdcoloringdestroy_            MATFDCOLORINGDESTROY
#define matfdcoloringsetfunctionsnes_    MATFDCOLORINGSETFUNCTIONSNES
#define matfdcoloringsetfunctionts_      MATFDCOLORINGSETFUNCTIONTS
#define matcopy_                         MATCOPY
#define matgetsubmatrices_               MATGETSUBMATRICES
#define matgetcoloring_                  MATGETCOLORING
#define matpartitioningsettype_          MATPARTITIONINGSETTYPE
#define matduplicate_                    MATDUPLICATE
#define matzerorows_                     MATZEROROWS
#define matzerorowslocal_                MATZEROROWSLOCAL
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matpartitioningsettype_          matpartitioningsettype
#define matsetvalue_                     matsetvalue
#define matgetrow_                       matgetrow
#define matrestorerow_                   matrestorerow
#define matview_                         matview
#define matgetinfo_                      matgetinfo
#define matgettype_                      matgettype
#define matgettypefromoptions_           matgettypefromoptions
#define matdestroy_                      matdestroy
#define matcreatempiaij_                 matcreatempiaij
#define matcreateseqaij_                 matcreateseqaij
#define matcreatempibaij_                matcreatempibaij
#define matcreateseqbaij_                matcreateseqbaij
#define matcreate_                       matcreate
#define matcreateshell_                  matcreateshell
#define matorderingregisterdestroy_      matorderingregisterdestroy
#define matgetordering_                  matgetordering
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
#define matfdcoloringcreate_             matfdcoloringcreate
#define matfdcoloringdestroy_            matfdcoloringdestroy
#define matfdcoloringsetfunctionsnes_    matfdcoloringsetfunctionsnes
#define matfdcoloringsetfunctionts_      matfdcoloringsetfunctionts
#define matcopy_                         matcopy
#define matgetsubmatrices_               matgetsubmatrices
#define matgetcoloring_                  matgetcoloring
#define matduplicate_                    matduplicate
#define matzerorows_                     matzerorows
#define matzerorowslocal_                matzerorowslocal
#endif

EXTERN_C_BEGIN


void PETSC_STDCALL matpartitioningsettype_(MatPartitioning *part,CHAR type PETSC_MIXED_LEN(len),
                                           int *__ierr PETSC_END_LEN(len) )
{
  char *t;
  FIXCHAR(type,len,t);
  *__ierr = MatPartitioningSetType(*part,t);
  FREECHAR(type,t);
}

void PETSC_STDCALL matgetcoloring_(Mat *mat,CHAR type PETSC_MIXED_LEN(len),ISColoring *iscoloring,
                                   int *__ierr PETSC_END_LEN(len) )
{
  char *t;
  FIXCHAR(type,len,t);
  *__ierr = MatGetColoring(*mat,t,iscoloring);
  FREECHAR(type,t);
}

void PETSC_STDCALL matsetvalue_(Mat *mat,int *i,int *j,Scalar *va,InsertMode *mode)
{
  /* cannot use MatSetValue() here since that usesCHKERRQ() which has a return in it */
  MatSetValues(*mat,1,i,1,j,va,*mode);
}

void PETSC_STDCALL matfdcoloringcreate_(Mat *mat,ISColoring *iscoloring,MatFDColoring *color,int *__ierr)
{
  *__ierr = MatFDColoringCreate(*mat,*iscoloring,color);
}

/*
   This is a poor way of storing the column and value pointers 
  generated by MatGetRow() to be returned with MatRestoreRow()
  but there is not natural, good place else to store them. Hence
  Fortran programmers can only have one outstanding MatGetRows()
  at a time.
*/
static int    matgetrowactive = 0, *my_ocols = 0;
static Scalar *my_ovals = 0;

void PETSC_STDCALL matgetrow_(Mat *mat,int *row,int *ncols,int *cols,Scalar *vals,int *ierr)
{
  int    **oocols = &my_ocols;
  Scalar **oovals = &my_ovals;

  if (matgetrowactive) {
     PetscError(__LINE__,"MatGetRow_Fortran",__FILE__,__SDIR__,1,0,
               "Cannot have two MatGetRow() active simultaneously\n\
               call MatRestoreRow() before calling MatGetRow() a second time");
     *ierr = 1;
     return;
  }
  if (FORTRANNULLINTEGER(cols)) oocols = PETSC_NULL;
  if (FORTRANNULLSCALAR(vals))  oovals = PETSC_NULL;

  *ierr = MatGetRow(*mat,*row,ncols,oocols,oovals); 
  if (*ierr) return;

  if (oocols) { *ierr = PetscMemcpy(cols,my_ocols,(*ncols)*sizeof(int)); if (*ierr) return;}
  if (oovals) { *ierr = PetscMemcpy(vals,my_ovals,(*ncols)*sizeof(Scalar)); if (*ierr) return; }
  matgetrowactive = 1;
}

void PETSC_STDCALL matrestorerow_(Mat *mat,int *row,int *ncols,int *cols,Scalar *vals,int *ierr)
{
  int    **oocols = &my_ocols;
  Scalar **oovals = &my_ovals;
  if (!matgetrowactive) {
     PetscError(__LINE__,"MatRestoreRow_Fortran",__FILE__,__SDIR__,1,0,
               "Must call MatGetRow() first");
     *ierr = 1;
     return;
  }
  if (FORTRANNULLINTEGER(cols)) oocols = PETSC_NULL;
  if (FORTRANNULLSCALAR(vals))  oovals = PETSC_NULL;
  *ierr = MatRestoreRow(*mat,*row,ncols,oocols,oovals); 
  matgetrowactive = 0;
}

void PETSC_STDCALL matview_(Mat *mat,Viewer *vin, int *__ierr )
{
  Viewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *__ierr = MatView(*mat,v);
}

void PETSC_STDCALL matcopy_(Mat *A,Mat *B, MatStructure *str,int *__ierr )
{
  *__ierr = MatCopy(*A,*B,*str);
}

void PETSC_STDCALL matgetinfo_(Mat *mat,MatInfoType *flag,double *finfo,int *__ierr )
{
  MatInfo info;
  *__ierr = MatGetInfo(*mat,*flag,&info);
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

void PETSC_STDCALL matgettypefromoptions_(MPI_Comm *comm,CHAR prefix PETSC_MIXED_LEN(len),MatType *type,
                              PetscTruth *set,int *__ierr PETSC_END_LEN(len) )
{
  char *t;
  FIXCHAR(prefix,len,t);
  *__ierr = MatGetTypeFromOptions((MPI_Comm)PetscToPointerComm( *comm ),t,type,set);
  FREECHAR(prefix,t);
}


void PETSC_STDCALL matgetarray_(Mat *mat,Scalar *fa,long *ia, int *__ierr)
{
  Scalar *mm;
  int    m,n;

  *__ierr = MatGetArray(*mat,&mm); if (*__ierr) return;
  *__ierr = MatGetSize(*mat,&m,&n);  if (*__ierr) return;
  *__ierr = PetscScalarAddressToFortran((PetscObject)*mat,fa,mm,m*n,ia); if (*__ierr) return;
}

void PETSC_STDCALL matrestorearray_(Mat *mat,Scalar *fa,long *ia,int *__ierr)
{
  Scalar               *lx;
  int                  m,n;

  *__ierr = MatGetSize(*mat,&m,&n); if (*__ierr) return;
  *__ierr = PetscScalarAddressFromFortran((PetscObject)*mat,fa,*ia,m*n,&lx);if (*__ierr) return;
  *__ierr = MatRestoreArray(*mat,&lx);if (*__ierr) return;
}

void PETSC_STDCALL mattranspose_(Mat *mat,Mat *B, int *__ierr )
{
  if (FORTRANNULLINTEGER(B)) B = PETSC_NULL;
  *__ierr = MatTranspose(*mat,B);
}

void PETSC_STDCALL matload_(Viewer *viewer,MatType *outtype,Mat *newmat, int *__ierr )
{
  *__ierr = MatLoad(*viewer,*outtype,newmat);
}

void PETSC_STDCALL matconvert_(Mat *mat,MatType *newtype,Mat *M, int *__ierr )
{
  *__ierr = MatConvert(*mat,*newtype,M);
}

void PETSC_STDCALL matcreateseqdense_(MPI_Comm *comm,int *m,int *n,Scalar *data,Mat *newmat,int *__ierr )
{
  if (FORTRANNULLSCALAR(data)) data = PETSC_NULL;
  *__ierr = MatCreateSeqDense((MPI_Comm)PetscToPointerComm(*comm),*m,*n,data,newmat);
}

void PETSC_STDCALL matcreatempidense_(MPI_Comm *comm,int *m,int *n,int *M,int *N,Scalar *data,Mat *newmat,
                        int *__ierr )
{
  if (FORTRANNULLSCALAR(data)) data = PETSC_NULL;
  *__ierr = MatCreateMPIDense((MPI_Comm)PetscToPointerComm( *comm ),*m,*n,*M,*N,data,newmat);
}

/* Fortran ignores diagv */
void PETSC_STDCALL matcreatempibdiag_(MPI_Comm *comm,int *m,int *M,int *N,int *nd,int *bs,
                        int *diag,Scalar **diagv,Mat *newmat, int *__ierr )
{
  *__ierr = MatCreateMPIBDiag((MPI_Comm)PetscToPointerComm( *comm ),
                              *m,*M,*N,*nd,*bs,diag,PETSC_NULL,newmat);
}

/* Fortran ignores diagv */
void PETSC_STDCALL matcreateseqbdiag_(MPI_Comm *comm,int *m,int *n,int *nd,int *bs,
                        int *diag,Scalar **diagv,Mat *newmat, int *__ierr )
{
  *__ierr = MatCreateSeqBDiag((MPI_Comm)PetscToPointerComm( *comm ),*m,*n,*nd,*bs,diag,
                               PETSC_NULL,newmat);
}

#if defined(PETSC_HAVE_BLOCKSOLVE) && !defined(PETSC_USE_COMPLEX)
/*  Fortran cannot pass in procinfo, hence ignored */
void PETSC_STDCALL matcreatempirowbs_(MPI_Comm *comm,int *m,int *M,int *nz,int *nnz,
                       void *procinfo,Mat *newmat, int *__ierr )
{
  if (FORTRANNULLINTEGER(nnz)) nnz = PETSC_NULL;
  *__ierr = MatCreateMPIRowbs((MPI_Comm)PetscToPointerComm( *comm ),
                               *m,*M,*nz,nnz,PETSC_NULL,newmat);
}
#endif

void PETSC_STDCALL matgetordering_(Mat *mat,CHAR type PETSC_MIXED_LEN(len),IS *rperm,IS *cperm, 
                       int *__ierr PETSC_END_LEN(len) )
{
  char *t;
  FIXCHAR(type,len,t);
  *__ierr = MatGetOrdering(*mat,t,rperm,cperm);
  FREECHAR(type,t);
}

void PETSC_STDCALL matorderingregisterdestroy_(int *__ierr)
{
  *__ierr = MatOrderingRegisterDestroy();
}

void PETSC_STDCALL matgettype_(Mat *mm,MatType *type,CHAR name PETSC_MIXED_LEN(len),
                               int *__ierr PETSC_END_LEN(len) )
{
  char *tname;

  if (FORTRANNULLINTEGER(type)) type = PETSC_NULL;
  *__ierr = MatGetType(*mm,type,&tname);
#if defined(PETSC_USES_CPTOFCD)
  {
    char *t = _fcdtocp(name); int len1 = _fcdlen(name);
    if (t != PETSC_NULL_CHARACTER_Fortran) {
      *__ierr = PetscStrncpy(t,tname,len1);if (*__ierr) return;
    }
  }
#else
  if (name != PETSC_NULL_CHARACTER_Fortran) {
    *__ierr = PetscStrncpy(name,tname,len);if (*__ierr) return;
  }
#endif
}

void PETSC_STDCALL matcreate_(MPI_Comm *comm,int *m,int *n,int *M,int *N,Mat *V, int *__ierr )
{
  *__ierr = MatCreate((MPI_Comm)PetscToPointerComm( *comm),*m,*n,*M,*N,V);
}

void PETSC_STDCALL matcreateseqaij_(MPI_Comm *comm,int *m,int *n,int *nz,
                           int *nnz,Mat *newmat, int *__ierr )
{
  if (FORTRANNULLSCALAR(nnz)) {
    PetscError(__LINE__,"MatCreateSeqAIJ_Fortran",__FILE__,__SDIR__,1,0,
               "Cannot pass PETSC_NULL_SCALAR, use PETSC_NULL_INT");
    *__ierr = 1;
    return;
  }
  if (FORTRANNULLINTEGER(nnz)) nnz = PETSC_NULL;
  *__ierr = MatCreateSeqAIJ((MPI_Comm)PetscToPointerComm(*comm),*m,*n,*nz,nnz,newmat);
}

void PETSC_STDCALL matcreateseqbaij_(MPI_Comm *comm,int *bs,int *m,int *n,int *nz,
                           int *nnz,Mat *newmat, int *__ierr )
{
  if (FORTRANNULLSCALAR(nnz)) {
    PetscError(__LINE__,"MatCreateSeqBAIJ_Fortran",__FILE__,__SDIR__,1,0,
               "Cannot pass PETSC_NULL_SCALAR, use PETSC_NULL_INT");
    *__ierr = 1;
    return;
  }
  if (FORTRANNULLINTEGER(nnz)) nnz = PETSC_NULL;
  *__ierr = MatCreateSeqBAIJ((MPI_Comm)PetscToPointerComm(*comm),*bs,*m,*n,*nz,nnz,newmat);
}

void PETSC_STDCALL matfdcoloringdestroy_(MatFDColoring *mat, int *__ierr )
{
  *__ierr = MatFDColoringDestroy(*mat);
}

void PETSC_STDCALL matdestroy_(Mat *mat, int *__ierr )
{
  *__ierr = MatDestroy(*mat);
}

void PETSC_STDCALL matcreatempiaij_(MPI_Comm *comm,int *m,int *n,int *M,int *N,
         int *d_nz,int *d_nnz,int *o_nz,int *o_nnz,Mat *newmat, int *__ierr )
{
  if (FORTRANNULLSCALAR(d_nnz) || FORTRANNULLSCALAR(o_nnz)) {
    PetscError(__LINE__,"MatCreateMPIAIJ_Fortran",__FILE__,__SDIR__,1,0,
               "Cannot pass PETSC_NULL_SCALAR, use PETSC_NULL_INT");
    *__ierr = 1;
    return;
  }
  if (FORTRANNULLINTEGER(d_nnz)) d_nnz = PETSC_NULL;
  if (FORTRANNULLINTEGER(o_nnz)) o_nnz = PETSC_NULL;
  *__ierr = MatCreateMPIAIJ((MPI_Comm)PetscToPointerComm(*comm),
                             *m,*n,*M,*N,*d_nz,d_nnz,*o_nz,o_nnz,newmat);
}
void PETSC_STDCALL matcreatempibaij_(MPI_Comm *comm,int *bs,int *m,int *n,int *M,int *N,
         int *d_nz,int *d_nnz,int *o_nz,int *o_nnz,Mat *newmat, int *__ierr )
{
  if (FORTRANNULLSCALAR(d_nnz) || FORTRANNULLSCALAR(o_nnz)) {
    PetscError(__LINE__,"MatCreateMPIBAIJ_Fortran",__FILE__,__SDIR__,1,0,
               "Cannot pass PETSC_NULL_SCALAR, use PETSC_NULL_INT");
    *__ierr = 1;
    return;
  }
  if (FORTRANNULLINTEGER(d_nnz)) d_nnz = PETSC_NULL;
  if (FORTRANNULLINTEGER(o_nnz)) o_nnz = PETSC_NULL;
  *__ierr = MatCreateMPIBAIJ((MPI_Comm)PetscToPointerComm(*comm),
                             *bs,*m,*n,*M,*N,*d_nz,d_nnz,*o_nz,o_nnz,newmat);
}

/*
      The MatShell Matrix Vector product requires a C routine.
   This C routine then calls the corresponding Fortran routine that was
   set by the user.
*/
void PETSC_STDCALL matcreateshell_(MPI_Comm *comm,int *m,int *n,int *M,int *N,void *ctx,Mat *mat,int *__ierr)
{
  *__ierr = MatCreateShell((MPI_Comm)PetscToPointerComm(*comm),*m,*n,*M,*N,ctx,mat);
  if (*__ierr) return;
  ((PetscObject)*mat)->fortran_func_pointers = (void **) PetscMalloc(sizeof(void *));
  if (!((PetscObject)*mat)->fortran_func_pointers) {*__ierr = 1; return;}
}

static int ourmult(Mat mat, Vec x, Vec y)
{
  int              ierr = 0;
  (*(int (*)(Mat*,Vec*,Vec*,int*))(((PetscObject)mat)->fortran_func_pointers[0]))(&mat,&x,&y,&ierr);
  return ierr;
}

void PETSC_STDCALL matshellsetoperation_(Mat *mat,MatOperation *op,int (*f)(Mat*,Vec*,Vec*,int*), int *__ierr )
{
  if (*op == MATOP_MULT) {
    *__ierr = MatShellSetOperation(*mat,*op,(void *)ourmult);
    ((PetscObject)*mat)->fortran_func_pointers[0] = (void *) f;
  } else {
    PetscError(__LINE__,"MatShellSetOperation_Fortran",__FILE__,__SDIR__,1,0,
               "Cannot set that matrix operation");
    *__ierr = 1;
  }
}

#include "ts.h"
/*
        MatFDColoringSetFunction sticks the Fortran function into the fortran_func_pointers
    this function is then accessed by ourmatfdcoloringfunction()

   NOTE: FORTRAN USER CANNOT PUT IN A NEW J OR B currently.

   USER CAN HAVE ONLY ONE MatFDColoring in code Because there is no place to hang f7!
*/

static void (*f7)(TS*,double*,Vec*,Vec*,void*,int*);

static int ourmatfdcoloringfunctionts(TS ts,double t,Vec x,Vec y, void *ctx)
{
  int ierr = 0;
  (*f7)(&ts,&t,&x,&y,ctx,&ierr);
  return ierr;
}

void PETSC_STDCALL matfdcoloringsetfunctionts_(MatFDColoring *fd,void (*f)(TS*,double*,Vec*,Vec*,void*,int*),
                                 void *ctx, int *__ierr )
{
  f7 = f;
  *__ierr = MatFDColoringSetFunction(*fd,(int (*)(void))ourmatfdcoloringfunctionts,ctx);
}

static void (*f8)(SNES*,Vec*,Vec*,void*,int*);

static int ourmatfdcoloringfunctionsnes(SNES ts,Vec x,Vec y, void *ctx)
{
  int ierr = 0;
  (*f8)(&ts,&x,&y,ctx,&ierr);
  return ierr;
}


void PETSC_STDCALL matfdcoloringsetfunctionsnes_(MatFDColoring *fd,void (*f)(SNES*,Vec*,Vec*,void*,int*),
                                 void *ctx, int *__ierr )
{
  f8 = f;
  *__ierr = MatFDColoringSetFunction(*fd,(int (*)(void))ourmatfdcoloringfunctionsnes,ctx);
}

/*
    MatGetSubmatrices() is slightly different from C since the 
    Fortran provides the array to hold the submatrix objects, while in C that 
    array is allocated by the MatGetSubmatrices()
*/
void PETSC_STDCALL matgetsubmatrices_(Mat *mat, int *n, IS *isrow, IS *iscol,MatReuse *scall,
                        Mat *smat, int *__ierr)
{
  Mat *lsmat;
  int i;

  if (*scall == MAT_INITIAL_MATRIX) {
    *__ierr = MatGetSubMatrices(*mat,*n,isrow,iscol,*scall,&lsmat);
    for (i=0; i<*n; i++) {
      smat[i] = lsmat[i];
    }
    PetscFree(lsmat); 
  } else {
    *__ierr = MatGetSubMatrices(*mat,*n,isrow,iscol,*scall,&smat);
  }
}

void PETSC_STDCALL matduplicate_(Mat *matin,MatDuplicateOption *op,Mat *matout, int *__ierr )
{
  *__ierr = MatDuplicate(*matin,*op,matout);
}

void PETSC_STDCALL matzerorows_(Mat *mat,IS *is,Scalar *diag, int *__ierr )
{
  if (FORTRANNULLSCALAR(diag))  diag = PETSC_NULL;
  *__ierr = MatZeroRows(*mat,*is,diag);
}

void PETSC_STDCALL matzerorowslocal_(Mat *mat,IS *is,Scalar *diag, int *__ierr )
{
  if (FORTRANNULLSCALAR(diag))  diag = PETSC_NULL;
  *__ierr = MatZeroRowsLocal(*mat,*is,diag);
}

EXTERN_C_END

