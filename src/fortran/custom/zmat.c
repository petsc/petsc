/*$Id: zmat.c,v 1.100 2001/08/07 03:05:11 balay Exp $*/

#include "src/mat/impls/adj/mpi/mpiadj.h"
#include "src/fortran/custom/zpetsc.h"
#include "petscmat.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define matsettype_                      MATSETTYPE
#define matmpiaijgetseqaij_              MATMPIAIJGETSEQAIJ
#define matmpibaijgetseqbaij_            MATMPIBAIJGETSEQBAIJ
#define matgetrowij_                     MATGETROWIJ
#define matrestorerowij_                 MATRESTOREROWIJ
#define matsetfromoptions_               MATSETFROMOPTIONS
#define matcreateseqaijwitharrays_       MATCREATESEQAIJWITHARRAYS
#define matpartitioningdestroy_          MATPARTITIONINGDESTROY
#define matsetvalue_                     MATSETVALUE
#define matsetvaluelocal_                MATSETVALUELOCAL
#define matgetrow_                       MATGETROW
#define matrestorerow_                   MATRESTOREROW
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
#define matpartitioningview_             MATPARTITIONINGVIEW
#define matpartitioningcreate_           MATPARTITIONINGCREATE
#define matpartitioningsetadjacency_     MATPARTITIONINGSETADJACENCY
#define matpartitioningapply_            MATPARTITIONINGAPPLY
#define matcreatempiadj_                 MATCREATEMPIADJ
#define matsetvaluesstencil_             MATSETVALUESSTENCIL
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matsettype_                      matsettype
#define matmpiaijgetseqaij_              matmpiaijgetseqaij
#define matmpibaijgetseqbaij_            matmpibaijgetseqbaij          
#define matrestorerowij_                 matrestorerowij
#define matgetrowij_                     matgetrowij
#define matcreateseqaijwitharrays_       matcreateseqaijwitharrays
#define matpartitioningdestroy_          matpartitioningdestroy
#define matpartitioningsettype_          matpartitioningsettype
#define matsetvalue_                     matsetvalue
#define matsetvaluelocal_                matsetvaluelocal
#define matgetrow_                       matgetrow
#define matrestorerow_                   matrestorerow
#define matview_                         matview
#define matgetinfo_                      matgetinfo
#define matgettype_                      matgettype
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
#define matpartitioningview_             matpartitioningview
#define matpartitioningcreate_           matpartitioningcreate
#define matpartitioningsetadjacency_     matpartitioningsetadjacency
#define matpartitioningapply_            matpartitioningapply            
#define matcreatempiadj_                 matcreatempiadj
#define matsetfromoptions_               matsetfromoptions
#define matsetvaluesstencil_             matsetvaluesstencil
#endif


EXTERN_C_BEGIN

void PETSC_STDCALL matsettype_(Mat *x,CHAR type_name PETSC_MIXED_LEN(len),int *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type_name,len,t);
  *ierr = MatSetType(*x,t);
  FREECHAR(type_name,t);
}

void PETSC_STDCALL matsetvaluesstencil_(Mat *mat,int *m,MatStencil *idxm,int *n,MatStencil *idxn,PetscScalar *v,InsertMode *addv,
                                        int *ierr)
{
  *ierr = MatSetValuesStencil(*mat,*m,idxm,*n,idxn,v,*addv);
}

void PETSC_STDCALL matmpiaijgetseqaij_(Mat *A,Mat *Ad,Mat *Ao,int *ic,long *iic,int *ierr)
{
  int *i;
  *ierr = MatMPIAIJGetSeqAIJ(*A,Ad,Ao,&i);if (*ierr) return;
  *iic  = PetscIntAddressToFortran(ic,i);
}

void PETSC_STDCALL matmpibaijgetseqbaij_(Mat *A,Mat *Ad,Mat *Ao,int *ic,long *iic,int *ierr)
{
  int *i;
  *ierr = MatMPIBAIJGetSeqBAIJ(*A,Ad,Ao,&i);if (*ierr) return;
  *iic  = PetscIntAddressToFortran(ic,i);
}

void PETSC_STDCALL matgetrowij_(Mat *B,int *shift,PetscTruth *sym,int *n,int *ia,long *iia,int *ja,long *jja,
                                PetscTruth *done,int *ierr)
{
  int *IA,*JA;
  *ierr = MatGetRowIJ(*B,*shift,*sym,n,&IA,&JA,done);if (*ierr) return;
  *iia  = PetscIntAddressToFortran(ia,IA);
  *jja  = PetscIntAddressToFortran(ja,JA);
}

void PETSC_STDCALL matrestorerowij_(Mat *B,int *shift,PetscTruth *sym,int *n,int *ia,long *iia,int *ja,long *jja,
                                    PetscTruth *done,int *ierr)
{
  int *IA = PetscIntAddressFromFortran(ia,*iia),*JA = PetscIntAddressFromFortran(ja,*jja);
  *ierr = MatRestoreRowIJ(*B,*shift,*sym,n,&IA,&JA,done);
}

void PETSC_STDCALL matsetfromoptions_(Mat *B,int *ierr)
{
  *ierr = MatSetFromOptions(*B);
}

void PETSC_STDCALL matcreateseqaijwitharrays_(MPI_Comm *comm,int *m,int *n,int *i,int *j,PetscScalar *a,Mat *mat,int *ierr)
{
  *ierr = MatCreateSeqAIJWithArrays((MPI_Comm)PetscToPointerComm(*comm),*m,*n,i,j,a,mat);
}

void PETSC_STDCALL matcreatempiadj_(MPI_Comm *comm,int *m,int *n,int *i,int *j,int *values,Mat *A,int *ierr)
{
  Mat_MPIAdj *adj;

  CHKFORTRANNULLINTEGER(values);
  *ierr = MatCreateMPIAdj((MPI_Comm)PetscToPointerComm(*comm),*m,*n,i,j,values,A);
  adj = (Mat_MPIAdj*)(*A)->data;
  adj->freeaij = PETSC_FALSE;
}

void PETSC_STDCALL matpartitioningdestroy_(MatPartitioning *part,int *ierr)
{
  *ierr = MatPartitioningDestroy(*part);
}

void PETSC_STDCALL matpartitioningcreate_(MPI_Comm *comm,MatPartitioning *part, int *ierr)
{
  *ierr = MatPartitioningCreate((MPI_Comm)PetscToPointerComm(*comm),part);
}

void PETSC_STDCALL matpartitioningapply_(MatPartitioning *part,IS *is,int *ierr)
{
  *ierr = MatPartitioningApply(*part,is);
}

void PETSC_STDCALL matpartitioningsetadjacency_(MatPartitioning *part,Mat *mat,int *ierr)
{
  *ierr = MatPartitioningSetAdjacency(*part,*mat);
}

void PETSC_STDCALL matpartitioningview_(MatPartitioning  *part,PetscViewer *viewer, int *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = MatPartitioningView(*part,v);
}

void PETSC_STDCALL matpartitioningsettype_(MatPartitioning *part,CHAR type PETSC_MIXED_LEN(len),
                                           int *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(type,len,t);
  *ierr = MatPartitioningSetType(*part,t);
  FREECHAR(type,t);
}

void PETSC_STDCALL matgetcoloring_(Mat *mat,CHAR type PETSC_MIXED_LEN(len),ISColoring *iscoloring,
                                   int *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(type,len,t);
  *ierr = MatGetColoring(*mat,t,iscoloring);
  FREECHAR(type,t);
}

void PETSC_STDCALL matsetvalue_(Mat *mat,int *i,int *j,PetscScalar *va,InsertMode *mode,int *ierr)
{
  /* cannot use MatSetValue() here since that usesCHKERRQ() which has a return in it */
  *ierr = MatSetValues(*mat,1,i,1,j,va,*mode);
}

void PETSC_STDCALL matsetvaluelocal_(Mat *mat,int *i,int *j,PetscScalar *va,InsertMode *mode,int *ierr)
{
  /* cannot use MatSetValueLocal() here since that usesCHKERRQ() which has a return in it */
  *ierr = MatSetValuesLocal(*mat,1,i,1,j,va,*mode);
}

void PETSC_STDCALL matfdcoloringcreate_(Mat *mat,ISColoring *iscoloring,MatFDColoring *color,int *ierr)
{
  *ierr = MatFDColoringCreate(*mat,*iscoloring,color);
}

/*
   This is a poor way of storing the column and value pointers 
  generated by MatGetRow() to be returned with MatRestoreRow()
  but there is not natural,good place else to store them. Hence
  Fortran programmers can only have one outstanding MatGetRows()
  at a time.
*/
static int    matgetrowactive = 0,*my_ocols = 0;
static PetscScalar *my_ovals = 0;

void PETSC_STDCALL matgetrow_(Mat *mat,int *row,int *ncols,int *cols,PetscScalar *vals,int *ierr)
{
  int         **oocols = &my_ocols;
  PetscScalar **oovals = &my_ovals;

  if (matgetrowactive) {
     PetscError(__LINE__,"MatGetRow_Fortran",__FILE__,__SDIR__,1,0,
               "Cannot have two MatGetRow() active simultaneously\n\
               call MatRestoreRow() before calling MatGetRow() a second time");
     *ierr = 1;
     return;
  }

  CHKFORTRANNULLINTEGER(cols); if (!cols) oocols = PETSC_NULL;
  CHKFORTRANNULLSCALAR(vals);  if (!vals) oovals = PETSC_NULL;

  *ierr = MatGetRow(*mat,*row,ncols,oocols,oovals); 
  if (*ierr) return;

  if (oocols) { *ierr = PetscMemcpy(cols,my_ocols,(*ncols)*sizeof(int)); if (*ierr) return;}
  if (oovals) { *ierr = PetscMemcpy(vals,my_ovals,(*ncols)*sizeof(PetscScalar)); if (*ierr) return; }
  matgetrowactive = 1;
}

void PETSC_STDCALL matrestorerow_(Mat *mat,int *row,int *ncols,int *cols,PetscScalar *vals,int *ierr)
{
  int    **oocols = &my_ocols;
  PetscScalar **oovals = &my_ovals;
  if (!matgetrowactive) {
     PetscError(__LINE__,"MatRestoreRow_Fortran",__FILE__,__SDIR__,1,0,
               "Must call MatGetRow() first");
     *ierr = 1;
     return;
  }
  CHKFORTRANNULLINTEGER(cols); if (!cols) oocols = PETSC_NULL;
  CHKFORTRANNULLSCALAR(vals);  if (!vals) oovals = PETSC_NULL;

  *ierr = MatRestoreRow(*mat,*row,ncols,oocols,oovals); 
  matgetrowactive = 0;
}

void PETSC_STDCALL matview_(Mat *mat,PetscViewer *vin,int *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = MatView(*mat,v);
}

void PETSC_STDCALL matcopy_(Mat *A,Mat *B,MatStructure *str,int *ierr)
{
  *ierr = MatCopy(*A,*B,*str);
}

void PETSC_STDCALL matgetinfo_(Mat *mat,MatInfoType *flag,double *finfo,int *ierr)
{
  *ierr = MatGetInfo(*mat,*flag,(MatInfo*)finfo);
}

void PETSC_STDCALL matgetarray_(Mat *mat,PetscScalar *fa,long *ia,int *ierr)
{
  PetscScalar *mm;
  int    m,n;

  *ierr = MatGetArray(*mat,&mm); if (*ierr) return;
  *ierr = MatGetSize(*mat,&m,&n);  if (*ierr) return;
  *ierr = PetscScalarAddressToFortran((PetscObject)*mat,fa,mm,m*n,ia); if (*ierr) return;
}

void PETSC_STDCALL matrestorearray_(Mat *mat,PetscScalar *fa,long *ia,int *ierr)
{
  PetscScalar          *lx;
  int                  m,n;

  *ierr = MatGetSize(*mat,&m,&n); if (*ierr) return;
  *ierr = PetscScalarAddressFromFortran((PetscObject)*mat,fa,*ia,m*n,&lx);if (*ierr) return;
  *ierr = MatRestoreArray(*mat,&lx);if (*ierr) return;
}

void PETSC_STDCALL mattranspose_(Mat *mat,Mat *B,int *ierr)
{
  CHKFORTRANNULLINTEGER(B);
  *ierr = MatTranspose(*mat,B);
}

void PETSC_STDCALL matload_(PetscViewer *viewer,CHAR outtype PETSC_MIXED_LEN(len),Mat *newmat,int *ierr PETSC_END_LEN(len))
{
  char *t;
  PetscViewer v;
  FIXCHAR(outtype,len,t);
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = MatLoad(v,t,newmat);
  FREECHAR(outtype,t);
}

void PETSC_STDCALL matconvert_(Mat *mat,CHAR outtype PETSC_MIXED_LEN(len),Mat *M,int *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(outtype,len,t);
  *ierr = MatConvert(*mat,t,M);
  FREECHAR(outtype,t);
}

void PETSC_STDCALL matcreateseqdense_(MPI_Comm *comm,int *m,int *n,PetscScalar *data,Mat *newmat,int *ierr)
{
  CHKFORTRANNULLSCALAR(data);
  *ierr = MatCreateSeqDense((MPI_Comm)PetscToPointerComm(*comm),*m,*n,data,newmat);
}

void PETSC_STDCALL matcreatempidense_(MPI_Comm *comm,int *m,int *n,int *M,int *N,PetscScalar *data,Mat *newmat,
                        int *ierr)
{
  CHKFORTRANNULLSCALAR(data);
  *ierr = MatCreateMPIDense((MPI_Comm)PetscToPointerComm(*comm),*m,*n,*M,*N,data,newmat);
}

/* Fortran ignores diagv */
void PETSC_STDCALL matcreatempibdiag_(MPI_Comm *comm,int *m,int *M,int *N,int *nd,int *bs,
                        int *diag,PetscScalar **diagv,Mat *newmat,int *ierr)
{
  *ierr = MatCreateMPIBDiag((MPI_Comm)PetscToPointerComm(*comm),
                              *m,*M,*N,*nd,*bs,diag,PETSC_NULL,newmat);
}

/* Fortran ignores diagv */
void PETSC_STDCALL matcreateseqbdiag_(MPI_Comm *comm,int *m,int *n,int *nd,int *bs,
                        int *diag,PetscScalar **diagv,Mat *newmat,int *ierr)
{
  *ierr = MatCreateSeqBDiag((MPI_Comm)PetscToPointerComm(*comm),*m,*n,*nd,*bs,diag,
                               PETSC_NULL,newmat);
}

#if defined(PETSC_HAVE_BLOCKSOLVE) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE)
/*  Fortran cannot pass in procinfo,hence ignored */
void PETSC_STDCALL matcreatempirowbs_(MPI_Comm *comm,int *m,int *M,int *nz,int *nnz,Mat *newmat,int *ierr)
{
  CHKFORTRANNULLINTEGER(nnz);
  *ierr = MatCreateMPIRowbs((MPI_Comm)PetscToPointerComm(*comm),*m,*M,*nz,nnz,newmat);
}
#endif

void PETSC_STDCALL matgetordering_(Mat *mat,CHAR type PETSC_MIXED_LEN(len),IS *rperm,IS *cperm,
                       int *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(type,len,t);
  *ierr = MatGetOrdering(*mat,t,rperm,cperm);
  FREECHAR(type,t);
}

void PETSC_STDCALL matorderingregisterdestroy_(int *ierr)
{
  *ierr = MatOrderingRegisterDestroy();
}

void PETSC_STDCALL matgettype_(Mat *mm,CHAR name PETSC_MIXED_LEN(len),int *ierr PETSC_END_LEN(len))
{
  char *tname;

  *ierr = MatGetType(*mm,&tname);
#if defined(PETSC_USES_CPTOFCD)
  {
    char *t = _fcdtocp(name); int len1 = _fcdlen(name);
    if (t != PETSC_NULL_CHARACTER_Fortran) {
      *ierr = PetscStrncpy(t,tname,len1);if (*ierr) return;
    }
  }
#else
  if (name != PETSC_NULL_CHARACTER_Fortran) {
    *ierr = PetscStrncpy(name,tname,len);if (*ierr) return;
  }
#endif
}

void PETSC_STDCALL matcreate_(MPI_Comm *comm,int *m,int *n,int *M,int *N,Mat *V,int *ierr)
{
  *ierr = MatCreate((MPI_Comm)PetscToPointerComm(*comm),*m,*n,*M,*N,V);
}

void PETSC_STDCALL matcreateseqaij_(MPI_Comm *comm,int *m,int *n,int *nz,
                           int *nnz,Mat *newmat,int *ierr)
{
  CHKFORTRANNULLINTEGER(nnz);
  *ierr = MatCreateSeqAIJ((MPI_Comm)PetscToPointerComm(*comm),*m,*n,*nz,nnz,newmat);
}

void PETSC_STDCALL matcreateseqbaij_(MPI_Comm *comm,int *bs,int *m,int *n,int *nz,
                           int *nnz,Mat *newmat,int *ierr)
{
  CHKFORTRANNULLINTEGER(nnz);
  *ierr = MatCreateSeqBAIJ((MPI_Comm)PetscToPointerComm(*comm),*bs,*m,*n,*nz,nnz,newmat);
}

void PETSC_STDCALL matfdcoloringdestroy_(MatFDColoring *mat,int *ierr)
{
  *ierr = MatFDColoringDestroy(*mat);
}

void PETSC_STDCALL matdestroy_(Mat *mat,int *ierr)
{
  *ierr = MatDestroy(*mat);
}

void PETSC_STDCALL matcreatempiaij_(MPI_Comm *comm,int *m,int *n,int *M,int *N,
         int *d_nz,int *d_nnz,int *o_nz,int *o_nnz,Mat *newmat,int *ierr)
{
  CHKFORTRANNULLINTEGER(d_nnz);
  CHKFORTRANNULLINTEGER(o_nnz);

  *ierr = MatCreateMPIAIJ((MPI_Comm)PetscToPointerComm(*comm),
                             *m,*n,*M,*N,*d_nz,d_nnz,*o_nz,o_nnz,newmat);
}
void PETSC_STDCALL matcreatempibaij_(MPI_Comm *comm,int *bs,int *m,int *n,int *M,int *N,
         int *d_nz,int *d_nnz,int *o_nz,int *o_nnz,Mat *newmat,int *ierr)
{
  CHKFORTRANNULLINTEGER(d_nnz);
  CHKFORTRANNULLINTEGER(o_nnz);
  *ierr = MatCreateMPIBAIJ((MPI_Comm)PetscToPointerComm(*comm),
                             *bs,*m,*n,*M,*N,*d_nz,d_nnz,*o_nz,o_nnz,newmat);
}


/*
      The MatShell Matrix Vector product requires a C routine.
   This C routine then calls the corresponding Fortran routine that was
   set by the user.
*/
void PETSC_STDCALL matcreateshell_(MPI_Comm *comm,int *m,int *n,int *M,int *N,void **ctx,Mat *mat,int *ierr)
{
  *ierr = MatCreateShell((MPI_Comm)PetscToPointerComm(*comm),*m,*n,*M,*N,*ctx,mat);
  if (*ierr) return;
  *ierr = PetscMalloc(4*sizeof(void *),&((PetscObject)*mat)->fortran_func_pointers);
}

static int ourmult(Mat mat,Vec x,Vec y)
{
  int              ierr = 0;
  (*(int (PETSC_STDCALL *)(Mat*,Vec*,Vec*,int*))(((PetscObject)mat)->fortran_func_pointers[0]))(&mat,&x,&y,&ierr);
  return ierr;
}

static int ourmulttranspose(Mat mat,Vec x,Vec y)
{
  int              ierr = 0;
  (*(int (PETSC_STDCALL *)(Mat*,Vec*,Vec*,int*))(((PetscObject)mat)->fortran_func_pointers[2]))(&mat,&x,&y,&ierr);
  return ierr;
}

static int ourmultadd(Mat mat,Vec x,Vec y,Vec z)
{
  int              ierr = 0;
  (*(int (PETSC_STDCALL *)(Mat*,Vec*,Vec*,Vec*,int*))(((PetscObject)mat)->fortran_func_pointers[1]))(&mat,&x,&y,&z,&ierr);
  return ierr;
}

static int ourmulttransposeadd(Mat mat,Vec x,Vec y,Vec z)
{
  int              ierr = 0;
  (*(int (PETSC_STDCALL *)(Mat*,Vec*,Vec*,Vec*,int*))(((PetscObject)mat)->fortran_func_pointers[3]))(&mat,&x,&y,&z,&ierr);
  return ierr;
}

void PETSC_STDCALL matshellsetoperation_(Mat *mat,MatOperation *op,int (PETSC_STDCALL *f)(Mat*,Vec*,Vec*,int*),int *ierr)
{
  if (*op == MATOP_MULT) {
    *ierr = MatShellSetOperation(*mat,*op,(void(*)(void))ourmult);
    ((PetscObject)*mat)->fortran_func_pointers[0] = (void(*)(void))f;
  } else if (*op == MATOP_MULT_TRANSPOSE) {
    *ierr = MatShellSetOperation(*mat,*op,(void(*)(void))ourmulttranspose);
    ((PetscObject)*mat)->fortran_func_pointers[2] = (void(*)(void))f;
  } else if (*op == MATOP_MULT_ADD) {
    *ierr = MatShellSetOperation(*mat,*op,(void(*)(void))ourmultadd);
    ((PetscObject)*mat)->fortran_func_pointers[1] = (void(*)(void))f;
  } else if (*op == MATOP_MULT_TRANSPOSE_ADD) {
    *ierr = MatShellSetOperation(*mat,*op,(void(*)(void))ourmulttransposeadd);
    ((PetscObject)*mat)->fortran_func_pointers[3] = (void(*)(void))f;
  } else {
    PetscError(__LINE__,"MatShellSetOperation_Fortran",__FILE__,__SDIR__,1,0,
               "Cannot set that matrix operation");
    *ierr = 1;
  }
}

#include "petscts.h"
/*
        MatFDColoringSetFunction sticks the Fortran function into the fortran_func_pointers
    this function is then accessed by ourmatfdcoloringfunction()

   NOTE: FORTRAN USER CANNOT PUT IN A NEW J OR B currently.

   USER CAN HAVE ONLY ONE MatFDColoring in code Because there is no place to hang f7!
*/

static void (PETSC_STDCALL *f7)(TS*,double*,Vec*,Vec*,void*,int*);

static int ourmatfdcoloringfunctionts(TS ts,double t,Vec x,Vec y,void *ctx)
{
  int ierr = 0;
  (*f7)(&ts,&t,&x,&y,ctx,&ierr);
  return ierr;
}

void PETSC_STDCALL matfdcoloringsetfunctionts_(MatFDColoring *fd,void (PETSC_STDCALL *f)(TS*,double*,Vec*,Vec*,void*,int*),
                                 void *ctx,int *ierr)
{
  f7 = f;
  *ierr = MatFDColoringSetFunction(*fd,(int (*)(void))ourmatfdcoloringfunctionts,ctx);
}

static void (PETSC_STDCALL *f8)(SNES*,Vec*,Vec*,void*,int*);

static int ourmatfdcoloringfunctionsnes(SNES ts,Vec x,Vec y,void *ctx)
{
  int ierr = 0;
  (*f8)(&ts,&x,&y,ctx,&ierr);
  return ierr;
}


void PETSC_STDCALL matfdcoloringsetfunctionsnes_(MatFDColoring *fd,void (PETSC_STDCALL *f)(SNES*,Vec*,Vec*,void*,int*),
                                 void *ctx,int *ierr)
{
  f8 = f;
  *ierr = MatFDColoringSetFunction(*fd,(int (*)(void))ourmatfdcoloringfunctionsnes,ctx);
}

/*
    MatGetSubmatrices() is slightly different from C since the 
    Fortran provides the array to hold the submatrix objects,while in C that 
    array is allocated by the MatGetSubmatrices()
*/
void PETSC_STDCALL matgetsubmatrices_(Mat *mat,int *n,IS *isrow,IS *iscol,MatReuse *scall,Mat *smat,int *ierr)
{
  Mat *lsmat;
  int i;

  if (*scall == MAT_INITIAL_MATRIX) {
    *ierr = MatGetSubMatrices(*mat,*n,isrow,iscol,*scall,&lsmat);
    for (i=0; i<*n; i++) {
      smat[i] = lsmat[i];
    }
    PetscFree(lsmat); 
  } else {
    *ierr = MatGetSubMatrices(*mat,*n,isrow,iscol,*scall,&smat);
  }
}

void PETSC_STDCALL matduplicate_(Mat *matin,MatDuplicateOption *op,Mat *matout,int *ierr)
{
  *ierr = MatDuplicate(*matin,*op,matout);
}

void PETSC_STDCALL matzerorows_(Mat *mat,IS *is,PetscScalar *diag,int *ierr)
{
  CHKFORTRANNULLSCALAR(diag);
  *ierr = MatZeroRows(*mat,*is,diag);
}

void PETSC_STDCALL matzerorowslocal_(Mat *mat,IS *is,PetscScalar *diag,int *ierr)
{
  CHKFORTRANNULLSCALAR(diag);
  *ierr = MatZeroRowsLocal(*mat,*is,diag);
}

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define matdiagonalscalelocal_           MATDIAGONALSCALELOCAL
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matdiagonalscalelocal_           matdiagonalscalelocal
#endif


void PETSC_STDCALL matdiagonalscalelocal_(Mat *A,Vec *scale, int *ierr )
{
  *ierr = MatDiagonalScaleLocal(*A,*scale);
}

EXTERN_C_END




