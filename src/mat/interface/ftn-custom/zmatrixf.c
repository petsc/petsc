#include <petsc/private/fortranimpl.h>
#include <petsc/private/f90impl.h>
#include <petscmat.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matsetvalues_                    MATSETVALUES
#define matsetvaluesnnnn_                MATSETVALUESNNNN
#define matsetvalues0_                   MATSETVALUES0
#define matsetvaluesnn1_                 MATSETVALUESNN1
#define matsetvalues11_                  MATSETVALUES11
#define matsetvalues1n_                  MATSETVALUES1N
#define matsetvaluesn1_                  MATSETVALUESN1
#define matsetvaluesblocked0_            MATSETVALUESBLOCKED0
#define matsetvaluesblocked2_            MATSETVALUESBLOCKED2
#define matsetvaluesblocked11_           MATSETVALUESBLOCKED11
#define matsetvaluesblocked111_          MATSETVALUESBLOCKED111
#define matsetvaluesblocked1n_           MATSETVALUESBLOCKED1N
#define matsetvaluesblockedn1_           MATSETVALUESBLOCKEDN1
#define matsetvaluesblockedlocal_        MATSETVALUESBLOCKEDLOCAL
#define matsetvaluesblockedlocal0_       MATSETVALUESBLOCKEDLOCAL0
#define matsetvaluesblockedlocal11_      MATSETVALUESBLOCKEDLOCAL11
#define matsetvaluesblockedlocal111_     MATSETVALUESBLOCKEDLOCAL111
#define matsetvaluesblockedlocal1n_      MATSETVALUESBLOCKEDLOCAL1N
#define matsetvaluesblockedlocaln1_      MATSETVALUESBLOCKEDLOCALN1
#define matsetvalueslocal_               MATSETVALUESLOCAL
#define matsetvalueslocal0_              MATSETVALUESLOCAL0
#define matsetvalueslocal11_             MATSETVALUESLOCAL11
#define matsetvalueslocal11nn_           MATSETVALUESLOCAL11NN
#define matsetvalueslocal111_            MATSETVALUESLOCAL111
#define matsetvalueslocal1n_             MATSETVALUESLOCAL1N
#define matsetvalueslocaln1_             MATSETVALUESLOCALN1
#define matgetrowmin_                    MATGETROWMIN
#define matgetrowminabs_                 MATGETROWMINABS
#define matgetrowmax_                    MATGETROWMAX
#define matgetrowmaxabs_                 MATGETROWMAXABS
#define matdestroymatrices_              MATDESTROYMATRICES
#define matdestroysubmatrices_           MATDESTROYSUBMATRICES
#define matgetfactor_                    MATGETFACTOR
#define matfactorgetsolverpackage_       MATFACTORGETSOLVERPACKAGE
#define matgetrowij_                     MATGETROWIJ
#define matrestorerowij_                 MATRESTOREROWIJ
#define matgetrow_                       MATGETROW
#define matrestorerow_                   MATRESTOREROW
#define matload_                         MATLOAD
#define matview_                         MATVIEW
#define matseqaijgetarray_               MATSEQAIJGETARRAY
#define matseqaijrestorearray_           MATSEQAIJRESTOREARRAY
#define matdensegetarray_                MATDENSEGETARRAY
#define matdensegetarrayread_            MATDENSEGETARRAYREAD
#define matdenserestorearray_            MATDENSERESTOREARRAY
#define matdenserestorearrayread_        MATDENSERESTOREARRAYREAD
#define matconvert_                      MATCONVERT
#define matcreatesubmatrices_            MATCREATESUBMATRICES
#define matcreatesubmatricesmpi_         MATCREATESUBMATRICESMPI
#define matzerorowscolumns_              MATZEROROWSCOLUMNS
#define matzerorowscolumnsis_            MATZEROROWSCOLUMNSIS
#define matzerorowsstencil_              MATZEROROWSSTENCIL
#define matzerorowscolumnsstencil_       MATZEROROWSCOLUMNSSTENCIL
#define matzerorows_                     MATZEROROWS
#define matzerorowsis_                   MATZEROROWSIS
#define matzerorowslocal_                MATZEROROWSLOCAL
#define matzerorowslocal0_               MATZEROROWSLOCAL0
#define matzerorowslocal1_               MATZEROROWSLOCAL1
#define matzerorowslocalis_              MATZEROROWSLOCALIS
#define matzerorowscolumnslocal_         MATZEROROWSCOLUMNSLOCAL
#define matzerorowscolumnslocalis_       MATZEROROWSCOLUMNSLOCALIS
#define matsetoptionsprefix_             MATSETOPTIONSPREFIX
#define matcreatevecs_                   MATCREATEVECS
#define matnullspaceremove_              MATNULLSPACEREMOVE
#define matgetinfo_                      MATGETINFO
#define matlufactor_                     MATLUFACTOR
#define matilufactor_                    MATILUFACTOR
#define matlufactorsymbolic_             MATLUFACTORSYMBOLIC
#define matlufactornumeric_              MATLUFACTORNUMERIC
#define matcholeskyfactor_               MATCHOLESKYFACTOR
#define matcholeskyfactorsymbolic_       MATCHOLESKYFACTORSYMBOLIC
#define matcholeskyfactornumeric_        MATCHOLESKYFACTORNUMERIC
#define matilufactorsymbolic_            MATILUFACTORSYMBOLIC
#define maticcfactorsymbolic_            MATICCFACTORSYMBOLIC
#define maticcfactor_                    MATICCFACTOR
#define matfactorinfoinitialize_         MATFACTORINFOINITIALIZE
#define matnullspacesetfunction_         MATNULLSPACESETFUNCTION
#define matfindnonzerorows_              MATFINDNONZEROROWS
#define matgetsize_                      MATGETSIZE
#define matgetsize00_                    MATGETSIZE00
#define matgetsize10_                    MATGETSIZE10
#define matgetsize01_                    MATGETSIZE01
#define matgetlocalsize_                 MATGETLOCALSIZE
#define matgetlocalsize00_               MATGETLOCALSIZE00
#define matgetlocalsize10_               MATGETLOCALSIZE10
#define matgetlocalsize01_               MATGETLOCALSIZE01
#define matsetnullspace_                 MATSETNULLSPACE
#define matgetownershiprange_            MATGETOWNERSHIPRANGE
#define matgetownershiprange00_          MATGETOWNERSHIPRANGE00
#define matgetownershiprange10_          MATGETOWNERSHIPRANGE10
#define matgetownershiprange01_          MATGETOWNERSHIPRANGE01
#define matgetownershiprange11_          MATGETOWNERSHIPRANGE11
#define matgetownershipis_               MATGETOWNERSHIPIS
#define matgetownershiprangecolumn_      MATGETOWNERSHIPRANGECOLUMN
#define matviewfromoptions_              MATVIEWFROMOPTIONS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matsetvalues_                    matsetvalues
#define matsetvaluesnnnn_                matsetvaluesnnnn
#define matsetvalues0_                   matsetvalues0
#define matsetvaluesnn1_                 matsetvaluesnn1
#define matsetvalues11_                  matsetvalues11
#define matsetvaluesn1_                  matsetvaluesn1
#define matsetvalues1n_                  matsetvalues1n
#define matsetvalueslocal_               matsetvalueslocal
#define matsetvalueslocal0_              matsetvalueslocal0
#define matsetvalueslocal11_             matsetvalueslocal11
#define matsetvalueslocal11nn_           matsetvalueslocal11nn
#define matsetvalueslocal111_            matsetvalueslocal111
#define matsetvalueslocal1n_             matsetvalueslocal1n
#define matsetvalueslocaln1_             matsetvalueslocaln1
#define matsetvaluesblocked_             matsetvaluesblocked
#define matsetvaluesblocked0_            matsetvaluesblocked0
#define matsetvaluesblocked2_            matsetvaluesblocked2
#define matsetvaluesblocked11_           matsetvaluesblocked11
#define matsetvaluesblocked111_          matsetvaluesblocked111
#define matsetvaluesblocked1n_           matsetvaluesblocked1n
#define matsetvaluesblockedn1_           matsetvaluesblockedn1
#define matsetvaluesblockedlocal_        matsetvaluesblockedlocal
#define matsetvaluesblockedlocal0_       matsetvaluesblockedlocal0
#define matsetvaluesblockedlocal11_      matsetvaluesblockedlocal11
#define matsetvaluesblockedlocal111_     matsetvaluesblockedlocal111
#define matsetvaluesblockedlocal1n_      matsetvaluesblockedlocal1n
#define matsetvaluesblockedlocaln1_      matsetvaluesblockedlocaln1
#define matgetrowmin_                    matgetrowmin
#define matgetrowminabs_                 matgetrowminabs
#define matgetrowmax_                    matgetrowmax
#define matgetrowmaxabs_                 matgetrowmaxabs
#define matdestroymatrices_              matdestroymatrices
#define matdestroysubmatrices_           matdestroysubmatrices
#define matgetfactor_                    matgetfactor
#define matfactorgetsolverpackage_       matfactorgetsolverpackage
#define matcreatevecs_                   matcreatevecs
#define matgetrowij_                     matgetrowij
#define matrestorerowij_                 matrestorerowij
#define matgetrow_                       matgetrow
#define matrestorerow_                   matrestorerow
#define matview_                         matview
#define matload_                         matload
#define matseqaijgetarray_               matseqaijgetarray
#define matseqaijrestorearray_           matseqaijrestorearray
#define matdensegetarray_                matdensegetarray
#define matdensegetarrayread_            matdensegetarrayread
#define matdenserestorearray_            matdenserestorearray
#define matdenserestorearrayread_        matdenserestorearrayread
#define matconvert_                      matconvert
#define matcreatesubmatrices_            matcreatesubmatrices
#define matcreatesubmatricesmpi_         matcreatesubmatricesmpi
#define matzerorowscolumns_              matzerorowscolumns
#define matzerorowscolumnsis_            matzerorowscolumnsis
#define matzerorowsstencil_              matzerorowsstencil
#define matzerorowscolumnsstencil_       matzerorowscolumnsstencil
#define matzerorows_                     matzerorows
#define matzerorowsis_                   matzerorowsis
#define matzerorowslocal_                matzerorowslocal
#define matzerorowslocalis_              matzerorowslocalis
#define matzerorowscolumnslocal_         matzerorowscolumnslocal
#define matzerorowscolumnslocalis_       matzerorowscolumnslocalis
#define matsetoptionsprefix_             matsetoptionsprefix
#define matnullspaceremove_              matnullspaceremove
#define matgetinfo_                      matgetinfo
#define matlufactor_                     matlufactor
#define matilufactor_                    matilufactor
#define matlufactorsymbolic_             matlufactorsymbolic
#define matlufactornumeric_              matlufactornumeric
#define matcholeskyfactor_               matcholeskyfactor
#define matcholeskyfactorsymbolic_       matcholeskyfactorsymbolic
#define matcholeskyfactornumeric_        matcholeskyfactornumeric
#define matilufactorsymbolic_            matilufactorsymbolic
#define maticcfactorsymbolic_            maticcfactorsymbolic
#define maticcfactor_                    maticcfactor
#define matfactorinfoinitialize_         matfactorinfoinitialize
#define matnullspacesetfunction_         matnullspacesetfunction
#define matfindnonzerorows_              matfindnonzerorows
#define matgetsize_                      matgetsize
#define matgetsize00_                    matgetsize00
#define matgetsize10_                    matgetsize10
#define matgetsize01_                    matgetsize01
#define matgetlocalsize_                 matgetlocalsize
#define matgetlocalsize00_               matgetlocalsize00
#define matgetlocalsize10_               matgetlocalsize10
#define matgetlocalsize01_               matgetlocalsize01
#define matsetnullspace_                 matsetnullspace
#define matgetownershiprange_            matgetownershiprange
#define matgetownershiprange00_          matgetownershiprange00
#define matgetownershiprange10_          matgetownershiprange10
#define matgetownershiprange01_          matgetownershiprange01
#define matgetownershiprange11_          matgetownershiprange11
#define matgetownershipis_               matgetownershipis
#define matgetownershiprangecolumn_      matgetownershiprangecolumn
#define matviewfromoptions_              matviewfromoptions
#endif

PETSC_EXTERN void PETSC_STDCALL  matgetownershiprange_(Mat *mat,PetscInt *m,PetscInt *n, int *ierr )
{
  CHKFORTRANNULLINTEGER(m);
  CHKFORTRANNULLINTEGER(n);
  *ierr = MatGetOwnershipRange(*mat,m,n);
}

PETSC_EXTERN void PETSC_STDCALL  matgetownershiprange00_(Mat *mat,PetscInt *m,PetscInt *n, int *ierr )
{
  CHKFORTRANNULLINTEGER(m);
  CHKFORTRANNULLINTEGER(n);
  *ierr = MatGetOwnershipRange(*mat,m,n);
}

PETSC_EXTERN void PETSC_STDCALL  matgetownershiprange10_(Mat *mat,PetscInt *m,PetscInt *n, int *ierr )
{
  CHKFORTRANNULLINTEGER(m);
  CHKFORTRANNULLINTEGER(n);
  *ierr = MatGetOwnershipRange(*mat,m,n);
}

PETSC_EXTERN void PETSC_STDCALL  matgetownershiprange01_(Mat *mat,PetscInt *m,PetscInt *n, int *ierr )
{
  CHKFORTRANNULLINTEGER(m);
  CHKFORTRANNULLINTEGER(n);
  *ierr = MatGetOwnershipRange(*mat,m,n);
}

PETSC_EXTERN void PETSC_STDCALL  matgetownershiprange11_(Mat *mat,PetscInt *m,PetscInt *n, int *ierr )
{
  CHKFORTRANNULLINTEGER(m);
  CHKFORTRANNULLINTEGER(n);
  *ierr = MatGetOwnershipRange(*mat,m,n);
}

PETSC_EXTERN void PETSC_STDCALL  matgetownershipis_(Mat *mat,IS *m,IS *n, int *ierr )
{
  CHKFORTRANNULLOBJECT(m);
  CHKFORTRANNULLOBJECT(n);
  *ierr = MatGetOwnershipIS(*mat,m,n);
}

PETSC_EXTERN void PETSC_STDCALL  matgetownershiprangecolumn_(Mat *mat,PetscInt *m,PetscInt *n, int *ierr )
{
  CHKFORTRANNULLINTEGER(m);
  CHKFORTRANNULLINTEGER(n);
  *ierr = MatGetOwnershipRangeColumn(*mat,m,n);
}

PETSC_EXTERN void PETSC_STDCALL  matgetsize_(Mat *mat,PetscInt *m,PetscInt *n, int *ierr )
{
  CHKFORTRANNULLINTEGER(m);
  CHKFORTRANNULLINTEGER(n);
  *ierr = MatGetSize(*mat,m,n);
}

PETSC_EXTERN void PETSC_STDCALL  matgetsize00_(Mat *mat,PetscInt *m,PetscInt *n, int *ierr )
{
  matgetsize_(mat,m,n,ierr);
}

PETSC_EXTERN void PETSC_STDCALL  matgetsize10_(Mat *mat,PetscInt *m,PetscInt *n, int *ierr )
{
  matgetsize_(mat,m,n,ierr);
}

PETSC_EXTERN void PETSC_STDCALL  matgetsize01_(Mat *mat,PetscInt *m,PetscInt *n, int *ierr )
{
  matgetsize_(mat,m,n,ierr);
}

PETSC_EXTERN void PETSC_STDCALL  matgetlocalsize_(Mat *mat,PetscInt *m,PetscInt *n, int *ierr )
{
  CHKFORTRANNULLINTEGER(m);
  CHKFORTRANNULLINTEGER(n);
  *ierr = MatGetLocalSize(*mat,m,n);
}

PETSC_EXTERN void PETSC_STDCALL  matgetlocalsize00_(Mat *mat,PetscInt *m,PetscInt *n, int *ierr )
{
  matgetlocalsize_(mat,m,n,ierr);
}

PETSC_EXTERN void PETSC_STDCALL  matgetlocalsize10_(Mat *mat,PetscInt *m,PetscInt *n, int *ierr )
{
  matgetlocalsize_(mat,m,n,ierr);
}

PETSC_EXTERN void PETSC_STDCALL  matgetlocalsize01_(Mat *mat,PetscInt *m,PetscInt *n, int *ierr )
{
  matgetlocalsize_(mat,m,n,ierr);
}

PETSC_EXTERN void PETSC_STDCALL  matsetvaluesblocked_(Mat *mat,PetscInt *m, PetscInt idxm[],PetscInt *n, PetscInt idxn[], PetscScalar v[],InsertMode *addv, int *ierr ){
  *ierr = MatSetValuesBlocked(*mat,*m,idxm,*n,idxn,v,*addv);
}

PETSC_EXTERN void PETSC_STDCALL  matsetvaluesblocked2_(Mat *mat,PetscInt *m, PetscInt idxm[],PetscInt *n, PetscInt idxn[], F90Array2d *y,InsertMode *addv, int *ierr PETSC_F90_2PTR_PROTO(ptrd)){
  PetscScalar *fa;
  *ierr = F90Array2dAccess(y,MPIU_SCALAR,(void**)&fa PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
  matsetvaluesblocked_(mat,m,idxm,n,idxn,fa,addv,ierr);
}

PETSC_EXTERN void PETSC_STDCALL  matsetvaluesblocked0_(Mat *mat,PetscInt *m, PetscInt idxm[],PetscInt *n, PetscInt idxn[], PetscScalar v[],InsertMode *addv, int *ierr ){
  matsetvaluesblocked_(mat,m,idxm,n,idxn,v,addv,ierr);
}

PETSC_EXTERN void PETSC_STDCALL  matsetvaluesblocked11_(Mat *mat,PetscInt *m, PetscInt idxm[],PetscInt *n, PetscInt idxn[], PetscScalar v[],InsertMode *addv, int *ierr ){
  matsetvaluesblocked_(mat,m,idxm,n,idxn,v,addv,ierr);
}

PETSC_EXTERN void PETSC_STDCALL  matsetvaluesblocked111_(Mat *mat,PetscInt *m, PetscInt idxm[],PetscInt *n, PetscInt idxn[], PetscScalar v[],InsertMode *addv, int *ierr ){
  matsetvaluesblocked_(mat,m,idxm,n,idxn,v,addv,ierr);
}

PETSC_EXTERN void PETSC_STDCALL  matsetvaluesblocked1n_(Mat *mat,PetscInt *m, PetscInt idxm[],PetscInt *n, PetscInt idxn[], PetscScalar v[],InsertMode *addv, int *ierr ){
  matsetvaluesblocked_(mat,m,idxm,n,idxn,v,addv,ierr);
}

PETSC_EXTERN void PETSC_STDCALL  matsetvaluesblockedn1_(Mat *mat,PetscInt *m, PetscInt idxm[],PetscInt *n, PetscInt idxn[], PetscScalar v[],InsertMode *addv, int *ierr ){
  matsetvaluesblocked_(mat,m,idxm,n,idxn,v,addv,ierr);
}

PETSC_EXTERN void PETSC_STDCALL  matsetvaluesblockedlocal_(Mat *mat,PetscInt *nrow, PetscInt irow[],PetscInt *ncol, PetscInt icol[], PetscScalar y[],InsertMode *addv, int *ierr )
{
  *ierr = MatSetValuesBlockedLocal(*mat,*nrow,irow,*ncol,icol,y,*addv);
}

PETSC_EXTERN void PETSC_STDCALL  matsetvaluesblockedlocal0_(Mat *mat,PetscInt *m, PetscInt idxm[],PetscInt *n, PetscInt idxn[], PetscScalar v[],InsertMode *addv, int *ierr ){
  matsetvaluesblockedlocal_(mat,m,idxm,n,idxn,v,addv,ierr);
}

PETSC_EXTERN void PETSC_STDCALL  matsetvaluesblockedlocal11_(Mat *mat,PetscInt *m, PetscInt idxm[],PetscInt *n, PetscInt idxn[], PetscScalar v[],InsertMode *addv, int *ierr ){
  matsetvaluesblockedlocal_(mat,m,idxm,n,idxn,v,addv,ierr);
}

PETSC_EXTERN void PETSC_STDCALL  matsetvaluesblockedlocal111_(Mat *mat,PetscInt *m, PetscInt idxm[],PetscInt *n, PetscInt idxn[], PetscScalar v[],InsertMode *addv, int *ierr ){
  matsetvaluesblockedlocal_(mat,m,idxm,n,idxn,v,addv,ierr);
}

PETSC_EXTERN void PETSC_STDCALL  matsetvaluesblockedlocal1n_(Mat *mat,PetscInt *m, PetscInt idxm[],PetscInt *n, PetscInt idxn[], PetscScalar v[],InsertMode *addv, int *ierr ){
  matsetvaluesblockedlocal_(mat,m,idxm,n,idxn,v,addv,ierr);
}

PETSC_EXTERN void PETSC_STDCALL  matsetvaluesblockedlocaln1_(Mat *mat,PetscInt *m, PetscInt idxm[],PetscInt *n, PetscInt idxn[], PetscScalar v[],InsertMode *addv, int *ierr ){
  matsetvaluesblockedlocal_(mat,m,idxm,n,idxn,v,addv,ierr);
}

PETSC_EXTERN void PETSC_STDCALL  matsetvalues_(Mat *mat,PetscInt *m, PetscInt idxm[],PetscInt *n, PetscInt idxn[], PetscScalar v[],InsertMode *addv, int *ierr )
{
  *ierr = MatSetValues(*mat,*m,idxm,*n,idxn,v,*addv);
}

PETSC_EXTERN void PETSC_STDCALL  matsetvaluesnnnn_(Mat *mat,PetscInt *m, PetscInt idxm[],PetscInt *n, PetscInt idxn[], PetscScalar v[],InsertMode *addv, int *ierr )
{
  matsetvalues_(mat,m,idxm,n,idxn,v,addv,ierr);
}

PETSC_EXTERN void PETSC_STDCALL  matsetvalues0_(Mat *mat,PetscInt *m, PetscInt idxm[],PetscInt *n, PetscInt idxn[], PetscScalar v[],InsertMode *addv, int *ierr )
{
  matsetvalues_(mat,m,idxm,n,idxn,v,addv,ierr);
}

PETSC_EXTERN void PETSC_STDCALL  matsetvaluesnn1_(Mat *mat,PetscInt *m, PetscInt idxm[],PetscInt *n, PetscInt idxn[], PetscScalar v[],InsertMode *addv, int *ierr )
{
  matsetvalues_(mat,m,idxm,n,idxn,v,addv,ierr);
}

PETSC_EXTERN void PETSC_STDCALL  matsetvalues11_(Mat *mat,PetscInt *m, PetscInt idxm[],PetscInt *n, PetscInt idxn[], PetscScalar v[],InsertMode *addv, int *ierr )
{
  matsetvalues_(mat,m,idxm,n,idxn,v,addv,ierr);
}

PETSC_EXTERN void PETSC_STDCALL  matsetvaluesn1_(Mat *mat,PetscInt *m, PetscInt idxm[],PetscInt *n, PetscInt idxn[], PetscScalar v[],InsertMode *addv, int *ierr )
{
  matsetvalues_(mat,m,idxm,n,idxn,v,addv,ierr);
}

PETSC_EXTERN void PETSC_STDCALL  matsetvalues1n_(Mat *mat,PetscInt *m, PetscInt idxm[],PetscInt *n, PetscInt idxn[], PetscScalar v[],InsertMode *addv, int *ierr )
{
  matsetvalues_(mat,m,idxm,n,idxn,v,addv,ierr);
}

PETSC_EXTERN void PETSC_STDCALL  matsetvalueslocal_(Mat *mat,PetscInt *nrow, PetscInt irow[],PetscInt *ncol, PetscInt icol[], PetscScalar y[],InsertMode *addv, int *ierr )
{
  *ierr = MatSetValuesLocal(*mat,*nrow,irow,*ncol,icol,y,*addv);
}

PETSC_EXTERN void PETSC_STDCALL  matsetvalueslocal0_(Mat *mat,PetscInt *nrow, PetscInt irow[],PetscInt *ncol, PetscInt icol[], PetscScalar y[],InsertMode *addv, int *ierr )
{
  matsetvalueslocal_(mat,nrow,irow,ncol,icol,y,addv,ierr);
}

PETSC_EXTERN void PETSC_STDCALL  matsetvalueslocal11_(Mat *mat,PetscInt *nrow, PetscInt irow[],PetscInt *ncol, PetscInt icol[], PetscScalar y[],InsertMode *addv, int *ierr )
{
  matsetvalueslocal_(mat,nrow,irow,ncol,icol,y,addv,ierr);
}

PETSC_EXTERN void PETSC_STDCALL  matsetvalueslocal11nn_(Mat *mat,PetscInt *nrow, PetscInt irow[],PetscInt *ncol, PetscInt icol[], PetscScalar y[],InsertMode *addv, int *ierr )
{
  matsetvalueslocal_(mat,nrow,irow,ncol,icol,y,addv,ierr);
}

PETSC_EXTERN void PETSC_STDCALL  matsetvalueslocal111_(Mat *mat,PetscInt *nrow, PetscInt irow[],PetscInt *ncol, PetscInt icol[], PetscScalar y[],InsertMode *addv, int *ierr )
{
  matsetvalueslocal_(mat,nrow,irow,ncol,icol,y,addv,ierr);
}

PETSC_EXTERN void PETSC_STDCALL  matsetvalueslocal1n_(Mat *mat,PetscInt *nrow, PetscInt irow[],PetscInt *ncol, PetscInt icol[], PetscScalar y[],InsertMode *addv, int *ierr )
{
  matsetvalueslocal_(mat,nrow,irow,ncol,icol,y,addv,ierr);
}

PETSC_EXTERN void PETSC_STDCALL  matsetvalueslocaln1_(Mat *mat,PetscInt *nrow, PetscInt irow[],PetscInt *ncol, PetscInt icol[], PetscScalar y[],InsertMode *addv, int *ierr )
{
  matsetvalueslocal_(mat,nrow,irow,ncol,icol,y,addv,ierr);
}

PETSC_EXTERN void PETSC_STDCALL  matgetrowmin_(Mat *mat,Vec *v,PetscInt idx[], int *ierr )
{
  CHKFORTRANNULLINTEGER(idx);
  *ierr = MatGetRowMin(*mat,*v,idx);
}

PETSC_EXTERN void PETSC_STDCALL  matgetrowminabs_(Mat *mat,Vec *v,PetscInt idx[], int *ierr )
{
  CHKFORTRANNULLINTEGER(idx);
  *ierr = MatGetRowMinAbs(*mat,*v,idx);
}

PETSC_EXTERN void PETSC_STDCALL  matgetrowmax_(Mat *mat,Vec *v,PetscInt idx[], int *ierr )
{
  CHKFORTRANNULLINTEGER(idx);
  *ierr = MatGetRowMax(*mat,*v,idx);
}

PETSC_EXTERN void PETSC_STDCALL  matgetrowmaxabs_(Mat *mat,Vec *v,PetscInt idx[], int *ierr )
{
  CHKFORTRANNULLINTEGER(idx);
  *ierr = MatGetRowMaxAbs(*mat,*v,idx);
}

static PetscErrorCode ournullfunction(MatNullSpace sp,Vec x,void *ctx)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(MatNullSpace*,Vec*,void*,PetscErrorCode*))(((PetscObject)sp)->fortran_func_pointers[0]))(&sp,&x,ctx,&ierr);CHKERRQ(ierr);
  return 0;
}

PETSC_EXTERN void PETSC_STDCALL matnullspacesetfunction_(MatNullSpace *sp, PetscErrorCode (*rem)(MatNullSpace,Vec,void*),void *ctx,PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*sp,1);
  ((PetscObject)*sp)->fortran_func_pointers[0] = (PetscVoidFunction)rem;

  *ierr = MatNullSpaceSetFunction(*sp,ournullfunction,ctx);
}

PETSC_EXTERN void PETSC_STDCALL matcreatevecs_(Mat *mat,Vec *right,Vec *left, int *ierr)
{
  CHKFORTRANNULLOBJECT(right);
  CHKFORTRANNULLOBJECT(left);
  *ierr = MatCreateVecs(*mat,right,left);
}

PETSC_EXTERN void PETSC_STDCALL matgetrowij_(Mat *B,PetscInt *shift,PetscBool *sym,PetscBool *blockcompressed,PetscInt *n,PetscInt *ia,size_t *iia,
                                PetscInt *ja,size_t *jja,PetscBool  *done,PetscErrorCode *ierr)
{
  const PetscInt *IA,*JA;
  *ierr = MatGetRowIJ(*B,*shift,*sym,*blockcompressed,n,&IA,&JA,done);if (*ierr) return;
  *iia  = PetscIntAddressToFortran(ia,(PetscInt*)IA);
  *jja  = PetscIntAddressToFortran(ja,(PetscInt*)JA);
}

PETSC_EXTERN void PETSC_STDCALL matrestorerowij_(Mat *B,PetscInt *shift,PetscBool *sym,PetscBool *blockcompressed, PetscInt *n,PetscInt *ia,size_t *iia,
                                    PetscInt *ja,size_t *jja,PetscBool  *done,PetscErrorCode *ierr)
{
  const PetscInt *IA = PetscIntAddressFromFortran(ia,*iia),*JA = PetscIntAddressFromFortran(ja,*jja);
  *ierr = MatRestoreRowIJ(*B,*shift,*sym,*blockcompressed,n,&IA,&JA,done);
}

/*
   This is a poor way of storing the column and value pointers
  generated by MatGetRow() to be returned with MatRestoreRow()
  but there is not natural,good place else to store them. Hence
  Fortran programmers can only have one outstanding MatGetRows()
  at a time.
*/
static PetscErrorCode    matgetrowactive = 0;
static const PetscInt    *my_ocols       = 0;
static const PetscScalar *my_ovals       = 0;

PETSC_EXTERN void PETSC_STDCALL matgetrow_(Mat *mat,PetscInt *row,PetscInt *ncols,PetscInt *cols,PetscScalar *vals,PetscErrorCode *ierr)
{
  const PetscInt    **oocols = &my_ocols;
  const PetscScalar **oovals = &my_ovals;

  if (matgetrowactive) {
    PetscError(PETSC_COMM_SELF,__LINE__,"MatGetRow_Fortran",__FILE__,PETSC_ERR_ARG_WRONGSTATE,PETSC_ERROR_INITIAL,
               "Cannot have two MatGetRow() active simultaneously\n\
               call MatRestoreRow() before calling MatGetRow() a second time");
    *ierr = 1;
    return;
  }

  CHKFORTRANNULLINTEGER(cols); if (!cols) oocols = NULL;
  CHKFORTRANNULLSCALAR(vals);  if (!vals) oovals = NULL;

  *ierr = MatGetRow(*mat,*row,ncols,oocols,oovals);
  if (*ierr) return;

  if (oocols) { *ierr = PetscArraycpy(cols,my_ocols,*ncols); if (*ierr) return;}
  if (oovals) { *ierr = PetscArraycpy(vals,my_ovals,*ncols); if (*ierr) return;}
  matgetrowactive = 1;
}

PETSC_EXTERN void PETSC_STDCALL matrestorerow_(Mat *mat,PetscInt *row,PetscInt *ncols,PetscInt *cols,PetscScalar *vals,PetscErrorCode *ierr)
{
  const PetscInt    **oocols = &my_ocols;
  const PetscScalar **oovals = &my_ovals;
  if (!matgetrowactive) {
    PetscError(PETSC_COMM_SELF,__LINE__,"MatRestoreRow_Fortran",__FILE__,PETSC_ERR_ARG_WRONGSTATE,PETSC_ERROR_INITIAL,
               "Must call MatGetRow() first");
    *ierr = 1;
    return;
  }
  CHKFORTRANNULLINTEGER(cols); if (!cols) oocols = NULL;
  CHKFORTRANNULLSCALAR(vals);  if (!vals) oovals = NULL;

  *ierr           = MatRestoreRow(*mat,*row,ncols,oocols,oovals);
  matgetrowactive = 0;
}

PETSC_EXTERN void PETSC_STDCALL matview_(Mat *mat,PetscViewer *vin,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = MatView(*mat,v);
}

PETSC_EXTERN void PETSC_STDCALL matload_(Mat *mat,PetscViewer *vin,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = MatLoad(*mat,v);
}

PETSC_EXTERN void PETSC_STDCALL matseqaijgetarray_(Mat *mat,PetscScalar *fa,size_t *ia,PetscErrorCode *ierr)
{
  PetscScalar *mm;
  PetscInt    m,n;

  *ierr = MatSeqAIJGetArray(*mat,&mm); if (*ierr) return;
  *ierr = MatGetSize(*mat,&m,&n);  if (*ierr) return;
  *ierr = PetscScalarAddressToFortran((PetscObject)*mat,1,fa,mm,m*n,ia); if (*ierr) return;
}

PETSC_EXTERN void PETSC_STDCALL matseqaijrestorearray_(Mat *mat,PetscScalar *fa,size_t *ia,PetscErrorCode *ierr)
{
  PetscScalar *lx;
  PetscInt    m,n;

  *ierr = MatGetSize(*mat,&m,&n); if (*ierr) return;
  *ierr = PetscScalarAddressFromFortran((PetscObject)*mat,fa,*ia,m*n,&lx);if (*ierr) return;
  *ierr = MatSeqAIJRestoreArray(*mat,&lx);if (*ierr) return;
}

PETSC_EXTERN void PETSC_STDCALL matdensegetarray_(Mat *mat,PetscScalar *fa,size_t *ia,PetscErrorCode *ierr)
{
  PetscScalar *mm;
  PetscInt    m,n;

  *ierr = MatDenseGetArray(*mat,&mm); if (*ierr) return;
  *ierr = MatGetSize(*mat,&m,&n);  if (*ierr) return;
  *ierr = PetscScalarAddressToFortran((PetscObject)*mat,1,fa,mm,m*n,ia); if (*ierr) return;
}

PETSC_EXTERN void PETSC_STDCALL matdenserestorearray_(Mat *mat,PetscScalar *fa,size_t *ia,PetscErrorCode *ierr)
{
  PetscScalar *lx;
  PetscInt    m,n;

  *ierr = MatGetSize(*mat,&m,&n); if (*ierr) return;
  *ierr = PetscScalarAddressFromFortran((PetscObject)*mat,fa,*ia,m*n,&lx);if (*ierr) return;
  *ierr = MatDenseRestoreArray(*mat,&lx);if (*ierr) return;
}

PETSC_EXTERN void PETSC_STDCALL matdensegetarrayread_(Mat *mat,PetscScalar *fa,size_t *ia,PetscErrorCode *ierr)
{
  const PetscScalar *mm;
  PetscInt         m,n;

  *ierr = MatDenseGetArrayRead(*mat,&mm); if (*ierr) return;
  *ierr = MatGetSize(*mat,&m,&n);  if (*ierr) return;
  *ierr = PetscScalarAddressToFortran((PetscObject)*mat,1,fa,(PetscScalar*)mm,m*n,ia); if (*ierr) return;
}


PETSC_EXTERN void PETSC_STDCALL matdenserestorearrayread_(Mat *mat,PetscScalar *fa,size_t *ia,PetscErrorCode *ierr)
{
  const PetscScalar *lx;
  PetscInt          m,n;

  *ierr = MatGetSize(*mat,&m,&n); if (*ierr) return;
  *ierr = PetscScalarAddressFromFortran((PetscObject)*mat,fa,*ia,m*n,(PetscScalar**)&lx);if (*ierr) return;
  *ierr = MatDenseRestoreArrayRead(*mat,&lx);if (*ierr) return;
}

PETSC_EXTERN void PETSC_STDCALL matfactorgetsolverpackage_(Mat *mat,char* name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = MatFactorGetSolverType(*mat,&tname);if (*ierr) return;
  if (name != PETSC_NULL_CHARACTER_Fortran) {
    *ierr = PetscStrncpy(name,tname,len);if (*ierr) return;
  }
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}

PETSC_EXTERN void PETSC_STDCALL matgetfactor_(Mat *mat,char* outtype PETSC_MIXED_LEN(len),MatFactorType *ftype,Mat *M,PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(outtype,len,t);
  *ierr = MatGetFactor(*mat,t,*ftype,M);if (*ierr) return;
  FREECHAR(outtype,t);
}

PETSC_EXTERN void PETSC_STDCALL matconvert_(Mat *mat,char* outtype PETSC_MIXED_LEN(len),MatReuse *reuse,Mat *M,PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(outtype,len,t);
  *ierr = MatConvert(*mat,t,*reuse,M);if (*ierr) return;
  FREECHAR(outtype,t);
}

/*
    MatCreateSubmatrices() is slightly different from C since the
    Fortran provides the array to hold the submatrix objects,while in C that
    array is allocated by the MatCreateSubmatrices()
*/
PETSC_EXTERN void PETSC_STDCALL matcreatesubmatrices_(Mat *mat,PetscInt *n,IS *isrow,IS *iscol,MatReuse *scall,Mat *smat,PetscErrorCode *ierr)
{
  Mat      *lsmat;
  PetscInt i;

  if (*scall == MAT_INITIAL_MATRIX) {
    *ierr = MatCreateSubMatrices(*mat,*n,isrow,iscol,*scall,&lsmat);
    for (i=0; i<=*n; i++) { /* lsmat[*n] might be a dummy matrix for saving data struc */
      smat[i] = lsmat[i];
    }
    *ierr = PetscFree(lsmat);
  } else {
    *ierr = MatCreateSubMatrices(*mat,*n,isrow,iscol,*scall,&smat);
  }
}

/*
    MatCreateSubmatrices() is slightly different from C since the
    Fortran provides the array to hold the submatrix objects,while in C that
    array is allocated by the MatCreateSubmatrices()
*/
PETSC_EXTERN void PETSC_STDCALL matcreatesubmatricesmpi_(Mat *mat,PetscInt *n,IS *isrow,IS *iscol,MatReuse *scall,Mat *smat,PetscErrorCode *ierr)
{
  Mat      *lsmat;
  PetscInt i;

  if (*scall == MAT_INITIAL_MATRIX) {
    *ierr = MatCreateSubMatricesMPI(*mat,*n,isrow,iscol,*scall,&lsmat);
    for (i=0; i<=*n; i++) { /* lsmat[*n] might be a dummy matrix for saving data struc */
      smat[i] = lsmat[i];
    }
    *ierr = PetscFree(lsmat);
  } else {
    *ierr = MatCreateSubMatricesMPI(*mat,*n,isrow,iscol,*scall,&smat);
  }
}

/*
    MatDestroyMatrices() is slightly different from C since the
    Fortran does not free the array of matrix objects, while in C that
    the array is freed
*/
PETSC_EXTERN void PETSC_STDCALL matdestroymatrices_(PetscInt *n,Mat *smat,PetscErrorCode *ierr)
{
  PetscInt i;

  for (i=0; i<*n; i++) {
    *ierr = MatDestroy(&smat[i]);if (*ierr) return;
  }
}

/*
    MatDestroySubMatrices() is slightly different from C since the
    Fortran provides the array to hold the submatrix objects, while in C that
    array is allocated by the MatCreateSubmatrices()
*/
PETSC_EXTERN void PETSC_STDCALL matdestroysubmatrices_(PetscInt *n,Mat *smat,PetscErrorCode *ierr)
{
  Mat      *lsmat;
  PetscInt i;

  *ierr = PetscMalloc1(*n+1,&lsmat);
  for (i=0; i<=*n; i++) {
      lsmat[i] = smat[i];
  }
  *ierr = MatDestroySubMatrices(*n,&lsmat);
}

PETSC_EXTERN void PETSC_STDCALL matsetoptionsprefix_(Mat *mat,char* prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = MatSetOptionsPrefix(*mat,t);if (*ierr) return;
  FREECHAR(prefix,t);
}

PETSC_EXTERN void PETSC_STDCALL matnullspaceremove_(MatNullSpace *sp,Vec *vec,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(*sp)
  *ierr = MatNullSpaceRemove(*sp,*vec);
}

PETSC_EXTERN void PETSC_STDCALL matgetinfo_(Mat *mat,MatInfoType *flag,MatInfo *info, int *ierr)
{
  *ierr = MatGetInfo(*mat,*flag,info);
}

PETSC_EXTERN void PETSC_STDCALL matlufactor_(Mat *mat,IS *row,IS *col,const MatFactorInfo *info, int *ierr)
{
  *ierr = MatLUFactor(*mat,*row,*col,info);
}

PETSC_EXTERN void PETSC_STDCALL matilufactor_(Mat *mat,IS *row,IS *col,const MatFactorInfo *info, int *ierr)
{
  *ierr = MatILUFactor(*mat,*row,*col,info);
}

PETSC_EXTERN void PETSC_STDCALL matlufactorsymbolic_(Mat *fact,Mat *mat,IS *row,IS *col,const MatFactorInfo *info, int *ierr)
{
  *ierr = MatLUFactorSymbolic(*fact,*mat,*row,*col,info);
}

PETSC_EXTERN void PETSC_STDCALL matlufactornumeric_(Mat *fact,Mat *mat,const MatFactorInfo *info, int *ierr)
{
  *ierr = MatLUFactorNumeric(*fact,*mat,info);
}

PETSC_EXTERN void PETSC_STDCALL matcholeskyfactor_(Mat *mat,IS *perm,const MatFactorInfo *info, int *ierr)
{
  *ierr = MatCholeskyFactor(*mat,*perm,info);
}

PETSC_EXTERN void PETSC_STDCALL matcholeskyfactorsymbolic_(Mat *fact,Mat *mat,IS *perm,const MatFactorInfo *info, int *ierr)
{
  *ierr = MatCholeskyFactorSymbolic(*fact,*mat,*perm,info);
}

PETSC_EXTERN void PETSC_STDCALL matcholeskyfactornumeric_(Mat *fact,Mat *mat,const MatFactorInfo *info, int *ierr)
{
  *ierr = MatCholeskyFactorNumeric(*fact,*mat,info);
}

PETSC_EXTERN void PETSC_STDCALL matilufactorsymbolic_(Mat *fact,Mat *mat,IS *row,IS *col,const MatFactorInfo *info, int *ierr)
{
  *ierr = MatILUFactorSymbolic(*fact,*mat,*row,*col,info);
}

PETSC_EXTERN void PETSC_STDCALL maticcfactorsymbolic_(Mat *fact,Mat *mat,IS *perm,const MatFactorInfo *info, int *ierr)
{
  *ierr = MatICCFactorSymbolic(*fact,*mat,*perm,info);
}

PETSC_EXTERN void PETSC_STDCALL maticcfactor_(Mat *mat,IS *row,const MatFactorInfo *info, int *ierr)
{
  *ierr = MatICCFactor(*mat,*row,info);
}

PETSC_EXTERN void PETSC_STDCALL matfactorinfoinitialize_(MatFactorInfo *info, int *ierr)
{
  *ierr = MatFactorInfoInitialize(info);
}
PETSC_EXTERN void PETSC_STDCALL  matzerorowslocal_(Mat *mat,PetscInt *numRows, PetscInt rows[],PetscScalar *diag,Vec *x,Vec *b, int *ierr)
{
  *ierr = MatZeroRowsLocal(*mat,*numRows,rows,*diag,*x,*b);
}
PETSC_EXTERN void PETSC_STDCALL  matzerorowslocal0_(Mat *mat,PetscInt *numRows, PetscInt rows[],PetscScalar *diag,Vec *x,Vec *b, int *ierr)
{
  matzerorowslocal_(mat,numRows,rows,diag,x,b,ierr);
}
PETSC_EXTERN void PETSC_STDCALL  matzerorowslocal1_(Mat *mat,PetscInt *numRows, PetscInt rows[],PetscScalar *diag,Vec *x,Vec *b, int *ierr)
{
  matzerorowslocal_(mat,numRows,rows,diag,x,b,ierr);
}
PETSC_EXTERN void PETSC_STDCALL matviewfromoptions_(Mat *ao,PetscObject obj,char* type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = MatViewFromOptions(*ao,obj,t);if (*ierr) return;
  FREECHAR(type,t);
}
