#define PETSCMAT_DLL

/*
     Routines that call the kernel minpack coloring subroutines
*/

#include "src/mat/matimpl.h"
#include "src/mat/color/color.h"

/*
    MatFDColoringDegreeSequence_Minpack - Calls the MINPACK routine seqr() that
      computes the degree sequence required by MINPACK coloring routines.
*/
#undef __FUNCT__  
#define __FUNCT__ "MatFDColoringDegreeSequence_Minpack" 
PetscErrorCode MatFDColoringDegreeSequence_Minpack(PetscInt m,PetscInt *cja, PetscInt *cia, PetscInt *rja, PetscInt *ria, PetscInt **seq)
{
  PetscInt       *work;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc(m*sizeof(PetscInt),&work);CHKERRQ(ierr);  
  ierr = PetscMalloc(m*sizeof(PetscInt),seq);CHKERRQ(ierr);

  MINPACKdegr(&m,cja,cia,rja,ria,*seq,work);

  ierr = PetscFree(work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    MatFDColoringMinimumNumberofColors_Private - For a given sparse 
        matrix computes the minimum number of colors needed.

*/
#undef __FUNCT__  
#define __FUNCT__ "MatFDColoringMinimumNumberofColors_Private" 
PetscErrorCode MatFDColoringMinimumNumberofColors_Private(PetscInt m,PetscInt *ia,PetscInt *minc)
{
  PetscInt i,c = 0;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    c = PetscMax(c,ia[i+1]-ia[i]);
  }
  *minc = c;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
/* ----------------------------------------------------------------------------*/
/*
    MatFDColoringSL_Minpack - Uses the smallest-last (SL) coloring of minpack
*/
#undef __FUNCT__  
#define __FUNCT__ "MatFDColoringSL_Minpack" 
PetscErrorCode PETSCMAT_DLLEXPORT MatFDColoringSL_Minpack(Mat mat,MatColoringType name,ISColoring *iscoloring)
{
  PetscErrorCode ierr;
  PetscInt        *list,*work,clique,*ria,*rja,*cia,*cja,*seq,*coloring,n;
  PetscInt        ncolors,i;
  PetscTruth      done;

  PetscFunctionBegin;
  ierr = MatGetRowIJ(mat,1,PETSC_FALSE,&n,&ria,&rja,&done);CHKERRQ(ierr);
  ierr = MatGetColumnIJ(mat,1,PETSC_FALSE,&n,&cia,&cja,&done);CHKERRQ(ierr);
  if (!done) SETERRQ(PETSC_ERR_SUP,"Ordering requires IJ");

  ierr = MatFDColoringDegreeSequence_Minpack(n,cja,cia,rja,ria,&seq);CHKERRQ(ierr);

  ierr = PetscMalloc(5*n*sizeof(PetscInt),&list);CHKERRQ(ierr);
  work = list + n;

  MINPACKslo(&n,cja,cia,rja,ria,seq,list,&clique,work,work+n,work+2*n,work+3*n);

  ierr = PetscMalloc(n*sizeof(PetscInt),&coloring);CHKERRQ(ierr);
  MINPACKseq(&n,cja,cia,rja,ria,list,coloring,&ncolors,work);

  ierr = PetscFree(list);CHKERRQ(ierr);
  ierr = PetscFree(seq);CHKERRQ(ierr);
  ierr = MatRestoreRowIJ(mat,1,PETSC_FALSE,&n,&ria,&rja,&done);CHKERRQ(ierr);
  ierr = MatRestoreColumnIJ(mat,1,PETSC_FALSE,&n,&cia,&cja,&done);CHKERRQ(ierr);

  /* shift coloring numbers to start at zero and shorten */
  if (ncolors > IS_COLORING_MAX-1) SETERRQ(PETSC_ERR_SUP,"Maximum color size exceeded");
  { 
    ISColoringValue *s = (ISColoringValue*) coloring;
    for (i=0; i<n; i++) {
      s[i] = (ISColoringValue) (coloring[i]-1);
    }
    ierr = MatColoringPatch(mat,n,ncolors,s,iscoloring);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
/* ----------------------------------------------------------------------------*/
/*
    MatFDColoringLF_Minpack - 
*/
#undef __FUNCT__  
#define __FUNCT__ "MatFDColoringLF_Minpack" 
PetscErrorCode PETSCMAT_DLLEXPORT MatFDColoringLF_Minpack(Mat mat,MatColoringType name,ISColoring *iscoloring)
{
  PetscErrorCode ierr;
  PetscInt       *list,*work,*ria,*rja,*cia,*cja,*seq,*coloring,n;
  PetscInt       n1, none,ncolors,i;
  PetscTruth     done;

  PetscFunctionBegin;
  ierr = MatGetRowIJ(mat,1,PETSC_FALSE,&n,&ria,&rja,&done);CHKERRQ(ierr);
  ierr = MatGetColumnIJ(mat,1,PETSC_FALSE,&n,&cia,&cja,&done);CHKERRQ(ierr);
  if (!done) SETERRQ(PETSC_ERR_SUP,"Ordering requires IJ");

  ierr = MatFDColoringDegreeSequence_Minpack(n,cja,cia,rja,ria,&seq);CHKERRQ(ierr);

  ierr = PetscMalloc(5*n*sizeof(PetscInt),&list);CHKERRQ(ierr);
  work = list + n;

  n1   = n - 1;
  none = -1;
  MINPACKnumsrt(&n,&n1,seq,&none,list,work+2*n,work+n);
  ierr = PetscMalloc(n*sizeof(PetscInt),&coloring);CHKERRQ(ierr);
  MINPACKseq(&n,cja,cia,rja,ria,list,coloring,&ncolors,work);

  ierr = PetscFree(list);CHKERRQ(ierr);
  ierr = PetscFree(seq);CHKERRQ(ierr);

  ierr = MatRestoreRowIJ(mat,1,PETSC_FALSE,&n,&ria,&rja,&done);CHKERRQ(ierr);
  ierr = MatRestoreColumnIJ(mat,1,PETSC_FALSE,&n,&cia,&cja,&done);CHKERRQ(ierr);

  /* shift coloring numbers to start at zero and shorten */
  if (ncolors > IS_COLORING_MAX-1) SETERRQ(PETSC_ERR_SUP,"Maximum color size exceeded");
  { 
    ISColoringValue *s = (ISColoringValue*) coloring;
    for (i=0; i<n; i++) {
      s[i] = (ISColoringValue) (coloring[i]-1);
    }
    ierr = MatColoringPatch(mat,n,ncolors,s,iscoloring);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
/* ----------------------------------------------------------------------------*/
/*
    MatFDColoringID_Minpack - 
*/
#undef __FUNCT__  
#define __FUNCT__ "MatFDColoringID_Minpack" 
PetscErrorCode PETSCMAT_DLLEXPORT MatFDColoringID_Minpack(Mat mat,MatColoringType name,ISColoring *iscoloring)
{
  PetscErrorCode ierr;
  PetscInt       *list,*work,clique,*ria,*rja,*cia,*cja,*seq,*coloring,n;
  PetscInt       ncolors,i;
  PetscTruth     done;

  PetscFunctionBegin;
  ierr = MatGetRowIJ(mat,1,PETSC_FALSE,&n,&ria,&rja,&done);CHKERRQ(ierr);
  ierr = MatGetColumnIJ(mat,1,PETSC_FALSE,&n,&cia,&cja,&done);CHKERRQ(ierr);
  if (!done) SETERRQ(PETSC_ERR_SUP,"Ordering requires IJ");

  ierr = MatFDColoringDegreeSequence_Minpack(n,cja,cia,rja,ria,&seq);CHKERRQ(ierr);

  ierr = PetscMalloc(5*n*sizeof(PetscInt),&list);CHKERRQ(ierr);
  work = list + n;

  MINPACKido(&n,&n,cja,cia,rja,ria,seq,list,&clique,work,work+n,work+2*n,work+3*n);

  ierr = PetscMalloc(n*sizeof(PetscInt),&coloring);CHKERRQ(ierr);
  MINPACKseq(&n,cja,cia,rja,ria,list,coloring,&ncolors,work);

  ierr = PetscFree(list);CHKERRQ(ierr);
  ierr = PetscFree(seq);CHKERRQ(ierr);

  ierr = MatRestoreRowIJ(mat,1,PETSC_FALSE,&n,&ria,&rja,&done);CHKERRQ(ierr);
  ierr = MatRestoreColumnIJ(mat,1,PETSC_FALSE,&n,&cia,&cja,&done);CHKERRQ(ierr);

  /* shift coloring numbers to start at zero and shorten */
  if (ncolors > IS_COLORING_MAX-1) SETERRQ(PETSC_ERR_SUP,"Maximum color size exceeded");
  { 
    ISColoringValue *s = (ISColoringValue*) coloring;
    for (i=0; i<n; i++) {
      s[i] = (ISColoringValue) (coloring[i]-1);
    }
    ierr = MatColoringPatch(mat,n,ncolors,s,iscoloring);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
/*
   Simplest coloring, each column of the matrix gets its own unique color.
*/
#undef __FUNCT__  
#define __FUNCT__ "MatColoring_Natural" 
PetscErrorCode PETSCMAT_DLLEXPORT MatColoring_Natural(Mat mat,const MatColoringType color, ISColoring *iscoloring)
{
  PetscErrorCode  ierr;
  PetscInt        start,end,i;
  ISColoringValue *colors;
  MPI_Comm        comm;

  PetscFunctionBegin;
  ierr = MatGetOwnershipRange(mat,&start,&end);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)mat,&comm);CHKERRQ(ierr);
  ierr = PetscMalloc((end-start+1)*sizeof(PetscInt),&colors);CHKERRQ(ierr);
  for (i=start; i<end; i++) {
    colors[i-start] = i;
  }
  ierr = ISColoringCreate(comm,end-start,colors,iscoloring);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END
  
/* ===========================================================================================*/

#include "petscsys.h"

PetscFList MatColoringList = 0;
PetscTruth MatColoringRegisterAllCalled = PETSC_FALSE;

#undef __FUNCT__  
#define __FUNCT__ "MatColoringRegister" 
PetscErrorCode PETSCMAT_DLLEXPORT MatColoringRegister(const char sname[],const char path[],const char name[],PetscErrorCode (*function)(Mat,const MatColoringType,ISColoring*))
{
  PetscErrorCode ierr;
  char           fullname[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  ierr = PetscFListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = PetscFListAdd(&MatColoringList,sname,fullname,(void (*)(void))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatColoringRegisterDestroy" 
/*@C
   MatColoringRegisterDestroy - Frees the list of coloringing routines.

   Not Collective

   Level: developer

.keywords: matrix, register, destroy

.seealso: MatColoringRegisterDynamic(), MatColoringRegisterAll()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatColoringRegisterDestroy(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (MatColoringList) {
    ierr = PetscFListDestroy(&MatColoringList);CHKERRQ(ierr);
    MatColoringList = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetColoring" 
/*@C
   MatGetColoring - Gets a coloring for a matrix to reduce the number of function evaluations
   needed to compute a sparse Jacobian via differencing.

   Collective on Mat

   Input Parameters:
.  mat - the matrix
.  type - type of coloring, one of the following:
$      MATCOLORING_NATURAL - natural (one color for each column, very slow)
$      MATCOLORING_SL - smallest-last
$      MATCOLORING_LF - largest-first
$      MATCOLORING_ID - incidence-degree

   Output Parameters:
.   iscoloring - the coloring

   Options Database Keys:
   To specify the coloring through the options database, use one of
   the following 
$    -mat_coloring_type natural, -mat_coloring_type sl, -mat_coloring_type lf,
$    -mat_coloring_type id
   To see the coloring use
$    -mat_coloring_view

   Level: intermediate

   Notes:
     These compute the graph coloring of the graph of A^{T}A. The coloring used 
   for efficient (parallel or thread based) triangular solves etc is NOT yet 
   available. 

   The user can define additional colorings; see MatColoringRegisterDynamic().

   The sequential colorings SL, LF, and ID are obtained via the Minpack software that was
   converted to C using f2c.

.keywords: matrix, get, coloring

.seealso:  MatGetColoringTypeFromOptions(), MatColoringRegisterDynamic(), MatFDColoringCreate(),
           SNESDefaultComputeJacobianColor()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatGetColoring(Mat mat,const MatColoringType type,ISColoring *iscoloring)
{
  PetscTruth     flag;
  PetscErrorCode ierr,(*r)(Mat,const MatColoringType,ISColoring *);
  char           tname[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidPointer(iscoloring,3);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (!MatColoringRegisterAllCalled) {
    ierr = MatColoringRegisterAll(PETSC_NULL);CHKERRQ(ierr);
  }
  
  /* look for type on command line */
  ierr = PetscOptionsGetString(mat->prefix,"-mat_coloring_type",tname,256,&flag);CHKERRQ(ierr);
  if (flag) {
    type = tname;
  }

  ierr = PetscLogEventBegin(MAT_GetColoring,mat,0,0,0);CHKERRQ(ierr);
  ierr =  PetscFListFind(mat->comm, MatColoringList, type,(void (**)(void)) &r);CHKERRQ(ierr);
  if (!r) {SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Unknown or unregistered type: %s",type);}
  ierr = (*r)(mat,type,iscoloring);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_GetColoring,mat,0,0,0);CHKERRQ(ierr);

  ierr = PetscVerboseInfo(((PetscObject)mat,"MatGetColoring:Number of colors %d\n",(*iscoloring)->n));CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-mat_coloring_view",&flag);CHKERRQ(ierr);
  if (flag) {
    ierr = ISColoringView(*iscoloring,PETSC_VIEWER_STDOUT_((*iscoloring)->comm));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
 
