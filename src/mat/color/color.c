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
int MatFDColoringDegreeSequence_Minpack(int m,int *cja, int *cia, int *rja, int *ria, int **seq)
{
  int *work;
  int ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc(m*sizeof(int),&work);CHKERRQ(ierr);  
  ierr = PetscMalloc(m*sizeof(int),seq);CHKERRQ(ierr);

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
int MatFDColoringMinimumNumberofColors_Private(int m,int *ia,int *minc)
{
  int i,c = 0;

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
int MatFDColoringSL_Minpack(Mat mat,MatColoringType name,ISColoring *iscoloring)
{
  int        *list,*work,clique,ierr,*ria,*rja,*cia,*cja,*seq,*coloring,n;
  int        ncolors,i;
  PetscTruth done;

  PetscFunctionBegin;
  ierr = MatGetRowIJ(mat,1,PETSC_FALSE,&n,&ria,&rja,&done);CHKERRQ(ierr);
  ierr = MatGetColumnIJ(mat,1,PETSC_FALSE,&n,&cia,&cja,&done);CHKERRQ(ierr);
  if (!done) SETERRQ(PETSC_ERR_SUP,"Ordering requires IJ");

  ierr = MatFDColoringDegreeSequence_Minpack(n,cja,cia,rja,ria,&seq);CHKERRQ(ierr);

  ierr = PetscMalloc(5*n*sizeof(int),&list);CHKERRQ(ierr);
  work = list + n;

  MINPACKslo(&n,cja,cia,rja,ria,seq,list,&clique,work,work+n,work+2*n,work+3*n);

  ierr = PetscMalloc(n*sizeof(int),&coloring);CHKERRQ(ierr);
  MINPACKseq(&n,cja,cia,rja,ria,list,coloring,&ncolors,work);

  ierr = PetscFree(list);CHKERRQ(ierr);
  ierr = PetscFree(seq);CHKERRQ(ierr);
  ierr = MatRestoreRowIJ(mat,1,PETSC_FALSE,&n,&ria,&rja,&done);CHKERRQ(ierr);
  ierr = MatRestoreColumnIJ(mat,1,PETSC_FALSE,&n,&cia,&cja,&done);CHKERRQ(ierr);

  /* shift coloring numbers to start at zero and shorten */
  if (ncolors > IS_COLORING_MAX-1) SETERRQ(1,"Maximum color size exceeded");
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
int MatFDColoringLF_Minpack(Mat mat,MatColoringType name,ISColoring *iscoloring)
{
  int        *list,*work,ierr,*ria,*rja,*cia,*cja,*seq,*coloring,n;
  int        n1, none,ncolors,i;
  PetscTruth done;

  PetscFunctionBegin;
  ierr = MatGetRowIJ(mat,1,PETSC_FALSE,&n,&ria,&rja,&done);CHKERRQ(ierr);
  ierr = MatGetColumnIJ(mat,1,PETSC_FALSE,&n,&cia,&cja,&done);CHKERRQ(ierr);
  if (!done) SETERRQ(PETSC_ERR_SUP,"Ordering requires IJ");

  ierr = MatFDColoringDegreeSequence_Minpack(n,cja,cia,rja,ria,&seq);CHKERRQ(ierr);

  ierr = PetscMalloc(5*n*sizeof(int),&list);CHKERRQ(ierr);
  work = list + n;

  n1   = n - 1;
  none = -1;
  MINPACKnumsrt(&n,&n1,seq,&none,list,work+2*n,work+n);
  ierr = PetscMalloc(n*sizeof(int),&coloring);CHKERRQ(ierr);
  MINPACKseq(&n,cja,cia,rja,ria,list,coloring,&ncolors,work);

  ierr = PetscFree(list);CHKERRQ(ierr);
  ierr = PetscFree(seq);CHKERRQ(ierr);

  ierr = MatRestoreRowIJ(mat,1,PETSC_FALSE,&n,&ria,&rja,&done);CHKERRQ(ierr);
  ierr = MatRestoreColumnIJ(mat,1,PETSC_FALSE,&n,&cia,&cja,&done);CHKERRQ(ierr);

  /* shift coloring numbers to start at zero and shorten */
  if (ncolors > IS_COLORING_MAX-1) SETERRQ(1,"Maximum color size exceeded");
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
int MatFDColoringID_Minpack(Mat mat,MatColoringType name,ISColoring *iscoloring)
{
  int        *list,*work,clique,ierr,*ria,*rja,*cia,*cja,*seq,*coloring,n;
  int        ncolors,i;
  PetscTruth done;

  PetscFunctionBegin;
  ierr = MatGetRowIJ(mat,1,PETSC_FALSE,&n,&ria,&rja,&done);CHKERRQ(ierr);
  ierr = MatGetColumnIJ(mat,1,PETSC_FALSE,&n,&cia,&cja,&done);CHKERRQ(ierr);
  if (!done) SETERRQ(PETSC_ERR_SUP,"Ordering requires IJ");

  ierr = MatFDColoringDegreeSequence_Minpack(n,cja,cia,rja,ria,&seq);CHKERRQ(ierr);

  ierr = PetscMalloc(5*n*sizeof(int),&list);CHKERRQ(ierr);
  work = list + n;

  MINPACKido(&n,&n,cja,cia,rja,ria,seq,list,&clique,work,work+n,work+2*n,work+3*n);

  ierr = PetscMalloc(n*sizeof(int),&coloring);CHKERRQ(ierr);
  MINPACKseq(&n,cja,cia,rja,ria,list,coloring,&ncolors,work);

  ierr = PetscFree(list);CHKERRQ(ierr);
  ierr = PetscFree(seq);CHKERRQ(ierr);

  ierr = MatRestoreRowIJ(mat,1,PETSC_FALSE,&n,&ria,&rja,&done);CHKERRQ(ierr);
  ierr = MatRestoreColumnIJ(mat,1,PETSC_FALSE,&n,&cia,&cja,&done);CHKERRQ(ierr);

  /* shift coloring numbers to start at zero and shorten */
  if (ncolors > IS_COLORING_MAX-1) SETERRQ(1,"Maximum color size exceeded");
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
int MatColoring_Natural(Mat mat,const MatColoringType color, ISColoring *iscoloring)
{
  int             start,end,ierr,i;
  ISColoringValue *colors;
  MPI_Comm        comm;

  PetscFunctionBegin;
  ierr = MatGetOwnershipRange(mat,&start,&end);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)mat,&comm);CHKERRQ(ierr);
  ierr = PetscMalloc((end-start+1)*sizeof(int),&colors);CHKERRQ(ierr);
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
int MatColoringRegister(const char sname[],const char path[],const char name[],int (*function)(Mat,const MatColoringType,ISColoring*))
{
  int  ierr;
  char fullname[256];

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
int MatColoringRegisterDestroy(void)
{
  int ierr;

  PetscFunctionBegin;
  if (MatColoringList) {
    ierr = PetscFListDestroy(&MatColoringList);CHKERRQ(ierr);
    MatColoringList = 0;
  }
  PetscFunctionReturn(0);
}

EXTERN int MatAdjustForInodes(Mat,IS *,IS *);

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
int MatGetColoring(Mat mat,const MatColoringType type,ISColoring *iscoloring)
{
  PetscTruth flag;
  int        ierr,(*r)(Mat,const MatColoringType,ISColoring *);
  char       tname[256];

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

  PetscLogInfo((PetscObject)mat,"MatGetColoring:Number of colors %d\n",(*iscoloring)->n);
  ierr = PetscOptionsHasName(PETSC_NULL,"-mat_coloring_view",&flag);CHKERRQ(ierr);
  if (flag) {
    ierr = ISColoringView(*iscoloring,PETSC_VIEWER_STDOUT_((*iscoloring)->comm));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
 
