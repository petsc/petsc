#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: color.c,v 1.37 1999/05/04 20:32:47 balay Exp balay $";
#endif
 
/*
     Routines that call the kernel minpack coloring subroutines
*/

#include "petsc.h"
#include "src/mat/matimpl.h"
#include "src/mat/impls/color/color.h"

/*
    MatFDColoringDegreeSequence_Minpack - Calls the MINPACK routine seqr() that
      computes the degree sequence required by MINPACK coloring routines.
*/
#undef __FUNC__  
#define __FUNC__ "MatFDColoringDegreeSequence_Minpack" 
int MatFDColoringDegreeSequence_Minpack(int m,int *cja, int *cia, int *rja, int *ria, int **seq)
{
  int *work;

  PetscFunctionBegin;
  work = (int *) PetscMalloc( m*sizeof(int) );CHKPTRQ(work);  
  *seq = (int *) PetscMalloc( m*sizeof(int) );CHKPTRQ(*seq);

  MINPACKdegr(&m,cja,cia,rja,ria,*seq,work);

  PetscFree(work);
  PetscFunctionReturn(0);
}

/*
    MatFDColoringMinimumNumberofColors_Private - For a given sparse 
        matrix computes the minimum number of colors needed.

*/
#undef __FUNC__  
#define __FUNC__ "MatFDColoringMinimumNumberofColors_Private" 
int MatFDColoringMinimumNumberofColors_Private(int m,int *ia,int *minc)
{
  int i,c = 0;

  PetscFunctionBegin;
  for ( i=0; i<m; i++ ) {
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
#undef __FUNC__  
#define __FUNC__ "MatFDColoringSL_Minpack" 
int MatFDColoringSL_Minpack(Mat mat,MatColoringType name,ISColoring *iscoloring)
{
  int        *list,*work,clique,ierr,*ria,*rja,*cia,*cja,*seq,*coloring,n;
  int        ncolors;
  PetscTruth done;

  PetscFunctionBegin;
  ierr = MatGetRowIJ(mat,1,PETSC_FALSE,&n,&ria,&rja,&done);CHKERRQ(ierr);
  ierr = MatGetColumnIJ(mat,1,PETSC_FALSE,&n,&cia,&cja,&done);CHKERRQ(ierr);
  if (!done) SETERRQ(PETSC_ERR_SUP,0,"Ordering requires IJ");

  ierr = MatFDColoringDegreeSequence_Minpack(n,cja,cia,rja,ria,&seq);CHKERRQ(ierr);

  list = (int*) PetscMalloc( 5*n*sizeof(int) );CHKPTRQ(list);
  work = list + n;

  MINPACKslo(&n,cja,cia,rja,ria,seq,list,&clique,work,work+n,work+2*n,work+3*n);

  coloring = (int *) PetscMalloc(n*sizeof(int));CHKPTRQ(coloring);
  MINPACKseq(&n,cja,cia,rja,ria,list,coloring,&ncolors,work);

  PetscFree(list);
  PetscFree(seq);
  ierr = MatRestoreRowIJ(mat,1,PETSC_FALSE,&n,&ria,&rja,&done);CHKERRQ(ierr);
  ierr = MatRestoreColumnIJ(mat,1,PETSC_FALSE,&n,&cia,&cja,&done);CHKERRQ(ierr);

  ierr = MatColoringPatch(mat,ncolors,coloring,iscoloring);CHKERRQ(ierr);
  PetscFree(coloring);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
/* ----------------------------------------------------------------------------*/
/*
    MatFDColoringLF_Minpack - 
*/
#undef __FUNC__  
#define __FUNC__ "MatFDColoringLF_Minpack" 
int MatFDColoringLF_Minpack(Mat mat,MatColoringType name,ISColoring *iscoloring)
{
  int        *list,*work,ierr,*ria,*rja,*cia,*cja,*seq,*coloring,n;
  int        n1, none,ncolors;
  PetscTruth done;

  PetscFunctionBegin;
  ierr = MatGetRowIJ(mat,1,PETSC_FALSE,&n,&ria,&rja,&done);CHKERRQ(ierr);
  ierr = MatGetColumnIJ(mat,1,PETSC_FALSE,&n,&cia,&cja,&done);CHKERRQ(ierr);
  if (!done) SETERRQ(PETSC_ERR_SUP,0,"Ordering requires IJ");

  ierr = MatFDColoringDegreeSequence_Minpack(n,cja,cia,rja,ria,&seq);CHKERRQ(ierr);

  list = (int*) PetscMalloc( 5*n*sizeof(int) );CHKPTRQ(list);
  work = list + n;

  n1   = n - 1;
  none = -1;
  MINPACKnumsrt(&n,&n1,seq,&none,list,work+2*n,work+n);
  coloring = (int *) PetscMalloc(n*sizeof(int));CHKPTRQ(coloring);
  MINPACKseq(&n,cja,cia,rja,ria,list,coloring,&ncolors,work);

  PetscFree(list);
  PetscFree(seq);

  ierr = MatRestoreRowIJ(mat,1,PETSC_FALSE,&n,&ria,&rja,&done);CHKERRQ(ierr);
  ierr = MatRestoreColumnIJ(mat,1,PETSC_FALSE,&n,&cia,&cja,&done);CHKERRQ(ierr);

  ierr = MatColoringPatch(mat,ncolors,coloring,iscoloring);CHKERRQ(ierr);
  PetscFree(coloring);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
/* ----------------------------------------------------------------------------*/
/*
    MatFDColoringID_Minpack - 
*/
#undef __FUNC__  
#define __FUNC__ "MatFDColoringID_Minpack" 
int MatFDColoringID_Minpack(Mat mat,MatColoringType name,ISColoring *iscoloring)
{
  int        *list,*work,clique,ierr,*ria,*rja,*cia,*cja,*seq,*coloring,n;
  int        ncolors;
  PetscTruth done;

  PetscFunctionBegin;
  ierr = MatGetRowIJ(mat,1,PETSC_FALSE,&n,&ria,&rja,&done);CHKERRQ(ierr);
  ierr = MatGetColumnIJ(mat,1,PETSC_FALSE,&n,&cia,&cja,&done);CHKERRQ(ierr);
  if (!done) SETERRQ(PETSC_ERR_SUP,0,"Ordering requires IJ");

  ierr = MatFDColoringDegreeSequence_Minpack(n,cja,cia,rja,ria,&seq);CHKERRQ(ierr);

  list = (int*) PetscMalloc( 5*n*sizeof(int) );CHKPTRQ(list);
  work = list + n;

  MINPACKido(&n,&n,cja,cia,rja,ria,seq,list,&clique,work,work+n,work+2*n,work+3*n);

  coloring = (int *) PetscMalloc(n*sizeof(int));CHKPTRQ(coloring);
  MINPACKseq(&n,cja,cia,rja,ria,list,coloring,&ncolors,work);

  PetscFree(list);
  PetscFree(seq);

  ierr = MatRestoreRowIJ(mat,1,PETSC_FALSE,&n,&ria,&rja,&done);CHKERRQ(ierr);
  ierr = MatRestoreColumnIJ(mat,1,PETSC_FALSE,&n,&cia,&cja,&done);CHKERRQ(ierr);

  ierr = MatColoringPatch(mat,ncolors,coloring,iscoloring);CHKERRQ(ierr);

  PetscFree(coloring);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
/*
   Simplest coloring, each column of the matrix gets its own unique color.
*/
#undef __FUNC__  
#define __FUNC__ "MatColoring_Natural" 
int MatColoring_Natural(Mat mat,MatColoringType color, ISColoring *iscoloring)
{
  int      N,start,end,ierr,i,tag;
  IS       *is;
  MPI_Comm comm;

  PetscFunctionBegin;
  ierr = MatGetSize(mat,&N,&N);CHKERRQ(ierr);
  is  = (IS *) PetscMalloc( N*sizeof(IS*) );CHKPTRQ(is); 
  *iscoloring       = (ISColoring) PetscMalloc(sizeof(struct _p_ISColoring));CHKPTRQ(*iscoloring);
  (*iscoloring)->n  = N;
  (*iscoloring)->is = is;
  
  ierr = MatGetOwnershipRange(mat,&start,&end);CHKERRQ(ierr);
  for ( i=0; i<start; i++ ) {
    ierr = ISCreateGeneral(PETSC_COMM_SELF,0,PETSC_NULL,is+i);CHKERRQ(ierr);
  }
  for ( i=start; i<end; i++ ) {
    ierr = ISCreateGeneral(PETSC_COMM_SELF,1,&i,is+i);CHKERRQ(ierr);
  }
  for ( i=end; i<N; i++ ) {
    ierr = ISCreateGeneral(PETSC_COMM_SELF,0,PETSC_NULL,is+i);CHKERRQ(ierr);
  }
  ierr = PetscObjectGetComm((PetscObject)mat,&comm);CHKERRQ(ierr);
  ierr = PetscCommDuplicate_Private(comm,&(*iscoloring)->comm,&tag);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
  
/* ===========================================================================================*/

#include "sys.h"

FList MatColoringList = 0;
int   MatColoringRegisterAllCalled = 0;

/*MC
   MatColoringRegister - Adds a new sparse matrix coloring to the 
                               matrix package. 

   Synopsis:
   MatColoringRegister(char *name_coloring,char *path,char *name_create,int (*routine_create)(MatColoring))

   Not Collective

   Input Parameters:
+  sname - name of Coloring (for example MATCOLORING_SL)
.  path - location of library where creation routine is 
.  name - name of function that creates the Coloring type, a string
-  function - function pointer that creates the coloring

   Level: developer

   If dynamic libraries are used, then the fourth input argument (function)
   is ignored.

   Sample usage:
.vb
   MatColoringRegister("my_color",/home/username/my_lib/lib/libO/solaris/mylib.a,
               "MyColor",MyColor);
.ve

   Then, your partitioner can be chosen with the procedural interface via
$     MatColoringSetType(part,"my_color")
   or at runtime via the option
$     -mat_coloring_type my_color

   $PETSC_ARCH and $BOPT occuring in pathname will be replaced with appropriate values.

.keywords: matrix, Coloring, register

.seealso: MatColoringRegisterDestroy(), MatColoringRegisterAll()
M*/

#undef __FUNC__  
#define __FUNC__ "MatColoringRegister_Private" 
int MatColoringRegister_Private(char *sname,char *path,char *name,int (*function)(Mat,MatColoringType,ISColoring*))
{
  int  ierr;
  char fullname[256];

  PetscFunctionBegin;
  ierr = PetscStrcpy(fullname,path);CHKERRQ(ierr);
  PetscStrcat(fullname,":");PetscStrcat(fullname,name);
  ierr = FListAdd_Private(&MatColoringList,sname,fullname,(int (*)(void*))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatColoringRegisterDestroy" 
/*@C
   MatColoringRegisterDestroy - Frees the list of coloringing routines.

   Not Collective

   Level: developer

.keywords: matrix, register, destroy

.seealso: MatColoringRegister(), MatColoringRegisterAll()
@*/
int MatColoringRegisterDestroy(void)
{
  int ierr;

  PetscFunctionBegin;
  if (MatColoringList) {
    ierr = FListDestroy( MatColoringList );CHKERRQ(ierr);
    MatColoringList = 0;
  }
  PetscFunctionReturn(0);
}

extern int MatAdjustForInodes(Mat,IS *,IS *);

#undef __FUNC__  
#define __FUNC__ "MatGetColoring" 
/*@C
   MatGetColoring - Gets a coloring for a matrix to reduce fill or to
   improve numerical stability of LU factorization.

   Collective on Mat

   Input Parameters:
.  mat - the matrix
.  type - type of coloring, one of the following:
$      MATCOLORING_NATURAL - natural
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

   The user can define additional colorings; see MatColoringRegister().

.keywords: matrix, get, coloring

.seealso:  MatGetColoringTypeFromOptions(), MatColoringRegister()
@*/
int MatGetColoring(Mat mat,MatColoringType type,ISColoring *iscoloring)
{
  int         ierr,flag;
  int         (*r)(Mat,MatColoringType,ISColoring *);
  char        tname[256];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for factored matrix"); 
  if (!MatColoringRegisterAllCalled) {
    ierr = MatColoringRegisterAll(PETSC_NULL);CHKERRQ(ierr);
  }
  
  /* look for type on command line */
  ierr = OptionsGetString(mat->prefix,"-mat_coloring_type",tname,256,&flag);CHKERRQ(ierr);
  if (flag) {
    type = tname;
  }

  PLogEventBegin(MAT_GetColoring,mat,0,0,0);
  ierr =  FListFind(mat->comm, MatColoringList, type,(int (**)(void *)) &r );CHKERRQ(ierr);
  if (!r) {SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,0,"Unknown or unregistered type: %s",type);}
  ierr = (*r)(mat,type,iscoloring);CHKERRQ(ierr);
  PLogEventEnd(MAT_GetColoring,mat,0,0,0);

  PLogInfo((PetscObject)mat,"MatGetColoring:Number of colors %d\n",(*iscoloring)->n);
  ierr = OptionsHasName(PETSC_NULL,"-mat_coloring_view",&flag);CHKERRQ(ierr);
  if (flag) {
    ierr = ISColoringView(*iscoloring,VIEWER_STDOUT_((*iscoloring)->comm));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
 
