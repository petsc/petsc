#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: color.c,v 1.34 1999/02/28 20:06:44 bsmith Exp bsmith $";
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
  work = (int *) PetscMalloc( m*sizeof(int) ); CHKPTRQ(work);  
  *seq = (int *) PetscMalloc( m*sizeof(int) ); CHKPTRQ(*seq);

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

  ierr = MatFDColoringDegreeSequence_Minpack(n,cja,cia,rja,ria,&seq); CHKERRQ(ierr);

  list = (int*) PetscMalloc( 5*n*sizeof(int) ); CHKPTRQ(list);
  work = list + n;

  MINPACKslo(&n,cja,cia,rja,ria,seq,list,&clique,work,work+n,work+2*n,work+3*n);

  coloring = (int *) PetscMalloc(n*sizeof(int)); CHKPTRQ(coloring);
  MINPACKseq(&n,cja,cia,rja,ria,list,coloring,&ncolors,work);

  PetscFree(list);
  PetscFree(seq);
  ierr = MatRestoreRowIJ(mat,1,PETSC_FALSE,&n,&ria,&rja,&done);CHKERRQ(ierr);
  ierr = MatRestoreColumnIJ(mat,1,PETSC_FALSE,&n,&cia,&cja,&done);CHKERRQ(ierr);

  ierr = MatColoringPatch(mat,ncolors,coloring,iscoloring); CHKERRQ(ierr);
  PetscFree(coloring);
  PetscFunctionReturn(0);
}

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

  ierr = MatFDColoringDegreeSequence_Minpack(n,cja,cia,rja,ria,&seq); CHKERRQ(ierr);

  list = (int*) PetscMalloc( 5*n*sizeof(int) ); CHKPTRQ(list);
  work = list + n;

  n1   = n - 1;
  none = -1;
  MINPACKnumsrt(&n,&n1,seq,&none,list,work+2*n,work+n);
  coloring = (int *) PetscMalloc(n*sizeof(int)); CHKPTRQ(coloring);
  MINPACKseq(&n,cja,cia,rja,ria,list,coloring,&ncolors,work);

  PetscFree(list);
  PetscFree(seq);

  ierr = MatRestoreRowIJ(mat,1,PETSC_FALSE,&n,&ria,&rja,&done);CHKERRQ(ierr);
  ierr = MatRestoreColumnIJ(mat,1,PETSC_FALSE,&n,&cia,&cja,&done);CHKERRQ(ierr);

  ierr = MatColoringPatch(mat,ncolors,coloring,iscoloring); CHKERRQ(ierr);
  PetscFree(coloring);
  PetscFunctionReturn(0);
}

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

  ierr = MatFDColoringDegreeSequence_Minpack(n,cja,cia,rja,ria,&seq); CHKERRQ(ierr);

  list = (int*) PetscMalloc( 5*n*sizeof(int) ); CHKPTRQ(list);
  work = list + n;

  MINPACKido(&n,&n,cja,cia,rja,ria,seq,list,&clique,work,work+n,work+2*n,work+3*n);

  coloring = (int *) PetscMalloc(n*sizeof(int)); CHKPTRQ(coloring);
  MINPACKseq(&n,cja,cia,rja,ria,list,coloring,&ncolors,work);

  PetscFree(list);
  PetscFree(seq);

  ierr = MatRestoreRowIJ(mat,1,PETSC_FALSE,&n,&ria,&rja,&done);CHKERRQ(ierr);
  ierr = MatRestoreColumnIJ(mat,1,PETSC_FALSE,&n,&cia,&cja,&done);CHKERRQ(ierr);

  ierr = MatColoringPatch(mat,ncolors,coloring,iscoloring); CHKERRQ(ierr);

  PetscFree(coloring);
  PetscFunctionReturn(0);
}

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
  ierr = MatGetSize(mat,&N,&N); CHKERRQ(ierr);
  is  = (IS *) PetscMalloc( N*sizeof(IS*) ); CHKPTRQ(is); 
  *iscoloring       = (ISColoring) PetscMalloc(sizeof(struct _p_ISColoring));CHKPTRQ(*iscoloring);
  (*iscoloring)->n  = N;
  (*iscoloring)->is = is;
  
  ierr = MatGetOwnershipRange(mat,&start,&end); CHKERRQ(ierr);
  for ( i=0; i<start; i++ ) {
    ierr = ISCreateGeneral(PETSC_COMM_SELF,0,PETSC_NULL,is+i); CHKERRQ(ierr);
  }
  for ( i=start; i<end; i++ ) {
    ierr = ISCreateGeneral(PETSC_COMM_SELF,1,&i,is+i); CHKERRQ(ierr);
  }
  for ( i=end; i<N; i++ ) {
    ierr = ISCreateGeneral(PETSC_COMM_SELF,0,PETSC_NULL,is+i); CHKERRQ(ierr);
  }
  ierr = PetscObjectGetComm((PetscObject)mat,&comm);CHKERRQ(ierr);
  ierr = PetscCommDuplicate_Private(comm,&(*iscoloring)->comm,&tag);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
  
/* ===========================================================================================*/

#include "src/sys/nreg.h"
#include "sys.h"

static NRList *__MatColoringList = 0;
int MatColoringRegisterAllCalled = 0;

#undef __FUNC__  
#define __FUNC__ "MatColoringRegister" 
/*@C
   MatColoringRegister - Adds a new sparse matrix coloring to the 
   matrix package. 

   Input Parameters:
.  name - name of coloring (for example COLORING_SL) or COLORING_NEW
.  sname -  corresponding string for name
.  order - routine that does coloring

   Output Parameters:
.  oname - number associated with the coloring (for example COLORING_SL)

   Level: developer

   Not Collective

.keywords: matrix, coloring, register

.seealso: MatColoringRegisterDestroy(), MatColoringRegisterAll()
@*/
int MatColoringRegister(MatColoringType name,MatColoringType *oname,char *sname,int (*color)(Mat,MatColoringType,ISColoring*))
{
  int         ierr;
  static int  numberregistered = 0;

  PetscFunctionBegin;
  if (!__MatColoringList) {
    ierr = NRCreate(&__MatColoringList); CHKERRQ(ierr);
  }

  if (name == COLORING_NEW) name = (MatColoringType) ((int) COLORING_NEW + numberregistered++);
  if (oname) *oname = name;
  ierr = NRRegister(__MatColoringList,(int)name,sname,(int (*)(void*))color);CHKERRQ(ierr);
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
  PetscFunctionBegin;
  if (__MatColoringList) {
    NRDestroy( __MatColoringList );
    __MatColoringList = 0;
  }
  MatColoringRegisterAllCalled = 0;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetColoringTypeFromOptions" 
/*@C
   MatGetColoringTypeFromOptions - Gets matrix coloring method from the
   options database.

   Not Collective

   Input Parameter:
.  prefix - optional database prefix

   Output Parameter:
.  type - coloring method

   Options Database Keys:
$    -mat_coloring natural, -mat_coloring sl, -mat_coloring id, 
$    -mat_coloring lf

   Level: intermediate

.keywords: matrix, coloring, 

.seealso: MatGetColoring()
@*/
int MatGetColoringTypeFromOptions(char *prefix,MatColoringType *type)
{
  char sbuf[50];
  int  ierr,flg;
  
  PetscFunctionBegin;
  ierr = OptionsGetString(prefix,"-mat_coloring", sbuf, 50,&flg); CHKERRQ(ierr);
  if (flg) {
    if (!__MatColoringList) MatColoringRegisterAll();
    *type = (MatColoringType)NRFindID( __MatColoringList, sbuf );
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatColoringGetName" 
/*@C
   MatColoringGetName - Gets the name associated with a coloring.

   Not Collective

   Input Parameter:
.  coloringing - integer name of coloring

   Output Parameter:
.  name - name of coloring

   Level: advanced

.keywords: matrix, get, coloring, name
@*/
int MatColoringGetName(MatColoringType meth,char **name)
{
  int ierr;

  PetscFunctionBegin;
  if (!__MatColoringList) {ierr = MatColoringRegisterAll(); CHKERRQ(ierr);}
   *name = NRFindName( __MatColoringList, (int)meth );
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
$      COLORING_NATURAL - natural
$      COLORING_SL - smallest-last
$      COLORING_LF - largest-first
$      COLORING_ID - incidence-degree

   Output Parameters:
.   iscoloring - the coloring

   Options Database Keys:
   To specify the coloring through the options database, use one of
   the following 
$    -mat_coloring natural, -mat_coloring sl, -mat_coloring lf,
$    -mat_coloring id
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for factored matrix"); 
  if (!MatColoringRegisterAllCalled) {
    ierr = MatColoringRegisterAll();CHKERRQ(ierr);
  }

  ierr = MatGetColoringTypeFromOptions(0,&type); CHKERRQ(ierr);
  PLogEventBegin(MAT_GetColoring,mat,0,0,0);
  r =  (int (*)(Mat,MatColoringType,ISColoring*))NRFindRoutine(__MatColoringList,(int)type,(char *)0);
  if (!r) {SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Unknown or unregistered type");}
  ierr = (*r)(mat,type,iscoloring); CHKERRQ(ierr);
  PLogEventEnd(MAT_GetColoring,mat,0,0,0);

  PLogInfo((PetscObject)mat,"MatGetColoring:Number of colors %d\n",(*iscoloring)->n);
  ierr = OptionsHasName(PETSC_NULL,"-mat_coloring_view",&flag);
  if (flag) {
    ierr = ISColoringView(*iscoloring,VIEWER_STDOUT_((*iscoloring)->comm));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
 
