#ifndef lint
static char vcid[] = "$Id: color.c,v 1.15 1997/02/22 02:26:05 bsmith Exp curfman $";
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
#define __FUNC__ "MatFDColoringDegreeSequence_Minpack" /* ADIC Ignore */
int MatFDColoringDegreeSequence_Minpack(int m,int *cja, int *cia, int *rja, int *ria, int **seq)
{
  int *work;

  work = (int *) PetscMalloc( m*sizeof(int) ); CHKPTRQ(work);  
  *seq = (int *) PetscMalloc( m*sizeof(int) ); CHKPTRQ(*seq);

  MINPACKdegr(&m,cja,cia,rja,ria,*seq,work);

  PetscFree(work);
  return 0;
}

/*
    MatFDColoringMinimumNumberofColors_Private - For a given sparse 
        matrix computes the minimum number of colors needed.

*/
#undef __FUNC__  
#define __FUNC__ "MatFDColoringMinimumNumberofColors_Private" /* ADIC Ignore */
int MatFDColoringMinimumNumberofColors_Private(int m,int *ia,int *minc)
{
  int i,c = 0;

  for ( i=0; i<m; i++ ) {
    c = PetscMax(c,ia[i+1]-ia[i]);
  }
  *minc = c;
  return 0;
}

/* ----------------------------------------------------------------------------*/
/*
    MatFDColoringSL_Minpack - Uses the smallest-last (SL) coloring of minpack
*/
#undef __FUNC__  
#define __FUNC__ "MatFDColoringSL_Minpack" /* ADIC Ignore */
int MatFDColoringSL_Minpack(Mat mat,MatColoring name,ISColoring *iscoloring)
{
  int        *list,*work,clique,ierr,*ria,*rja,*cia,*cja,*seq,*coloring,n;
  int        ncolors;
  PetscTruth done;

  ierr = MatGetRowIJ(mat,1,PETSC_FALSE,&n,&ria,&rja,&done);CHKERRQ(ierr);
  ierr = MatGetColumnIJ(mat,1,PETSC_FALSE,&n,&cia,&cja,&done);CHKERRQ(ierr);
  if (!done) SETERRQ(1,0,"Ordering requires IJ");

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
  return 0;
}

/* ----------------------------------------------------------------------------*/
/*
    MatFDColoringLF_Minpack - 
*/
#undef __FUNC__  
#define __FUNC__ "MatFDColoringLF_Minpack" /* ADIC Ignore */
int MatFDColoringLF_Minpack(Mat mat,MatColoring name,ISColoring *iscoloring)
{
  int        *list,*work,ierr,*ria,*rja,*cia,*cja,*seq,*coloring,n;
  int        n1, none,ncolors;
  PetscTruth done;

  ierr = MatGetRowIJ(mat,1,PETSC_FALSE,&n,&ria,&rja,&done);CHKERRQ(ierr);
  ierr = MatGetColumnIJ(mat,1,PETSC_FALSE,&n,&cia,&cja,&done);CHKERRQ(ierr);
  if (!done) SETERRQ(1,0,"Ordering requires IJ");

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
  return 0;
}

/* ----------------------------------------------------------------------------*/
/*
    MatFDColoringID_Minpack - 
*/
#undef __FUNC__  
#define __FUNC__ "MatFDColoringID_Minpack" /* ADIC Ignore */
int MatFDColoringID_Minpack(Mat mat,MatColoring name,ISColoring *iscoloring)
{
  int        *list,*work,clique,ierr,*ria,*rja,*cia,*cja,*seq,*coloring,n;
  int        ncolors;
  PetscTruth done;

  ierr = MatGetRowIJ(mat,1,PETSC_FALSE,&n,&ria,&rja,&done);CHKERRQ(ierr);
  ierr = MatGetColumnIJ(mat,1,PETSC_FALSE,&n,&cia,&cja,&done);CHKERRQ(ierr);
  if (!done) SETERRQ(1,0,"Ordering requires IJ");

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
  return 0;
}

/*
   Simplest coloring, each column of the matrix gets its own unique color.
*/
#undef __FUNC__  
#define __FUNC__ "MatColoring_Natural" /* ADIC Ignore */
int MatColoring_Natural(Mat mat,MatColoring color, ISColoring *iscoloring)
{
  int N,start,end,ierr,i;
  IS  *is;

  ierr = MatGetSize(mat,&N,&N); CHKERRQ(ierr);
  is  = (IS *) PetscMalloc( N*sizeof(IS*) ); CHKPTRQ(is); 
  *iscoloring       = (ISColoring) PetscMalloc(sizeof(struct _ISColoring));CHKPTRQ(*iscoloring);
  (*iscoloring)->n  = N;
  (*iscoloring)->is = is;
  
  ierr = MatGetOwnershipRange(mat,&start,&end); CHKERRQ(ierr);
  for ( i=0; i<start; i++ ) {
    ierr = ISCreateGeneral(MPI_COMM_SELF,0,PETSC_NULL,is+i); CHKERRQ(ierr);
  }
  for ( i=start; i<end; i++ ) {
    ierr = ISCreateGeneral(MPI_COMM_SELF,1,&i,is+i); CHKERRQ(ierr);
  }
  for ( i=end; i<N; i++ ) {
    ierr = ISCreateGeneral(MPI_COMM_SELF,0,PETSC_NULL,is+i); CHKERRQ(ierr);
  }
  return 0;
}
  
/* ===========================================================================================*/

#include "src/sys/nreg.h"
#include "sys.h"

static NRList *__MatColoringList = 0;
int MatColoringRegisterAllCalled = 0;

#undef __FUNC__  
#define __FUNC__ "MatColoringRegister" /* ADIC Ignore */
/*@C
   MatColoringRegister - Adds a new sparse matrix coloring to the 
   matrix package. 

   Input Parameters:
.  name - name of coloring (for example COLORING_SL) or COLORING_NEW
.  sname -  corresponding string for name
.  order - routine that does coloring

   Output Parameters:
.  oname - number associated with the coloring (for example COLORING_SL)

.keywords: matrix, coloring, register

.seealso: MatColoringRegisterDestroy(), MatColoringRegisterAll()
@*/
int MatColoringRegister(MatColoring name,MatColoring *oname,char *sname,int (*color)(Mat,MatColoring,ISColoring*))
{
  int         ierr;
  static int  numberregistered = 0;

  if (!__MatColoringList) {
    ierr = NRCreate(&__MatColoringList); CHKERRQ(ierr);
  }

  if (name == COLORING_NEW) name = (MatColoring) ((int) COLORING_NEW + numberregistered++);
  if (oname) *oname = name;
  ierr = NRRegister(__MatColoringList,(int)name,sname,(int (*)(void*))color);CHKERRQ(ierr);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatColoringRegisterDestroy" /* ADIC Ignore */
/*@C
   MatColoringRegisterDestroy - Frees the list of coloringing routines.

.keywords: matrix, register, destroy

.seealso: MatColoringRegister(), MatColoringRegisterAll()
@*/
int MatColoringRegisterDestroy()
{
  if (__MatColoringList) {
    NRDestroy( __MatColoringList );
    __MatColoringList = 0;
  }
  MatColoringRegisterAllCalled = 0;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatGetColoringTypeFromOptions" /* ADIC Ignore */
/*@C
   MatGetColoringTypeFromOptions - Gets matrix coloring method from the
   options database.

   Input Parameter:
.  prefix - optional database prefix

   Output Parameter:
.  type - coloring method

   Options Database Keys:
   To specify the coloringing through the options database, use one of
   the following 
$    -mat_coloring natural, -mat_coloring sl, -mat_coloring id, 
$    -mat_coloring lf

.keywords: matrix, coloring, 

.seealso: MatGetColoring()
@*/
int MatGetColoringTypeFromOptions(char *prefix,MatColoring *type)
{
  char sbuf[50];
  int  ierr,flg;
  
  ierr = OptionsGetString(prefix,"-mat_coloring", sbuf, 50,&flg); CHKERRQ(ierr);
  if (flg) {
    if (!__MatColoringList) MatColoringRegisterAll();
    *type = (MatColoring)NRFindID( __MatColoringList, sbuf );
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatColoringGetName" /* ADIC Ignore */
/*@C
   MatColoringGetName - Gets the name associated with a coloring.

   Input Parameter:
.  coloringing - integer name of coloring

   Output Parameter:
.  name - name of coloring

.keywords: matrix, get, coloring, name
@*/
int MatColoringGetName(MatColoring meth,char **name)
{
  int ierr;
  if (!__MatColoringList) {ierr = MatColoringRegisterAll(); CHKERRQ(ierr);}
   *name = NRFindName( __MatColoringList, (int)meth );
  return 0;
}

#include "src/mat/matimpl.h"
extern int MatAdjustForInodes(Mat,IS *,IS *);

#undef __FUNC__  
#define __FUNC__ "MatGetColoring" /* ADIC Ignore */
/*@C
   MatGetColoring - Gets a coloring for a matrix to reduce fill or to
   improve numerical stability of LU factorization.

   Input Parameters:
.  mat - the matrix
.  type - type of coloring, one of the following:
$      COLORING_NATURAL - natural
$      COLORING_SL - smallest last
$      COLORING_LF - largest-first
$      COLORING_ID - incidence-degree

   Output Parameters:
.   iscoloring - the coloring

   Options Database Keys:
   To specify the coloring through the options database, use one of
   the following 
$    -mat_coloring natural, -mat_coloring sl, -mat_coloring lf,
$    -mat_coloring id

   The user can define additional colorings; see MatColoringRegister().

.keywords: matrix, get, coloring

.seealso:  MatGetColoringTypeFromOptions(), MatColoringRegister()
@*/
int MatGetColoring(Mat mat,MatColoring type,ISColoring *iscoloring)
{
  int         ierr,flag;
  int         (*r)(Mat,MatColoring,ISColoring *);

  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (!mat->assembled) SETERRQ(1,0,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(1,0,"Not for factored matrix"); 
  if (!MatColoringRegisterAllCalled) {
    ierr = MatColoringRegisterAll();CHKERRQ(ierr);
  }

  ierr = MatGetColoringTypeFromOptions(0,&type); CHKERRQ(ierr);
  PLogEventBegin(MAT_GetColoring,mat,0,0,0);
  r =  (int (*)(Mat,MatColoring,ISColoring*))NRFindRoutine(__MatColoringList,(int)type,(char *)0);
  if (!r) {SETERRQ(1,0,"Unknown or unregistered type");}
  ierr = (*r)(mat,type,iscoloring); CHKERRQ(ierr);
  PLogEventEnd(MAT_GetColoring,mat,0,0,0);

  PLogInfo((PetscObject)mat,"MatGetColoring:Number of colors %d\n",(*iscoloring)->n);
  ierr = OptionsHasName(PETSC_NULL,"-matcoloring_view",&flag);
  if (flag) {
    Viewer viewer;
    ierr = ViewerFileOpenASCII(mat->comm,"stdout",&viewer);CHKERRQ(ierr);
    ierr = ISColoringView(*iscoloring,viewer);CHKERRQ(ierr);
    ierr = ViewerDestroy(viewer); CHKERRQ(ierr);
  }
  return 0;
}
 
