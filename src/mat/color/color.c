 
/*
     Routines that call the kernel minpack coloring subroutines
*/

#include "petsc.h"
#include "mat.h"
#include "src/mat/impls/color/color.h"

/*
    MatFDColoringDegreeSequence_Minpack - Calls the MINPACK routine seqr() that
      computes the degree sequence required by MINPACK coloring routines.
*/
int MatFDColoringDegreeSequence_Minpack(int m,int *ria, int *rja, int *cia, int *cja, int **seq)
{
  int *work;

  work = (int *) PetscMalloc( m*sizeof(int) ); CHKPTRQ(work);  
  *seq = (int *) PetscMalloc( m*sizeof(int) ); CHKPTRQ(*seq);

  MINPACKdegr(&m,rja,ria,cja,cia,*seq,work);

  PetscFree(work);
  return 0;
}

/*
    MatFDColoringMinimumNumberofColors_Private - For a given sparse 
        matrix computes the minimum number of colors needed.

*/
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
int MatFDColoringSL_Minpack(Mat mat,MatColoring name,int *ncolors,IS **is)
{
  int        *list,*work,clique,ierr,*ria,*rja,*cia,*cja,*seq,*coloring,n,*sizes,**ii,i;
  PetscTruth done;

  ierr = MatGetRowIJ(mat,1,PETSC_FALSE,&n,&ria,&rja,&done);CHKERRQ(ierr);
  ierr = MatGetColumnIJ(mat,1,PETSC_FALSE,&n,&cia,&cja,&done);CHKERRQ(ierr);
  if (!done) SETERRQ(1,"MatFDColoringSL_Minpack:Ordering requires IJ");

  ierr = MatFDColoringDegreeSequence_Minpack(n,ria,rja,cia,cja,&seq); CHKERRQ(ierr);

  list = (int*) PetscMalloc( 5*n*sizeof(int) ); CHKPTRQ(list);
  work = list + n;

  MINPACKslo(&n,rja,ria,cja,cia,seq,list,&clique,work,work+n,work+2*n,work+3*n);

  coloring = (int *) PetscMalloc(n*sizeof(int)); CHKPTRQ(coloring);
  MINPACKseq(&n,rja,ria,cja,cia,list,coloring,ncolors,work);

  PetscFree(list);
  PetscFree(seq);

  ierr = MatRestoreRowIJ(mat,1,PETSC_FALSE,&n,&ria,&rja,&done);CHKERRQ(ierr);
  ierr = MatRestoreColumnIJ(mat,1,PETSC_FALSE,&n,&cia,&cja,&done);CHKERRQ(ierr);

  /* construct the index sets from the coloring array */
  sizes = (int *) PetscMalloc( *ncolors*sizeof(int) ); CHKPTRQ(sizes);
  PetscMemzero(sizes,*ncolors*sizeof(int));
  for ( i=0; i<n; i++ ) {
    sizes[coloring[i]-1]++;
  }
  ii    = (int **) PetscMalloc( *ncolors*sizeof(int*) ); CHKPTRQ(ii);
  ii[0] = (int *) PetscMalloc( n*sizeof(int) ); CHKPTRQ(ii[0]);
  for ( i=1; i<*ncolors; i++ ) {
    ii[i] = ii[i-1] + sizes[i-1];
  }
  PetscMemzero(sizes,*ncolors*sizeof(int));
  for ( i=0; i<n; i++ ) {
    ii[coloring[i]-1][sizes[coloring[i]-1]++] = i;
  }
  *is  = (IS *) PetscMalloc( *ncolors*sizeof(is) ); CHKPTRQ(*is);
  for ( i=0; i<*ncolors; i++ ) {
    ierr = ISCreateGeneral(MPI_COMM_SELF,sizes[i],ii[i],(*is)+i); CHKERRQ(ierr);
  }

  PetscFree(sizes);
  PetscFree(ii[0]);
  PetscFree(ii);
  PetscFree(coloring);
  return 0;
}


/*
   Simplest coloring, each column of the matrix gets its own unique color.
*/
int MatColoring_Natural(Mat mat,MatColoring color, int *n,IS** is)
{
  int N,start,end,ierr,i;

  ierr = MatGetSize(mat,&N,&N); CHKERRQ(ierr);
  *n   = N;
  *is  = (IS *) PetscMalloc( N*sizeof(is) ); CHKPTRQ(*is);
  ierr = MatGetOwnershipRange(mat,&start,&end); CHKERRQ(ierr);
  for ( i=0; i<start; i++ ) {
    ierr = ISCreateGeneral(MPI_COMM_SELF,0,PETSC_NULL,(*is)+i); CHKERRQ(ierr);
  }
  for ( i=start; i<end; i++ ) {
    ierr = ISCreateGeneral(MPI_COMM_SELF,1,&i,(*is)+i); CHKERRQ(ierr);
  }
  for ( i=end; i<N; i++ ) {
    ierr = ISCreateGeneral(MPI_COMM_SELF,0,PETSC_NULL,(*is)+i); CHKERRQ(ierr);
  }
  return 0;
}
  
/* ===========================================================================================*/

#include "src/sys/nreg.h"
#include "sys.h"

static NRList *__MatColoringList = 0;

/*@C
   MatColoringRegister - Adds a new sparse matrix coloring to the 
   matrix package. 

   Input Parameters:
.  sname -  corresponding string for name
.  order - routine that does coloring

   Output Parameters:
.  name - number associated with the coloring (for example COLORING_SL)

.keywords: matrix, coloring, register

.seealso: MatColoringRegisterDestroy(), MatColoringRegisterAll()
@*/
int  MatColoringRegister(MatColoring *name,char *sname,int (*color)(Mat,MatColoring,int*,IS**))
{
  int         ierr;
  static int  numberregistered = 0;

  if (!__MatColoringList) {
    ierr = NRCreate(&__MatColoringList); CHKERRQ(ierr);
    numberregistered = 0;
  }

  /*
       This is tacky, it forces the standard coloringing routines to 
     be registered before any user provided. This is so the predefined 
     types like COLORING_NATURAL match their positions in the list of 
     registered coloringings.
  */
  if (numberregistered == 0 && color != MatColoring_Natural) {
    MatColoringRegisterAll();
  }

  *name = (MatColoring) numberregistered++;
  ierr = NRRegister(__MatColoringList,(int)*name,sname,(int (*)(void*))color);CHKERRQ(ierr);
  return 0;
}

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
  return 0;
}

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
$    -mat_coloring natural, -mat_coloring sl, -mat_coloring ld, 
$    -mat_coloring if

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

/*@C
   MatColoringGetName - Gets the name associated with a coloring.

   Input Parameter:
.  coloringing - integer name of coloring

   Output Parameter:
.  name - name of coloring

.keywords: PC, get, method, name, type
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

/*@C
   MatGetColoring - Gets a coloring for a matrix to reduce fill or to
   improve numerical stability of LU factorization.

   Input Parameters:
.  mat - the matrix
.  type - type of coloring, one of the following:
$      COLORING_NATURAL - Natural
$      COLORING_LD
$      COLORING_SL
$      COLORING_IF

   Output Parameters:
.  n - the number of colors
.  is - array of index sets defining colors

   Options Database Keys:
   To specify the coloringing through the options database, use one of
   the following 
$    -mat_coloring natural, -mat_coloring sl, -mat_coloring ld,
$    -mat_coloring if

   The user can define additional coloringings; see MatColoringRegister().

.keywords: matrix, coloringing

.seealso:  MatGetColoringTypeFromOptions(), MatColoringRegister()
@*/
int MatGetColoring(Mat mat,MatColoring type,int *ncolors,IS **is)
{
  int         ierr;
  int         (*r)(Mat,MatColoring,int*,IS**);

  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (!mat->assembled) SETERRQ(1,"MatGetColoring:Not for unassembled matrix");
  if (mat->factor) SETERRQ(1,"MatGetColoring:Not for factored matrix"); 
  if (!__MatColoringList) {
    ierr = MatColoringRegisterAll();CHKERRQ(ierr);
  }

  ierr = MatGetColoringTypeFromOptions(0,&type); CHKERRQ(ierr);
  PLogEventBegin(MAT_GetColoring,mat,0,0,0);
  r =  (int (*)(Mat,MatColoring,int*,IS**))NRFindRoutine(__MatColoringList,(int)type,(char *)0);
  if (!r) {SETERRQ(1,"MatGetColoring:Unknown type");}

  ierr = (*r)(mat,type,ncolors,is); CHKERRQ(ierr);

  PLogEventEnd(MAT_GetColoring,mat,0,0,0);
  return 0;
}

extern int MatColoring_Natural(Mat,MatColoring,int*,IS**);
extern int MatFDColoringSL_Minpack(Mat,MatColoring,int *,IS **);

/*@C
  MatColoringRegisterAll - Registers all of the matrix coloring routines in PETSc.


.keywords: matrix, coloring, register, all

.seealso: MatColoringRegister(), MatColoringRegisterDestroy()
@*/
int MatColoringRegisterAll()
{
  int         ierr;
  MatColoring name;
  static int  called = 0;
  if (called) return 0; else called = 1;

  /*
       Do not change the order of these unless similarly changing 
    them in include/mat.h
  */
  ierr = MatColoringRegister(&name,"natural",MatColoring_Natural);CHKERRQ(ierr);
  ierr = MatColoringRegister(&name,"sl",MatFDColoringSL_Minpack);CHKERRQ(ierr);

  return 0;
}



