#ifndef lint
static char vcid[] = "$Id: sorder.c,v 1.22 1996/08/08 14:43:21 bsmith Exp bsmith $";
#endif
/*
     Provides the code that allows PETSc users to register their own
  sequential matrix reordering routines.
*/
#include "src/mat/matimpl.h"
#include "src/sys/nreg.h"
#include "sys.h"

static NRList *__MatReorderingList = 0;

PetscTruth MatReorderingRequiresSymmetric[25];
int        MatReorderingIndexShift[25];

int MatGetReordering_IJ(int n,int *ia,int* ja,MatReordering type,IS *rperm,IS *cperm)
{
  int  ierr,*permr,*permc,(*r)(int*,int*,int*,int*,int*);

  permr = (int *) PetscMalloc( (2*n+1)*sizeof(int) ); CHKPTRQ(permr);
  permc = permr + n;

  /* Get the function pointers for the method requested */
  if (!__MatReorderingList) {MatReorderingRegisterAll();}
  if (!__MatReorderingList) {
    SETERRQ(1,"MatGetReordering_IJ:Could not get list of methods"); 
  }
  r =  (int (*)(int*,int*,int*,int*,int*))NRFindRoutine( 
                              __MatReorderingList,(int)type,(char *)0 );
  if (!r) {SETERRQ(1,"MatGetReordering_IJ:Unknown method");}

  ierr = (*r)(&n,ia,ja,permr,permc); CHKERRQ(ierr);

  ierr = ISCreateGeneral(MPI_COMM_SELF,n,permr,rperm); CHKERRQ(ierr);
  ISSetPermutation(*rperm);
  ierr = ISCreateGeneral(MPI_COMM_SELF,n,permc,cperm); CHKERRQ(ierr);
  ISSetPermutation(*cperm);
  PetscFree(permr); 

  /* 
     this is tacky: In the future when we have written special factorization
     and solve routines for the identity permutation we should use a 
     stride index set instead of the general one.
  */
  if (type == ORDER_NATURAL) {
    ISSetIdentity(*rperm);
    ISSetIdentity(*cperm);
  }

  return 0; 
}

int MatOrder_Natural(int *N,int *ia,int* ja, int* permr, int* permc)
{
  int n = *N, i;
  for ( i=0; i<n; i++ ) permr[i] = permc[i] = i;
  return 0;
}

/*
     Orders the rows (and columns) by the lengths of the rows. 
   This produces a symmetric reordering but does not require a 
   matrix with symmetric non-zero structure.
*/
int MatOrder_RowLength(int *N,int *ia,int* ja, int* permr, int* permc)
{
  int n = *N, i, *lens,ierr;

  lens = (int *) PetscMalloc( n*sizeof(int) ); CHKPTRQ(lens);
  for ( i=0; i<n; i++ ) { 
    lens[i]  = ia[i+1] - ia[i];
    permr[i] = i;
  }

  ierr = PetscSortIntWithPermutation(n, lens, permr); CHKERRQ(ierr);
  PetscFree(lens);

  /* column permutations get same as row */
  for ( i=0; i<n; i++ ) { 
    permc[i] = permr[i];
  }
  return 0;
}

extern int MatOrder_Natural(int*,int*,int*,int*,int*);
/*@C
   MatReorderingRegister - Adds a new sparse matrix reordering to the 
   matrix package. This is only for adding reordering for sequential 
   matrices. The reordering routine has input in the usual compressed
   row storage format with indices starting at zero (indexshift == 0)
   or one (indexshift == 1).

   Input Parameters:
.  sname -  corresponding string for name
.  sym - PETSC_TRUE if requires symmetric nonzero structure, else PETSC_FALSE.
.  indexshift - 0 or 1 depending on first index your program expects.
.  order - routine that does reordering

   Output Parameters:
.  name - number associated with the reordering (for example ORDER_ND)

.keywords: matrix, reordering, register

.seealso: MatReorderingRegisterDestroy(), MatReorderingRegisterAll()
@*/
int  MatReorderingRegister(MatReordering *name,char *sname,PetscTruth sym,int shift,
                           int (*order)(int*,int*,int*,int*,int*))
{
  int         ierr;
  static int  numberregistered = 0;

  if (!__MatReorderingList) {
    ierr = NRCreate(&__MatReorderingList); CHKERRQ(ierr);
    numberregistered = 0;
  }

  /*
       This is tacky, it forces the standard ordering routines to 
     be registered before any user provided. This is so the predefined 
     types like ORDER_NATURAL match their positions in the list of 
     registered orderings.
  */
  if (numberregistered == 0 && order != MatOrder_Natural) {
    MatReorderingRegisterAll();
  }

  *name = (MatReordering) numberregistered++;
  ierr = NRRegister(__MatReorderingList,(int)*name,sname,(int (*)(void*))order);
  CHKERRQ(ierr);
  MatReorderingRequiresSymmetric[(int)*name] = sym;
  MatReorderingIndexShift[(int)*name]        = shift;
  return 0;
}

/*@C
   MatReorderingRegisterDestroy - Frees the list of ordering routines.

.keywords: matrix, register, destroy

.seealso: MatReorderingRegister(), MatReorderingRegisterAll()
@*/
int MatReorderingRegisterDestroy()
{
  if (__MatReorderingList) {
    NRDestroy( __MatReorderingList );
    __MatReorderingList = 0;
  }
  return 0;
}

/*@C
   MatGetReorderingTypeFromOptions - Gets matrix reordering method from the
   options database.

   Input Parameter:
.  prefix - optional database prefix

   Output Parameter:
.  type - reordering method

   Options Database Keys:
   To specify the ordering through the options database, use one of
   the following 
$    -mat_order natural, -mat_order nd, -mat_order 1wd, 
$    -mat_order rcm, -mat_order qmd

.keywords: matrix, set, ordering, factorization, direct, ILU, LU,
           fill, reordering, natural, Nested Dissection,
           One-way Dissection, Cholesky, Reverse Cuthill-McGee, 
           Quotient Minimum Degree

.seealso: MatGetReordering()
@*/
int MatGetReorderingTypeFromOptions(char *prefix,MatReordering *type)
{
  char sbuf[50];
  int  ierr,flg;
  
  ierr = OptionsGetString(prefix,"-mat_order", sbuf, 50,&flg); CHKERRQ(ierr);
  if (flg) {
    if (!__MatReorderingList) MatReorderingRegisterAll();
    *type = (MatReordering)NRFindID( __MatReorderingList, sbuf );
  }
  return 0;
}

/*@C
   MatReorderingGetName - Gets the name associated with a reordering.

   Input Parameter:
.  ordering - integer name of reordering

   Output Parameter:
.  name - name of reordering

.keywords: PC, get, method, name, type
@*/
int MatReorderingGetName(MatReordering meth,char **name)
{
  int ierr;
  if (!__MatReorderingList) {ierr = MatReorderingRegisterAll(); CHKERRQ(ierr);}
   *name = NRFindName( __MatReorderingList, (int)meth );
  return 0;
}








