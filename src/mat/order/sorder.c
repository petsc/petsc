#ifndef lint
static char vcid[] = "$Id: sorder.c,v 1.13 1995/11/01 23:19:07 bsmith Exp bsmith $";
#endif
/*
     Provides the code that allows PETSc users to register their own
  sequential matrix reordering routines.
*/
#include "../../matimpl.h"
#include "sys/nreg.h"
#include "sys.h"

static NRList *__MatReorderingList = 0;

int MatGetReordering_IJ(int n,int *ia,int* ja,MatOrdering type,IS *rperm, IS *cperm)
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

  ierr = ISCreateSeq(MPI_COMM_SELF,n,permr,rperm); CHKERRQ(ierr);
  ISSetPermutation(*rperm);
  ierr = ISCreateSeq(MPI_COMM_SELF,n,permc,cperm); CHKERRQ(ierr);
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

int MatOrderNatural(int *N,int *ia,int* ja, int* permr, int* permc)
{
  int n = *N, i;
  for ( i=0; i<n; i++ ) permr[i] = permc[i] = i;
  return 0;
}

/*@C
   MatReorderingRegister - Adds a new sparse matrix reordering to the 
   matrix package.

   Input Parameters:
.  name - for instance ORDER_ND, ...
.  sname -  corresponding string for name
.  order - routine that does reordering

.keywords: matrix, reordering, register

.seealso: MatReorderingRegisterDestroy(), MatReorderingRegisterAll()
@*/
int  MatReorderingRegister(MatOrdering name,char *sname,int (*order)(int*,int*,int*,int*,int*))
{
  int ierr;
  if (!__MatReorderingList) {
    ierr = NRCreate(&__MatReorderingList); CHKERRQ(ierr);
  }
  return NRRegister(__MatReorderingList,(int)name,sname,(int (*)(void*))order);
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

/*@
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
int MatGetReorderingTypeFromOptions(char *prefix,MatOrdering *type)
{
  char sbuf[50];
  if (OptionsGetString(prefix,"-mat_order", sbuf, 50 )) {
    if (!__MatReorderingList) MatReorderingRegisterAll();
    *type = (MatOrdering)NRFindID( __MatReorderingList, sbuf );
  }
  return 0;
}
