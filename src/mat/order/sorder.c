
#include "../../../../matimpl.h"
#include "sys/nreg.h"
#include "sys.h"

static NRList *__MatReorderingList = 0;

int MatGetReorder_IJ(int n,int *ia,int* ja,MatOrdering type,
                            IS *rperm, IS *cperm)
{
  int  ierr,*permr,*permc;
  int  (*r)(int*,int*,int*,int*,int*);

  permr = (int *) PETSCMALLOC( 2*n*sizeof(int) ); CHKPTRQ(permr);
  permc = permr + n;

  /* Get the function pointers for the method requested */
  if (!__MatReorderingList) {MatReorderingRegisterAll();}
  if (!__MatReorderingList) {
    SETERRQ(1,"MatGetReorder_IJ: Could not acquire list of methods"); 
  }
  r =  (int (*)(int*,int*,int*,int*,int*))NRFindRoutine( 
                              __MatReorderingList,(int)type,(char *)0 );
  if (!r) {SETERRQ(1,"MatGetReorder_IJ: Unknown method");}

  ierr = (*r)(&n,ia,ja,permr,permc); CHKERRQ(ierr);

  ierr = ISCreateSequential(MPI_COMM_SELF,n,permr,rperm); CHKERRQ(ierr);
  ISSetPermutation(*rperm);
  ierr = ISCreateSequential(MPI_COMM_SELF,n,permc,cperm); CHKERRQ(ierr);
  ISSetPermutation(*cperm);
  PETSCFREE(permr); 
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

.seealso: MatReorderingRegisterDestroy()
@*/
int  MatReorderingRegister(MatOrdering name,char *sname,
                          int (*order)(int*,int*,int*,int*,int*))
{
  int ierr;
  if (!__MatReorderingList) {
    ierr = NRCreate(&__MatReorderingList); CHKERRQ(ierr);
  }
  return NRRegister(__MatReorderingList,(int)name,sname,(int (*)(void*))order);
}

/*@
   MatReorderingRegisterDestroy - Frees the list of ordering routines.

.keywords: matrix, register, destroy

.seealso: MatReorderingRegister()
@*/
int MatReorderingRegisterDestroy()
{
  if (__MatReorderingList) {
    NRDestroy( __MatReorderingList );
    __MatReorderingList = 0;
  }
  return 0;
}

/*
   MatGetReorderingMethodFromOptions_Private - Gets reorder method 
     from the options database.

   Output Parameter:
.  method - reordering method

   Returns:
   Returns 1 if the method is found; 0 otherwise.
*/
int MatGetReorderingMethodFromOptions_Private(MatOrdering *type)
{
  char sbuf[50];
  if (OptionsGetString(0,"-mat_order", sbuf, 50 )) {
    if (!__MatReorderingList) MatReorderingRegisterAll();
    *type = (MatOrdering)NRFindID( __MatReorderingList, sbuf );
    return 1;
  }
  return 0;
}
