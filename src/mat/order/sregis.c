
#include "../../../../matimpl.h"

extern int SpOrderND(int,int*,int*,int*,int*);
extern int SpOrder1WD(int,int*,int*,int*,int*);
extern int SpOrderQMD(int,int*,int*,int*,int*);
extern int SpOrderRCM(int,int*,int*,int*,int*);

int MatGetReorder_IJ(int n,int *ia,int* ja,MatOrdering type,
                            IS *rperm, IS *cperm)
{
  int  i,ierr,*permr,*permc;

  permr = (int *) PETSCMALLOC( 2*n*sizeof(int) ); CHKPTRQ(permr);
  permc = permr + n;

  if (type == ORDER_NATURAL) {
    for ( i=0; i<n; i++ ) permr[i] = i; ierr = 0;
  }
  else if (type == ORDER_ND) {
    ierr = SpOrderND( n, ia, ja, permr, permc );
  }
  else if (type == ORDER_1WD) {
    ierr = SpOrder1WD( n, ia, ja, permr, permc );
  }
  else if (type == ORDER_RCM) {
    ierr = SpOrderRCM( n, ia, ja, permr, permc );
  }
  else if (type == ORDER_QMD) {
    ierr = SpOrderQMD( n, ia, ja, permr, permc );
  }
  else SETERRQ(1,"MatGetReordering_AIJ:Cannot performing ordering requested");
  CHKERRQ(ierr);

  ierr = ISCreateSequential(MPI_COMM_SELF,n,permr,rperm); CHKERRQ(ierr);
  ISSetPermutation(*rperm);
  ierr = ISCreateSequential(MPI_COMM_SELF,n,permc,cperm); CHKERRQ(ierr);
  ISSetPermutation(*cperm);
  PETSCFREE(permr); 
  return 0; 
}
