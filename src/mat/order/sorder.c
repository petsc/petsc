extern int SpOrderND(int,int*,int*,int*);
extern int SpOrder1WD(int,int*,int*,int*);
extern int SpOrderQMD(int,int*,int*,int*);
extern int SpOrderRCM(int,int*,int*,int*);

static int MatGetReordering_AIJ(Mat mat,MatOrdering type,IS *rperm, IS *cperm)
{
  Mat_AIJ *aij = (Mat_AIJ *) mat->data;
  int     i, ierr, *ia, *ja, *perma;

  if (!aij->assembled) 
    SETERRQ(1,"MatGetReordering_AIJ:Cannot reorder unassembled matrix");

  ierr = MatToSymmetricIJ_AIJ( aij, &ia, &ja ); CHKERRQ(ierr);
  ierr = MatGetReorder_IJ(aij->n,ia,ja,type,rperm,cperm); CHKERRQ(ierr);
  PETSCFREE(ia); PETSCFREE(ja);

  perma = (int *) PETSCMALLOC( aij->n*sizeof(int) ); CHKPTRQ(perma);

  ierr = SpToSymmetricIJ_AIJ( aij, &ia, &ja ); CHKERRQ(ierr);

  if (type == ORDER_NATURAL) {
    for ( i=0; i<aij->n; i++ ) perma[i] = i;
  }
  else if (type == ORDER_ND) {
    ierr = SpOrderND( aij->n, ia, ja, perma );
  }
  else if (type == ORDER_1WD) {
    ierr = SpOrder1WD( aij->n, ia, ja, perma );
  }
  else if (type == ORDER_RCM) {
    ierr = SpOrderRCM( aij->n, ia, ja, perma );
  }
  else if (type == ORDER_QMD) {
    ierr = SpOrderQMD( aij->n, ia, ja, perma );
  }
  else SETERRQ(1,"MatGetReordering_AIJ:Cannot performing ordering requested");
  CHKERRQ(ierr);
  PETSCFREE(ia); PETSCFREE(ja);

  ierr = ISCreateSequential(MPI_COMM_SELF,aij->n,perma,rperm); CHKERRQ(ierr);
  ISSetPermutation(*rperm);
  ierr = ISCreateSequential(MPI_COMM_SELF,aij->n,perma,cperm); CHKERRQ(ierr);
  ISSetPermutation(*cperm);
  PETSCFREE(perma); 
  return 0; 
}
