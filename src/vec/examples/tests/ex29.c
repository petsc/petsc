#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex16.c,v 1.2 1999/02/02 23:41:41 bsmith Exp $";
#endif

static char help[] = "Tests VecSetValues and VecSetValuesBlocked() on MPI vectors\n\
where atleast a couple of mallocs will occur in the stash code.\n\n";

#include "vec.h"
#include "sys.h"

int main(int argc,char **argv)
{
  int          i,j,n = 50,ierr,flg,bs,size,*rows;
  Scalar       val,*vals;
  Vec          x;

  PetscInitialize(&argc,&argv,(char*)0,help);
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  bs = size;

  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,&flg); CHKERRA(ierr);
  ierr = VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,n*bs,&x); CHKERRA(ierr);

  for ( i=0; i<n*bs; i++ ) {
    val  = i*1.0;
    ierr = VecSetValues(x,1,&i,&val,INSERT_VALUES); CHKERRA(ierr);
  }
  ierr = VecAssemblyBegin(x); CHKERRA(ierr);
  ierr = VecAssemblyEnd(x); CHKERRA(ierr);

  ierr = VecView(x,VIEWER_STDOUT_WORLD); CHKERRA(ierr);

  /* Now do the blocksetvalues */
  ierr = VecSetBlockSize(x,bs); CHKERRA(ierr);
  rows = (int *)PetscMalloc(bs*sizeof(int)); CHKPTRA(rows);
  vals = (Scalar *)PetscMalloc(bs*sizeof(Scalar)); CHKPTRA(vals);
  for ( i=0; i<n; i++ ) {
    for ( j=0; j<bs; j++ ) {
      rows[j] = i*bs + j;
      vals[j] = rows[j]*1.0;
    }
    ierr = VecSetValues(x,1,rows,vals,INSERT_VALUES); CHKERRA(ierr);
  }

  ierr = VecView(x,VIEWER_STDOUT_WORLD); CHKERRA(ierr);

  ierr = VecDestroy(x); CHKERRA(ierr);
  PetscFree(rows);
  PetscFree(vals);
  PetscFinalize();
  return 0;
}
 
