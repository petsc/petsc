#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex29.c,v 1.5 1999/04/19 22:11:24 bsmith Exp balay $";
#endif

static char help[] = "Tests VecSetValues and VecSetValuesBlocked() on MPI vectors\n\
where atleast a couple of mallocs will occur in the stash code.\n\n";

#include "vec.h"
#include "sys.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int          i,j,n = 50,ierr,flg,bs,size;
  Scalar       val,*vals,zero=0.0;
  Vec          x;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);
  bs = size;

  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,&flg);CHKERRA(ierr);
  ierr = VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,n*bs,&x);CHKERRA(ierr);
  ierr = VecSetBlockSize(x,bs);CHKERRA(ierr);

  for ( i=0; i<n*bs; i++ ) {
    val  = i*1.0;
    ierr = VecSetValues(x,1,&i,&val,INSERT_VALUES);CHKERRA(ierr);
  }
  ierr = VecAssemblyBegin(x);CHKERRA(ierr);
  ierr = VecAssemblyEnd(x);CHKERRA(ierr);

  ierr = VecView(x,VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  /* Now do the blocksetvalues */
  ierr = VecSet(&zero,x);CHKERRQ(ierr);
  vals = (Scalar *)PetscMalloc(bs*sizeof(Scalar));CHKPTRA(vals);
  for ( i=0; i<n; i++ ) {
    for ( j=0; j<bs; j++ ) {
      vals[j] = (i*bs+j)*1.0;
    }
    ierr = VecSetValuesBlocked(x,1,&i,vals,INSERT_VALUES);CHKERRA(ierr);
  }

  ierr = VecAssemblyBegin(x);CHKERRA(ierr);
  ierr = VecAssemblyEnd(x);CHKERRA(ierr);

  ierr = VecView(x,VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  ierr = VecDestroy(x);CHKERRA(ierr);
  PetscFree(vals);
  PetscFinalize();
  return 0;
}
 
