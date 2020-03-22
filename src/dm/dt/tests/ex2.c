static char help[] = "Tests 1D cell-based discretization tools.\n\n";

#include <petscdt.h>
#include <petscviewer.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       i,j,degrees[1000],ndegrees,nsrc_points,ntarget_points;
  PetscReal      src_points[1000],target_points[1000],*R;
  PetscBool      flg;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Discretization tools test options",NULL);CHKERRQ(ierr);
  {
    ndegrees   = 1000;
    degrees[0] = 1;
    degrees[1] = 2;
    degrees[2] = 3;
    ierr = PetscOptionsIntArray("-degrees","list of max degrees to evaluate","",degrees,&ndegrees,&flg);CHKERRQ(ierr);
    if (!flg) ndegrees = 3;

    nsrc_points   = 1000;
    src_points[0] = -1.;
    src_points[1] = 0.;
    src_points[2] = 1.;
    ierr = PetscOptionsRealArray("-src_points","list of points defining intervals on which to integrate","",src_points,&nsrc_points,&flg);CHKERRQ(ierr);
    if (!flg) nsrc_points = 3;

    ntarget_points   = 1000;
    target_points[0] = -1.;
    target_points[1] = 0.;
    target_points[2] = 1.;
    ierr = PetscOptionsRealArray("-target_points","list of points defining intervals on which to integrate","",target_points,&ntarget_points,&flg);CHKERRQ(ierr);
    if (!flg) ntarget_points = 3;
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = PetscMalloc1((nsrc_points-1)*(ntarget_points-1),&R);CHKERRQ(ierr);
  for (i=0; i<ndegrees; i++) {
    ierr = PetscDTReconstructPoly(degrees[i],nsrc_points-1,src_points,ntarget_points-1,target_points,R);CHKERRQ(ierr);
    for (j=0; j<(ntarget_points-1)*(nsrc_points-1); j++) { /* Truncate to zero for nicer output */
      if (PetscAbs(R[j]) < 10*PETSC_MACHINE_EPSILON) R[j] = 0;
    }
    for (j=0; j<ntarget_points-1; j++) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Degree %D target interval (%g,%g)\n",degrees[i],(double)target_points[j],(double)target_points[j+1]);CHKERRQ(ierr);
      ierr = PetscRealView(nsrc_points-1,R+j*(nsrc_points-1),PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(R);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
  test:
    suffix: 1
    args: -degrees 1,2,3 -target_points -0.3,0,.2 -src_points -1,-.3,0,.2,1
TEST*/
