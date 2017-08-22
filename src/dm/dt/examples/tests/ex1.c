static char help[] = "Tests 1D discretization tools.\n\n";

#include <petscdt.h>
#include <petscviewer.h>

static PetscErrorCode CheckPoints(const char *name,PetscInt npoints,const PetscReal *points,PetscInt ndegrees,const PetscInt *degrees)
{
  PetscErrorCode ierr;
  PetscReal      *B,*D,*D2;
  PetscInt       i,j;

  PetscFunctionBegin;
  ierr = PetscMalloc3(npoints*ndegrees,&B,npoints*ndegrees,&D,npoints*ndegrees,&D2);CHKERRQ(ierr);
  ierr = PetscDTLegendreEval(npoints,points,ndegrees,degrees,B,D,D2);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%s\n",name);CHKERRQ(ierr);
  for (i=0; i<npoints; i++) {
    for (j=0; j<ndegrees; j++) {
      PetscReal b,d,d2;
      b = B[i*ndegrees+j];
      d = D[i*ndegrees+j];
      d2 = D2[i*ndegrees+j];
      if (PetscAbsReal(b) < PETSC_SMALL) b   = 0;
      if (PetscAbsReal(d) < PETSC_SMALL) d   = 0;
      if (PetscAbsReal(d2) < PETSC_SMALL) d2 = 0;
      ierr = PetscPrintf(PETSC_COMM_WORLD,"degree %D at %12.4g: B=%12.4g  D=%12.4g  D2=%12.4g\n",degrees[j],(double)points[i],(double)b,(double)d,(double)d2);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree3(B,D,D2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       degrees[1000],ndegrees,npoints,two;
  PetscReal      points[1000],weights[1000],interval[2];
  PetscBool      flg;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Discretization tools test options",NULL);CHKERRQ(ierr);
  {
    ndegrees   = 1000;
    degrees[0] = 0;
    degrees[1] = 1;
    degrees[2] = 2;
    ierr       = PetscOptionsIntArray("-degrees","list of degrees to evaluate","",degrees,&ndegrees,&flg);CHKERRQ(ierr);

    if (!flg) ndegrees = 3;
    npoints   = 1000;
    points[0] = 0.0;
    points[1] = -0.5;
    points[2] = 1.0;
    ierr      = PetscOptionsRealArray("-points","list of points at which to evaluate","",points,&npoints,&flg);CHKERRQ(ierr);

    if (!flg) npoints = 3;
    two         = 2;
    interval[0] = -1.;
    interval[1] = 1.;
    ierr        = PetscOptionsRealArray("-interval","interval on which to construct quadrature","",interval,&two,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = CheckPoints("User-provided points",npoints,points,ndegrees,degrees);CHKERRQ(ierr);

  ierr = PetscDTGaussQuadrature(npoints,interval[0],interval[1],points,weights);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Quadrature weights\n");CHKERRQ(ierr);
  ierr = PetscRealView(npoints,weights,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  {
    PetscReal a = interval[0],b = interval[1],zeroth,first,second;
    PetscInt  i;
    zeroth = b - a;
    first  = (b*b - a*a)/2;
    second = (b*b*b - a*a*a)/3;
    for (i=0; i<npoints; i++) {
      zeroth -= weights[i];
      first  -= weights[i] * points[i];
      second -= weights[i] * PetscSqr(points[i]);
    }
    if (PetscAbs(zeroth) < 1e-10) zeroth = 0.;
    if (PetscAbs(first)  < 1e-10) first  = 0.;
    if (PetscAbs(second) < 1e-10) second = 0.;
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Moment error: zeroth=%g, first=%g, second=%g\n",(double)(-zeroth),(double)(-first),(double)(-second));CHKERRQ(ierr);
  }
  ierr = CheckPoints("Gauss points",npoints,points,ndegrees,degrees);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
  test:
    suffix: 1
    args: -degrees 1,2,3,4,5 -points 0,.2,-.5,.8,.9,1 -interval -.5,1
TEST*/
