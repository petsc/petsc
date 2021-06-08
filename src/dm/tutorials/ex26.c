static char help[] = "Calculates moments for Gaussian functions.\n\n";

#include <petscviewer.h>
#include <petscdt.h>
#include <petscvec.h>

#include <gsl/gsl_sf_hermite.h>
#include <gsl/gsl_randist.h>

int main(int argc, char **argv)
{
  PetscErrorCode ierr;
  int            s,n = 15;
  PetscInt       tick, moment = 0,momentummax = 7;
  PetscReal      *zeros,*weights,scale,h,sigma = 1/sqrt(2), g = 0, mu = 0;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;

  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-moment_max",&momentummax,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-sigma",&sigma,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-mu",&mu,NULL);CHKERRQ(ierr);

  /* calulate zeros and roots of Hermite Gauss quadrature */
  ierr = PetscMalloc1(n,&zeros);CHKERRQ(ierr);
  zeros[0] = 0;
  tick = n % 2;
  for (s=0; s<n/2; s++) {
    zeros[2*s+tick]   =  -gsl_sf_hermite_zero(n,s+1);
    zeros[2*s+1+tick] =  gsl_sf_hermite_zero(n,s+1);
  }

  ierr = PetscDTFactorial(n, &scale);CHKERRQ(ierr);
  scale = exp2(n-1)*scale*PetscSqrtReal(PETSC_PI)/(n*n);
  ierr = PetscMalloc1(n+1,&weights);CHKERRQ(ierr);
  for (s=0; s<n; s++) {
    h          = gsl_sf_hermite(n-1, (double) zeros[s]);
    weights[s] = scale/(h*h);
  }
  /* zeros and weights verfied up to n = 5 with http://mathworld.wolfram.com/Hermite-GaussQuadrature.html */

  for (moment=0; moment<momentummax; moment++) {

    /* http://www.wouterdenhaan.com/numerical/integrationslides.pdf */
    /* https://en.wikipedia.org/wiki/Gauss-Hermite_quadrature */
    /*
       int_{-infinity}^{infinity} \frac{1}{sigma sqrt(2pi)} exp(- \frac{(x - mu)^2}{2 sigma^2) h(x) dx

       then approx equals 1/pi sum_i w_i h( sqrt(2) sigma x_i + mu)
    */
    g = 0;
    for (s=0; s<n; s++) {
      g += weights[s]*PetscPowRealInt(sqrt(2)*sigma*zeros[s] + mu,moment);
    }
    g /= sqrt(PETSC_PI);
    /* results confirmed with https://en.wikipedia.org/wiki/Normal_distribution#Moments sigma^p * (p-1)!!*/
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Moment %D %g \n",moment,(double)g);CHKERRQ(ierr);

  }
  ierr = PetscFree(zeros);CHKERRQ(ierr);
  ierr = PetscFree(weights);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  build:
    requires: gsl double

  test:

TEST*/

