
/*
    Run with -google_refresh_token XXX to allow access to your Google Drive or else it will prompt you to enter log in information for Google Drive.
*/

#include <petscsys.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  char           access_token[512];

  ierr = PetscInitialize(&argc,&argv,NULL,NULL);if (ierr) return ierr;
  CHKERRQ(PetscGoogleDriveRefresh(PETSC_COMM_WORLD,NULL,access_token,sizeof(access_token)));
  CHKERRQ(PetscGoogleDriveUpload(PETSC_COMM_WORLD,access_token,"googledriveupload.c"));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
     requires: ssl

   test:
     TODO: determine how to run this test without making a google refresh token public

TEST*/
