
static char help[] = "Tests PetscGetFullPath().\n\n";

#include <petscsys.h>

/* for windows - fix up path - so that we can do diff test */
PetscErrorCode  path_to_unix(char filein[])
{
  PetscErrorCode ierr;
  size_t         i,n;

  PetscFunctionBegin;
  ierr = PetscStrlen(filein,&n);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    if (filein[i] == '\\') filein[i] = '/';
  }
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  char           fpath[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscGetFullPath("~/somefile",fpath,sizeof(fpath));CHKERRQ(ierr);
  ierr = path_to_unix(fpath);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%s\n",fpath);CHKERRQ(ierr);
  ierr = PetscGetFullPath("someotherfile",fpath,sizeof(fpath));CHKERRQ(ierr);
  ierr = path_to_unix(fpath);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%s\n",fpath);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      requires: !windows_compilers
      filter: sed "s?$(pwd -P)??g" |  sed "s?${HOME}??g"

   test:
      suffix: 2
      requires: windows_compilers
      output_file: output/ex31_1.out
      filter: sed "s?`cygpath -m ${PWD}`??g" |  sed "s?`cygpath -m ${HOME}`??g"

TEST*/
