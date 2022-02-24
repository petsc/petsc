
static char help[] = "Tests PetscGetFullPath().\n\n";

#include <petscsys.h>

/* for windows - fix up path - so that we can do diff test */
PetscErrorCode  path_to_unix(char filein[])
{
  size_t         i,n;

  PetscFunctionBegin;
  CHKERRQ(PetscStrlen(filein,&n));
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
  CHKERRQ(PetscGetFullPath("~/somefile",fpath,sizeof(fpath)));
  CHKERRQ(path_to_unix(fpath));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"%s\n",fpath));
  CHKERRQ(PetscGetFullPath("someotherfile",fpath,sizeof(fpath)));
  CHKERRQ(path_to_unix(fpath));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"%s\n",fpath));
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
