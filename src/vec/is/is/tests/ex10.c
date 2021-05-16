
static char help[] = "Tests ISFilter().\n\n";

#include <petscis.h>
#include <petscviewer.h>

static PetscErrorCode CreateIS(MPI_Comm comm, PetscInt n, PetscInt first, PetscInt step, IS *is)
{
  PetscInt       *idx, i, j;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  *is = NULL;
  first += rank;
  ierr = PetscMalloc1(n,&idx);CHKERRQ(ierr);
  for (i=0,j=first; i<n; i++,j+=step) idx[i] = j;
  ierr = ISCreateGeneral(comm,n,idx,PETSC_OWN_POINTER,is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  IS             is;
  PetscInt       n=10, N, first=0, step=0, start, end;
  PetscMPIInt    rank;
  MPI_Comm       comm;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-first",&first,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-step",&step,NULL);CHKERRQ(ierr);
  start = 0; end = n;
  ierr = PetscOptionsGetInt(NULL,NULL,"-start",&start,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-end",&end,NULL);CHKERRQ(ierr);

  ierr = CreateIS(comm, n, first, step, &is);CHKERRQ(ierr);
  ierr = ISGeneralFilter(is, start, end);CHKERRQ(ierr);
  ierr = ISView(is,PETSC_VIEWER_STDOUT_(comm));CHKERRQ(ierr);
  ierr = ISGetSize(is, &N);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_(comm), "global size: %D\n", N);CHKERRQ(ierr);

  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

    test:
      suffix: 1
      nsize: 4
      args: -n 6
      args: -first -2
      args: -step 1
      args: -start {{-2 4}separate output} -end {{2 6}separate output}

TEST*/
