
static char help[] = "Tests ISFilter().\n\n";

#include <petscis.h>
#include <petscviewer.h>

static PetscErrorCode CreateIS(MPI_Comm comm, PetscInt n, PetscInt first, PetscInt step, IS *is)
{
  PetscInt       *idx, i, j;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(comm,&rank));
  *is = NULL;
  first += rank;
  CHKERRQ(PetscMalloc1(n,&idx));
  for (i=0,j=first; i<n; i++,j+=step) idx[i] = j;
  CHKERRQ(ISCreateGeneral(comm,n,idx,PETSC_OWN_POINTER,is));
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
  CHKERRMPI(MPI_Comm_rank(comm,&rank));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-first",&first,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-step",&step,NULL));
  start = 0; end = n;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-start",&start,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-end",&end,NULL));

  CHKERRQ(CreateIS(comm, n, first, step, &is));
  CHKERRQ(ISGeneralFilter(is, start, end));
  CHKERRQ(ISView(is,PETSC_VIEWER_STDOUT_(comm)));
  CHKERRQ(ISGetSize(is, &N));
  CHKERRQ(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_(comm), "global size: %" PetscInt_FMT "\n", N));

  CHKERRQ(ISDestroy(&is));
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
