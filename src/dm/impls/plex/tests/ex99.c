static char help[] = "Tests DMPlex Gmsh reader.\n\n";

#include <petscdmplex.h>

#if !defined(PETSC_GMSH_EXE)
#define PETSC_GMSH_EXE "gmsh"
#endif

#include <petscds.h>

static void one(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar value[])
{
  value[0] = (PetscReal)1;
}

static PetscErrorCode CreateFE(DM dm)
{
  DM             cdm;
  PetscSpace     P;
  PetscDualSpace Q;
  DM             K;
  PetscFE        fe;
  DMPolytopeType ptype;

  PetscInt       dim,k;
  PetscBool      isSimplex;

  PetscDS        ds;

  PetscFunctionBeginUser;
  CHKERRQ(DMGetCoordinateDM(dm, &cdm));
  CHKERRQ(DMGetField(cdm, 0, NULL, (PetscObject*) &fe));
  CHKERRQ(PetscFEGetBasisSpace(fe, &P));
  CHKERRQ(PetscFEGetDualSpace(fe, &Q));
  CHKERRQ(PetscDualSpaceGetDM(Q,&K));
  CHKERRQ(DMGetDimension(K,&dim));
  CHKERRQ(PetscSpaceGetDegree(P, &k, NULL));
  CHKERRQ(DMPlexGetCellType(K, 0, &ptype));
  switch (ptype) {
  case DM_POLYTOPE_QUADRILATERAL:
  case DM_POLYTOPE_HEXAHEDRON:
    isSimplex = PETSC_FALSE; break;
  default:
    isSimplex = PETSC_TRUE; break;
  }

  CHKERRQ(PetscFECreateLagrange(PETSC_COMM_SELF, dim, 1, isSimplex, k, PETSC_DETERMINE, &fe));
  CHKERRQ(PetscFESetName(fe, "scalar"));
  CHKERRQ(DMAddField(dm, NULL, (PetscObject) fe));
  CHKERRQ(PetscFEDestroy(&fe));
  CHKERRQ(DMCreateDS(dm));

  CHKERRQ(DMGetDS(dm, &ds));
  CHKERRQ(PetscDSSetObjective(ds, 0, one));
  PetscFunctionReturn(0);
}

static PetscErrorCode CheckIntegral(DM dm, PetscReal integral, PetscReal tol)
{
  Vec            u;
  PetscReal      rval;
  PetscScalar    result;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  CHKERRQ(DMGetGlobalVector(dm, &u));
  CHKERRQ(DMPlexComputeIntegralFEM(dm, u, &result, NULL));
  CHKERRQ(DMRestoreGlobalVector(dm, &u));
  rval = PetscRealPart(result);
  if (integral > 0 && PetscAbsReal(integral - rval) > tol) {
    ierr = PetscPrintf(PetscObjectComm((PetscObject) dm), "Calculated value %g != %g actual value (error %g > %g tol)\n",
                       (double) rval, (double) integral, (double) PetscAbsReal(integral - rval), (double) tol);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM                dm;
  const char *const mshlist[] = {"seg", "tri", "qua", "tet", "wed", "hex",
                                 "vtx", "B2tri", "B2qua", "B3tet", "B3hex"};
  const char *const fmtlist[] = {"msh22", "msh40", "msh41"};
  PetscInt          msh = 5;
  PetscInt          fmt = 2;
  PetscBool         bin = PETSC_TRUE;
  PetscInt          dim = 3;
  PetscInt          order = 2;

  const char        cmdtemplate[] = "%s -format %s %s -%d -order %d %s -o %s";
  char              gmsh[PETSC_MAX_PATH_LEN] = PETSC_GMSH_EXE;
  char              tag[PETSC_MAX_PATH_LEN], path[PETSC_MAX_PATH_LEN];
  char              geo[PETSC_MAX_PATH_LEN], geodir[PETSC_MAX_PATH_LEN] = ".";
  char              out[PETSC_MAX_PATH_LEN], outdir[PETSC_MAX_PATH_LEN] = ".";
  char              cmd[PETSC_MAX_PATH_LEN*4];
  PetscBool         set,flg;
  FILE              *fp;

  CHKERRQ(PetscInitialize(&argc, &argv, NULL, help));

  CHKERRQ(PetscStrncpy(geodir, "${PETSC_DIR}/share/petsc/datafiles/meshes", sizeof(geodir)));
  CHKERRQ(PetscOptionsGetenv(PETSC_COMM_SELF, "GMSH", path, sizeof(path), &set));
  if (set) CHKERRQ(PetscStrncpy(gmsh, path, sizeof(gmsh)));
  CHKERRQ(PetscOptionsGetString(NULL, NULL, "-gmsh", gmsh, sizeof(gmsh), NULL));
  CHKERRQ(PetscOptionsGetString(NULL, NULL, "-dir", geodir, sizeof(geodir), NULL));
  CHKERRQ(PetscOptionsGetString(NULL, NULL, "-out", outdir, sizeof(outdir), NULL));
  CHKERRQ(PetscOptionsGetEList(NULL, NULL, "-msh", mshlist, (int)(sizeof(mshlist)/sizeof(mshlist[0])), &msh, NULL));
  CHKERRQ(PetscOptionsGetEList(NULL, NULL, "-fmt", fmtlist, (int)(sizeof(fmtlist)/sizeof(fmtlist[0])), &fmt, NULL));
  CHKERRQ(PetscOptionsGetBool(NULL, NULL, "-bin", &bin, NULL));
  CHKERRQ(PetscOptionsGetInt(NULL, NULL, "-dim", &dim, NULL));
  CHKERRQ(PetscOptionsGetInt(NULL, NULL, "-order", &order, NULL));
  if (fmt == 1) bin = PETSC_FALSE; /* Recent Gmsh releases cannot generate msh40+binary format*/

  { /* This test requires Gmsh >= 4.2.0 */
    int inum = 0, major = 0, minor = 0, micro = 0;
    CHKERRQ(PetscSNPrintf(cmd, sizeof(cmd), "%s -info", gmsh));
    CHKERRQ(PetscPOpen(PETSC_COMM_SELF, NULL, cmd, "r", &fp));
    if (fp) {inum = fscanf(fp, "Version : %d.%d.%d", &major, &minor, &micro);}
    CHKERRQ(PetscPClose(PETSC_COMM_SELF, fp));
    if (inum != 3 || major < 4 || (major == 4 && minor < 2)) {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "Gmsh>=4.2.0 not available\n")); goto finish;
    }
  }

  CHKERRQ(PetscSNPrintf(tag, sizeof(tag), "%s-%d-%d-%s%s", mshlist[msh], (int)dim, (int)order, fmtlist[fmt], bin?"-bin":""));
  CHKERRQ(PetscSNPrintf(geo, sizeof(geo), "%s/gmsh-%s.geo", geodir, mshlist[msh]));
  CHKERRQ(PetscSNPrintf(out, sizeof(out), "%s/mesh-%s.msh", outdir, tag));
  CHKERRQ(PetscStrreplace(PETSC_COMM_SELF, geo, path, sizeof(path)));
  CHKERRQ(PetscFixFilename(path, geo));
  CHKERRQ(PetscStrreplace(PETSC_COMM_SELF, out, path, sizeof(path)));
  CHKERRQ(PetscFixFilename(path, out));
  CHKERRQ(PetscTestFile(geo, 'r', &flg));
  PetscCheck(flg,PETSC_COMM_SELF, PETSC_ERR_USER_INPUT, "File not found: %s", geo);

  CHKERRQ(PetscSNPrintf(cmd, sizeof(cmd), cmdtemplate, gmsh, fmtlist[fmt], bin?"-bin":"", (int)dim, (int)order, geo, out));
  CHKERRQ(PetscPOpen(PETSC_COMM_SELF, NULL, cmd, "r", &fp));
  CHKERRQ(PetscPClose(PETSC_COMM_SELF, fp));

  CHKERRQ(DMPlexCreateFromFile(PETSC_COMM_SELF, out, "ex99_plex", PETSC_TRUE, &dm));
  CHKERRQ(PetscSNPrintf(tag, sizeof(tag), "mesh-%s", mshlist[msh]));
  CHKERRQ(PetscObjectSetName((PetscObject)dm, tag));
  CHKERRQ(DMViewFromOptions(dm, NULL, "-dm_view"));
  CHKERRQ(DMSetFromOptions(dm));
  {
    PetscBool check;
    PetscReal integral = 0, tol = (PetscReal)1.0e-4;
    CHKERRQ(PetscOptionsGetReal(NULL, NULL, "-integral", &integral, &check));
    CHKERRQ(PetscOptionsGetReal(NULL, NULL, "-tol", &tol, NULL));
    if (check) {
      CHKERRQ(CreateFE(dm));
      CHKERRQ(CheckIntegral(dm, integral, tol));
    }
  }
  CHKERRQ(DMDestroy(&dm));

finish:
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: defined(PETSC_HAVE_POPEN)

  test:
    requires: defined(PETSC_GMSH_EXE)
    args: -dir ${wPETSC_DIR}/share/petsc/datafiles/meshes
    args: -msh {{vtx}separate_output}
    args: -order 1
    args: -fmt {{msh22 msh40 msh41}} -bin {{0 1}}
    args: -dm_view ::ascii_info_detail
    args: -dm_plex_check_all
    args: -dm_plex_gmsh_highorder false
    args: -dm_plex_gmsh_use_marker true
    args: -dm_plex_gmsh_spacedim 3

  test:
    requires: defined(PETSC_GMSH_EXE)
    args: -dir ${wPETSC_DIR}/share/petsc/datafiles/meshes
    args: -msh {{seg tri qua tet wed hex}separate_output}
    args: -order {{1 2 3 7}}
    args: -fmt {{msh22 msh40 msh41}} -bin {{0 1}}
    args: -dm_view ::ascii_info_detail
    args: -dm_plex_check_all
    args: -dm_plex_gmsh_highorder false
    args: -dm_plex_gmsh_use_marker true

  testset:
    suffix: B2 # 2D ball
    requires: defined(PETSC_GMSH_EXE)
    args: -dir ${wPETSC_DIR}/share/petsc/datafiles/meshes
    args: -msh {{B2tri B2qua}}
    args: -dim 2 -integral 3.141592653589793 # pi
    args: -order {{2 3 4 5 6 7 8 9}} -tol 0.05

  testset:
    suffix: B2_bnd # 2D ball boundary
    requires: defined(PETSC_GMSH_EXE)
    args: -dir ${wPETSC_DIR}/share/petsc/datafiles/meshes
    args: -dm_plex_gmsh_spacedim 2
    args: -msh {{B2tri B2qua}}
    args: -dim 1 -integral 6.283185307179586 # 2*pi
    args: -order {{2 3 4 5 6 7 8 9}} -tol 0.05

  testset:
    suffix: B3 # 3D ball
    requires: defined(PETSC_GMSH_EXE)
    args: -dir ${wPETSC_DIR}/share/petsc/datafiles/meshes
    args: -msh {{B3tet B3hex}}
    args: -dim 3 -integral 4.1887902047863905 # 4/3*pi
    args: -order {{2 3 4 5}} -tol 0.20

  testset:
    suffix: B3_bnd # 3D ball boundary
    requires: defined(PETSC_GMSH_EXE)
    args: -dir ${wPETSC_DIR}/share/petsc/datafiles/meshes
    args: -dm_plex_gmsh_spacedim 3
    args: -msh {{B3tet B3hex}}
    args: -dim 2 -integral 12.566370614359172 # 4*pi
    args: -order {{2 3 4 5 6 7 8 9}} -tol 0.25

  testset:
    suffix: B_lin # linear discretizations
    requires: defined(PETSC_GMSH_EXE)
    args: -dir ${wPETSC_DIR}/share/petsc/datafiles/meshes
    args: -dm_plex_gmsh_highorder true
    args: -dm_plex_gmsh_project true
    args: -dm_plex_gmsh_project_petscspace_degree {{1 2 3}separate_output}
    args: -dm_plex_gmsh_fe_view
    args: -dm_plex_gmsh_project_fe_view
    args: -order 1 -tol 1e-4
    test:
      suffix: dim-1
      args: -dm_plex_gmsh_spacedim 2
      args: -msh {{B2tri B2qua}separate_output}
      args: -dim 1 -integral 5.656854249492381 # 4*sqrt(2)
    test:
      suffix: dim-2
      args: -dm_plex_gmsh_spacedim 2
      args: -msh {{B2tri B2qua}separate_output}
      args: -dim 2 -integral 2.000000000000000 # 2
    test:
      suffix: dim-2_msh-B3tet
      args: -dm_plex_gmsh_spacedim 3
      args: -msh B3tet -dim 2 -integral 9.914478
    test:
      suffix: dim-2_msh-B3hex
      args: -dm_plex_gmsh_spacedim 3
      args: -msh B3hex -dim 2 -integral 8.000000
    test:
      suffix: dim-3_msh-B3tet
      args: -dm_plex_gmsh_spacedim 3
      args: -msh B3tet -dim 3 -integral 2.666649
    test:
      suffix: dim-3_msh-B3hex
      args: -dm_plex_gmsh_spacedim 3
      args: -msh B3hex -dim 3 -integral 1.539600

TEST*/
