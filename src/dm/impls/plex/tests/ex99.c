static char help[] = "Tests DMPlex Gmsh reader.\n\n";

#include <petscdmplex.h>

#if !defined(PETSC_GMSH_EXE)
#define PETSC_GMSH_EXE "gmsh"
#endif

int main(int argc, char **argv)
{
  DM                dm;
  const char *const mshlist[] = {"seg", "tri", "qua", "tet", "wed", "hex"};
  const char *const fmtlist[] = {"msh22", "msh40", "msh41"};
  PetscInt          msh = 5;
  PetscInt          fmt = 2;
  PetscBool         bin = PETSC_FALSE;
  PetscInt          order = 2;

  const char        cmdtemplate[] = "%s -3 -format %s %s -order %d %s -o %s";
  char              gmsh[PETSC_MAX_PATH_LEN] = PETSC_GMSH_EXE;
  char              tag[PETSC_MAX_PATH_LEN], path[PETSC_MAX_PATH_LEN];
  char              geo[PETSC_MAX_PATH_LEN], geodir[PETSC_MAX_PATH_LEN] = ".";
  char              out[PETSC_MAX_PATH_LEN], outdir[PETSC_MAX_PATH_LEN] = ".";
  char              cmd[PETSC_MAX_PATH_LEN*4];
  PetscBool         set,flg;
  FILE              *fp;
  PetscErrorCode    ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;

  ierr = PetscStrncpy(geodir, "${PETSC_DIR}/share/petsc/datafiles/meshes", sizeof(geodir));CHKERRQ(ierr);
  ierr = PetscOptionsGetenv(PETSC_COMM_SELF, "GMSH", path, sizeof(path), &set);CHKERRQ(ierr);
  if (set) {ierr = PetscStrncpy(gmsh, path, sizeof(gmsh));CHKERRQ(ierr);}
  ierr = PetscOptionsGetString(NULL, NULL, "-gmsh", gmsh, sizeof(gmsh), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL, NULL, "-dir", geodir, sizeof(geodir), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL, NULL, "-out", outdir, sizeof(outdir), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetEList(NULL, NULL, "-msh", mshlist, 6, &msh, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetEList(NULL, NULL, "-fmt", fmtlist, 3, &fmt, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL, NULL, "-bin", &bin, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-order", &order, NULL);CHKERRQ(ierr);
  if (fmt == 1) bin = PETSC_FALSE; /* Recent Gmsh releases cannot generate msh40+binary format*/

  { /* This test requires Gmsh >= 4.2.0 */
    int inum = 0, major = 0, minor = 0, micro = 0;
    ierr = PetscSNPrintf(cmd, sizeof(cmd), "%s -info", gmsh);CHKERRQ(ierr);
    ierr = PetscPOpen(PETSC_COMM_SELF, NULL, cmd, "r", &fp);CHKERRQ(ierr);
    if (fp) {inum = fscanf(fp, "Version : %d.%d.%d", &major, &minor, &micro);}
    ierr = PetscPClose(PETSC_COMM_SELF, fp);CHKERRQ(ierr);
    if (inum != 3 || major < 4 || (major == 4 && minor < 2)) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "Gmsh>=4.2.0 not available\n");CHKERRQ(ierr); goto finish;
    }
  }

  ierr = PetscSNPrintf(tag, sizeof(tag), "%s-%d-%s%s", mshlist[msh], order, fmtlist[fmt], bin?"-bin":"");CHKERRQ(ierr);
  ierr = PetscSNPrintf(geo, sizeof(geo), "%s/gmsh-%s.geo", geodir, mshlist[msh]);CHKERRQ(ierr);
  ierr = PetscSNPrintf(out, sizeof(out), "%s/mesh-%s.msh", outdir, tag);CHKERRQ(ierr);
  ierr = PetscStrreplace(PETSC_COMM_SELF, geo, path, sizeof(path));CHKERRQ(ierr);
  ierr = PetscFixFilename(path, geo);CHKERRQ(ierr);
  ierr = PetscStrreplace(PETSC_COMM_SELF, out, path, sizeof(path));CHKERRQ(ierr);
  ierr = PetscFixFilename(path, out);CHKERRQ(ierr);
  ierr = PetscTestFile(geo, 'r', &flg);CHKERRQ(ierr);
  if (!flg) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_USER_INPUT, "File not found: %s", geo);

  ierr = PetscSNPrintf(cmd, sizeof(cmd), cmdtemplate, gmsh, fmtlist[fmt], bin?"-bin":"", order, geo, out);CHKERRQ(ierr);
  ierr = PetscPOpen(PETSC_COMM_SELF, NULL, cmd, "r", &fp);CHKERRQ(ierr);
  ierr = PetscPClose(PETSC_COMM_SELF, fp);CHKERRQ(ierr);

  ierr = DMPlexCreateFromFile(PETSC_COMM_SELF, out, PETSC_TRUE, &dm);CHKERRQ(ierr);
  ierr = PetscSNPrintf(tag, sizeof(tag), "mesh-%s", mshlist[msh]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)dm, tag);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);

finish:
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  build:
    requires: define(PETSC_HAVE_POPEN)

  test:
    requires: define(PETSC_GMSH_EXE)
    args: -dir ${wPETSC_DIR}/share/petsc/datafiles/meshes
    args: -msh {{seg tri qua tet wed hex}separate_output}
    args: -order {{1 2 3 7}}
    args: -fmt {{msh22 msh40 msh41}} -bin {{0 1}}
    args: -dm_view ::ascii_info_detail
    args: -dm_plex_check_all
    args: -dm_plex_gmsh_highorder false

TEST*/
