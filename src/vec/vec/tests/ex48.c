
static char help[] = "Tests HDF5 attribute I/O.\n\n";

#include <petscviewerhdf5.h>
#include <petscvec.h>

static PetscInt  n       = 5; /* testing vector size */
static PetscBool verbose = PETSC_FALSE;
#define SLEN 128

/* sequence of unique absolute paths */
#define nap 9
static const char *apaths[nap] = {
  /* 0 */
  "/", "/g1", "/g1/g2", "/g1/nonExistingGroup1", "/g1/g3",
  /* 5 */
  "/g1/g3/g4", "/g1/nonExistingGroup2", "/g1/nonExistingGroup2/g5", "/g1/g6/g7"};

#define np 21
/* sequence of paths (absolute or relative); "<" encodes Pop */
static const char *paths[np] = {
  /* 0 */
  "/",
  "/g1",
  "/g1/g2",
  "/g1/nonExistingGroup1",
  "<",
  /* 5 */
  ".", /* /g1/g2 */
  "<",
  "<",
  "g3", /* /g1/g3 */
  "g4", /* /g1/g3/g4 */
        /* 10 */
  "<",
  "<",
  ".", /* /g1 */
  "<",
  "nonExistingGroup2", /* /g1/nonExistingG2 */
                       /* 15 */
  "g5",                /* /g1/nonExistingG2/g5 */
  "<",
  "<",
  "g6/g7", /* /g1/g6/g7 */
  "<",
  /* 20 */
  "<",
};
/* corresponding expected absolute paths - positions in abspath */
static const PetscInt paths2apaths[np] = {
  /* 0 */
  0,
  1,
  2,
  3,
  2,
  /* 5 */
  2,
  2,
  1,
  4,
  5,
  /* 10 */
  4,
  1,
  1,
  1,
  6,
  /* 15 */
  7,
  6,
  1,
  8,
  1,
  /* 20 */
  0,
};

#define ns 4
/* for "" attribute will be stored to group, otherwise to given dataset */
static const char *datasets[ns] = {"", "x", "nonExistingVec", "y"};

/* beware this yields PETSC_FALSE for "" but group "" is interpreted as "/" */
static inline PetscErrorCode shouldExist(const char name[], PetscBool emptyExists, PetscBool *has)
{
  size_t len = 0;

  PetscFunctionBegin;
  PetscCall(PetscStrlen(name, &len));
  *has = emptyExists;
  if (len) {
    char *loc = NULL;
    PetscCall(PetscStrstr(name, "nonExisting", &loc));
    *has = PetscNot(loc);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode isPop(const char path[], PetscBool *has)
{
  PetscFunctionBegin;
  PetscCall(PetscStrcmp(path, "<", has));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode isDot(const char path[], PetscBool *has)
{
  PetscFunctionBegin;
  PetscCall(PetscStrcmp(path, ".", has));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode isRoot(const char path[], PetscBool *flg)
{
  size_t len;

  PetscFunctionBegin;
  PetscCall(PetscStrlen(path, &len));
  *flg = PetscNot(len);
  if (!*flg) PetscCall(PetscStrcmp(path, "/", flg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode compare(PetscDataType dt, void *ptr0, void *ptr1, PetscBool *flg)
{
  PetscFunctionBegin;
  switch (dt) {
  case PETSC_INT:
    *flg = (PetscBool)(*(PetscInt *)ptr0 == *(PetscInt *)ptr1);
    if (verbose) {
      if (*flg) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "%" PetscInt_FMT, *(PetscInt *)ptr0));
      } else {
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "%" PetscInt_FMT " != %" PetscInt_FMT "\n", *(PetscInt *)ptr0, *(PetscInt *)ptr1));
      }
    }
    break;
  case PETSC_REAL:
    *flg = (PetscBool)(*(PetscReal *)ptr0 == *(PetscReal *)ptr1);
    if (verbose) {
      if (*flg) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "%f", *(PetscReal *)ptr0));
      } else {
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "%f != %f\n", *(PetscReal *)ptr0, *(PetscReal *)ptr1));
      }
    }
    break;
  case PETSC_BOOL:
    *flg = (PetscBool)(*(PetscBool *)ptr0 == *(PetscBool *)ptr1);
    if (verbose) {
      if (*flg) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "%s", PetscBools[*(PetscBool *)ptr0]));
      } else {
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "%s != %s\n", PetscBools[*(PetscBool *)ptr0], PetscBools[*(PetscBool *)ptr1]));
      }
    }
    break;
  case PETSC_STRING:
    PetscCall(PetscStrcmp((const char *)ptr0, (const char *)ptr1, flg));
    if (verbose) {
      if (*flg) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "%s", (char *)ptr0));
      } else {
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "%s != %s\n", (char *)ptr0, (char *)ptr1));
      }
    }
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "PetscDataType %s not handled here", PetscDataTypes[dt]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode alterString(const char oldstr[], char str[])
{
  size_t i, n;

  PetscFunctionBegin;
  PetscCall(PetscStrlen(oldstr, &n));
  PetscCall(PetscStrncpy(str, oldstr, n + 1));
  for (i = 0; i < n; i++) {
    if (('A' <= str[i] && str[i] < 'Z') || ('a' <= str[i] && str[i] < 'z')) {
      str[i]++;
      break;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* if name given, check dataset with this name exists under current group, otherwise just check current group exists */
/* flg: 0 doesn't exist, 1 group, 2 dataset */
static PetscErrorCode hasGroupOrDataset(PetscViewer viewer, const char path[], int *flg)
{
  PetscBool has;

  PetscFunctionBegin;
  *flg = 0;
  PetscCall(PetscViewerHDF5HasGroup(viewer, path, &has));
  if (has) *flg = 1;
  else {
    PetscCall(PetscViewerHDF5HasDataset(viewer, path, &has));
    if (has) *flg = 2;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define nt 5 /* number of datatypes */
typedef struct _n_Capsule *Capsule;
struct _n_Capsule {
  char          names[nt][SLEN];
  PetscDataType types[nt];
  char          typeNames[nt][SLEN];
  size_t        sizes[nt];
  void         *vals[nt];
  PetscInt      id, ntypes;
};

static PetscErrorCode CapsuleCreate(Capsule old, Capsule *newcapsule)
{
  Capsule       c;
  PetscBool     bool0      = PETSC_TRUE;
  PetscInt      int0       = -1;
  PetscReal     real0      = -1.1;
  char          str0[]     = "Test String";
  char          nestr0[]   = "NONEXISTING STRING"; /* this attribute shall be skipped for writing */
  void         *vals[nt]   = {&bool0, &int0, &real0, str0, nestr0};
  size_t        sizes[nt]  = {sizeof(bool0), sizeof(int0), sizeof(real0), sizeof(str0), sizeof(str0)};
  PetscDataType types[nt]  = {PETSC_BOOL, PETSC_INT, PETSC_REAL, PETSC_STRING, PETSC_STRING};
  const char   *tNames[nt] = {"bool", "int", "real", "str", "nonExisting"};
  PetscInt      t;

  PetscFunctionBegin;
  PetscCall(PetscNew(&c));
  c->id     = 0;
  c->ntypes = nt;
  if (old) {
    /* alter values */
    t     = 0;
    bool0 = PetscNot(*((PetscBool *)old->vals[t]));
    t++;
    int0 = *((PetscInt *)old->vals[t]) * -2;
    t++;
    real0 = *((PetscReal *)old->vals[t]) * -2.0;
    t++;
    PetscCall(alterString((const char *)old->vals[t], str0));
    t++;
    c->id = old->id + 1;
  }
  for (t = 0; t < nt; t++) {
    c->sizes[t] = sizes[t];
    c->types[t] = types[t];
    PetscCall(PetscStrncpy(c->typeNames[t], tNames[t], sizeof(c->typeNames[t])));
    PetscCall(PetscSNPrintf(c->names[t], SLEN, "attr_%" PetscInt_FMT "_%s", c->id, tNames[t]));
    PetscCall(PetscMalloc(sizes[t], &c->vals[t]));
    PetscCall(PetscMemcpy(c->vals[t], vals[t], sizes[t]));
  }
  *newcapsule = c;
  PetscFunctionReturn(PETSC_SUCCESS);
}
#undef nt

static PetscErrorCode CapsuleWriteAttributes(Capsule c, PetscViewer v, const char parent[])
{
  PetscInt  t;
  PetscBool flg = PETSC_FALSE;

  PetscFunctionBegin;
  for (t = 0; t < c->ntypes; t++) {
    PetscCall(shouldExist(c->names[t], PETSC_FALSE, &flg));
    if (!flg) continue;
    PetscCall(PetscViewerHDF5WriteAttribute(v, parent, c->names[t], c->types[t], c->vals[t]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CapsuleReadAndCompareAttributes(Capsule c, PetscViewer v, const char parent[])
{
  char     *group;
  int       gd = 0;
  PetscInt  t;
  PetscBool flg = PETSC_FALSE, hasAttr = PETSC_FALSE;
  MPI_Comm  comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)v, &comm));
  PetscCall(PetscViewerHDF5GetGroup(v, NULL, &group));
  PetscCall(hasGroupOrDataset(v, parent, &gd));
  /* check correct existence of attributes */
  for (t = 0; t < c->ntypes; t++) {
    const char *attribute = c->names[t];
    PetscCall(shouldExist(attribute, PETSC_FALSE, &flg));
    PetscCall(PetscViewerHDF5HasAttribute(v, parent, attribute, &hasAttr));
    if (verbose) {
      PetscCall(PetscPrintf(comm, "    %-24s = ", attribute));
      if (!hasAttr) PetscCall(PetscPrintf(comm, "---"));
    }
    PetscCheck(gd || !hasAttr, comm, PETSC_ERR_PLIB, "Attribute %s/%s/%s exists while its parent %s/%s doesn't exist", group, parent, attribute, group, parent);
    PetscCheck(flg == hasAttr, comm, PETSC_ERR_PLIB, "Attribute %s/%s should exist? %s Exists? %s", parent, attribute, PetscBools[flg], PetscBools[hasAttr]);

    /* check loaded attributes are the same as original */
    if (hasAttr) {
      char  buffer[SLEN];
      char *str;
      void *ptr0;
      /* check the stored data is the same as original */
      //TODO datatype should better be output arg, not input
      //TODO string attributes should probably have a separate function since the handling is different;
      //TODO   or maybe it should just accept string buffer rather than pointer to string
      if (c->types[t] == PETSC_STRING) {
        PetscCall(PetscViewerHDF5ReadAttribute(v, parent, attribute, c->types[t], NULL, &str));
        ptr0 = str;
      } else {
        PetscCall(PetscViewerHDF5ReadAttribute(v, parent, attribute, c->types[t], NULL, &buffer));
        ptr0 = &buffer;
      }
      PetscCall(compare(c->types[t], ptr0, c->vals[t], &flg));
      PetscCheck(flg, comm, PETSC_ERR_PLIB, "Value of attribute %s/%s/%s is not equal to the original value", group, parent, attribute);
      if (verbose) PetscCall(PetscPrintf(comm, " (=)"));
      if (c->types[t] == PETSC_STRING) PetscCall(PetscFree(str));
    }
    if (verbose && gd) PetscCall(PetscPrintf(comm, "\n"));
  }
  PetscCall(PetscFree(group));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CapsuleDestroy(Capsule *c)
{
  PetscInt t;

  PetscFunctionBegin;
  if (!*c) PetscFunctionReturn(PETSC_SUCCESS);
  for (t = 0; t < (*c)->ntypes; t++) PetscCall(PetscFree((*c)->vals[t]));
  PetscCall(PetscFree(*c));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode testGroupsDatasets(PetscViewer viewer)
{
  char        buf[PETSC_MAX_PATH_LEN];
  Vec         vecs[nap][ns];
  PetscInt    p, s;
  PetscBool   flg = PETSC_FALSE, flg1 = PETSC_FALSE, flg2 = PETSC_FALSE;
  PetscRandom rand;
  const char *filename;
  MPI_Comm    comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));
  PetscCall(PetscViewerFileGetName(viewer, &filename));
  if (verbose) PetscCall(PetscPrintf(comm, "# TEST testGroupsDatasets\n"));
  /* store random vectors */
  PetscCall(PetscRandomCreate(comm, &rand));
  PetscCall(PetscRandomSetInterval(rand, 0.0, 10.0));
  PetscCall(PetscRandomSetFromOptions(rand));
  PetscCall(PetscMemzero(vecs, nap * ns * sizeof(Vec)));

  /* test dataset writing */
  if (verbose) PetscCall(PetscPrintf(comm, "## WRITE PHASE\n"));
  for (p = 0; p < np; p++) {
    PetscCall(isPop(paths[p], &flg));
    PetscCall(isDot(paths[p], &flg1));
    PetscCall(shouldExist(apaths[paths2apaths[p]], PETSC_FALSE, &flg2));
    if (flg) {
      PetscCall(PetscViewerHDF5PopGroup(viewer));
    } else {
      PetscCall(PetscViewerHDF5PushGroup(viewer, paths[p]));
    }
    if (verbose) PetscCall(PetscPrintf(comm, "%-32s => %4s => %-32s  should exist? %s\n", paths[p], flg ? "pop" : "push", apaths[paths2apaths[p]], PetscBools[flg2]));
    if (flg || flg1 || !flg2) continue;

    for (s = 0; s < ns; s++) {
      Vec v;

      PetscCall(shouldExist(datasets[s], PETSC_FALSE, &flg));
      if (!flg) continue;

      PetscCall(VecCreate(comm, &v));
      PetscCall(PetscObjectSetName((PetscObject)v, datasets[s]));
      PetscCall(VecSetSizes(v, n, PETSC_DECIDE));
      PetscCall(VecSetFromOptions(v));
      PetscCall(VecSetRandom(v, rand));
      if (verbose) {
        PetscReal min, max;
        PetscCall(VecMin(v, NULL, &min));
        PetscCall(VecMax(v, NULL, &max));
        PetscCall(PetscPrintf(comm, "  Create dataset %s/%s, keep in memory in vecs[%" PetscInt_FMT "][%" PetscInt_FMT "], min %.3e max %.3e\n", apaths[paths2apaths[p]], datasets[s], paths2apaths[p], s, min, max));
      }

      PetscCall(VecView(v, viewer));
      vecs[paths2apaths[p]][s] = v;
    }
  }
  PetscCall(PetscViewerFlush(viewer));
  PetscCall(PetscRandomDestroy(&rand));

  if (verbose) PetscCall(PetscPrintf(comm, "\n## READ PHASE\n"));
  /* check correct existence of groups in file */
  for (p = 0; p < np; p++) {
    char       *group;
    const char *expected = apaths[paths2apaths[p]];

    /* check Push/Pop is correct */
    PetscCall(isPop(paths[p], &flg));
    if (flg) {
      PetscCall(PetscViewerHDF5PopGroup(viewer));
    } else {
      PetscCall(PetscViewerHDF5PushGroup(viewer, paths[p]));
    }
    PetscCall(PetscViewerHDF5GetGroup(viewer, NULL, &group));
    PetscCall(PetscViewerHDF5HasGroup(viewer, NULL, &flg1));
    if (verbose) PetscCall(PetscPrintf(comm, "%-32s => %4s => %-32s  exists? %s\n", paths[p], flg ? "pop" : "push", group, PetscBools[flg1]));
    PetscCall(PetscStrcmp(group, expected, &flg2));
    PetscCheck(flg2, comm, PETSC_ERR_PLIB, "Current group %s not equal to expected %s", group, expected);
    PetscCall(shouldExist(group, PETSC_TRUE, &flg2));
    PetscCheck(flg1 == flg2, comm, PETSC_ERR_PLIB, "Group %s should exist? %s Exists in %s? %s", group, PetscBools[flg2], filename, PetscBools[flg1]);
    PetscCall(PetscFree(group));
  }

  /* check existence of datasets; compare loaded vectors with original ones */
  for (p = 0; p < np; p++) {
    char *group;

    /* check Push/Pop is correct */
    PetscCall(isPop(paths[p], &flg));
    if (flg) {
      PetscCall(PetscViewerHDF5PopGroup(viewer));
    } else {
      PetscCall(PetscViewerHDF5PushGroup(viewer, paths[p]));
    }
    PetscCall(PetscViewerHDF5GetGroup(viewer, NULL, &group));
    PetscCall(PetscViewerHDF5HasGroup(viewer, NULL, &flg));
    if (verbose) PetscCall(PetscPrintf(comm, "Has %s group? %s\n", group, PetscBools[flg]));
    for (s = 0; s < ns; s++) {
      const char *name     = datasets[s];
      char       *fullname = buf;

      /* check correct existence of datasets in file */
      PetscCall(PetscSNPrintf(fullname, sizeof(buf), "%s/%s", group, name));
      PetscCall(shouldExist(name, PETSC_FALSE, &flg1));
      flg1 = (PetscBool)(flg && flg1); /* both group and dataset need to exist */
      PetscCall(PetscViewerHDF5HasDataset(viewer, name, &flg2));
      if (verbose) PetscCall(PetscPrintf(comm, "    %s dataset? %s", fullname, PetscBools[flg2]));
      PetscCheck(flg2 == flg1, comm, PETSC_ERR_PLIB, "Dataset %s should exist? %s Exists in %s? %s", fullname, PetscBools[flg1], filename, PetscBools[flg2]);

      if (flg2) {
        Vec v;
        /* check loaded Vec is the same as original */
        PetscCall(VecCreate(comm, &v));
        PetscCall(PetscObjectSetName((PetscObject)v, name));
        PetscCall(VecLoad(v, viewer));
        PetscCall(VecEqual(v, vecs[paths2apaths[p]][s], &flg1));
        PetscCheck(flg1, comm, PETSC_ERR_PLIB, "Dataset %s in %s is not equal to the original Vec", fullname, filename);
        if (verbose) PetscCall(PetscPrintf(comm, " (=)"));
        PetscCall(VecDestroy(&v));
      }
      if (verbose) PetscCall(PetscPrintf(comm, "\n"));
    }
    PetscCall(PetscFree(group));
  }
  PetscCall(PetscViewerFlush(viewer));
  for (p = 0; p < nap; p++)
    for (s = 0; s < ns; s++) PetscCall(VecDestroy(&vecs[p][s]));
  if (verbose) PetscCall(PetscPrintf(comm, "# END  testGroupsDatasets\n\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode formPath(PetscBool relativize, const char path[], const char dataset[], char buf[], size_t bufsize)
{
  PetscBool isroot = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(isRoot(path, &isroot));
  if (relativize) {
    if (isroot) {
      PetscCall(PetscStrncpy(buf, dataset, bufsize));
    } else {
      /* skip initial '/' in paths[p] if prefix given */
      PetscCall(PetscSNPrintf(buf, bufsize, "%s/%s", path + 1, dataset));
    }
  } else {
    PetscCall(PetscSNPrintf(buf, bufsize, "%s/%s", isroot ? "" : path, dataset));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* test attribute writing, existence checking and reading, use absolute paths */
static PetscErrorCode testAttributesAbsolutePath(PetscViewer viewer, const char prefix[])
{
  char      buf[PETSC_MAX_PATH_LEN];
  Capsule   capsules[nap][ns], c = NULL, old = NULL;
  PetscInt  p, s;
  PetscBool flg = PETSC_FALSE, flg1 = PETSC_FALSE;
  MPI_Comm  comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));
  if (verbose) {
    if (prefix) {
      PetscCall(PetscPrintf(comm, "# TEST testAttributesAbsolutePath, prefix=\"%s\"\n", prefix));
    } else {
      PetscCall(PetscPrintf(comm, "# TEST testAttributesAbsolutePath\n"));
    }
    PetscCall(PetscPrintf(comm, "## WRITE PHASE\n"));
  }
  PetscCall(PetscMemzero(capsules, nap * ns * sizeof(Capsule)));

  /* test attribute writing */
  if (prefix) PetscCall(PetscViewerHDF5PushGroup(viewer, prefix));
  for (p = 0; p < np; p++)
    for (s = 0; s < ns; s++) {
      /* we test only absolute paths here */
      PetscCall(PetscViewerHDF5PathIsRelative(paths[p], PETSC_FALSE, &flg));
      if (flg) continue;
      {
        char *group;

        PetscCall(PetscViewerHDF5GetGroup(viewer, NULL, &group));
        PetscCall(PetscStrcmp(group, prefix, &flg));
        PetscCheck(flg, comm, PETSC_ERR_PLIB, "prefix %s not equal to pushed group %s", prefix, group);
        PetscCall(PetscFree(group));
      }
      PetscCall(formPath((PetscBool) !!prefix, paths[p], datasets[s], buf, sizeof(buf)));
      PetscCall(shouldExist(buf, PETSC_TRUE, &flg));
      if (!flg) continue;

      if (verbose) {
        if (prefix) {
          PetscCall(PetscPrintf(comm, "Write attributes to %s/%s\n", prefix, buf));
        } else {
          PetscCall(PetscPrintf(comm, "Write attributes to %s\n", buf));
        }
      }

      PetscCall(CapsuleCreate(old, &c));
      PetscCall(CapsuleWriteAttributes(c, viewer, buf));
      PetscCheck(!capsules[paths2apaths[p]][s], comm, PETSC_ERR_PLIB, "capsules[%" PetscInt_FMT "][%" PetscInt_FMT "] gets overwritten for %s", paths2apaths[p], s, buf);
      capsules[paths2apaths[p]][s] = c;
      old                          = c;
    }
  if (prefix) PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(PetscViewerFlush(viewer));

  if (verbose) PetscCall(PetscPrintf(comm, "\n## READ PHASE\n"));
  if (prefix) PetscCall(PetscViewerHDF5PushGroup(viewer, prefix));
  for (p = 0; p < np; p++)
    for (s = 0; s < ns; s++) {
      /* we test only absolute paths here */
      PetscCall(PetscViewerHDF5PathIsRelative(paths[p], PETSC_FALSE, &flg));
      if (flg) continue;

      /* check existence of given group/dataset */
      PetscCall(formPath((PetscBool) !!prefix, paths[p], datasets[s], buf, sizeof(buf)));
      PetscCall(shouldExist(buf, PETSC_TRUE, &flg));
      if (verbose) {
        if (prefix) {
          PetscCall(PetscPrintf(comm, "Has %s/%s? %s\n", prefix, buf, PetscBools[flg]));
        } else {
          PetscCall(PetscPrintf(comm, "Has %s? %s\n", buf, PetscBools[flg]));
        }
      }

      /* check attribute capsule has been created for given path */
      c    = capsules[paths2apaths[p]][s];
      flg1 = (PetscBool) !!c;
      PetscCheck(flg == flg1, comm, PETSC_ERR_PLIB, "Capsule should exist for %s? %s Exists? %s", buf, PetscBools[flg], PetscBools[flg1]);
      if (!flg) continue;

      /* check correct existence and fidelity of attributes in file */
      PetscCall(CapsuleReadAndCompareAttributes(c, viewer, buf));
    }
  if (prefix) PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(PetscViewerFlush(viewer));
  for (p = 0; p < nap; p++)
    for (s = 0; s < ns; s++) PetscCall(CapsuleDestroy(&capsules[p][s]));
  if (verbose) PetscCall(PetscPrintf(comm, "# END  testAttributesAbsolutePath\n\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* test attribute writing, existence checking and reading, use group push/pop */
static PetscErrorCode testAttributesPushedPath(PetscViewer viewer)
{
  Capsule   capsules[nap][ns], c = NULL, old = NULL;
  PetscInt  p, s;
  int       gd;
  PetscBool flg = PETSC_FALSE, flg1 = PETSC_FALSE;
  MPI_Comm  comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));
  if (verbose) {
    PetscCall(PetscPrintf(comm, "# TEST testAttributesPushedPath\n"));
    PetscCall(PetscPrintf(comm, "## WRITE PHASE\n"));
  }
  PetscCall(PetscMemzero(capsules, nap * ns * sizeof(Capsule)));

  /* test attribute writing */
  for (p = 0; p < np; p++) {
    PetscCall(isPop(paths[p], &flg));
    PetscCall(isDot(paths[p], &flg1));
    if (flg) {
      PetscCall(PetscViewerHDF5PopGroup(viewer));
    } else {
      PetscCall(PetscViewerHDF5PushGroup(viewer, paths[p]));
    }
    /* < and . have been already visited => skip */
    if (flg || flg1) continue;

    /* assume here that groups and datasets are already in the file */
    for (s = 0; s < ns; s++) {
      PetscCall(hasGroupOrDataset(viewer, datasets[s], &gd));
      if (!gd) continue;
      if (verbose) PetscCall(PetscPrintf(comm, "Write attributes to %s/%s\n", apaths[paths2apaths[p]], datasets[s]));
      PetscCall(CapsuleCreate(old, &c));
      PetscCall(CapsuleWriteAttributes(c, viewer, datasets[s]));
      PetscCheck(!capsules[paths2apaths[p]][s], comm, PETSC_ERR_PLIB, "capsules[%" PetscInt_FMT "][%" PetscInt_FMT "] gets overwritten for %s/%s", paths2apaths[p], s, paths[p], datasets[s]);
      capsules[paths2apaths[p]][s] = c;
      old                          = c;
    }
  }
  PetscCall(PetscViewerFlush(viewer));

  if (verbose) PetscCall(PetscPrintf(comm, "\n## READ PHASE\n"));
  for (p = 0; p < np; p++) {
    char *group;

    PetscCall(isPop(paths[p], &flg1));
    if (flg1) {
      PetscCall(PetscViewerHDF5PopGroup(viewer));
    } else {
      PetscCall(PetscViewerHDF5PushGroup(viewer, paths[p]));
    }
    PetscCall(PetscViewerHDF5GetGroup(viewer, NULL, &group));
    for (s = 0; s < ns; s++) {
      PetscCall(hasGroupOrDataset(viewer, datasets[s], &gd));
      if (verbose) PetscCall(PetscPrintf(comm, "%s/%s   %s\n", group, datasets[s], gd ? (gd == 1 ? "is group" : "is dataset") : "does not exist"));

      /* check attribute capsule has been created for given path */
      c    = capsules[paths2apaths[p]][s];
      flg  = (PetscBool) !!gd;
      flg1 = (PetscBool) !!c;
      PetscCheck(flg == flg1, comm, PETSC_ERR_PLIB, "Capsule should exist for %s/%s? %s Exists? %s", group, datasets[s], PetscBools[flg], PetscBools[flg1]);
      if (!flg) continue;

      /* check correct existence of attributes in file */
      PetscCall(CapsuleReadAndCompareAttributes(c, viewer, datasets[s]));
    }
    PetscCall(PetscFree(group));
  }
  PetscCall(PetscViewerFlush(viewer));
  for (p = 0; p < nap; p++)
    for (s = 0; s < ns; s++) PetscCall(CapsuleDestroy(&capsules[p][s]));
  if (verbose) PetscCall(PetscPrintf(comm, "# END  testAttributesPushedPath\n\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* test attribute writing, existence checking and reading, use group push/pop */
static PetscErrorCode testObjectAttributes(PetscViewer viewer)
{
  Capsule   capsules[nap][ns], c = NULL, old = NULL;
  PetscInt  p, s;
  PetscBool flg = PETSC_FALSE, flg1 = PETSC_FALSE;
  MPI_Comm  comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));
  if (verbose) {
    PetscCall(PetscPrintf(comm, "# TEST testObjectAttributes\n"));
    PetscCall(PetscPrintf(comm, "## WRITE PHASE\n"));
  }
  PetscCall(PetscMemzero(capsules, nap * ns * sizeof(Capsule)));

  /* test attribute writing */
  for (p = 0; p < np; p++) {
    PetscCall(isPop(paths[p], &flg));
    PetscCall(isDot(paths[p], &flg1));
    if (flg) {
      PetscCall(PetscViewerHDF5PopGroup(viewer));
    } else {
      PetscCall(PetscViewerHDF5PushGroup(viewer, paths[p]));
    }
    /* < and . have been already visited => skip */
    if (flg || flg1) continue;

    /* assume here that groups and datasets are already in the file */
    for (s = 0; s < ns; s++) {
      Vec         v;
      size_t      len;
      const char *name = datasets[s];

      PetscCall(PetscStrlen(name, &len));
      if (!len) continue;
      PetscCall(VecCreate(comm, &v));
      PetscCall(PetscObjectSetName((PetscObject)v, name));
      PetscCall(PetscViewerHDF5HasObject(viewer, (PetscObject)v, &flg));
      if (flg) {
        if (verbose) PetscCall(PetscPrintf(comm, "Write attributes to %s/%s\n", apaths[paths2apaths[p]], name));
        PetscCall(CapsuleCreate(old, &c));
        PetscCall(CapsuleWriteAttributes(c, viewer, name));
        PetscCheck(!capsules[paths2apaths[p]][s], comm, PETSC_ERR_PLIB, "capsules[%" PetscInt_FMT "][%" PetscInt_FMT "] gets overwritten for %s/%s", paths2apaths[p], s, paths[p], name);
        capsules[paths2apaths[p]][s] = c;
        old                          = c;
      }
      PetscCall(VecDestroy(&v));
    }
  }
  PetscCall(PetscViewerFlush(viewer));

  if (verbose) PetscCall(PetscPrintf(comm, "\n## READ PHASE\n"));
  for (p = 0; p < np; p++) {
    char *group;

    PetscCall(isPop(paths[p], &flg));
    if (flg) {
      PetscCall(PetscViewerHDF5PopGroup(viewer));
    } else {
      PetscCall(PetscViewerHDF5PushGroup(viewer, paths[p]));
    }
    PetscCall(PetscViewerHDF5GetGroup(viewer, NULL, &group));
    for (s = 0; s < ns; s++) {
      Vec         v;
      size_t      len;
      const char *name = datasets[s];

      PetscCall(PetscStrlen(name, &len));
      if (!len) continue;
      PetscCall(VecCreate(comm, &v));
      PetscCall(PetscObjectSetName((PetscObject)v, name));
      PetscCall(PetscViewerHDF5HasObject(viewer, (PetscObject)v, &flg));
      if (verbose) PetscCall(PetscPrintf(comm, "Is %s/%s dataset? %s\n", group, name, PetscBools[flg]));

      /* check attribute capsule has been created for given path */
      c    = capsules[paths2apaths[p]][s];
      flg1 = (PetscBool) !!c;
      PetscCheck(flg == flg1, comm, PETSC_ERR_PLIB, "Capsule should exist for %s/%s? %s Exists? %s", group, name, PetscBools[flg], PetscBools[flg1]);

      /* check correct existence of attributes in file */
      if (flg) PetscCall(CapsuleReadAndCompareAttributes(c, viewer, name));
      PetscCall(VecDestroy(&v));
    }
    PetscCall(PetscFree(group));
  }
  PetscCall(PetscViewerFlush(viewer));
  for (p = 0; p < nap; p++)
    for (s = 0; s < ns; s++) PetscCall(CapsuleDestroy(&capsules[p][s]));
  if (verbose) PetscCall(PetscPrintf(comm, "# END  testObjectAttributes\n\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode testAttributesDefaultValue(PetscViewer viewer)
{
#define nv 4
  PetscBool bools[nv];
  PetscInt  ints[nv];
  PetscReal reals[nv];
  char     *strings[nv];
  PetscBool flg;
  PetscInt  i;
  MPI_Comm  comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));
  if (verbose) PetscCall(PetscPrintf(comm, "# TEST testAttributesDefaultValue\n"));

  PetscCall(PetscViewerHDF5ReadAttribute(viewer, "/", "attr_0_bool", PETSC_BOOL, NULL, &bools[0]));
  bools[1] = PetscNot(bools[0]);
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, "/", "attr_0_bool", PETSC_BOOL, &bools[1], &bools[2]));
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, "/", "attr_nonExisting_bool", PETSC_BOOL, &bools[1], &bools[3]));
  PetscCheck(bools[2] == bools[0], comm, PETSC_ERR_PLIB, "%s = bools[2] != bools[0] = %s", PetscBools[bools[2]], PetscBools[bools[0]]);
  PetscCheck(bools[3] == bools[1], comm, PETSC_ERR_PLIB, "%s = bools[3] != bools[1] = %s", PetscBools[bools[3]], PetscBools[bools[1]]);

  PetscCall(PetscViewerHDF5ReadAttribute(viewer, "/", "attr_0_int", PETSC_INT, NULL, &ints[0]));
  ints[1] = ints[0] * -333;
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, "/", "attr_0_int", PETSC_INT, &ints[1], &ints[2]));
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, "/", "attr_nonExisting_int", PETSC_INT, &ints[1], &ints[3]));
  PetscCheck(ints[2] == ints[0], comm, PETSC_ERR_PLIB, "%" PetscInt_FMT " = ints[2] != ints[0] = %" PetscInt_FMT, ints[2], ints[0]);
  PetscCheck(ints[3] == ints[1], comm, PETSC_ERR_PLIB, "%" PetscInt_FMT " = ints[3] != ints[1] = %" PetscInt_FMT, ints[3], ints[1]);
  if (verbose) PetscCall(PetscIntView(nv, ints, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(PetscViewerHDF5ReadAttribute(viewer, "/", "attr_0_real", PETSC_REAL, NULL, &reals[0]));
  reals[1] = reals[0] * -11.1;
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, "/", "attr_0_real", PETSC_REAL, &reals[1], &reals[2]));
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, "/", "attr_nonExisting_real", PETSC_REAL, &reals[1], &reals[3]));
  PetscCheck(reals[2] == reals[0], comm, PETSC_ERR_PLIB, "%f = reals[2] != reals[0] = %f", reals[2], reals[0]);
  PetscCheck(reals[3] == reals[1], comm, PETSC_ERR_PLIB, "%f = reals[3] != reals[1] = %f", reals[3], reals[1]);
  if (verbose) PetscCall(PetscRealView(nv, reals, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(PetscViewerHDF5ReadAttribute(viewer, "/", "attr_0_str", PETSC_STRING, NULL, &strings[0]));
  PetscCall(PetscStrallocpy(strings[0], &strings[1]));
  PetscCall(alterString(strings[0], strings[1]));
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, "/", "attr_0_str", PETSC_STRING, &strings[1], &strings[2]));
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, "/", "attr_nonExisting_str", PETSC_STRING, &strings[1], &strings[3]));
  PetscCall(PetscStrcmp(strings[2], strings[0], &flg));
  PetscCheck(flg, comm, PETSC_ERR_PLIB, "%s = strings[2] != strings[0] = %s", strings[2], strings[0]);
  PetscCall(PetscStrcmp(strings[3], strings[1], &flg));
  PetscCheck(flg, comm, PETSC_ERR_PLIB, "%s = strings[3] != strings[1] = %s", strings[3], strings[1]);
  for (i = 0; i < nv; i++) PetscCall(PetscFree(strings[i]));

  PetscCall(PetscViewerFlush(viewer));
  if (verbose) PetscCall(PetscPrintf(comm, "# END  testAttributesDefaultValue\n"));
#undef nv
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  static char filename[PETSC_MAX_PATH_LEN] = "ex48.h5";
  PetscMPIInt rank;
  MPI_Comm    comm;
  PetscViewer viewer;

  PetscFunctionBegin;
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-verbose", &verbose, NULL));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-filename", filename, sizeof(filename), NULL));
  if (verbose) PetscCall(PetscPrintf(comm, "np ns " PetscStringize(np) " " PetscStringize(ns) "\n"));

  PetscCall(PetscViewerHDF5Open(comm, filename, FILE_MODE_WRITE, &viewer));
  PetscCall(testGroupsDatasets(viewer));
  PetscCall(testAttributesAbsolutePath(viewer, "/"));
  PetscCall(testAttributesAbsolutePath(viewer, "/prefix"));
  PetscCall(PetscViewerDestroy(&viewer));

  /* test reopening in update mode */
  PetscCall(PetscViewerHDF5Open(comm, filename, FILE_MODE_UPDATE, &viewer));
  PetscCall(testAttributesPushedPath(viewer));
  PetscCall(testObjectAttributes(viewer));
  PetscCall(testAttributesDefaultValue(viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

     build:
       requires: hdf5

     test:
       nsize: {{1 4}}

TEST*/
