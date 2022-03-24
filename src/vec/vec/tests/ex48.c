
static char help[] = "Tests HDF5 attribute I/O.\n\n";

#include <petscviewerhdf5.h>
#include <petscvec.h>

static PetscInt       n = 5;    /* testing vector size */
static PetscBool      verbose = PETSC_FALSE;
#define SLEN 128

/* sequence of unique absolute paths */
#define nap 9
static const char *apaths[nap] =
{
/* 0 */
  "/",
  "/g1",
  "/g1/g2",
  "/g1/nonExistingGroup1",
  "/g1/g3",
/* 5 */
  "/g1/g3/g4",
  "/g1/nonExistingGroup2",
  "/g1/nonExistingGroup2/g5",
  "/g1/g6/g7"
};

#define np 21
/* sequence of paths (absolute or relative); "<" encodes Pop */
static const char *paths[np] =
{
/* 0 */
  "/",
  "/g1",
  "/g1/g2",
  "/g1/nonExistingGroup1",
  "<",
/* 5 */
  ".",                      /* /g1/g2 */
  "<",
  "<",
  "g3",                     /* /g1/g3 */
  "g4",                     /* /g1/g3/g4 */
/* 10 */
  "<",
  "<",
  ".",                      /* /g1 */
  "<",
  "nonExistingGroup2",      /* /g1/nonExistingG2 */
/* 15 */
  "g5",                     /* /g1/nonExistingG2/g5 */
  "<",
  "<",
  "g6/g7",                  /* /g1/g6/g7 */
  "<",
/* 20 */
  "<",
};
/* corresponsing expected absolute paths - positions in abspath */
static const PetscInt paths2apaths[np] =
{
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
static const char *datasets[ns]  =
{
  "",
  "x",
  "nonExistingVec",
  "y"
};

/* beware this yields PETSC_FALSE for "" but group "" is interpreted as "/" */
static inline PetscErrorCode shouldExist(const char name[], PetscBool emptyExists, PetscBool *has)
{
  size_t         len=0;

  PetscFunctionBegin;
  CHKERRQ(PetscStrlen(name, &len));
  *has = emptyExists;
  if (len) {
    char *loc=NULL;
    CHKERRQ(PetscStrstr(name,"nonExisting",&loc));
    *has = PetscNot(loc);
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode isPop(const char path[], PetscBool *has)
{
  PetscFunctionBegin;
  CHKERRQ(PetscStrcmp(path, "<", has));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode isDot(const char path[], PetscBool *has)
{
  PetscFunctionBegin;
  CHKERRQ(PetscStrcmp(path, ".", has));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode isRoot(const char path[], PetscBool *flg)
{
  size_t         len;

  PetscFunctionBegin;
  CHKERRQ(PetscStrlen(path, &len));
  *flg = PetscNot(len);
  if (!*flg) {
    CHKERRQ(PetscStrcmp(path, "/", flg));
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode compare(PetscDataType dt, void *ptr0, void *ptr1, PetscBool *flg)
{
  PetscFunctionBegin;
  switch (dt) {
    case PETSC_INT:
      *flg = (PetscBool)(*(PetscInt*)ptr0 == *(PetscInt*)ptr1);
      if (verbose) {
        if (*flg) {
          CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "%" PetscInt_FMT, *(PetscInt*)ptr0));
        } else {
          CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "%" PetscInt_FMT " != %" PetscInt_FMT "\n", *(PetscInt*)ptr0, *(PetscInt*)ptr1));
        }
      }
      break;
    case PETSC_REAL:
      *flg = (PetscBool)(*(PetscReal*)ptr0 == *(PetscReal*)ptr1);
      if (verbose) {
        if (*flg) {
          CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "%f", *(PetscReal*)ptr0));
        } else {
          CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "%f != %f\n", *(PetscReal*)ptr0, *(PetscReal*)ptr1));
        }
      }
      break;
    case PETSC_BOOL:
      *flg = (PetscBool)(*(PetscBool*)ptr0 == *(PetscBool*)ptr1);
      if (verbose) {
        if (*flg) {
          CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "%s", PetscBools[*(PetscBool*)ptr0]));
        } else {
          CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "%s != %s\n", PetscBools[*(PetscBool*)ptr0], PetscBools[*(PetscBool*)ptr1]));
        }
      }
      break;
    case PETSC_STRING:
      CHKERRQ(PetscStrcmp((const char*)ptr0, (const char*)ptr1, flg));
      if (verbose) {
        if (*flg) {
          CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "%s", (char*)ptr0));
        } else {
          CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "%s != %s\n", (char*)ptr0, (char*)ptr1));
        }
      }
      break;
    default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "PetscDataType %s not handled here", PetscDataTypes[dt]);
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode alterString(const char oldstr[], char str[])
{
  size_t          i,n;

  PetscFunctionBegin;
  CHKERRQ(PetscStrcpy(str, oldstr));
  CHKERRQ(PetscStrlen(oldstr, &n));
  for (i=0; i<n; i++) {
    if (('A' <= str[i] && str[i] < 'Z') || ('a' <= str[i] && str[i] < 'z')) {
      str[i]++;
      break;
    }
  }
  PetscFunctionReturn(0);
}

/* if name given, check dataset with this name exists under current group, otherwise just check current group exists */
/* flg: 0 doesn't exist, 1 group, 2 dataset */
static PetscErrorCode hasGroupOrDataset(PetscViewer viewer, const char path[], int *flg)
{
  PetscBool      has;

  PetscFunctionBegin;
  *flg = 0;
  CHKERRQ(PetscViewerHDF5HasGroup(viewer, path, &has));
  if (has) *flg = 1;
  else {
    CHKERRQ(PetscViewerHDF5HasDataset(viewer, path, &has));
    if (has) *flg = 2;
  }
  PetscFunctionReturn(0);
}

#define nt 5   /* number of datatypes */
typedef struct _n_Capsule* Capsule;
struct _n_Capsule {
  char           names[nt][SLEN];
  PetscDataType  types[nt];
  char           typeNames[nt][SLEN];
  size_t         sizes[nt];
  void           *vals[nt];
  PetscInt       id, ntypes;
};

static PetscErrorCode CapsuleCreate(Capsule old, Capsule *newcapsule)
{
  Capsule        c;
  PetscBool      bool0        = PETSC_TRUE;
  PetscInt       int0         = -1;
  PetscReal      real0        = -1.1;
  char           str0[]       = "Test String";
  char           nestr0[]     = "NONEXISTING STRING"; /* this attribute shall be skipped for writing */
  void           *vals[nt]    = {&bool0,        &int0,        &real0,        str0,         nestr0};
  size_t         sizes[nt]    = {sizeof(bool0), sizeof(int0), sizeof(real0), sizeof(str0), sizeof(str0)};
  PetscDataType  types[nt]    = {PETSC_BOOL,    PETSC_INT,    PETSC_REAL,    PETSC_STRING, PETSC_STRING};
  const char     *tNames[nt]  = {"bool",        "int",        "real",        "str",        "nonExisting"};
  PetscInt       t;

  PetscFunctionBegin;
  CHKERRQ(PetscNew(&c));
  c->id = 0;
  c->ntypes = nt;
  if (old) {
    /* alter values */
    t=0;
    bool0 = PetscNot(*((PetscBool*)old->vals[t]));                      t++;
    int0  = *((PetscInt*) old->vals[t]) * -2;                           t++;
    real0 = *((PetscReal*)old->vals[t]) * -2.0;                         t++;
    CHKERRQ(alterString((const char*)old->vals[t], str0)); t++;
    c->id = old->id+1;
  }
  for (t=0; t<nt; t++) {
    c->sizes[t] = sizes[t];
    c->types[t] = types[t];
    CHKERRQ(PetscStrcpy(c->typeNames[t], tNames[t]));
    CHKERRQ(PetscSNPrintf(c->names[t], SLEN, "attr_%" PetscInt_FMT "_%s", c->id, tNames[t]));
    CHKERRQ(PetscMalloc(sizes[t], &c->vals[t]));
    CHKERRQ(PetscMemcpy(c->vals[t], vals[t], sizes[t]));
  }
  *newcapsule = c;
  PetscFunctionReturn(0);
}
#undef nt

static PetscErrorCode CapsuleWriteAttributes(Capsule c, PetscViewer v, const char parent[])
{
  PetscInt       t;
  PetscBool      flg=PETSC_FALSE;

  PetscFunctionBegin;
  for (t=0; t < c->ntypes; t++) {
    CHKERRQ(shouldExist(c->names[t], PETSC_FALSE, &flg));
    if (!flg) continue;
    CHKERRQ(PetscViewerHDF5WriteAttribute(v, parent, c->names[t], c->types[t], c->vals[t]));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CapsuleReadAndCompareAttributes(Capsule c, PetscViewer v, const char parent[])
{
  const char     *group;
  int            gd=0;
  PetscInt       t;
  PetscBool      flg=PETSC_FALSE, hasAttr=PETSC_FALSE;
  MPI_Comm       comm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)v, &comm));
  CHKERRQ(PetscViewerHDF5GetGroup(v, &group));
  if (!group) group = "";
  CHKERRQ(hasGroupOrDataset(v, parent, &gd));
  /* check correct existence of attributes */
  for (t=0; t < c->ntypes; t++) {
    const char *attribute = c->names[t];
    CHKERRQ(shouldExist(attribute, PETSC_FALSE, &flg));
    CHKERRQ(PetscViewerHDF5HasAttribute(v, parent, attribute, &hasAttr));
    if (verbose) {
      CHKERRQ(PetscPrintf(comm, "    %-24s = ", attribute));
      if (!hasAttr) {
        CHKERRQ(PetscPrintf(comm, "---"));
      }
    }
    PetscCheckFalse(!gd && hasAttr,comm, PETSC_ERR_PLIB, "Attribute %s/%s/%s exists while its parent %s/%s doesn't exist", group, parent, attribute, group, parent);
    PetscCheckFalse(flg != hasAttr,comm, PETSC_ERR_PLIB, "Attribute %s/%s should exist? %s Exists? %s", parent, attribute, PetscBools[flg], PetscBools[hasAttr]);

    /* check loaded attributes are the same as original */
    if (hasAttr) {
      char buffer[SLEN];
      char *str;
      void *ptr0;
      /* check the stored data is the same as original */
      //TODO datatype should better be output arg, not input
      //TODO string attributes should probably have a separate function since the handling is different;
      //TODO   or maybe it should just accept string buffer rather than pointer to string
      if (c->types[t] == PETSC_STRING) {
        CHKERRQ(PetscViewerHDF5ReadAttribute(v, parent, attribute, c->types[t], NULL, &str));
        ptr0 = str;
      } else {
        CHKERRQ(PetscViewerHDF5ReadAttribute(v, parent, attribute, c->types[t], NULL, &buffer));
        ptr0 = &buffer;
      }
      CHKERRQ(compare(c->types[t], ptr0, c->vals[t], &flg));
      PetscCheck(flg,comm, PETSC_ERR_PLIB, "Value of attribute %s/%s/%s is not equal to the original value", group, parent, attribute);
      if (verbose) CHKERRQ(PetscPrintf(comm, " (=)"));
      if (c->types[t] == PETSC_STRING) {
        CHKERRQ(PetscFree(str));
      }
    }
    if (verbose && gd) CHKERRQ(PetscPrintf(comm, "\n"));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CapsuleDestroy(Capsule *c)
{
  PetscInt              t;

  PetscFunctionBegin;
  if (!*c) PetscFunctionReturn(0);
  for (t=0; t < (*c)->ntypes; t++) {
    CHKERRQ(PetscFree((*c)->vals[t]));
  }
  CHKERRQ(PetscFree(*c));
  PetscFunctionReturn(0);
}

static PetscErrorCode testGroupsDatasets(PetscViewer viewer)
{
  char           buf[PETSC_MAX_PATH_LEN];
  Vec            vecs[nap][ns];
  PetscInt       p,s;
  PetscBool      flg=PETSC_FALSE,flg1=PETSC_FALSE,flg2=PETSC_FALSE;
  PetscRandom    rand;
  const char    *filename;
  MPI_Comm       comm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)viewer, &comm));
  CHKERRQ(PetscViewerFileGetName(viewer, &filename));
  if (verbose) CHKERRQ(PetscPrintf(comm, "# TEST testGroupsDatasets\n"));
  /* store random vectors */
  CHKERRQ(PetscRandomCreate(comm, &rand));
  CHKERRQ(PetscRandomSetInterval(rand, 0.0, 10.0));
  CHKERRQ(PetscRandomSetFromOptions(rand));
  CHKERRQ(PetscMemzero(vecs, nap * ns * sizeof(Vec)));

  /* test dataset writing */
  if (verbose) CHKERRQ(PetscPrintf(comm, "## WRITE PHASE\n"));
  for (p=0; p<np; p++) {
    CHKERRQ(isPop(paths[p], &flg));
    CHKERRQ(isDot(paths[p], &flg1));
    CHKERRQ(shouldExist(apaths[paths2apaths[p]], PETSC_FALSE, &flg2));
    if (flg) {
      CHKERRQ(PetscViewerHDF5PopGroup(viewer));
    } else {
      CHKERRQ(PetscViewerHDF5PushGroup(viewer, paths[p]));
    }
    if (verbose) {
      CHKERRQ(PetscPrintf(comm, "%-32s => %4s => %-32s  should exist? %s\n", paths[p], flg?"pop":"push", apaths[paths2apaths[p]], PetscBools[flg2]));
    }
    if (flg || flg1 || !flg2) continue;

    for (s=0; s<ns; s++) {
      Vec       v;

      CHKERRQ(shouldExist(datasets[s], PETSC_FALSE, &flg));
      if (!flg) continue;

      CHKERRQ(VecCreate(comm, &v));
      CHKERRQ(PetscObjectSetName((PetscObject)v, datasets[s]));
      CHKERRQ(VecSetSizes(v, n, PETSC_DECIDE));
      CHKERRQ(VecSetFromOptions(v));
      CHKERRQ(VecSetRandom(v,rand));
      if (verbose) {
        PetscReal min,max;
        CHKERRQ(VecMin(v, NULL, &min));
        CHKERRQ(VecMax(v, NULL, &max));
        CHKERRQ(PetscPrintf(comm, "  Create dataset %s/%s, keep in memory in vecs[%" PetscInt_FMT "][%" PetscInt_FMT "], min %.3e max %.3e\n", apaths[paths2apaths[p]], datasets[s], paths2apaths[p], s, min, max));
      }

      CHKERRQ(VecView(v, viewer));
      vecs[paths2apaths[p]][s] = v;
    }
  }
  CHKERRQ(PetscViewerFlush(viewer));
  CHKERRQ(PetscRandomDestroy(&rand));

  if (verbose) CHKERRQ(PetscPrintf(comm, "\n## READ PHASE\n"));
  /* check correct existence of groups in file */
  for (p=0; p<np; p++) {
    const char *group;
    const char *expected = apaths[paths2apaths[p]];

    /* check Push/Pop is correct */
    CHKERRQ(isPop(paths[p], &flg));
    if (flg) {
      CHKERRQ(PetscViewerHDF5PopGroup(viewer));
    } else {
      CHKERRQ(PetscViewerHDF5PushGroup(viewer, paths[p]));
    }
    CHKERRQ(PetscViewerHDF5GetGroup(viewer, &group));
    CHKERRQ(PetscViewerHDF5HasGroup(viewer, NULL, &flg1));
    if (!group) group = "/";  /* "/" is stored as NULL */
    if (verbose) {
      CHKERRQ(PetscPrintf(comm, "%-32s => %4s => %-32s  exists? %s\n", paths[p], flg?"pop":"push", group, PetscBools[flg1]));
    }
    CHKERRQ(PetscStrcmp(group, expected, &flg2));
    PetscCheck(flg2,comm, PETSC_ERR_PLIB, "Current group %s not equal to expected %s", group, expected);
    CHKERRQ(shouldExist(group, PETSC_TRUE, &flg2));
    PetscCheckFalse(flg1 != flg2,comm, PETSC_ERR_PLIB, "Group %s should exist? %s Exists in %s? %s", group, PetscBools[flg2], filename, PetscBools[flg1]);
  }

  /* check existence of datasets; compare loaded vectors with original ones */
  for (p=0; p<np; p++) {
    const char *group;

    /* check Push/Pop is correct */
    CHKERRQ(isPop(paths[p], &flg));
    if (flg) {
      CHKERRQ(PetscViewerHDF5PopGroup(viewer));
    } else {
      CHKERRQ(PetscViewerHDF5PushGroup(viewer, paths[p]));
    }
    CHKERRQ(PetscViewerHDF5GetGroup(viewer, &group));
    CHKERRQ(PetscViewerHDF5HasGroup(viewer, NULL, &flg));
    if (verbose) CHKERRQ(PetscPrintf(comm, "Has %s group? %s\n", group ? group : "/", PetscBools[flg]));
    if (!group) group = "";  /* "/" is stored as NULL */
    for (s=0; s<ns; s++) {
      const char *name = datasets[s];
      char       *fullname = buf;

      /* check correct existence of datasets in file */
      CHKERRQ(PetscSNPrintf(fullname, sizeof(buf), "%s/%s", group, name));
      CHKERRQ(shouldExist(name,PETSC_FALSE,&flg1));
      flg1 = (PetscBool)(flg && flg1); /* both group and dataset need to exist */
      CHKERRQ(PetscViewerHDF5HasDataset(viewer, name, &flg2));
      if (verbose) CHKERRQ(PetscPrintf(comm, "    %s dataset? %s", fullname, PetscBools[flg2]));
      PetscCheckFalse(flg2 != flg1,comm, PETSC_ERR_PLIB, "Dataset %s should exist? %s Exists in %s? %s", fullname, PetscBools[flg1], filename, PetscBools[flg2]);

      if (flg2) {
        Vec v;
        /* check loaded Vec is the same as original */
        CHKERRQ(VecCreate(comm, &v));
        CHKERRQ(PetscObjectSetName((PetscObject)v, name));
        CHKERRQ(VecLoad(v, viewer));
        CHKERRQ(VecEqual(v, vecs[paths2apaths[p]][s], &flg1));
        PetscCheck(flg1,comm, PETSC_ERR_PLIB, "Dataset %s in %s is not equal to the original Vec", fullname, filename);
        if (verbose) CHKERRQ(PetscPrintf(comm, " (=)"));
        CHKERRQ(VecDestroy(&v));
      }
      if (verbose) CHKERRQ(PetscPrintf(comm, "\n"));
    }
  }
  CHKERRQ(PetscViewerFlush(viewer));
  for (p=0; p<nap; p++) for (s=0; s<ns; s++) {
    CHKERRQ(VecDestroy(&vecs[p][s]));
  }
  if (verbose) CHKERRQ(PetscPrintf(comm, "# END  testGroupsDatasets\n\n"));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode formPath(PetscBool relativize, const char path[], const char dataset[], char buf[], size_t bufsize)
{
  PetscBool      isroot=PETSC_FALSE;

  PetscFunctionBegin;
  CHKERRQ(isRoot(path, &isroot));
  if (relativize) {
    if (isroot) {
      CHKERRQ(PetscStrncpy(buf, dataset, bufsize));
    } else {
      /* skip initial '/' in paths[p] if prefix given */
      CHKERRQ(PetscSNPrintf(buf, bufsize, "%s/%s", path+1, dataset));
    }
  } else {
    CHKERRQ(PetscSNPrintf(buf, bufsize, "%s/%s", isroot ? "" : path, dataset));
  }
  PetscFunctionReturn(0);
}

/* test attribute writing, existence checking and reading, use absolute paths */
static PetscErrorCode testAttributesAbsolutePath(PetscViewer viewer, const char prefix[])
{
  char           buf[PETSC_MAX_PATH_LEN];
  Capsule        capsules[nap][ns], c=NULL, old=NULL;
  PetscInt       p,s;
  PetscBool      flg=PETSC_FALSE,flg1=PETSC_FALSE;
  MPI_Comm       comm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)viewer, &comm));
  if (verbose) {
    if (prefix) {
      CHKERRQ(PetscPrintf(comm, "# TEST testAttributesAbsolutePath, prefix=\"%s\"\n", prefix));
    } else {
      CHKERRQ(PetscPrintf(comm, "# TEST testAttributesAbsolutePath\n"));
    }
    CHKERRQ(PetscPrintf(comm, "## WRITE PHASE\n"));
  }
  CHKERRQ(PetscMemzero(capsules, nap * ns * sizeof(Capsule)));

  /* test attribute writing */
  if (prefix) {
    CHKERRQ(PetscViewerHDF5PushGroup(viewer, prefix));
  }
  for (p=0; p<np; p++) for (s=0; s<ns; s++) {
    /* we test only absolute paths here */
    CHKERRQ(PetscViewerHDF5PathIsRelative(paths[p], PETSC_FALSE, &flg));
    if (flg) continue;
    {
      const char *group;
      CHKERRQ(PetscViewerHDF5GetGroup(viewer, &group));
      CHKERRQ(PetscStrcmp(group, prefix, &flg));
      PetscCheck(flg,comm, PETSC_ERR_PLIB, "prefix %s not equal to pushed group %s", prefix, group);
    }
    CHKERRQ(formPath((PetscBool)!!prefix, paths[p], datasets[s], buf, sizeof(buf)));
    CHKERRQ(shouldExist(buf, PETSC_TRUE, &flg));
    if (!flg) continue;

    if (verbose) {
      if (prefix) {
        CHKERRQ(PetscPrintf(comm, "Write attributes to %s/%s\n", prefix, buf));
      } else {
        CHKERRQ(PetscPrintf(comm, "Write attributes to %s\n", buf));
      }
    }

    CHKERRQ(CapsuleCreate(old, &c));
    CHKERRQ(CapsuleWriteAttributes(c, viewer, buf));
    PetscCheckFalse(capsules[paths2apaths[p]][s],comm, PETSC_ERR_PLIB, "capsules[%" PetscInt_FMT "][%" PetscInt_FMT "] gets overwritten for %s", paths2apaths[p], s, buf);
    capsules[paths2apaths[p]][s] = c;
    old = c;
  }
  if (prefix) {
    CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  }
  CHKERRQ(PetscViewerFlush(viewer));

  if (verbose) CHKERRQ(PetscPrintf(comm, "\n## READ PHASE\n"));
  if (prefix) {
    CHKERRQ(PetscViewerHDF5PushGroup(viewer, prefix));
  }
  for (p=0; p<np; p++) for (s=0; s<ns; s++) {
    /* we test only absolute paths here */
    CHKERRQ(PetscViewerHDF5PathIsRelative(paths[p], PETSC_FALSE, &flg));
    if (flg) continue;

    /* check existence of given group/dataset */
    CHKERRQ(formPath((PetscBool)!!prefix, paths[p], datasets[s], buf, sizeof(buf)));
    CHKERRQ(shouldExist(buf, PETSC_TRUE, &flg));
    if (verbose) {
      if (prefix) {
        CHKERRQ(PetscPrintf(comm, "Has %s/%s? %s\n", prefix, buf, PetscBools[flg]));
      } else {
        CHKERRQ(PetscPrintf(comm, "Has %s? %s\n", buf, PetscBools[flg]));
      }
    }

    /* check attribute capsule has been created for given path */
    c = capsules[paths2apaths[p]][s];
    flg1 = (PetscBool) !!c;
    PetscCheckFalse(flg != flg1,comm, PETSC_ERR_PLIB, "Capsule should exist for %s? %s Exists? %s", buf, PetscBools[flg], PetscBools[flg1]);
    if (!flg) continue;

    /* check correct existence and fidelity of attributes in file */
    CHKERRQ(CapsuleReadAndCompareAttributes(c, viewer, buf));
  }
  if (prefix) {
    CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  }
  CHKERRQ(PetscViewerFlush(viewer));
  for (p=0; p<nap; p++) for (s=0; s<ns; s++) {
    CHKERRQ(CapsuleDestroy(&capsules[p][s]));
  }
  if (verbose) CHKERRQ(PetscPrintf(comm, "# END  testAttributesAbsolutePath\n\n"));
  PetscFunctionReturn(0);
}

/* test attribute writing, existence checking and reading, use group push/pop */
static PetscErrorCode testAttributesPushedPath(PetscViewer viewer)
{
  Capsule        capsules[nap][ns], c=NULL, old=NULL;
  PetscInt       p,s;
  int            gd;
  PetscBool      flg=PETSC_FALSE,flg1=PETSC_FALSE;
  MPI_Comm       comm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)viewer, &comm));
  if (verbose) {
    CHKERRQ(PetscPrintf(comm, "# TEST testAttributesPushedPath\n"));
    CHKERRQ(PetscPrintf(comm, "## WRITE PHASE\n"));
  }
  CHKERRQ(PetscMemzero(capsules, nap * ns * sizeof(Capsule)));

  /* test attribute writing */
  for (p=0; p<np; p++) {
    CHKERRQ(isPop(paths[p], &flg));
    CHKERRQ(isDot(paths[p], &flg1));
    if (flg) {
      CHKERRQ(PetscViewerHDF5PopGroup(viewer));
    } else {
      CHKERRQ(PetscViewerHDF5PushGroup(viewer, paths[p]));
    }
    /* < and . have been already visited => skip */
    if (flg || flg1) continue;

    /* assume here that groups and datasets are already in the file */
    for (s=0; s<ns; s++) {
      CHKERRQ(hasGroupOrDataset(viewer, datasets[s], &gd));
      if (!gd) continue;
      if (verbose) CHKERRQ(PetscPrintf(comm, "Write attributes to %s/%s\n", apaths[paths2apaths[p]], datasets[s]));
      CHKERRQ(CapsuleCreate(old, &c));
      CHKERRQ(CapsuleWriteAttributes(c, viewer, datasets[s]));
      PetscCheckFalse(capsules[paths2apaths[p]][s],comm, PETSC_ERR_PLIB, "capsules[%" PetscInt_FMT "][%" PetscInt_FMT "] gets overwritten for %s/%s", paths2apaths[p], s, paths[p], datasets[s]);
      capsules[paths2apaths[p]][s] = c;
      old = c;
    }
  }
  CHKERRQ(PetscViewerFlush(viewer));

  if (verbose) CHKERRQ(PetscPrintf(comm, "\n## READ PHASE\n"));
  for (p=0; p<np; p++) {
    const char *group;

    CHKERRQ(isPop(paths[p], &flg1));
    if (flg1) {
      CHKERRQ(PetscViewerHDF5PopGroup(viewer));
    } else {
      CHKERRQ(PetscViewerHDF5PushGroup(viewer, paths[p]));
    }
    CHKERRQ(PetscViewerHDF5GetGroup(viewer, &group));
    if (!group) group = "";
    for (s=0; s<ns; s++) {
      CHKERRQ(hasGroupOrDataset(viewer, datasets[s], &gd));
      if (verbose) CHKERRQ(PetscPrintf(comm, "%s/%s   %s\n", group, datasets[s], gd ? (gd==1 ? "is group" : "is dataset") : "does not exist"));

      /* check attribute capsule has been created for given path */
      c = capsules[paths2apaths[p]][s];
      flg  = (PetscBool) !!gd;
      flg1 = (PetscBool) !!c;
      PetscCheckFalse(flg != flg1,comm, PETSC_ERR_PLIB, "Capsule should exist for %s/%s? %s Exists? %s", group, datasets[s], PetscBools[flg], PetscBools[flg1]);
      if (!flg) continue;

      /* check correct existence of attributes in file */
      CHKERRQ(CapsuleReadAndCompareAttributes(c, viewer, datasets[s]));
    }
  }
  CHKERRQ(PetscViewerFlush(viewer));
  for (p=0; p<nap; p++) for (s=0; s<ns; s++) {
    CHKERRQ(CapsuleDestroy(&capsules[p][s]));
  }
  if (verbose) CHKERRQ(PetscPrintf(comm, "# END  testAttributesPushedPath\n\n"));
  PetscFunctionReturn(0);
}

/* test attribute writing, existence checking and reading, use group push/pop */
static PetscErrorCode testObjectAttributes(PetscViewer viewer)
{
  Capsule        capsules[nap][ns], c=NULL, old=NULL;
  PetscInt       p,s;
  PetscBool      flg=PETSC_FALSE,flg1=PETSC_FALSE;
  MPI_Comm       comm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)viewer, &comm));
  if (verbose) {
    CHKERRQ(PetscPrintf(comm, "# TEST testObjectAttributes\n"));
    CHKERRQ(PetscPrintf(comm, "## WRITE PHASE\n"));
  }
  CHKERRQ(PetscMemzero(capsules, nap * ns * sizeof(Capsule)));

  /* test attribute writing */
  for (p=0; p<np; p++) {
    CHKERRQ(isPop(paths[p], &flg));
    CHKERRQ(isDot(paths[p], &flg1));
    if (flg) {
      CHKERRQ(PetscViewerHDF5PopGroup(viewer));
    } else {
      CHKERRQ(PetscViewerHDF5PushGroup(viewer, paths[p]));
    }
    /* < and . have been already visited => skip */
    if (flg || flg1) continue;

    /* assume here that groups and datasets are already in the file */
    for (s=0; s<ns; s++) {
      Vec    v;
      size_t len;
      const char *name = datasets[s];

      CHKERRQ(PetscStrlen(name, &len));
      if (!len) continue;
      CHKERRQ(VecCreate(comm, &v));
      CHKERRQ(PetscObjectSetName((PetscObject)v, name));
      CHKERRQ(PetscViewerHDF5HasObject(viewer, (PetscObject)v, &flg));
      if (flg) {
        if (verbose) CHKERRQ(PetscPrintf(comm, "Write attributes to %s/%s\n", apaths[paths2apaths[p]], name));
        CHKERRQ(CapsuleCreate(old, &c));
        CHKERRQ(CapsuleWriteAttributes(c, viewer, name));
        PetscCheckFalse(capsules[paths2apaths[p]][s],comm, PETSC_ERR_PLIB, "capsules[%" PetscInt_FMT "][%" PetscInt_FMT "] gets overwritten for %s/%s", paths2apaths[p], s, paths[p], name);
        capsules[paths2apaths[p]][s] = c;
        old = c;
      }
      CHKERRQ(VecDestroy(&v));
    }
  }
  CHKERRQ(PetscViewerFlush(viewer));

  if (verbose) CHKERRQ(PetscPrintf(comm, "\n## READ PHASE\n"));
  for (p=0; p<np; p++) {
    const char *group;

    CHKERRQ(isPop(paths[p], &flg));
    if (flg) {
      CHKERRQ(PetscViewerHDF5PopGroup(viewer));
    } else {
      CHKERRQ(PetscViewerHDF5PushGroup(viewer, paths[p]));
    }
    CHKERRQ(PetscViewerHDF5GetGroup(viewer, &group));
    if (!group) group = "";
    for (s=0; s<ns; s++) {
      Vec        v;
      size_t     len;
      const char *name = datasets[s];

      CHKERRQ(PetscStrlen(name, &len));
      if (!len) continue;
      CHKERRQ(VecCreate(comm, &v));
      CHKERRQ(PetscObjectSetName((PetscObject)v, name));
      CHKERRQ(PetscViewerHDF5HasObject(viewer, (PetscObject)v, &flg));
      if (verbose) CHKERRQ(PetscPrintf(comm, "Is %s/%s dataset? %s\n", group, name, PetscBools[flg]));

      /* check attribute capsule has been created for given path */
      c = capsules[paths2apaths[p]][s];
      flg1 = (PetscBool) !!c;
      PetscCheckFalse(flg != flg1,comm, PETSC_ERR_PLIB, "Capsule should exist for %s/%s? %s Exists? %s", group, name, PetscBools[flg], PetscBools[flg1]);

      /* check correct existence of attributes in file */
      if (flg) {
        CHKERRQ(CapsuleReadAndCompareAttributes(c, viewer, name));
      }
      CHKERRQ(VecDestroy(&v));
    }
  }
  CHKERRQ(PetscViewerFlush(viewer));
  for (p=0; p<nap; p++) for (s=0; s<ns; s++) {
    CHKERRQ(CapsuleDestroy(&capsules[p][s]));
  }
  if (verbose) CHKERRQ(PetscPrintf(comm, "# END  testObjectAttributes\n\n"));
  PetscFunctionReturn(0);
}

static PetscErrorCode testAttributesDefaultValue(PetscViewer viewer)
{
#define nv 4
  PetscBool      bools[nv];
  PetscInt       ints[nv];
  PetscReal      reals[nv];
  char          *strings[nv];
  PetscBool      flg;
  PetscInt       i;
  MPI_Comm       comm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)viewer, &comm));
  if (verbose) {
    CHKERRQ(PetscPrintf(comm, "# TEST testAttributesDefaultValue\n"));
  }

  CHKERRQ(PetscViewerHDF5ReadAttribute(viewer, "/", "attr_0_bool", PETSC_BOOL, NULL, &bools[0]));
  bools[1] = PetscNot(bools[0]);
  CHKERRQ(PetscViewerHDF5ReadAttribute(viewer, "/", "attr_0_bool", PETSC_BOOL, &bools[1], &bools[2]));
  CHKERRQ(PetscViewerHDF5ReadAttribute(viewer, "/", "attr_nonExisting_bool", PETSC_BOOL, &bools[1], &bools[3]));
  PetscCheckFalse(bools[2] != bools[0],comm, PETSC_ERR_PLIB, "%s = bools[2] != bools[0] = %s", PetscBools[bools[2]], PetscBools[bools[0]]);
  PetscCheckFalse(bools[3] != bools[1],comm, PETSC_ERR_PLIB, "%s = bools[3] != bools[1] = %s", PetscBools[bools[3]], PetscBools[bools[1]]);

  CHKERRQ(PetscViewerHDF5ReadAttribute(viewer, "/", "attr_0_int", PETSC_INT, NULL, &ints[0]));
  ints[1] = ints[0] * -333;
  CHKERRQ(PetscViewerHDF5ReadAttribute(viewer, "/", "attr_0_int", PETSC_INT, &ints[1], &ints[2]));
  CHKERRQ(PetscViewerHDF5ReadAttribute(viewer, "/", "attr_nonExisting_int", PETSC_INT, &ints[1], &ints[3]));
  PetscCheckFalse(ints[2] != ints[0],comm, PETSC_ERR_PLIB, "%" PetscInt_FMT " = ints[2] != ints[0] = %" PetscInt_FMT, ints[2], ints[0]);
  PetscCheckFalse(ints[3] != ints[1],comm, PETSC_ERR_PLIB, "%" PetscInt_FMT " = ints[3] != ints[1] = %" PetscInt_FMT, ints[3], ints[1]);
  if (verbose) {
    CHKERRQ(PetscIntView(nv, ints, PETSC_VIEWER_STDOUT_WORLD));
  }

  CHKERRQ(PetscViewerHDF5ReadAttribute(viewer, "/", "attr_0_real", PETSC_REAL, NULL, &reals[0]));
  reals[1] = reals[0] * -11.1;
  CHKERRQ(PetscViewerHDF5ReadAttribute(viewer, "/", "attr_0_real", PETSC_REAL, &reals[1], &reals[2]));
  CHKERRQ(PetscViewerHDF5ReadAttribute(viewer, "/", "attr_nonExisting_real", PETSC_REAL, &reals[1], &reals[3]));
  PetscCheckFalse(reals[2] != reals[0],comm, PETSC_ERR_PLIB, "%f = reals[2] != reals[0] = %f", reals[2], reals[0]);
  PetscCheckFalse(reals[3] != reals[1],comm, PETSC_ERR_PLIB, "%f = reals[3] != reals[1] = %f", reals[3], reals[1]);
  if (verbose) {
    CHKERRQ(PetscRealView(nv, reals, PETSC_VIEWER_STDOUT_WORLD));
  }

  CHKERRQ(PetscViewerHDF5ReadAttribute(viewer, "/", "attr_0_str", PETSC_STRING, NULL, &strings[0]));
  CHKERRQ(PetscStrallocpy(strings[0], &strings[1]));
  CHKERRQ(alterString(strings[0], strings[1]));
  CHKERRQ(PetscViewerHDF5ReadAttribute(viewer, "/", "attr_0_str", PETSC_STRING, &strings[1], &strings[2]));
  CHKERRQ(PetscViewerHDF5ReadAttribute(viewer, "/", "attr_nonExisting_str", PETSC_STRING, &strings[1], &strings[3]));
  CHKERRQ(PetscStrcmp(strings[2], strings[0], &flg));
  PetscCheck(flg,comm, PETSC_ERR_PLIB, "%s = strings[2] != strings[0] = %s", strings[2], strings[0]);
  CHKERRQ(PetscStrcmp(strings[3], strings[1], &flg));
  PetscCheck(flg,comm, PETSC_ERR_PLIB, "%s = strings[3] != strings[1] = %s", strings[3], strings[1]);
  for (i=0; i<nv; i++) {
    CHKERRQ(PetscFree(strings[i]));
  }

  CHKERRQ(PetscViewerFlush(viewer));
  if (verbose) {
    CHKERRQ(PetscPrintf(comm, "# END  testAttributesDefaultValue\n"));
  }
#undef nv
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  static char    filename[PETSC_MAX_PATH_LEN] = "ex48.h5";
  PetscMPIInt    rank;
  MPI_Comm       comm;
  PetscViewer    viewer;

  PetscFunctionBegin;
  CHKERRQ(PetscInitialize(&argc, &argv, (char*) 0, help));
  comm = PETSC_COMM_WORLD;
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL, "-n", &n, NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL, "-verbose", &verbose, NULL));
  CHKERRQ(PetscOptionsGetString(NULL,NULL, "-filename", filename, sizeof(filename), NULL));
  if (verbose) {
    CHKERRQ(PetscPrintf(comm, "np ns " PetscStringize(np) " " PetscStringize(ns) "\n"));
  }

  CHKERRQ(PetscViewerHDF5Open(comm, filename, FILE_MODE_WRITE, &viewer));
  CHKERRQ(testGroupsDatasets(viewer));
  CHKERRQ(testAttributesAbsolutePath(viewer, NULL));
  CHKERRQ(testAttributesAbsolutePath(viewer, "/prefix"));
  CHKERRQ(PetscViewerDestroy(&viewer));

  /* test reopening in update mode */
  CHKERRQ(PetscViewerHDF5Open(comm, filename, FILE_MODE_UPDATE, &viewer));
  CHKERRQ(testAttributesPushedPath(viewer));
  CHKERRQ(testObjectAttributes(viewer));
  CHKERRQ(testAttributesDefaultValue(viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

     build:
       requires: hdf5

     test:
       nsize: {{1 4}}

TEST*/
