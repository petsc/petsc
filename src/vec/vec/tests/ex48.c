
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
PETSC_STATIC_INLINE PetscErrorCode shouldExist(const char name[], PetscBool emptyExists, PetscBool *has)
{
  size_t         len=0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrlen(name, &len);CHKERRQ(ierr);
  *has = emptyExists;
  if (len) {
    char *loc=NULL;
    ierr = PetscStrstr(name,"nonExisting",&loc);CHKERRQ(ierr);
    *has = PetscNot(loc);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode isPop(const char path[], PetscBool *has)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrcmp(path, "<", has);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode isDot(const char path[], PetscBool *has)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrcmp(path, ".", has);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode isRoot(const char path[], PetscBool *flg)
{
  size_t         len;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrlen(path, &len);CHKERRQ(ierr);
  *flg = PetscNot(len);
  if (!*flg) {
    ierr = PetscStrcmp(path, "/", flg);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode compare(PetscDataType dt, void *ptr0, void *ptr1, PetscBool *flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (dt) {
    case PETSC_INT:
      *flg = (PetscBool)(*(PetscInt*)ptr0 == *(PetscInt*)ptr1);
      if (verbose) {
        if (*flg) {
          ierr = PetscPrintf(PETSC_COMM_SELF, "%D", *(PetscInt*)ptr0);CHKERRQ(ierr);
        } else {
          ierr = PetscPrintf(PETSC_COMM_SELF, "%D != %D\n", *(PetscInt*)ptr0, *(PetscInt*)ptr1);CHKERRQ(ierr);
        }
      }
      break;
    case PETSC_REAL:
      *flg = (PetscBool)(*(PetscReal*)ptr0 == *(PetscReal*)ptr1);
      if (verbose) {
        if (*flg) {
          ierr = PetscPrintf(PETSC_COMM_SELF, "%f", *(PetscReal*)ptr0);CHKERRQ(ierr);
        } else {
          ierr = PetscPrintf(PETSC_COMM_SELF, "%f != %f\n", *(PetscReal*)ptr0, *(PetscReal*)ptr1);CHKERRQ(ierr);
        }
      }
      break;
    case PETSC_BOOL:
      *flg = (PetscBool)(*(PetscBool*)ptr0 == *(PetscBool*)ptr1);
      if (verbose) {
        if (*flg) {
          ierr = PetscPrintf(PETSC_COMM_SELF, "%s", PetscBools[*(PetscBool*)ptr0]);CHKERRQ(ierr);
        } else {
          ierr = PetscPrintf(PETSC_COMM_SELF, "%s != %s\n", PetscBools[*(PetscBool*)ptr0], PetscBools[*(PetscBool*)ptr1]);CHKERRQ(ierr);
        }
      }
      break;
    case PETSC_STRING:
      ierr = PetscStrcmp((const char*)ptr0, (const char*)ptr1, flg);CHKERRQ(ierr);
      if (verbose) {
        if (*flg) {
          ierr = PetscPrintf(PETSC_COMM_SELF, "%s", (char*)ptr0);CHKERRQ(ierr);
        } else {
          ierr = PetscPrintf(PETSC_COMM_SELF, "%s != %s\n", (char*)ptr0, (char*)ptr1);CHKERRQ(ierr);
        }
      }
      break;
    default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "PetscDataType %s not handled here", PetscDataTypes[dt]);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode alterString(const char oldstr[], char str[])
{
  size_t          i,n;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscStrcpy(str, oldstr);CHKERRQ(ierr);
  ierr = PetscStrlen(oldstr, &n);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *flg = 0;
  ierr = PetscViewerHDF5HasGroup(viewer, path, &has);CHKERRQ(ierr);
  if (has) *flg = 1;
  else {
    ierr = PetscViewerHDF5HasDataset(viewer, path, &has);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&c);CHKERRQ(ierr);
  c->id = 0;
  c->ntypes = nt;
  if (old) {
    /* alter values */
    t=0;
    bool0 = PetscNot(*((PetscBool*)old->vals[t]));                      t++;
    int0  = *((PetscInt*) old->vals[t]) * -2;                           t++;
    real0 = *((PetscReal*)old->vals[t]) * -2.0;                         t++;
    ierr  = alterString((const char*)old->vals[t], str0);CHKERRQ(ierr); t++;
    c->id = old->id+1;
  }
  for (t=0; t<nt; t++) {
    c->sizes[t] = sizes[t];
    c->types[t] = types[t];
    ierr = PetscStrcpy(c->typeNames[t], tNames[t]);CHKERRQ(ierr);
    ierr = PetscSNPrintf(c->names[t], SLEN, "attr_%D_%s", c->id, tNames[t]);CHKERRQ(ierr);
    ierr = PetscMalloc(sizes[t], &c->vals[t]);CHKERRQ(ierr);
    ierr = PetscMemcpy(c->vals[t], vals[t], sizes[t]);CHKERRQ(ierr);
  }
  *newcapsule = c;
  PetscFunctionReturn(0);
}
#undef nt

static PetscErrorCode CapsuleWriteAttributes(Capsule c, PetscViewer v, const char parent[])
{
  PetscInt       t;
  PetscBool      flg=PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (t=0; t < c->ntypes; t++) {
    ierr = shouldExist(c->names[t], PETSC_FALSE, &flg);CHKERRQ(ierr);
    if (!flg) continue;
    ierr = PetscViewerHDF5WriteAttribute(v, parent, c->names[t], c->types[t], c->vals[t]);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)v, &comm);CHKERRQ(ierr);
  ierr = PetscViewerHDF5GetGroup(v, &group);CHKERRQ(ierr);
  if (!group) group = "";
  ierr = hasGroupOrDataset(v, parent, &gd);CHKERRQ(ierr);
  /* check correct existence of attributes */
  for (t=0; t < c->ntypes; t++) {
    const char *attribute = c->names[t];
    ierr = shouldExist(attribute, PETSC_FALSE, &flg);CHKERRQ(ierr);
    ierr = PetscViewerHDF5HasAttribute(v, parent, attribute, &hasAttr);CHKERRQ(ierr);
    if (verbose) {
      ierr = PetscPrintf(comm, "    %-24s = ", attribute);CHKERRQ(ierr);
      if (!hasAttr) {
        ierr = PetscPrintf(comm, "---");CHKERRQ(ierr);
      }
    }
    if (!gd && hasAttr)  SETERRQ5(comm, PETSC_ERR_PLIB, "Attribute %s/%s/%s exists while its parent %s/%s doesn't exist", group, parent, attribute, group, parent);
    if (flg != hasAttr) SETERRQ4(comm, PETSC_ERR_PLIB, "Attribute %s/%s should exist? %s Exists? %s", parent, attribute, PetscBools[flg], PetscBools[hasAttr]);

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
        ierr = PetscViewerHDF5ReadAttribute(v, parent, attribute, c->types[t], NULL, &str);CHKERRQ(ierr);
        ptr0 = str;
      } else {
        ierr = PetscViewerHDF5ReadAttribute(v, parent, attribute, c->types[t], NULL, &buffer);CHKERRQ(ierr);
        ptr0 = &buffer;
      }
      ierr = compare(c->types[t], ptr0, c->vals[t], &flg);CHKERRQ(ierr);
      if (!flg) SETERRQ3(comm, PETSC_ERR_PLIB, "Value of attribute %s/%s/%s in %s is not equal to the original value", group, parent, attribute);
      if (verbose) {ierr = PetscPrintf(comm, " (=)");CHKERRQ(ierr);}
      if (c->types[t] == PETSC_STRING) {
        ierr = PetscFree(str);CHKERRQ(ierr);
      }
    }
    if (verbose && gd) {ierr = PetscPrintf(comm, "\n");CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CapsuleDestroy(Capsule *c)
{
  PetscInt              t;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  if (!*c) PetscFunctionReturn(0);
  for (t=0; t < (*c)->ntypes; t++) {
    ierr = PetscFree((*c)->vals[t]);CHKERRQ(ierr);
  }
  ierr = PetscFree(*c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode testGroupsDatasets(MPI_Comm comm, const char filename[], PetscBool overwrite)
{
  char           buf[PETSC_MAX_PATH_LEN];
  Vec            vecs[nap][ns];
  PetscInt       p,s;
  PetscBool      flg=PETSC_FALSE,flg1=PETSC_FALSE,flg2=PETSC_FALSE;
  PetscViewer    viewer;
  PetscRandom    rand;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (verbose) {ierr = PetscPrintf(comm, "# TEST testGroupsDatasets\n");CHKERRQ(ierr);}
  /* store random vectors */
  ierr = PetscRandomCreate(comm, &rand);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rand, 0.0, 10.0);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
  ierr = PetscMemzero(vecs, nap * ns * sizeof(Vec));CHKERRQ(ierr);

  /* test dataset writing */
  ierr = PetscViewerHDF5Open(comm, filename, overwrite ? FILE_MODE_WRITE : FILE_MODE_UPDATE, &viewer);CHKERRQ(ierr);
  if (verbose) {ierr = PetscPrintf(comm, "## WRITE PHASE\n");CHKERRQ(ierr);}
  for (p=0; p<np; p++) {
    ierr = isPop(paths[p], &flg);CHKERRQ(ierr);
    ierr = isDot(paths[p], &flg1);CHKERRQ(ierr);
    ierr = shouldExist(apaths[paths2apaths[p]], PETSC_FALSE, &flg2);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerHDF5PushGroup(viewer, paths[p]);CHKERRQ(ierr);
    }
    if (verbose) {
      ierr = PetscPrintf(comm, "%-32s => %4s => %-32s  should exist? %s\n", paths[p], flg?"pop":"push", apaths[paths2apaths[p]], PetscBools[flg2]);CHKERRQ(ierr);
    }
    if (flg || flg1 || !flg2) continue;

    for (s=0; s<ns; s++) {
      Vec       v;

      ierr = shouldExist(datasets[s], PETSC_FALSE, &flg);CHKERRQ(ierr);
      if (!flg) continue;

      ierr = VecCreate(comm, &v);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)v, datasets[s]);CHKERRQ(ierr);
      ierr = VecSetSizes(v, n, PETSC_DECIDE);CHKERRQ(ierr);
      ierr = VecSetFromOptions(v);CHKERRQ(ierr);
      ierr = VecSetRandom(v,rand);CHKERRQ(ierr);
      if (verbose) {
        PetscReal min,max;
        ierr = VecMin(v, NULL, &min);CHKERRQ(ierr);
        ierr = VecMax(v, NULL, &max);CHKERRQ(ierr);
        ierr = PetscPrintf(comm, "  Create dataset %s/%s, keep in memory in vecs[%d][%d], min %.3e max %.3e\n", apaths[paths2apaths[p]], datasets[s], paths2apaths[p], s, min, max);CHKERRQ(ierr);
      }

      ierr = VecView(v, viewer);CHKERRQ(ierr);
      vecs[paths2apaths[p]][s] = v;
    }
  }
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);

  if (verbose) {ierr = PetscPrintf(comm, "\n## READ PHASE\n");CHKERRQ(ierr);}
  ierr = PetscViewerHDF5Open(comm, filename, FILE_MODE_READ, &viewer);CHKERRQ(ierr);
  /* check correct existence of groups in file */
  for (p=0; p<np; p++) {
    const char *group;
    const char *expected = apaths[paths2apaths[p]];

    /* check Push/Pop is correct */
    ierr = isPop(paths[p], &flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerHDF5PushGroup(viewer, paths[p]);CHKERRQ(ierr);
    }
    ierr = PetscViewerHDF5GetGroup(viewer, &group);CHKERRQ(ierr);
    ierr = PetscViewerHDF5HasGroup(viewer, NULL, &flg1);CHKERRQ(ierr);
    if (!group) group = "/";  /* "/" is stored as NULL */
    if (verbose) {
      ierr = PetscPrintf(comm, "%-32s => %4s => %-32s  exists? %s\n", paths[p], flg?"pop":"push", group, PetscBools[flg1]);CHKERRQ(ierr);
    }
    ierr = PetscStrcmp(group, expected, &flg2);CHKERRQ(ierr);
    if (!flg2) SETERRQ2(comm, PETSC_ERR_PLIB, "Current group %s not equal to expected %s", group, expected);
    ierr = shouldExist(group, PETSC_TRUE, &flg2);CHKERRQ(ierr);
    if (flg1 != flg2) SETERRQ4(comm, PETSC_ERR_PLIB, "Group %s should exist? %s Exists in %s? %s", group, PetscBools[flg2], filename, PetscBools[flg1]);
  }

  /* check existence of datasets; compare loaded vectors with original ones */
  for (p=0; p<np; p++) {
    const char *group;

    /* check Push/Pop is correct */
    ierr = isPop(paths[p], &flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerHDF5PushGroup(viewer, paths[p]);CHKERRQ(ierr);
    }
    ierr = PetscViewerHDF5GetGroup(viewer, &group);CHKERRQ(ierr);
    ierr = PetscViewerHDF5HasGroup(viewer, NULL, &flg);CHKERRQ(ierr);
    if (verbose) {ierr = PetscPrintf(comm, "Has %s group? %s\n", group ? group : "/", PetscBools[flg]);CHKERRQ(ierr);}
    if (!group) group = "";  /* "/" is stored as NULL */
    for (s=0; s<ns; s++) {
      const char *name = datasets[s];
      char       *fullname = buf;

      /* check correct existence of datasets in file */
      ierr = PetscSNPrintf(fullname, sizeof(buf), "%s/%s", group, name);CHKERRQ(ierr);
      ierr = shouldExist(name,PETSC_FALSE,&flg1);CHKERRQ(ierr);
      flg1 = (PetscBool)(flg && flg1); /* both group and dataset need to exist */
      ierr = PetscViewerHDF5HasDataset(viewer, name, &flg2);CHKERRQ(ierr);
      if (verbose) {ierr = PetscPrintf(comm, "    %s dataset? %s", fullname, PetscBools[flg2]);CHKERRQ(ierr);}
      if (flg2 != flg1) SETERRQ4(comm, PETSC_ERR_PLIB, "Dataset %s should exist? %s Exists in %s? %s", fullname, PetscBools[flg1], filename, PetscBools[flg2]);

      if (flg2) {
        Vec v;
        /* check loaded Vec is the same as original */
        ierr = VecCreate(comm, &v);CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject)v, name);CHKERRQ(ierr);
        ierr = VecLoad(v, viewer);CHKERRQ(ierr);
        ierr = VecEqual(v, vecs[paths2apaths[p]][s], &flg1);CHKERRQ(ierr);
        if (!flg1) SETERRQ2(comm, PETSC_ERR_PLIB, "Dataset %s in %s is not equal to the original Vec", fullname, filename);
        if (verbose) {ierr = PetscPrintf(comm, " (=)");CHKERRQ(ierr);}
        ierr = VecDestroy(&v);CHKERRQ(ierr);
      }
      if (verbose) {ierr = PetscPrintf(comm, "\n");CHKERRQ(ierr);}
    }
  }
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  for (p=0; p<nap; p++) for (s=0; s<ns; s++) {
    ierr = VecDestroy(&vecs[p][s]);CHKERRQ(ierr);
  }
  if (verbose) {ierr = PetscPrintf(comm, "# END  testGroupsDatasets\n\n");CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode formPath(PetscBool relativize, const char path[], const char dataset[], char buf[], size_t bufsize)
{
  PetscBool      isroot=PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = isRoot(path, &isroot);CHKERRQ(ierr);
  if (relativize) {
    if (isroot) {
      ierr = PetscStrncpy(buf, dataset, bufsize);CHKERRQ(ierr);
    } else {
      /* skip initial '/' in paths[p] if prefix given */
      ierr = PetscSNPrintf(buf, bufsize, "%s/%s", path+1, dataset);CHKERRQ(ierr);
    }
  } else {
    ierr = PetscSNPrintf(buf, bufsize, "%s/%s", isroot ? "" : path, dataset);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* test attribute writing, existence checking and reading, use absolute paths */
static PetscErrorCode testAttributesAbsolutePath(MPI_Comm comm, const char filename[], const char prefix[], PetscBool overwrite)
{
  char           buf[PETSC_MAX_PATH_LEN];
  Capsule        capsules[nap][ns], c=NULL, old=NULL;
  PetscInt       p,s;
  PetscBool      flg=PETSC_FALSE,flg1=PETSC_FALSE;
  PetscViewer    viewer;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (verbose) {
    if (prefix) {
      ierr = PetscPrintf(comm, "# TEST testAttributesAbsolutePath, prefix=\"%s\"\n", prefix);CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(comm, "# TEST testAttributesAbsolutePath\n");CHKERRQ(ierr);
    }
    ierr = PetscPrintf(comm, "## WRITE PHASE\n");CHKERRQ(ierr);
  }
  ierr = PetscMemzero(capsules, nap * ns * sizeof(Capsule));CHKERRQ(ierr);

  /* test attribute writing */
  ierr = PetscViewerHDF5Open(comm, filename, overwrite ? FILE_MODE_WRITE : FILE_MODE_UPDATE, &viewer);CHKERRQ(ierr);
  if (prefix) {
    ierr = PetscViewerHDF5PushGroup(viewer, prefix);CHKERRQ(ierr);
  }
  for (p=0; p<np; p++) for (s=0; s<ns; s++) {
    /* we test only absolute paths here */
    ierr = PetscViewerHDF5PathIsRelative(paths[p], PETSC_FALSE, &flg);CHKERRQ(ierr);
    if (flg) continue;
    {
      const char *group;
      ierr = PetscViewerHDF5GetGroup(viewer, &group);CHKERRQ(ierr);
      ierr = PetscStrcmp(group, prefix, &flg);CHKERRQ(ierr);
      if (!flg) SETERRQ2(comm, PETSC_ERR_PLIB, "prefix %s not equal to pushed group %s", prefix, group);
    }
    ierr = formPath((PetscBool)!!prefix, paths[p], datasets[s], buf, sizeof(buf));CHKERRQ(ierr);
    ierr = shouldExist(buf, PETSC_TRUE, &flg);CHKERRQ(ierr);
    if (!flg) continue;

    if (verbose) {
      if (prefix) {
        ierr = PetscPrintf(comm, "Write attributes to %s/%s\n", prefix, buf);CHKERRQ(ierr);
      } else {
        ierr = PetscPrintf(comm, "Write attributes to %s\n", buf);CHKERRQ(ierr);
      }
    }

    ierr = CapsuleCreate(old, &c);CHKERRQ(ierr);
    ierr = CapsuleWriteAttributes(c, viewer, buf);CHKERRQ(ierr);
    if (capsules[paths2apaths[p]][s]) SETERRQ3(comm, PETSC_ERR_PLIB, "capsules[%D][%D] gets overwritten for %s", paths2apaths[p], s, buf);
    capsules[paths2apaths[p]][s] = c;
    old = c;
  }
  if (prefix) {
    ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  }
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  if (verbose) {ierr = PetscPrintf(comm, "\n## READ PHASE\n");CHKERRQ(ierr);}
  ierr = PetscViewerHDF5Open(comm, filename, FILE_MODE_READ, &viewer);CHKERRQ(ierr);
  if (prefix) {
    ierr = PetscViewerHDF5PushGroup(viewer, prefix);CHKERRQ(ierr);
  }
  for (p=0; p<np; p++) for (s=0; s<ns; s++) {
    /* we test only absolute paths here */
    ierr = PetscViewerHDF5PathIsRelative(paths[p], PETSC_FALSE, &flg);CHKERRQ(ierr);
    if (flg) continue;

    /* check existence of given group/dataset */
    ierr = formPath((PetscBool)!!prefix, paths[p], datasets[s], buf, sizeof(buf));CHKERRQ(ierr);
    ierr = shouldExist(buf, PETSC_TRUE, &flg);CHKERRQ(ierr);
    if (verbose) {
      if (prefix) {
        ierr = PetscPrintf(comm, "Has %s/%s? %s\n", prefix, buf, PetscBools[flg]);CHKERRQ(ierr);
      } else {
        ierr = PetscPrintf(comm, "Has %s? %s\n", buf, PetscBools[flg]);CHKERRQ(ierr);
      }
    }

    /* check attribute capsule has been created for given path */
    c = capsules[paths2apaths[p]][s];
    flg1 = (PetscBool) !!c;
    if (flg != flg1) SETERRQ3(comm, PETSC_ERR_PLIB, "Capsule should exist for %s? %s Exists? %s", buf, PetscBools[flg], PetscBools[flg1]);
    if (!flg) continue;

    /* check correct existence and fidelity of attributes in file */
    ierr = CapsuleReadAndCompareAttributes(c, viewer, buf);CHKERRQ(ierr);
  }
  if (prefix) {
    ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  }
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  for (p=0; p<nap; p++) for (s=0; s<ns; s++) {
    ierr = CapsuleDestroy(&capsules[p][s]);CHKERRQ(ierr);
  }
  if (verbose) {ierr = PetscPrintf(comm, "# END  testAttributesAbsolutePath\n\n");CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* test attribute writing, existence checking and reading, use group push/pop */
static PetscErrorCode testAttributesPushedPath(MPI_Comm comm, const char filename[], PetscBool overwrite)
{
  Capsule        capsules[nap][ns], c=NULL, old=NULL;
  PetscInt       p,s;
  int            gd;
  PetscBool      flg=PETSC_FALSE,flg1=PETSC_FALSE;
  PetscViewer    viewer;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (verbose) {
    ierr = PetscPrintf(comm, "# TEST testAttributesPushedPath\n");CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "## WRITE PHASE\n");CHKERRQ(ierr);
  }
  ierr = PetscMemzero(capsules, nap * ns * sizeof(Capsule));CHKERRQ(ierr);

  /* test attribute writing */
  ierr = PetscViewerHDF5Open(comm, filename, overwrite ? FILE_MODE_WRITE : FILE_MODE_UPDATE, &viewer);CHKERRQ(ierr);
  for (p=0; p<np; p++) {
    ierr = isPop(paths[p], &flg);CHKERRQ(ierr);
    ierr = isDot(paths[p], &flg1);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerHDF5PushGroup(viewer, paths[p]);CHKERRQ(ierr);
    }
    /* < and . have been already visited => skip */
    if (flg || flg1) continue;

    /* assume here that groups and datasets are already in the file */
    for (s=0; s<ns; s++) {
      ierr = hasGroupOrDataset(viewer, datasets[s], &gd);CHKERRQ(ierr);
      if (!gd) continue;
      if (verbose) {ierr = PetscPrintf(comm, "Write attributes to %s/%s\n", apaths[paths2apaths[p]], datasets[s]);CHKERRQ(ierr);}
      ierr = CapsuleCreate(old, &c);CHKERRQ(ierr);
      ierr = CapsuleWriteAttributes(c, viewer, datasets[s]);CHKERRQ(ierr);
      if (capsules[paths2apaths[p]][s]) SETERRQ4(comm, PETSC_ERR_PLIB, "capsules[%D][%D] gets overwritten for %s/%s", paths2apaths[p], s, paths[p], datasets[s]);
      capsules[paths2apaths[p]][s] = c;
      old = c;
    }
  }
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  if (verbose) {ierr = PetscPrintf(comm, "\n## READ PHASE\n");CHKERRQ(ierr);}
  ierr = PetscViewerHDF5Open(comm, filename, FILE_MODE_READ, &viewer);CHKERRQ(ierr);
  for (p=0; p<np; p++) {
    const char *group;

    ierr = isPop(paths[p], &flg1);CHKERRQ(ierr);
    if (flg1) {
      ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerHDF5PushGroup(viewer, paths[p]);CHKERRQ(ierr);
    }
    ierr = PetscViewerHDF5GetGroup(viewer, &group);CHKERRQ(ierr);
    if (!group) group = "";
    for (s=0; s<ns; s++) {
      ierr = hasGroupOrDataset(viewer, datasets[s], &gd);CHKERRQ(ierr);
      if (verbose) {ierr = PetscPrintf(comm, "%s/%s   %s\n", group, datasets[s], gd ? (gd==1 ? "is group" : "is dataset") : "does not exist");CHKERRQ(ierr);}

      /* check attribute capsule has been created for given path */
      c = capsules[paths2apaths[p]][s];
      flg  = (PetscBool) !!gd;
      flg1 = (PetscBool) !!c;
      if (flg != flg1) SETERRQ4(comm, PETSC_ERR_PLIB, "Capsule should exist for %s/%s? %s Exists? %s", group, datasets[s], PetscBools[flg], PetscBools[flg1]);
      if (!flg) continue;

      /* check correct existence of attributes in file */
      ierr = CapsuleReadAndCompareAttributes(c, viewer, datasets[s]);CHKERRQ(ierr);
    }
  }
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  for (p=0; p<nap; p++) for (s=0; s<ns; s++) {
    ierr = CapsuleDestroy(&capsules[p][s]);CHKERRQ(ierr);
  }
  if (verbose) {ierr = PetscPrintf(comm, "# END  testAttributesPushedPath\n\n");CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* test attribute writing, existence checking and reading, use group push/pop */
static PetscErrorCode testObjectAttributes(MPI_Comm comm, const char filename[], PetscBool overwrite)
{
  Capsule        capsules[nap][ns], c=NULL, old=NULL;
  PetscInt       p,s;
  PetscBool      flg=PETSC_FALSE,flg1=PETSC_FALSE;
  PetscViewer    viewer;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (verbose) {
    ierr = PetscPrintf(comm, "# TEST testObjectAttributes\n");CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "## WRITE PHASE\n");CHKERRQ(ierr);
  }
  ierr = PetscMemzero(capsules, nap * ns * sizeof(Capsule));CHKERRQ(ierr);

  /* test attribute writing */
  ierr = PetscViewerHDF5Open(comm, filename, overwrite ? FILE_MODE_WRITE : FILE_MODE_UPDATE, &viewer);CHKERRQ(ierr);
  for (p=0; p<np; p++) {
    ierr = isPop(paths[p], &flg);CHKERRQ(ierr);
    ierr = isDot(paths[p], &flg1);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerHDF5PushGroup(viewer, paths[p]);CHKERRQ(ierr);
    }
    /* < and . have been already visited => skip */
    if (flg || flg1) continue;

    /* assume here that groups and datasets are already in the file */
    for (s=0; s<ns; s++) {
      Vec    v;
      size_t len;
      const char *name = datasets[s];

      ierr = PetscStrlen(name, &len);CHKERRQ(ierr);
      if (!len) continue;
      ierr = VecCreate(comm, &v);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)v, name);CHKERRQ(ierr);
      ierr = PetscViewerHDF5HasObject(viewer, (PetscObject)v, &flg);CHKERRQ(ierr);
      if (flg) {
        if (verbose) {ierr = PetscPrintf(comm, "Write attributes to %s/%s\n", apaths[paths2apaths[p]], name);CHKERRQ(ierr);}
        ierr = CapsuleCreate(old, &c);CHKERRQ(ierr);
        ierr = CapsuleWriteAttributes(c, viewer, name);CHKERRQ(ierr);
        if (capsules[paths2apaths[p]][s]) SETERRQ4(comm, PETSC_ERR_PLIB, "capsules[%D][%D] gets overwritten for %s/%s", paths2apaths[p], s, paths[p], name);
        capsules[paths2apaths[p]][s] = c;
        old = c;
      }
      ierr = VecDestroy(&v);CHKERRQ(ierr);
    }
  }
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  if (verbose) {ierr = PetscPrintf(comm, "\n## READ PHASE\n");CHKERRQ(ierr);}
  ierr = PetscViewerHDF5Open(comm, filename, FILE_MODE_READ, &viewer);CHKERRQ(ierr);
  for (p=0; p<np; p++) {
    const char *group;

    ierr = isPop(paths[p], &flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerHDF5PushGroup(viewer, paths[p]);CHKERRQ(ierr);
    }
    ierr = PetscViewerHDF5GetGroup(viewer, &group);CHKERRQ(ierr);
    if (!group) group = "";
    for (s=0; s<ns; s++) {
      Vec        v;
      size_t     len;
      const char *name = datasets[s];

      ierr = PetscStrlen(name, &len);CHKERRQ(ierr);
      if (!len) continue;
      ierr = VecCreate(comm, &v);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)v, name);CHKERRQ(ierr);
      ierr = PetscViewerHDF5HasObject(viewer, (PetscObject)v, &flg);CHKERRQ(ierr);
      if (verbose) {ierr = PetscPrintf(comm, "Is %s/%s dataset? %s\n", group, name, PetscBools[flg]);CHKERRQ(ierr);}

      /* check attribute capsule has been created for given path */
      c = capsules[paths2apaths[p]][s];
      flg1 = (PetscBool) !!c;
      if (flg != flg1) SETERRQ4(comm, PETSC_ERR_PLIB, "Capsule should exist for %s/%s? %s Exists? %s", group, name, PetscBools[flg], PetscBools[flg1]);

      /* check correct existence of attributes in file */
      if (flg) {
        ierr = CapsuleReadAndCompareAttributes(c, viewer, name);CHKERRQ(ierr);
      }
      ierr = VecDestroy(&v);CHKERRQ(ierr);
    }
  }
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  for (p=0; p<nap; p++) for (s=0; s<ns; s++) {
    ierr = CapsuleDestroy(&capsules[p][s]);CHKERRQ(ierr);
  }
  if (verbose) {ierr = PetscPrintf(comm, "# END  testObjectAttributes\n\n");CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode testAttributesDefaultValue(MPI_Comm comm, const char filename[])
{
#define nv 4
  PetscViewer    viewer;
  PetscBool      bools[nv];
  PetscInt       ints[nv];
  PetscReal      reals[nv];
  char          *strings[nv];
  PetscBool      flg;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (verbose) {
    ierr = PetscPrintf(comm, "# TEST testAttributesDefaultValue\n");CHKERRQ(ierr);
  }
  ierr = PetscViewerHDF5Open(comm, filename, FILE_MODE_READ, &viewer);CHKERRQ(ierr);

  ierr = PetscViewerHDF5ReadAttribute(viewer, "/", "attr_0_bool", PETSC_BOOL, NULL, &bools[0]);CHKERRQ(ierr);
  bools[1] = PetscNot(bools[0]);
  ierr = PetscViewerHDF5ReadAttribute(viewer, "/", "attr_0_bool", PETSC_BOOL, &bools[1], &bools[2]);CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, "/", "attr_nonExisting_bool", PETSC_BOOL, &bools[1], &bools[3]);CHKERRQ(ierr);
  if (bools[2] != bools[0]) SETERRQ2(comm, PETSC_ERR_PLIB, "%s = bools[2] != bools[0] = %s", PetscBools[bools[2]], PetscBools[bools[0]]);
  if (bools[3] != bools[1]) SETERRQ2(comm, PETSC_ERR_PLIB, "%s = bools[3] != bools[1] = %s", PetscBools[bools[3]], PetscBools[bools[1]]);

  ierr = PetscViewerHDF5ReadAttribute(viewer, "/", "attr_0_int", PETSC_INT, NULL, &ints[0]);CHKERRQ(ierr);
  ints[1] = ints[0] * -333;
  ierr = PetscViewerHDF5ReadAttribute(viewer, "/", "attr_0_int", PETSC_INT, &ints[1], &ints[2]);CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, "/", "attr_nonExisting_int", PETSC_INT, &ints[1], &ints[3]);CHKERRQ(ierr);
  if (ints[2] != ints[0]) SETERRQ2(comm, PETSC_ERR_PLIB, "%D = ints[2] != ints[0] = %D", ints[2], ints[0]);
  if (ints[3] != ints[1]) SETERRQ2(comm, PETSC_ERR_PLIB, "%D = ints[3] != ints[1] = %D", ints[3], ints[1]);
  if (verbose) {
    ierr = PetscIntView(nv, ints, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  ierr = PetscViewerHDF5ReadAttribute(viewer, "/", "attr_0_real", PETSC_REAL, NULL, &reals[0]);CHKERRQ(ierr);
  reals[1] = reals[0] * -11.1;
  ierr = PetscViewerHDF5ReadAttribute(viewer, "/", "attr_0_real", PETSC_REAL, &reals[1], &reals[2]);CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, "/", "attr_nonExisting_real", PETSC_REAL, &reals[1], &reals[3]);CHKERRQ(ierr);
  if (reals[2] != reals[0]) SETERRQ2(comm, PETSC_ERR_PLIB, "%f = reals[2] != reals[0] = %f", reals[2], reals[0]);
  if (reals[3] != reals[1]) SETERRQ2(comm, PETSC_ERR_PLIB, "%f = reals[3] != reals[1] = %f", reals[3], reals[1]);
  if (verbose) {
    ierr = PetscRealView(nv, reals, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  ierr = PetscViewerHDF5ReadAttribute(viewer, "/", "attr_0_str", PETSC_STRING, NULL, &strings[0]);CHKERRQ(ierr);
  ierr = PetscStrallocpy(strings[0], &strings[1]);CHKERRQ(ierr);
  ierr = alterString(strings[0], strings[1]);CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, "/", "attr_0_str", PETSC_STRING, &strings[1], &strings[2]);CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, "/", "attr_nonExisting_str", PETSC_STRING, &strings[1], &strings[3]);CHKERRQ(ierr);
  ierr = PetscStrcmp(strings[2], strings[0], &flg);CHKERRQ(ierr);
  if (!flg) SETERRQ2(comm, PETSC_ERR_PLIB, "%s = strings[2] != strings[0] = %s", strings[2], strings[0]);
  ierr = PetscStrcmp(strings[3], strings[1], &flg);CHKERRQ(ierr);
  if (!flg) SETERRQ2(comm, PETSC_ERR_PLIB, "%s = strings[3] != strings[1] = %s", strings[3], strings[1]);
  for (i=0; i<nv; i++) {
    ierr = PetscFree(strings[i]);CHKERRQ(ierr);
  }

  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  if (verbose) {
    ierr = PetscPrintf(comm, "# END  testAttributesDefaultValue\n");CHKERRQ(ierr);
  }
#undef nv
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  static char    filename[PETSC_MAX_PATH_LEN] = "ex48.h5";
  PetscMPIInt    rank;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char*) 0, help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL, "-n", &n, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL, "-verbose", &verbose, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL, "-filename", filename, sizeof(filename), NULL);CHKERRQ(ierr);
  if (verbose) {ierr = PetscPrintf(comm, "np ns %D %D\n", np, ns);CHKERRQ(ierr);}

  ierr = testGroupsDatasets(comm, filename, PETSC_TRUE);CHKERRQ(ierr);
  ierr = testAttributesAbsolutePath(comm, filename, NULL, PETSC_FALSE);CHKERRQ(ierr);
  ierr = testAttributesAbsolutePath(comm, filename, "/prefix", PETSC_FALSE);CHKERRQ(ierr);
  ierr = testAttributesPushedPath(comm, filename, PETSC_FALSE);CHKERRQ(ierr);
  ierr = testObjectAttributes(comm, filename, PETSC_FALSE);CHKERRQ(ierr);
  ierr = testAttributesDefaultValue(comm, filename);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

     build:
       requires: hdf5

     test:
       nsize: {{1 4}}

TEST*/
