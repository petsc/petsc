(ch_versionchecking)=

# Checking the PETSc version

The PETSc version
is defined in `$PETSC_DIR/include/petscversion.h` with the three macros
`PETSC_VERSION_MAJOR`, `PETSC_VERSION_MINOR`, and `PETSC_VERSION_SUBMINOR`.

The shell commands `make getversion` or `$PETSC_DIR/lib/petsc/bin/petscversion` prints out the PETSc version.
The command

```console
$ $PETSC_DIR/lib/petsc/bin/petscversion <eq,gt,lt,ge,le> major.minor<.subminor>
```

allows one to add tests to make files, CMake files, configure scripts etc, to ensure the PETSc version is compatible with your applications. For example,

> ```console
> $ $PETSC_DIR/lib/petsc/bin/petscversion eq 3.22
> ```

returns 1 if the PETSc version is 3.22 (any subminor version is allowed). While

```console
$ $PETSC_DIR/lib/petsc/bin/petscversion ge 3.21
```

returns 1 if the PETSc version is 3.21 or higher.

Though we try to avoid making changes to the PETSc API, they are inevitable; thus we
provide tools to help manage one's application to be robust to such changes.

## During configure/make time

The command

```console
$ $PETSC_DIR/lib/petsc/bin/petscversion eq xxx.yyy[.zzz]
```

prints out 1 if the PETSc version matches `xxx.yyy[.zzz]` and 0 otherwise. The command works in a similar
way for `lt`, `le`, `gt`, and `ge`. This allows your application configure script, or `makefile` or `CMake` file
to check if the PETSc version is compatible with application even before beginning to compile your code.

## During compile time

The CPP macros

- `PETSC_VERSION_EQ(MAJOR,MINOR,SUBMINOR)`
- `PETSC_VERSION_LT(MAJOR,MINOR,SUBMINOR)`
- `PETSC_VERSION_LE(MAJOR,MINOR,SUBMINOR)`
- `PETSC_VERSION_GT(MAJOR,MINOR,SUBMINOR)`
- `PETSC_VERSION_GE(MAJOR,MINOR,SUBMINOR)`

may be used in the source code to choose different code paths or error out depending on the PETSc version.

## At Runtime

The command

```C
char  version(lengthofversion);
PetscErrorCode PetscGetVersion(char version[], size_t lengthofversion)
```

gives access to the version at runtime.
