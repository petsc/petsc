#include <petsc/private/fortranimpl.h>
#include <petscdm.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmview_                      DMVIEW
#define dmsetoptionsprefix_          DMSETOPTIONSPREFIX
#define dmsettype_                   DMSETTYPE
#define dmgettype_                   DMGETTYPE
#define dmsetmattype_                DMSETMATTYPE
#define dmsetvectype_                DMSETVECTYPE
#define dmgetmattype_                DMGETMATTYPE
#define dmgetvectype_                DMGETVECTYPE
#define dmlabelview_                 DMLABELVIEW
#define dmcreatelabel_               DMCREATELABEL
#define dmhaslabel_                  DMHASLABEL
#define dmgetlabelvalue_             DMGETLABELVALUE
#define dmsetlabelvalue_             DMSETLABELVALUE
#define dmgetlabelsize_              DMGETLABELSIZE
#define dmgetlabelidis_              DMGETLABELIDIS
#define dmgetlabelname_              DMGETLABELNAME
#define dmgetlabel_                  DMGETLABEL
#define dmgetstratumsize_            DMGETSTRATUMSIZE
#define dmgetstratumis_              DMGETSTRATUMIS
#define dmsetstratumis_              DMSETSTRATUMIS
#define dmremovelabel_               DMREMOVELABEL
#define dmviewfromoptions_           DMVIEWFROMOPTIONS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmview_                      dmview
#define dmsetoptionsprefix_          dmsetoptionsprefix
#define dmsettype_                   dmsettype
#define dmgettype_                   dmgettype
#define dmsetmattype_                dmsetmattype
#define dmsetvectype_                dmsetvectype
#define dmgetmattype_                dmgetmattype
#define dmgetvectype_                dmgetvectype
#define dmlabelview_                 dmlabelview
#define dmcreatelabel_               dmcreatelabel
#define dmhaslabel_                  dmhaslabel
#define dmgetlabelvalue_             dmgetlabelvalue
#define dmsetlabelvalue_             dmsetlabelvalue
#define dmgetlabelsize_              dmlabelsize
#define dmgetlabelidis_              dmlabelidis
#define dmgetlabelname_              dmgetlabelname
#define dmgetlabel_                  dmgetlabel
#define dmgetstratumsize_            dmgetstratumsize
#define dmgetstratumis_              dmgetstratumis
#define dmsetstratumis_              dmsetstratumis
#define dmremovelabel_               dmremovelabel
#define dmviewfromoptions_           dmviewfromoptions
#endif

PETSC_EXTERN void PETSC_STDCALL dmgetmattype_(DM *mm,char* name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = DMGetMatType(*mm,&tname);if (*ierr) return;
  if (name != PETSC_NULL_CHARACTER_Fortran) {
    *ierr = PetscStrncpy(name,tname,len);if (*ierr) return;
  }
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}

PETSC_EXTERN void PETSC_STDCALL dmgetvectype_(DM *mm,char* name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = DMGetVecType(*mm,&tname);if (*ierr) return;
  if (name != PETSC_NULL_CHARACTER_Fortran) {
    *ierr = PetscStrncpy(name,tname,len);if (*ierr) return;
  }
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}

PETSC_EXTERN void PETSC_STDCALL dmview_(DM *da,PetscViewer *vin,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = DMView(*da,v);
}

PETSC_EXTERN void PETSC_STDCALL dmsetoptionsprefix_(DM *dm,char* prefix PETSC_MIXED_LEN(len), PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = DMSetOptionsPrefix(*dm,t);if (*ierr) return;
  FREECHAR(prefix,t);
}

PETSC_EXTERN void PETSC_STDCALL dmsettype_(DM *x,char* type_name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type_name,len,t);
  *ierr = DMSetType(*x,t);if (*ierr) return;
  FREECHAR(type_name,t);
}

PETSC_EXTERN void PETSC_STDCALL dmgettype_(DM *mm,char* name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = DMGetType(*mm,&tname);if (*ierr) return;
  if (name != PETSC_NULL_CHARACTER_Fortran) {
    *ierr = PetscStrncpy(name,tname,len);if (*ierr) return;
  }
  FIXRETURNCHAR(PETSC_TRUE,name,len);

}

PETSC_EXTERN void PETSC_STDCALL dmsetmattype_(DM *dm,char* prefix PETSC_MIXED_LEN(len), PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = DMSetMatType(*dm,t);if (*ierr) return;
  FREECHAR(prefix,t);
}


PETSC_EXTERN void PETSC_STDCALL dmsetvectype_(DM *dm,char* prefix PETSC_MIXED_LEN(len), PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = DMSetVecType(*dm,t);if (*ierr) return;
  FREECHAR(prefix,t);
}

PETSC_EXTERN void PETSC_STDCALL dmcreatelabel_(DM *dm, char* name PETSC_MIXED_LEN(lenN), int *ierr PETSC_END_LEN(lenN))
{
  char *lname;

  FIXCHAR(name, lenN, lname);
  *ierr = DMCreateLabel(*dm, lname);if (*ierr) return;
  FREECHAR(name, lname);
}

PETSC_EXTERN void PETSC_STDCALL dmhaslabel_(DM *dm, char* name PETSC_MIXED_LEN(lenN), PetscBool *hasLabel, int *ierr PETSC_END_LEN(lenN))
{
  char *lname;

  FIXCHAR(name, lenN, lname);
  *ierr = DMHasLabel(*dm, lname, hasLabel);if (*ierr) return;
  FREECHAR(name, lname);
}

PETSC_EXTERN void PETSC_STDCALL dmgetlabelvalue_(DM *dm, char* name PETSC_MIXED_LEN(lenN), PetscInt *point, PetscInt *value, int *ierr PETSC_END_LEN(lenN))
{
  char *lname;

  FIXCHAR(name, lenN, lname);
  *ierr = DMGetLabelValue(*dm, lname, *point, value);if (*ierr) return;
  FREECHAR(name, lname);
}

PETSC_EXTERN void PETSC_STDCALL dmsetlabelvalue_(DM *dm, char* name PETSC_MIXED_LEN(lenN), PetscInt *point, PetscInt *value, int *ierr PETSC_END_LEN(lenN))
{
  char *lname;

  FIXCHAR(name, lenN, lname);
  *ierr = DMSetLabelValue(*dm, lname, *point, *value);if (*ierr) return;
  FREECHAR(name, lname);
}

PETSC_EXTERN void PETSC_STDCALL dmgetlabelsize_(DM *dm, char* name PETSC_MIXED_LEN(lenN), PetscInt *size, int *ierr PETSC_END_LEN(lenN))
{
  char *lname;

  FIXCHAR(name, lenN, lname);
  *ierr = DMGetLabelSize(*dm, lname, size);if (*ierr) return;
  FREECHAR(name, lname);
}

PETSC_EXTERN void PETSC_STDCALL dmgetlabelidis_(DM *dm, char* name PETSC_MIXED_LEN(lenN), IS *ids, int *ierr PETSC_END_LEN(lenN))
{
  char *lname;

  FIXCHAR(name, lenN, lname);
  *ierr = DMGetLabelIdIS(*dm, lname, ids);if (*ierr) return;
  FREECHAR(name, lname);
}

PETSC_EXTERN void PETSC_STDCALL dmgetlabelname_(DM *dm,PetscInt *n,char* name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tmp;
  *ierr = DMGetLabelName(*dm,*n,&tmp);
  *ierr = PetscStrncpy(name,tmp,len);if (*ierr) return;
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}

PETSC_EXTERN void PETSC_STDCALL dmgetlabel_(DM *dm, char* name PETSC_MIXED_LEN(lenN), DMLabel *label, int *ierr PETSC_END_LEN(lenN))
{
  char *lname;

  FIXCHAR(name, lenN, lname);
  *ierr = DMGetLabel(*dm, lname, label);if (*ierr) return;
  FREECHAR(name, lname);
}

PETSC_EXTERN void PETSC_STDCALL dmgetstratumsize_(DM *dm, char* name PETSC_MIXED_LEN(lenN), PetscInt *value, PetscInt *size, int *ierr PETSC_END_LEN(lenN))
{
  char *lname;

  FIXCHAR(name, lenN, lname);
  *ierr = DMGetStratumSize(*dm, lname, *value, size);if (*ierr) return;
  FREECHAR(name, lname);
}

PETSC_EXTERN void PETSC_STDCALL dmgetstratumis_(DM *dm, char* name PETSC_MIXED_LEN(lenN), PetscInt *value, IS *is, int *ierr PETSC_END_LEN(lenN))
{
  char *lname;

  FIXCHAR(name, lenN, lname);
  *ierr = DMGetStratumIS(*dm, lname, *value, is);if (*ierr) return;
  if (is && !*is) *is = (IS)0;
  FREECHAR(name, lname);
}

PETSC_EXTERN void PETSC_STDCALL dmsetstratumis_(DM *dm, char* name PETSC_MIXED_LEN(lenN), PetscInt *value, IS *is, int *ierr PETSC_END_LEN(lenN))
{
  char *lname;

  FIXCHAR(name, lenN, lname);
  *ierr = DMSetStratumIS(*dm, lname, *value, *is);if (*ierr) return;
  FREECHAR(name, lname);
}

PETSC_EXTERN void PETSC_STDCALL dmremovelabel_(DM *dm, char* name PETSC_MIXED_LEN(lenN), DMLabel *label, int *ierr PETSC_END_LEN(lenN))
{
  char *lname;

  FIXCHAR(name, lenN, lname);
  *ierr = DMRemoveLabel(*dm, lname, label);if (*ierr) return;
  FREECHAR(name, lname);
}

PETSC_EXTERN void PETSC_STDCALL dmviewfromoptions_(DM *dm,PetscObject obj,char* type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = DMViewFromOptions(*dm,obj,t);if (*ierr) return;
  FREECHAR(type,t);
}
