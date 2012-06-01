#include <petsc-private/meshimpl.h> /*I "petscdmmesh.h" I*/
#include <petscsnes.h>              /*I "petscsnes.h" I*/

#undef __FUNCT__
#define __FUNCT__ "DMMeshInterpolationCreate"
PetscErrorCode DMMeshInterpolationCreate(DM dm, DMMeshInterpolationInfo *ctx) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(ctx, 2);
  ierr = PetscMalloc(sizeof(struct _DMMeshInterpolationInfo), ctx);CHKERRQ(ierr);
  (*ctx)->dim    = -1;
  (*ctx)->nInput = 0;
  (*ctx)->points = PETSC_NULL;
  (*ctx)->cells  = PETSC_NULL;
  (*ctx)->n      = -1;
  (*ctx)->coords = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshInterpolationSetDim"
PetscErrorCode DMMeshInterpolationSetDim(DM dm, PetscInt dim, DMMeshInterpolationInfo ctx) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if ((dim < 1) || (dim > 3)) {SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid dimension for points: %d", dim);}
  ctx->dim = dim;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshInterpolationGetDim"
PetscErrorCode DMMeshInterpolationGetDim(DM dm, PetscInt *dim, DMMeshInterpolationInfo ctx) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(dim, 2);
  *dim = ctx->dim;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshInterpolationSetDof"
PetscErrorCode DMMeshInterpolationSetDof(DM dm, PetscInt dof, DMMeshInterpolationInfo ctx) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (dof < 1) {SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid number of components: %d", dof);}
  ctx->dof = dof;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshInterpolationGetDof"
PetscErrorCode DMMeshInterpolationGetDof(DM dm, PetscInt *dof, DMMeshInterpolationInfo ctx) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(dof, 2);
  *dof = ctx->dof;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshInterpolationAddPoints"
PetscErrorCode DMMeshInterpolationAddPoints(DM dm, PetscInt n, PetscReal points[], DMMeshInterpolationInfo ctx) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (ctx->dim < 0) {
    SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONGSTATE, "The spatial dimension has not been set");
  }
  if (ctx->points) {
    SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONGSTATE, "Cannot add points multiple times yet");
  }
  ctx->nInput = n;
  ierr = PetscMalloc(n*ctx->dim * sizeof(PetscReal), &ctx->points);CHKERRQ(ierr);
  ierr = PetscMemcpy(ctx->points, points, n*ctx->dim * sizeof(PetscReal));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshInterpolationSetUp"
PetscErrorCode DMMeshInterpolationSetUp(DM dm, DMMeshInterpolationInfo ctx, PetscBool redundantPoints) {
  ALE::Obj<PETSC_MESH_TYPE> m;
  MPI_Comm       comm = ((PetscObject) dm)->comm;
  PetscScalar   *a;
  PetscInt       p, q, i;
  PetscMPIInt    rank, size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  if (ctx->dim < 0) {
    SETERRQ(comm, PETSC_ERR_ARG_WRONGSTATE, "The spatial dimension has not been set");
  }
  // Locate points
  PetscLayout    layout;
  PetscReal      *globalPoints;
  const PetscInt *ranges;
  PetscMPIInt    *counts, *displs;
  PetscInt       *foundCells;
  PetscMPIInt    *foundProcs, *globalProcs;
  PetscInt        n = ctx->nInput, N;

  if (!redundantPoints) {
    ierr = PetscLayoutCreate(comm, &layout);CHKERRQ(ierr);
    ierr = PetscLayoutSetBlockSize(layout, 1);CHKERRQ(ierr);
    ierr = PetscLayoutSetLocalSize(layout, n);CHKERRQ(ierr);
    ierr = PetscLayoutSetUp(layout);CHKERRQ(ierr);
    ierr = PetscLayoutGetSize(layout, &N);CHKERRQ(ierr);
    /* Communicate all points to all processes */
    ierr = PetscMalloc3(N*ctx->dim,PetscReal,&globalPoints,size,PetscMPIInt,&counts,size,PetscMPIInt,&displs);CHKERRQ(ierr);
    ierr = PetscLayoutGetRanges(layout, &ranges);CHKERRQ(ierr);
    for(p = 0; p < size; ++p) {
      counts[p] = (ranges[p+1] - ranges[p])*ctx->dim;
      displs[p] = ranges[p]*ctx->dim;
    }
    ierr = MPI_Allgatherv(ctx->points, n*ctx->dim, MPIU_REAL, globalPoints, counts, displs, MPIU_REAL, comm);CHKERRQ(ierr);
  } else {
    N = n;
    globalPoints = ctx->points;
  }
  ierr = PetscMalloc3(N,PetscInt,&foundCells,N,PetscMPIInt,&foundProcs,N,PetscMPIInt,&globalProcs);CHKERRQ(ierr);
  for(p = 0; p < N; ++p) {
    foundCells[p] = m->locatePoint(&globalPoints[p*ctx->dim]);
    if (foundCells[p] >= 0) {
      foundProcs[p] = rank;
    } else {
      foundProcs[p] = size;
    }
  }
  /* Let the lowest rank process own each point */
  ierr = MPI_Allreduce(foundProcs, globalProcs, N, MPI_INT, MPI_MIN, comm);CHKERRQ(ierr);
  ctx->n = 0;
  for(p = 0; p < N; ++p) {
    if (globalProcs[p] == size) {
      SETERRQ4(comm, PETSC_ERR_PLIB, "Point %d: %g %g %g not located in mesh", p, globalPoints[p*ctx->dim+0], ctx->dim > 1 ? globalPoints[p*ctx->dim+1] : 0.0, ctx->dim > 2 ? globalPoints[p*ctx->dim+2] : 0.0);
    } else if (globalProcs[p] == rank) {
      ctx->n++;
    }
  }
  /* Create coordinates vector and array of owned cells */
  ierr = PetscMalloc(ctx->n * sizeof(PetscInt), &ctx->cells);CHKERRQ(ierr);
  ierr = VecCreate(comm, &ctx->coords);CHKERRQ(ierr);
  ierr = VecSetSizes(ctx->coords, ctx->n*ctx->dim, PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetBlockSize(ctx->coords, ctx->dim);CHKERRQ(ierr);
  ierr = VecSetFromOptions(ctx->coords);CHKERRQ(ierr);
  ierr = VecGetArray(ctx->coords, &a);CHKERRQ(ierr);
  for(p = 0, q = 0, i = 0; p < N; ++p) {
    if (globalProcs[p] == rank) {
      PetscInt d;

      for(d = 0; d < ctx->dim; ++d, ++i) {
        a[i] = globalPoints[p*ctx->dim+d];
      }
      ctx->cells[q++] = foundCells[p];
    }
  }
  ierr = VecRestoreArray(ctx->coords, &a);CHKERRQ(ierr);
  ierr = PetscFree3(foundCells,foundProcs,globalProcs);CHKERRQ(ierr);
  if (!redundantPoints) {
    ierr = PetscFree3(globalPoints,counts,displs);CHKERRQ(ierr);
    ierr = PetscLayoutDestroy(&layout);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshInterpolationGetCoordinates"
PetscErrorCode DMMeshInterpolationGetCoordinates(DM dm, Vec *coordinates, DMMeshInterpolationInfo ctx) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(coordinates, 2);
  if (!ctx->coords) {SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONGSTATE, "The interpolation context has not been setup.");}
  *coordinates = ctx->coords;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshInterpolationGetVector"
PetscErrorCode DMMeshInterpolationGetVector(DM dm, Vec *v, DMMeshInterpolationInfo ctx) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(v, 2);
  if (!ctx->coords) {SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONGSTATE, "The interpolation context has not been setup.");}
  ierr = VecCreate(((PetscObject) dm)->comm, v);CHKERRQ(ierr);
  ierr = VecSetSizes(*v, ctx->n*ctx->dof, PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*v, ctx->dof);CHKERRQ(ierr);
  ierr = VecSetFromOptions(*v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshInterpolationRestoreVector"
PetscErrorCode DMMeshInterpolationRestoreVector(DM dm, Vec *v, DMMeshInterpolationInfo ctx) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(v, 2);
  if (!ctx->coords) {SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONGSTATE, "The interpolation context has not been setup.");}
  ierr = VecDestroy(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshInterpolate_Simplex_Private"
PetscErrorCode DMMeshInterpolate_Simplex_Private(DM dm, SectionReal x, Vec v, DMMeshInterpolationInfo ctx) {
#ifdef PETSC_HAVE_SIEVE
  ALE::Obj<PETSC_MESH_TYPE> m;
  ALE::Obj<PETSC_MESH_TYPE::real_section_type> s;
  PetscInt       p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  ierr = SectionRealGetSection(x, s);CHKERRQ(ierr);
  const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& coordinates = m->getRealSection("coordinates");
  PetscReal   *v0, *J, *invJ, detJ;
  PetscScalar *a, *coords;

  ierr = PetscMalloc3(ctx->dim,PetscReal,&v0,ctx->dim*ctx->dim,PetscReal,&J,ctx->dim*ctx->dim,PetscReal,&invJ);CHKERRQ(ierr);
  ierr = VecGetArray(ctx->coords, &coords);CHKERRQ(ierr);
  ierr = VecGetArray(v, &a);CHKERRQ(ierr);
  for(p = 0; p < ctx->n; ++p) {
    PetscInt  e = ctx->cells[p];
    PetscReal xi[4];
    PetscInt  d, f, comp;

    if ((ctx->dim+1)*ctx->dof != m->sizeWithBC(s, e)) {SETERRQ2(((PetscObject) dm)->comm, PETSC_ERR_ARG_SIZ, "Invalid restrict size %d should be %d", m->sizeWithBC(s, e), (ctx->dim+1)*ctx->dof);}
    m->computeElementGeometry(coordinates, e, v0, J, invJ, detJ);
    const PetscScalar *c = m->restrictClosure(s, e); /* Must come after geom, since it uses closure temp space*/
    for(comp = 0; comp < ctx->dof; ++comp) {
      a[p*ctx->dof+comp] = c[0*ctx->dof+comp];
    }
    for(d = 0; d < ctx->dim; ++d) {
      xi[d] = 0.0;
      for(f = 0; f < ctx->dim; ++f) {
        xi[d] += invJ[d*ctx->dim+f]*0.5*(coords[p*ctx->dim+f] - v0[f]);
      }
      for(comp = 0; comp < ctx->dof; ++comp) {
        a[p*ctx->dof+comp] += (c[(d+1)*ctx->dof+comp] - c[0*ctx->dof+comp])*xi[d];
      }
    }
  }
  ierr = VecRestoreArray(v, &a);CHKERRQ(ierr);
  ierr = VecRestoreArray(ctx->coords, &coords);CHKERRQ(ierr);
  ierr = PetscFree3(v0, J, invJ);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#else
  SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_SUP, "Interpolation only work with DMMesh currently.");
#endif
}

#undef __FUNCT__
#define __FUNCT__ "QuadMap_Private"
PetscErrorCode QuadMap_Private(SNES snes, Vec Xref, Vec Xreal, void *ctx)
{
  const PetscScalar*vertices = (const PetscScalar *) ctx;
  const PetscScalar x0   = vertices[0];
  const PetscScalar y0   = vertices[1];
  const PetscScalar x1   = vertices[2];
  const PetscScalar y1   = vertices[3];
  const PetscScalar x2   = vertices[4];
  const PetscScalar y2   = vertices[5];
  const PetscScalar x3   = vertices[6];
  const PetscScalar y3   = vertices[7];
  const PetscScalar f_1  = x1 - x0;
  const PetscScalar g_1  = y1 - y0;
  const PetscScalar f_3  = x3 - x0;
  const PetscScalar g_3  = y3 - y0;
  const PetscScalar f_01 = x2 - x1 - x3 + x0;
  const PetscScalar g_01 = y2 - y1 - y3 + y0;
  PetscScalar      *ref, *real;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetArray(Xref,  &ref);CHKERRQ(ierr);
  ierr = VecGetArray(Xreal, &real);CHKERRQ(ierr);
  {
    const PetscScalar p0 = ref[0];
    const PetscScalar p1 = ref[1];

    real[0] = x0 + f_1 * p0 + f_3 * p1 + f_01 * p0 * p1;
    real[1] = y0 + g_1 * p0 + g_3 * p1 + g_01 * p0 * p1;
  }
  ierr = PetscLogFlops(28);CHKERRQ(ierr);
  ierr = VecRestoreArray(Xref,  &ref);CHKERRQ(ierr);
  ierr = VecRestoreArray(Xreal, &real);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QuadJacobian_Private"
PetscErrorCode QuadJacobian_Private(SNES snes, Vec Xref, Mat *J, Mat *M, MatStructure *flag, void *ctx)
{
  const PetscScalar*vertices = (const PetscScalar *) ctx;
  const PetscScalar x0   = vertices[0];
  const PetscScalar y0   = vertices[1];
  const PetscScalar x1   = vertices[2];
  const PetscScalar y1   = vertices[3];
  const PetscScalar x2   = vertices[4];
  const PetscScalar y2   = vertices[5];
  const PetscScalar x3   = vertices[6];
  const PetscScalar y3   = vertices[7];
  const PetscScalar f_01 = x2 - x1 - x3 + x0;
  const PetscScalar g_01 = y2 - y1 - y3 + y0;
  PetscScalar      *ref;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetArray(Xref,  &ref);CHKERRQ(ierr);
  {
    const PetscScalar x = ref[0];
    const PetscScalar y = ref[1];
    const PetscInt    rows[2]   = {0, 1};
    const PetscScalar values[4] = {(x1 - x0 + f_01*y) * 0.5, (x3 - x0 + f_01*x) * 0.5,
                                   (y1 - y0 + g_01*y) * 0.5, (y3 - y0 + g_01*x) * 0.5};
    ierr = MatSetValues(*J, 2, rows, 2, rows, values, INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscLogFlops(30);CHKERRQ(ierr);
  ierr = VecRestoreArray(Xref,  &ref);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*J, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*J, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshInterpolate_Quad_Private"
PetscErrorCode DMMeshInterpolate_Quad_Private(DM dm, SectionReal x, Vec v, DMMeshInterpolationInfo ctx) {
#ifdef PETSC_HAVE_SIEVE
  SNES        snes;
  KSP         ksp;
  PC          pc;
  Vec         r, ref, real;
  Mat         J;
  PetscScalar vertices[8];

  ALE::Obj<PETSC_MESH_TYPE> m;
  ALE::Obj<PETSC_MESH_TYPE::real_section_type> s;
  PetscInt       p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESCreate(PETSC_COMM_SELF, &snes);CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(snes, "quad_interp_");CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF, &r);CHKERRQ(ierr);
  ierr = VecSetSizes(r, 2, 2);CHKERRQ(ierr);
  ierr = VecSetFromOptions(r);CHKERRQ(ierr);
  ierr = VecDuplicate(r, &ref);CHKERRQ(ierr);
  ierr = VecDuplicate(r, &real);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_SELF, &J);CHKERRQ(ierr);
  ierr = MatSetSizes(J, 2, 2, 2, 2);CHKERRQ(ierr);
  ierr = MatSetType(J, MATSEQDENSE);CHKERRQ(ierr);
  ierr = MatSetUp(J);CHKERRQ(ierr);
  ierr = SNESSetFunction(snes, r, QuadMap_Private, vertices);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes, J, J, QuadJacobian_Private, vertices);CHKERRQ(ierr);
  ierr = SNESGetKSP(snes, &ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
  ierr = PCSetType(pc, PCLU);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  ierr = SectionRealGetSection(x, s);CHKERRQ(ierr);
  const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& coordinates = m->getRealSection("coordinates");
  PetscScalar *a, *coords;

  ierr = VecGetArray(ctx->coords, &coords);CHKERRQ(ierr);
  ierr = VecGetArray(v, &a);CHKERRQ(ierr);
  for(p = 0; p < ctx->n; ++p) {
    PetscScalar *xi;
    PetscInt     e = ctx->cells[p], comp;

    if (4*ctx->dof != m->sizeWithBC(s, e)) {SETERRQ2(((PetscObject) dm)->comm, PETSC_ERR_ARG_SIZ, "Invalid restrict size %d should be %d", m->sizeWithBC(s, e), 4*ctx->dof);}
    /* Can make this do all points at once */
    {
      const PetscReal *v = m->restrictClosure(coordinates, e);
      for(PetscInt i = 0; i < 8; ++i) vertices[i] = v[i];
    }
    const PetscScalar *c = m->restrictClosure(s, e); /* Must come after geom, since it uses closure temp space*/
    ierr = VecGetArray(real, &xi);CHKERRQ(ierr);
    xi[0] = coords[p*ctx->dim+0];
    xi[1] = coords[p*ctx->dim+1];
    ierr = VecRestoreArray(real, &xi);CHKERRQ(ierr);
    ierr = SNESSolve(snes, real, ref);CHKERRQ(ierr);
    ierr = VecGetArray(ref, &xi);CHKERRQ(ierr);
    for(comp = 0; comp < ctx->dof; ++comp) {
      a[p*ctx->dof+comp] = c[0*ctx->dof+comp]*(1 - xi[0])*(1 - xi[1]) + c[1*ctx->dof+comp]*xi[0]*(1 - xi[1]) + c[2*ctx->dof+comp]*xi[0]*xi[1] + c[3*ctx->dof+comp]*(1 - xi[0])*xi[1];
    }
    ierr = VecRestoreArray(ref, &xi);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(v, &a);CHKERRQ(ierr);
  ierr = VecRestoreArray(ctx->coords, &coords);CHKERRQ(ierr);

  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = VecDestroy(&ref);CHKERRQ(ierr);
  ierr = VecDestroy(&real);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#else
  SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_SUP, "Interpolation only work with DMMesh currently.");
#endif
}

#undef __FUNCT__
#define __FUNCT__ "HexMap_Private"
PetscErrorCode HexMap_Private(SNES snes, Vec Xref, Vec Xreal, void *ctx)
{
  const PetscScalar*vertices = (const PetscScalar *) ctx;
  const PetscScalar x0 = vertices[0];
  const PetscScalar y0 = vertices[1];
  const PetscScalar z0 = vertices[2];
  const PetscScalar x1 = vertices[3];
  const PetscScalar y1 = vertices[4];
  const PetscScalar z1 = vertices[5];
  const PetscScalar x2 = vertices[6];
  const PetscScalar y2 = vertices[7];
  const PetscScalar z2 = vertices[8];
  const PetscScalar x3 = vertices[9];
  const PetscScalar y3 = vertices[10];
  const PetscScalar z3 = vertices[11];
  const PetscScalar x4 = vertices[12];
  const PetscScalar y4 = vertices[13];
  const PetscScalar z4 = vertices[14];
  const PetscScalar x5 = vertices[15];
  const PetscScalar y5 = vertices[16];
  const PetscScalar z5 = vertices[17];
  const PetscScalar x6 = vertices[18];
  const PetscScalar y6 = vertices[19];
  const PetscScalar z6 = vertices[20];
  const PetscScalar x7 = vertices[21];
  const PetscScalar y7 = vertices[22];
  const PetscScalar z7 = vertices[23];
  const PetscScalar f_1 = x1 - x0;
  const PetscScalar g_1 = y1 - y0;
  const PetscScalar h_1 = z1 - z0;
  const PetscScalar f_3 = x3 - x0;
  const PetscScalar g_3 = y3 - y0;
  const PetscScalar h_3 = z3 - z0;
  const PetscScalar f_4 = x4 - x0;
  const PetscScalar g_4 = y4 - y0;
  const PetscScalar h_4 = z4 - z0;
  const PetscScalar f_01 = x2 - x1 - x3 + x0;
  const PetscScalar g_01 = y2 - y1 - y3 + y0;
  const PetscScalar h_01 = z2 - z1 - z3 + z0;
  const PetscScalar f_12 = x7 - x3 - x4 + x0;
  const PetscScalar g_12 = y7 - y3 - y4 + y0;
  const PetscScalar h_12 = z7 - z3 - z4 + z0;
  const PetscScalar f_02 = x5 - x1 - x4 + x0;
  const PetscScalar g_02 = y5 - y1 - y4 + y0;
  const PetscScalar h_02 = z5 - z1 - z4 + z0;
  const PetscScalar f_012 = x6 - x0 + x1 - x2 + x3 + x4 - x5 - x7;
  const PetscScalar g_012 = y6 - y0 + y1 - y2 + y3 + y4 - y5 - y7;
  const PetscScalar h_012 = z6 - z0 + z1 - z2 + z3 + z4 - z5 - z7;
  PetscScalar      *ref, *real;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetArray(Xref,  &ref);CHKERRQ(ierr);
  ierr = VecGetArray(Xreal, &real);CHKERRQ(ierr);
  {
    const PetscScalar p0 = ref[0];
    const PetscScalar p1 = ref[1];
    const PetscScalar p2 = ref[2];

    real[0] = x0 + f_1*p0 + f_3*p1 + f_4*p2 + f_01*p0*p1 + f_12*p1*p2 + f_02*p0*p2 + f_012*p0*p1*p2;
    real[1] = y0 + g_1*p0 + g_3*p1 + g_4*p2 + g_01*p0*p1 + g_01*p0*p1 + g_12*p1*p2 + g_02*p0*p2 + g_012*p0*p1*p2;
    real[2] = z0 + h_1*p0 + h_3*p1 + h_4*p2 + h_01*p0*p1 + h_01*p0*p1 + h_12*p1*p2 + h_02*p0*p2 + h_012*p0*p1*p2;
  }
  ierr = PetscLogFlops(114);CHKERRQ(ierr);
  ierr = VecRestoreArray(Xref,  &ref);CHKERRQ(ierr);
  ierr = VecRestoreArray(Xreal, &real);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "HexJacobian_Private"
PetscErrorCode HexJacobian_Private(SNES snes, Vec Xref, Mat *J, Mat *M, MatStructure *flag, void *ctx)
{
  const PetscScalar*vertices = (const PetscScalar *) ctx;
  const PetscScalar x0 = vertices[0];
  const PetscScalar y0 = vertices[1];
  const PetscScalar z0 = vertices[2];
  const PetscScalar x1 = vertices[3];
  const PetscScalar y1 = vertices[4];
  const PetscScalar z1 = vertices[5];
  const PetscScalar x2 = vertices[6];
  const PetscScalar y2 = vertices[7];
  const PetscScalar z2 = vertices[8];
  const PetscScalar x3 = vertices[9];
  const PetscScalar y3 = vertices[10];
  const PetscScalar z3 = vertices[11];
  const PetscScalar x4 = vertices[12];
  const PetscScalar y4 = vertices[13];
  const PetscScalar z4 = vertices[14];
  const PetscScalar x5 = vertices[15];
  const PetscScalar y5 = vertices[16];
  const PetscScalar z5 = vertices[17];
  const PetscScalar x6 = vertices[18];
  const PetscScalar y6 = vertices[19];
  const PetscScalar z6 = vertices[20];
  const PetscScalar x7 = vertices[21];
  const PetscScalar y7 = vertices[22];
  const PetscScalar z7 = vertices[23];
  const PetscScalar f_xy = x2 - x1 - x3 + x0;
  const PetscScalar g_xy = y2 - y1 - y3 + y0;
  const PetscScalar h_xy = z2 - z1 - z3 + z0;
  const PetscScalar f_yz = x7 - x3 - x4 + x0;
  const PetscScalar g_yz = y7 - y3 - y4 + y0;
  const PetscScalar h_yz = z7 - z3 - z4 + z0;
  const PetscScalar f_xz = x5 - x1 - x4 + x0;
  const PetscScalar g_xz = y5 - y1 - y4 + y0;
  const PetscScalar h_xz = z5 - z1 - z4 + z0;
  const PetscScalar f_xyz = x6 - x0 + x1 - x2 + x3 + x4 - x5 - x7;
  const PetscScalar g_xyz = y6 - y0 + y1 - y2 + y3 + y4 - y5 - y7;
  const PetscScalar h_xyz = z6 - z0 + z1 - z2 + z3 + z4 - z5 - z7;
  PetscScalar      *ref;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetArray(Xref,  &ref);CHKERRQ(ierr);
  {
    const PetscScalar x = ref[0];
    const PetscScalar y = ref[1];
    const PetscScalar z = ref[2];
    const PetscInt    rows[3]   = {0, 1, 2};
    const PetscScalar values[9] = {
      (x1 - x0 + f_xy*y + f_xz*z + f_xyz*y*z) / 2.0,
      (x3 - x0 + f_xy*x + f_yz*z + f_xyz*x*z) / 2.0,
      (x4 - x0 + f_yz*y + f_xz*x + f_xyz*x*y) / 2.0,
      (y1 - y0 + g_xy*y + g_xz*z + g_xyz*y*z) / 2.0,
      (y3 - y0 + g_xy*x + g_yz*z + g_xyz*x*z) / 2.0,
      (y4 - y0 + g_yz*y + g_xz*x + g_xyz*x*y) / 2.0,
      (z1 - z0 + h_xy*y + h_xz*z + h_xyz*y*z) / 2.0,
      (z3 - z0 + h_xy*x + h_yz*z + h_xyz*x*z) / 2.0,
      (z4 - z0 + h_yz*y + h_xz*x + h_xyz*x*y) / 2.0};
    ierr = MatSetValues(*J, 3, rows, 3, rows, values, INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscLogFlops(152);CHKERRQ(ierr);
  ierr = VecRestoreArray(Xref,  &ref);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*J, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*J, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshInterpolate_Hex_Private"
PetscErrorCode DMMeshInterpolate_Hex_Private(DM dm, SectionReal x, Vec v, DMMeshInterpolationInfo ctx) {
#ifdef PETSC_HAVE_SIEVE
  SNES        snes;
  KSP         ksp;
  PC          pc;
  Vec         r, ref, real;
  Mat         J;
  PetscScalar vertices[24];

  ALE::Obj<PETSC_MESH_TYPE> m;
  ALE::Obj<PETSC_MESH_TYPE::real_section_type> s;
  PetscInt       p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESCreate(PETSC_COMM_SELF, &snes);CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(snes, "hex_interp_");CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF, &r);CHKERRQ(ierr);
  ierr = VecSetSizes(r, 3, 3);CHKERRQ(ierr);
  ierr = VecSetFromOptions(r);CHKERRQ(ierr);
  ierr = VecDuplicate(r, &ref);CHKERRQ(ierr);
  ierr = VecDuplicate(r, &real);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_SELF, &J);CHKERRQ(ierr);
  ierr = MatSetSizes(J, 3, 3, 3, 3);CHKERRQ(ierr);
  ierr = MatSetType(J, MATSEQDENSE);CHKERRQ(ierr);
  ierr = MatSetUp(J);CHKERRQ(ierr);
  ierr = SNESSetFunction(snes, r, HexMap_Private, vertices);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes, J, J, HexJacobian_Private, vertices);CHKERRQ(ierr);
  ierr = SNESGetKSP(snes, &ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
  ierr = PCSetType(pc, PCLU);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  ierr = SectionRealGetSection(x, s);CHKERRQ(ierr);
  const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& coordinates = m->getRealSection("coordinates");
  PetscScalar *a, *coords;

  ierr = VecGetArray(ctx->coords, &coords);CHKERRQ(ierr);
  ierr = VecGetArray(v, &a);CHKERRQ(ierr);
  for(p = 0; p < ctx->n; ++p) {
    PetscScalar *xi;
    PetscInt     e = ctx->cells[p], comp;

    if (8*ctx->dof != m->sizeWithBC(s, e)) {SETERRQ2(((PetscObject) dm)->comm, PETSC_ERR_ARG_SIZ, "Invalid restrict size %d should be %d", m->sizeWithBC(s, e), 8*ctx->dof);}
    /* Can make this do all points at once */
    {
      const PetscReal *v = m->restrictClosure(coordinates, e);
      for(PetscInt i = 0; i < 24; ++i) vertices[i] = v[i];
    }
    const PetscScalar *c = m->restrictClosure(s, e); /* Must come after geom, since it uses closure temp space*/
    ierr = VecGetArray(real, &xi);CHKERRQ(ierr);
    xi[0] = coords[p*ctx->dim+0];
    xi[1] = coords[p*ctx->dim+1];
    xi[2] = coords[p*ctx->dim+2];
    ierr = VecRestoreArray(real, &xi);CHKERRQ(ierr);
    ierr = SNESSolve(snes, real, ref);CHKERRQ(ierr);
    ierr = VecGetArray(ref, &xi);CHKERRQ(ierr);
    for(comp = 0; comp < ctx->dof; ++comp) {
      a[p*ctx->dof+comp] =
        c[0*ctx->dof+comp]*(1-xi[0])*(1-xi[1])*(1-xi[2]) +
        c[1*ctx->dof+comp]*    xi[0]*(1-xi[1])*(1-xi[2]) +
        c[2*ctx->dof+comp]*    xi[0]*    xi[1]*(1-xi[2]) +
        c[3*ctx->dof+comp]*(1-xi[0])*    xi[1]*(1-xi[2]) +
        c[4*ctx->dof+comp]*(1-xi[0])*(1-xi[1])*   xi[2] +
        c[5*ctx->dof+comp]*    xi[0]*(1-xi[1])*   xi[2] +
        c[6*ctx->dof+comp]*    xi[0]*    xi[1]*   xi[2] +
        c[7*ctx->dof+comp]*(1-xi[0])*    xi[1]*   xi[2];
    }
    ierr = VecRestoreArray(ref, &xi);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(v, &a);CHKERRQ(ierr);
  ierr = VecRestoreArray(ctx->coords, &coords);CHKERRQ(ierr);

  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = VecDestroy(&ref);CHKERRQ(ierr);
  ierr = VecDestroy(&real);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#else
  SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_SUP, "Interpolation only work with DMMesh currently.");
#endif
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshInterpolationEvaluate"
PetscErrorCode DMMeshInterpolationEvaluate(DM dm, SectionReal x, Vec v, DMMeshInterpolationInfo ctx) {
  PetscInt       dim, coneSize, n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(x, SECTIONREAL_CLASSID, 2);
  PetscValidHeaderSpecific(v, VEC_CLASSID, 3);
  ierr = VecGetLocalSize(v, &n);CHKERRQ(ierr);
  if (n != ctx->n*ctx->dof) {SETERRQ2(((PetscObject) dm)->comm, PETSC_ERR_ARG_SIZ, "Invalid input vector size %d should be %d", n, ctx->n*ctx->dof);}
  if (n) {
    ierr = DMMeshGetDimension(dm, &dim);CHKERRQ(ierr);
    ierr = DMMeshGetConeSize(dm, ctx->cells[0], &coneSize);CHKERRQ(ierr);
    if (dim == 2) {
      if (coneSize == 3) {
        ierr = DMMeshInterpolate_Simplex_Private(dm, x, v, ctx);CHKERRQ(ierr);
      } else if (coneSize == 4) {
        ierr = DMMeshInterpolate_Quad_Private(dm, x, v, ctx);CHKERRQ(ierr);
      } else {
        SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Unsupported dimension %d for point interpolation", dim);
      }
    } else if (dim == 3) {
      if (coneSize == 4) {
        ierr = DMMeshInterpolate_Simplex_Private(dm, x, v, ctx);CHKERRQ(ierr);
      } else {
        ierr = DMMeshInterpolate_Hex_Private(dm, x, v, ctx);CHKERRQ(ierr);
      }
    } else {
      SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Unsupported dimension %d for point interpolation", dim);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshInterpolationDestroy"
PetscErrorCode DMMeshInterpolationDestroy(DM dm, DMMeshInterpolationInfo *ctx) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(ctx, 2);
  ierr = VecDestroy(&(*ctx)->coords);CHKERRQ(ierr);
  ierr = PetscFree((*ctx)->points);CHKERRQ(ierr);
  ierr = PetscFree((*ctx)->cells);CHKERRQ(ierr);
  ierr = PetscFree(*ctx);CHKERRQ(ierr);
  *ctx = PETSC_NULL;
  PetscFunctionReturn(0);
}
