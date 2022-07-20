const std = @import("std");
const p = @cImport({@cInclude("petsc.h");});

pub export fn main(argc: c_int, argv: [*c][*c]u8) c_int {
  var nargc: c_int = argc;
  var nargv: [*c][*c]u8 = argv;
  var ierr = p.PetscInitialize(&nargc,&nargv,"","");
  if (ierr != 0) return 0;

  ierr = p.PetscFinalize();
  return ierr;
}