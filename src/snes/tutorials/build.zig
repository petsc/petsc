// Uses zig to compiler ex1z.zig

const std = @import("std");

pub fn build(b: *std.build.Builder) void {
    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});

    // Standard release options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall.
    const mode = b.standardReleaseOptions();

    const exe = b.addExecutable("ex1z", "ex1z.zig");

    const PETSC_DIR = std.os.getenv("PETSC_DIR") orelse unreachable;
    const PETSC_ARCH = std.os.getenv("PETSC_ARCH") orelse  unreachable;
    var path = std.fs.path.join(std.heap.c_allocator, &[_][] const u8 { PETSC_DIR,PETSC_ARCH,"lib"});
    if (path) |value| {exe.addLibPath(value);}  else |_| {std.debug.print("Error bad path: {s}\n", .{path});}
    if (path) |value| {exe.addRPath(value);}  else |_| {}
    // This should not be needed but is export DYLD_LIBRARY_PATH=$PETSC_DIR/$PETSC_ARCH/lib
    path = std.fs.path.join(std.heap.c_allocator, &[_][] const u8 { PETSC_DIR,"include"});
    if (path) |value| {exe.addIncludeDir(value);}  else |_| {}
    path = std.fs.path.join(std.heap.c_allocator, &[_][] const u8 { PETSC_DIR,PETSC_ARCH,"include"});
    if (path) |value| {exe.addIncludeDir(value);}  else |_| {}

    exe.linkSystemLibrary("petsc");
    exe.setTarget(target);
    exe.setBuildMode(mode);
    exe.install();

    const run_cmd = exe.run();
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}
