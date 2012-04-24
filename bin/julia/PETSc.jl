#
#   Allows the PETSc dynamic library to be used from Julia. (http::/julialang.org)
#     PETSc must be configured with --with-shared-libraries
#     You can build with or without MPI, but cannot run on more than one process
#
#  Load the PETSc dynamic library
#
PETSC_DIR = getenv("PETSC_DIR");
PETSC_ARCH = getenv("PETSC_ARCH");
if (length(PETSC_DIR) == 0) 
  disp("Must have environmental variable PETSC_DIR set")
end
if (length(PETSC_ARCH) == 0) 
  disp("Must have environmental variable PETSC_ARCH set")
end
libpetsclocation = strcat(PETSC_DIR, "/", PETSC_ARCH, "/lib/", "libpetsc")
libpetsc = dlopen(libpetsclocation)

# -------------------------------------
function echodemo(filename)
  f = open(filename)
  h = readall(f)
  close(f)
  pos = 0
  while (pos <= length(h))
    (ex,pos)=parse(h,pos)
    str = string(ex.args)
    if _jl_have_color
      print("\033[1m\033[30m")
    end
    #  the next line doesn't work right for multiple line commands like for loops
    println(str[2:strlen(str)-1])
    if _jl_have_color
      print(_jl_answer_color())
      println(" ") # force a color change, otherwise answer color is not used
    end
    e = eval(ex)
    println(" ")
  end
end

# -------------------------------------

PETSC_INSERT_VALUES = 1;
PETSC_ADD_VALUES    = 2;
PETSC_COPY_VALUES   = 0;

PETSC_NORM_1         = 0;
PETSC_NORM_2         = 1;
PETSC_NORM_FROBENIUS = 2;
PETSC_NORM_INFINITY  = 3;
PETSC_NORM_MAX       = PETSC_NORM_INFINITY;


# -------------------------------------
#    These are the Julia interface methods for all the visible PETSc functions. Julia datatypes for PETSc objects simply contain the C pointer to the 
#    underlying PETSc object.
#   
# -------------------------------------
function PetscInitialize()
  PetscInitialize([])
end

function PetscInitialize(args)
  PetscInitialize(args,"","")
end

function PetscInitialize(args,filename,help)
  # argument list starts with program name
  args = ["julia",args]; 
  #
  #   If the user forgot to PetscFinalize() we do it for them, before restarting PETSc
  #
  init = 0;
  err = ccall(dlsym(libpetsc,:PetscInitialized),Int32,(Ptr{Int32},),&init);
  if (init != 0) 
    gc() # call garbage collection to force all PETSc objects be destroy that are queued up for destruction
    err = ccall(dlsym(libpetsc,:PetscFinalize),Int32,());if (err != 0) return err; end
  end
  arr = Array(ByteString, length(args))
  for i = 1:length(args)
    arr[i] = cstring(args[i])
  end
  ptrs = _jl_pre_exec(arr)
  err = ccall(dlsym(libpetsc, :PetscInitializeNoPointers),Int32,(Int32,Ptr{Ptr{Uint8}},Ptr{Uint8},Ptr{Uint8}), length(ptrs), ptrs,cstring(filename),cstring(help));
  return err
end

function PetscFinalize()
  gc() # call garbage collection to force all PETSc objects be destroy that are queued up for destruction
  return ccall(dlsym(libpetsc,:PetscFinalize),Int32,());
end

function PETSC_COMM_SELF()
  comm = Array(Int64, 1)
  err = ccall(dlsym(libpetsc, :PetscGetPETSC_COMM_SELF),Int32,(Ptr{Int64},),comm);
  return comm[1]
end

# -------------------------------------
#
abstract PetscObject

function PetscView(obj::PetscObject)
  PetscView(obj,0)
end

type PetscIS <: PetscObject
  pobj::Int64
  function PetscIS()
    comm = PETSC_COMM_SELF();
    is = Array(Int64,1)
    err = ccall(dlsym(libpetsc, :ISCreate),Int32,(Int64,Ptr{Int64}),comm,is);if (err != 0) return err;end
    is = new(is[1])
    finalizer(is,PetscDestroy) 
    # does not seem to be called immediately when is is no longer visible, is it called later during garbage collection?
    return is
  end
end

  function PetscDestroy(is::PetscIS)
    if (is.pobj != 0) then 
      err = ccall(dlsym(libpetsc, :ISDestroy),Int32,(Ptr{Int64},), &is.pobj);   
    end
    is.pobj = 0 
    return 0
  end

  function PetscIS(indices::Array{Int64})
    is = PetscIS()
    err = ccall(dlsym(libpetsc, :ISSetType),Int32,(Int64,Ptr{Uint8}), is.pobj,cstring("general"));
    err = ccall(dlsym(libpetsc, :ISGeneralSetIndices),Int32,(Int64,Int32,Ptr{Int32},Int32),is.pobj,length(indices),convert(Array{Int32},indices),PETSC_COPY_VALUES)
    return is
  end

  function PetscISSetType(vec::PetscIS,name)
    err = ccall(dlsym(libpetsc, :ISSetType),Int32,(Int64,Ptr{Uint8}), vec.pobj,cstring(name));
  end

  function PetscView(obj::PetscIS,viewer)
   err = ccall(dlsym(libpetsc, :ISView),Int32,(Int64,Int64),obj.pobj,0);
  end

  function PetscISGetSize(obj::PetscIS)
    n = Array(Int32,1)
    err = ccall(dlsym(libpetsc, :ISGetSize),Int32,(Int64,Ptr{Int32}), obj.pobj,n);
    return n[1]
  end

  function PetscISGetIndices(obj::PetscIS)
    len = PetscISGetSize(obj)
    indices = Array(Int32,len);
    err = ccall(dlsym(libpetsc,:ISGetIndicesCopy),Int32,(Int64,Ptr{Int32}),obj.pobj,indices);
    indices = indices + 1
    return indices  
  end

# -------------------------------------
#
type PetscVec <: PetscObject
  pobj::Int64
  function PetscVec()
    comm = PETSC_COMM_SELF();
    vec = Array(Int64,1)
    err = ccall(dlsym(libpetsc, :VecCreate),Int32,(Int64,Ptr{Int64}),comm,vec);
    vec = new(vec[1])
    finalizer(vec,PetscDestroy)  
    # does not seem to be called immediately when vec is no longer visible, is it called later during garbage collection?
    return vec
  end
end

  function PetscDestroy(vec::PetscVec)
    if (vec.pobj != 0) 
      err = ccall(dlsym(libpetsc, :VecDestroy),Int32,(Ptr{Int64},), &vec.pobj);    
    end
    vec.pobj = 0
  end

  function PetscVecSetType(vec::PetscVec,name)
    err = ccall(dlsym(libpetsc, :VecSetType),Int32,(Int64,Ptr{Uint8}), vec.pobj,cstring(name));
  end

  function PetscVec(array::Array{Float64})
    vec = PetscVec()
    err = ccall(dlsym(libpetsc, :VecSetType),Int32,(Int64,Ptr{Uint8}), vec.pobj,cstring("seq"));
    err = ccall(dlsym(libpetsc, :VecSetSizes),Int32,(Int64,Int32,Int32), vec.pobj,length(array),length(array));
    # want a 32 bit int array so build it ourselves
    idx = Array(Int32,length(array)); 
    for i=1:length(array);  idx[i] = i-1;  end
    err = ccall(dlsym(libpetsc, :VecSetValues), Int32,(Int64,Int32,Ptr{Int32},Ptr{Float64},Int32), vec.pobj,length(idx),idx,array,PETSC_INSERT_VALUES);
    err = ccall(dlsym(libpetsc, :VecAssemblyBegin),Int32,(Int64,), vec.pobj);
    err = ccall(dlsym(libpetsc, :VecAssemblyEnd),Int32,(Int64,), vec.pobj);
    return vec
  end

  function PetscVecSetValues(vec::PetscVec,idx::Array{Int64},array::Array{Float64},flag::Int)
    idx = idx - 1
    err = ccall(dlsym(libpetsc, :VecSetValues), Int32,(Int64,Int32,Ptr{Int32},Ptr{Float64},Int32), vec.pobj,length(idx),convert(Array{Int32},idx),array,flag);
    idx = idx + 1
    return err
  end
  function PetscVecSetValues(vec::PetscVec,idx::Array{Int64},array::Array{Float64})
    PetscVecSetValues(vec,idx,array,PETSC_INSERT_VALUES)
  end
  function PetscVecSetValues(vec::PetscVec,array::Array{Float64})
    idx = Array(Int64,length(array))
    for i=1:length(array);  idx[i] = i-1;  end
    PetscVecSetValues(vec,idx,array,PETSC_INSERT_VALUES)
  end

  function PetscVecAssemblyBegin(obj::PetscVec)
    err = ccall(dlsym(libpetsc, :VecAssemblyBegin),Int32,(Int64,), obj.pobj);
  end

  function PetscVecAssemblyEnd(obj::PetscVec)
    err = ccall(dlsym(libpetsc, :VecAssemblyEnd),Int32,(Int64,), obj.pobj);
  end

  function PetscVecSetSizes(vec::PetscVec,n::Int,N::Int)
    err = ccall(dlsym(libpetsc, :VecSetSizes),Int32,(Int64,Int32,Int32), vec.pobj,n,N);
  end

  function PetscView(obj::PetscVec,viewer)
    err = ccall(dlsym(libpetsc, :VecView),Int32,(Int64,Int64),obj.pobj,0);
  end

  function PetscVecGetSize(obj::PetscVec)
    n = Array(Int32,1)
    err = ccall(dlsym(libpetsc, :VecGetSize),Int32,(Int64,Ptr{Int32}), obj.pobj,n);
    return n[1]
  end

  function PetscVecNorm(obj::PetscVec,normtype::Int)
    n = Array(Float64,1)
    err = ccall(dlsym(libpetsc, :VecNorm),Int32,(Int64,Int32,Ptr{Int32}), obj.pobj,normtype, n);
    return n[1]
  end
  function PetscVecNorm(obj::PetscVec)
    return PetscVecNorm(obj,PETSC_NORM_2)
  end

# -------------------------------------
type PetscMat <: PetscObject
  pobj::Int64
  function PetscMat()
    comm = PETSC_COMM_SELF();
    vec = Array(Int64,1)
    err = ccall(dlsym(libpetsc, :MatCreate),Int32,(Int64,Ptr{Int64}),comm,vec);
    vec = new(vec[1])
    finalizer(vec,PetscDestroy)  
    # does not seem to be called immediately when vec is no longer visible, is it called later during garbage collection?
    return vec
  end
end

  function PetscDestroy(vec::PetscMat)
    if (vec.pobj != 0) 
      err = ccall(dlsym(libpetsc, :MatDestroy),Int32,(Ptr{Int64},), &vec.pobj);    
    end
    vec.pobj = 0
  end

  function PetscMatSetType(vec::PetscMat,name)
    err = ccall(dlsym(libpetsc, :MatSetType),Int32,(Int64,Ptr{Uint8}), vec.pobj,cstring(name));
  end

  function PetscSetUp(vec::PetscMat)
    err = ccall(dlsym(libpetsc, :MatSetUp),Int32,(Int64,), vec.pobj);
  end

  PETSC_MAT_FLUSH_ASSEMBLY = 1;
  PETSC_MAT_FINAL_ASSEMBLY = 0

  function PetscMatSetValues(vec::PetscMat,idi::Array{Int64},idj::Array{Int64},array::Array{Float64},flag::Int)
    idi = idi - 1
    idj = idj - 1
    err = ccall(dlsym(libpetsc, :MatSetValues), Int32,(Int64,Int32,Ptr{Int32},Int32,Ptr{Int32},Ptr{Float64},Int32), vec.pobj,length(idi),convert(Array{Int32},idi),length(idi),convert(Array{Int32},idj),array,flag);
    idi = idi + 1
    idj = idj + 1
    return err
  end

  function PetscMatAssemblyBegin(obj::PetscMat,flg::Int)
    err = ccall(dlsym(libpetsc, :MatAssemblyBegin),Int32,(Int64,Int32), obj.pobj,flg);
  end
  function PetscMatAssemblyBegin(obj::PetscMat)
    return PetscMatAssemblyBegin(obj,PETSC_MAT_FINAL_ASSEMBLY);
  end

  function PetscMatAssemblyEnd(obj::PetscMat,flg::Int)
    err = ccall(dlsym(libpetsc, :MatAssemblyEnd),Int32,(Int64,Int32), obj.pobj,flg);
  end
  function PetscMatAssemblyEnd(obj::PetscMat)
    return PetscMatAssemblyEnd(obj,PETSC_MAT_FINAL_ASSEMBLY);
  end

  function PetscMatSetSizes(vec::PetscMat,m::Int,n::Int,M::Int,N::Int)
    err = ccall(dlsym(libpetsc, :MatSetSizes),Int32,(Int64,Int32,Int32,Int32,Int32), vec.pobj,m,n,M,N);
  end

  function PetscView(obj::PetscMat,viewer)
    err = ccall(dlsym(libpetsc, :MatView),Int32,(Int64,Int64),obj.pobj,0);
  end

  function PetscMatGetSize(obj::PetscMat)
    m = Array(Int32,1)
    n = Array(Int32,1)
    err = ccall(dlsym(libpetsc, :MatGetSize),Int32,(Int64,Ptr{Int32},Ptr{Int32}), obj.pobj,m,n);
    return (m[1],n[1])
  end