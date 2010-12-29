classdef PetscViewer < PetscObject
%
%  PetscViewer - Abstract PETSc object for printing/displaying information about PETSc objects
%
%  Creation:
%    v = PetscViewer; 
%      v.SetType('ascii');
%      v.FileSetMode(Petsc.FILE_MODE_WRITE);
%      v.FileSetName('filename');
%
%    v = PetscViewer('filename');      Open ASCII file for writing
%
  methods
    function obj = PetscViewer(filename,mode)
      comm = PETSC_COMM_SELF();
      if (nargin == 2) 
        [err,dummy,obj.pobj] = calllib('libpetsc', 'PetscViewerBinaryOpen', comm,filename,mode,0);PetscCHKERRQ(err);
      elseif (nargin == 1) 
        [err,dummy,obj.pobj] = calllib('libpetsc', 'PetscViewerASCIIOpen', comm,filename,0);PetscCHKERRQ(err);
      else
        [err,obj.pobj] = calllib('libpetsc', 'PetscViewerCreate', comm,0);PetscCHKERRQ(err);
      end
    end
    function SetType(obj,name)
      err = calllib('libpetsc', 'PetscViewerSetType', obj.pobj,name);PetscCHKERRQ(err);
    end
    function FileSetMode(obj,mode)
      err = calllib('libpetsc', 'PetscViewerFileSetMode', obj.pobj,mode);PetscCHKERRQ(err);
    end
    function FileSetName(obj,name)
      err = calllib('libpetsc', 'PetscViewerFileSetName', obj.pobj,name);PetscCHKERRQ(err);
    end
    function View(obj,viewer)
      err = calllib('libpetsc', 'PetscViewerView', obj.pobj,viewer.pobj);PetscCHKERRQ(err);
    end
    function Destroy(obj)
      err = calllib('libpetsc', 'PetscViewerDestroy', obj.pobj);PetscCHKERRQ(err);
    end
  end
end

 
