classdef PetscViewer < PetscObject
  methods
    function obj = PetscViewer()
      [err,obj.pobj] = calllib('libpetsc', 'PetscViewerCreate', 0,0);
    end
    function SetType(obj,name)
      err = calllib('libpetsc', 'PetscViewerSetType', obj.pobj,name);
    end
    function View(obj,viewer)
      err = calllib('libpetsc', 'PetscViewerView', obj.pobj,viewer.pobj);
    end
    function Destroy(obj)
      err = calllib('libpetsc', 'PetscViewerDestroy', obj.pobj);
    end
  end
end

 
