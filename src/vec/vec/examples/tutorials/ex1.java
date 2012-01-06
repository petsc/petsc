class main {
     static public void main(String[] args) {
	 Petsc.InitializeNoArguments();
         Vec x = new Vec();
         PetscViewer v = new PetscViewer();
         x.SetSizes(2,2);
         x.SetFromOptions();
         x.Set(2.0);
         v.SetType("ascii");
         x.View(v);
         Petsc.Finalize();
     };
 }
