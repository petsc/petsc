class main {
     static public void main(String[] args) {
	 Petsc.InitializeNoArguments();
         Vec x = new Vec();
         x.SetFromOptions();
         Petsc.Finalize();
     };
 }
