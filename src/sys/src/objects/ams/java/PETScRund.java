/*$Id: PETScRund.java,v 1.8 2001/02/15 17:17:51 bsmith Exp $*/
/*
     Demon that serves requests to compile and run a PETSc program
*/

import java.lang.Thread;
import java.io.*;
import java.net.*;
import java.util.*;

/*
    This is the class that this file implements (must always be the same as
  the filename.
*/
public class PETScRund 
{
  Hashtable        systems[];
  static final int MACHINE = 0,DIRECTORY = 1,MAXNP = 2,EXAMPLES = 3, EXAMPLESHELP = 4;
  String           LOC = null;

  public PETScRund() {
    try {
      systems = load();
    } catch (java.io.IOException e) {;}
  }
  
  public void start(String s[]) throws java.io.IOException
  {
    if (s.length == 1) LOC = s[0];

    ServerSocket serve = new ServerSocket(2000);
    while (true) {
      Socket sock  = serve.accept();
      sock.setSoLinger(true,5);
      (new ThreadRunProgram(sock)).start();
    }
  }

  /*
          This handles a single clients request
  */
  class ThreadRunProgram extends Thread
  {
    Socket sock;
    public ThreadRunProgram(Socket s) {sock = s;}
    public void run() {
      try {
        Properties properties = (Properties) (new ObjectInputStream(sock.getInputStream())).readObject();
        if (properties.getProperty("Loadsystems") != null) {
  	  ObjectOutputStream os = new ObjectOutputStream(sock.getOutputStream());
          os.writeObject(systems[MACHINE]);
          os.writeObject(systems[DIRECTORY]);
          os.writeObject(systems[MAXNP]);
          os.writeObject(systems[EXAMPLES]);
          os.writeObject(systems[EXAMPLESHELP]);
          return;
        }

        String     arch       = properties.getProperty("PETSC_ARCH");
        String     bopt       = properties.getProperty("BOPT");
        int        np         = Integer.parseInt(properties.getProperty("NUMBERPROCESSORS"));
        String     directory  = properties.getProperty("DIRECTORY");

        /*
        if (!systems[EXAMPLES].containsKey(directory)) { 
          System.out.println("Requested example directory "+directory+" that does not exist");
          return;
	  } */
        String     example    = properties.getProperty("EXAMPLE");
        ArrayList  ex = (ArrayList) systems[EXAMPLES].get(directory);
        if (!ex.contains(example)) {
          System.out.println("Requested example "+example+" that does not exist");
          return;
        }

        String     command    = null,options = "";
        if (properties.getProperty("COMMAND").equals("make")) {
          command = "petscmake ";
        } else if (properties.getProperty("COMMAND").equals("maketest")) {
          command = "petscmake ";
          example = "run"+example;
        } else {
	  /* make sure number of processors is reasonable */
	  int maxnp = Integer.parseInt((String)systems[MAXNP].get(arch));
	  if (maxnp < np) np = maxnp;
          command = "petscrun "+np+" ";
          if (properties.getProperty("OPTIONS") != null) {
	      options += properties.getProperty("OPTIONS");
          }
          bopt = "";
        }
        command += systems[MACHINE].get(arch)+" "+
                   arch+" "+
                   bopt+" "+
                   systems[DIRECTORY].get(arch)+"/ "+
                   directory+" "+
                   example+" "+
                   options;
        System.out.println("Running command:"+command);
        Process make = Runtime.getRuntime().exec(command);
        PumpStream pump = new PumpStream(make,sock.getOutputStream());
        int len = pump.Pump();
        System.out.println("PETScRund: Done returning output, length = "+len);
      } catch (java.io.IOException ex) {System.out.println("PETScRund: bad running program");}
        catch (java.lang.ClassNotFoundException ex) {System.out.println("PETScRund: bad class running program");} 
    }
  }

  /*
        Loads information about the available systems
  */
  public Hashtable[] load() throws java.io.IOException
  {
    String           edirectory = null;
    ArrayList        examples = null,exampleshelp = null;
       
    systems = new Hashtable[5];
    systems[MACHINE]       = new Hashtable();
    systems[DIRECTORY]     = new Hashtable();
    systems[MAXNP]         = new Hashtable();
    systems[EXAMPLES]      = new Hashtable();
    systems[EXAMPLESHELP]  = new Hashtable();

    LineNumberReader input = new LineNumberReader(new InputStreamReader(new FileInputStream("PETScRun.systems")));
    String line = null;
    while (true) {
      try {
        line = input.readLine();
      } catch (java.io.IOException e) {break;}
      if (line == null) break;
      if (line.length() == 0) continue;
      if (line.charAt(0) == '#') continue;
      System.out.println(line);
      java.util.StringTokenizer toke = new java.util.StringTokenizer(line.trim());
      int cnt = toke.countTokens();
      if (cnt == 4) { /* found a new machine arch */
	String arch = toke.nextToken();
        systems[MACHINE].put(arch,toke.nextToken());
        systems[DIRECTORY].put(arch,toke.nextToken());
        systems[MAXNP].put(arch,toke.nextToken());
      }
    }

    input = new LineNumberReader(new InputStreamReader(new FileInputStream("PETScRun.examples")));
    line = null;
    String dir = null,sec;
    while (true) {
      try {
        line = input.readLine();
      } catch (java.io.IOException e) {break;}
      if (line == null) break;
      if (line.length() == 0) continue;
      if (line.charAt(0) == '#') continue;
      System.out.println(line);
      if (line.startsWith("src")) {
        /* save information from previous directory */
        if (dir != null) {
          systems[EXAMPLES].put(dir,examples);
          systems[EXAMPLESHELP].put(dir,exampleshelp);
          dir = null;
        }
        /* we have a new directory */
	int i = line.indexOf(' ');
        dir = line.substring(0,i-1);
        sec = line.substring(i+1);
        examples     = new ArrayList();
        exampleshelp = new ArrayList();
      } else {
          /* we have a new example */
        examples.add(line.substring(0,line.indexOf('.')));
	int i = line.indexOf(' ');
        if (i > 0) {
          sec = line.substring(i+1);
        } else {
	  sec = null;
        }
        exampleshelp.add(sec);
      }
    }
    if (dir != null) {
      systems[EXAMPLES].put(dir,examples);
      systems[EXAMPLESHELP].put(dir,exampleshelp);
    }
    return systems;
  }

  public static void main(String[] args)
  {
    PETScRund prun = new PETScRund();
    try {
      prun.start(args);
    } catch (java.io.IOException ex) {;}
  }
}








