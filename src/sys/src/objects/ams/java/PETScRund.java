/*$Id: PETScRund.java,v 1.6 2000/11/08 15:19:13 bsmith Exp $*/
/*
     Compiles and runs a PETSc program
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
  static final int MACHINE = 0,DIRECTORY = 1,MAXNP = 2,EXAMPLES = 3;

  public PETScRund() {
    try {
      systems = load();
    } catch (java.io.IOException e) {;}
  }
  
  public void start() throws java.io.IOException
  {
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
          return;
        }

        String     arch       = properties.getProperty("PETSC_ARCH");
        String     bopt       = properties.getProperty("BOPT");
        int        np         = Integer.parseInt(properties.getProperty("NUMBERPROCESSORS"));
        String     directory  = properties.getProperty("DIRECTORY");
        if (!systems[EXAMPLES].containsKey(directory)) { /* bad directory sent from client */
          System.out.println("Requested example directory "+directory+" that does not exist");
          return;
        }
        String     example    = properties.getProperty("EXAMPLE");
        ArrayList ex = (ArrayList) systems[EXAMPLES].get(directory);
        if (!ex.contains(example)) {
          System.out.println("Requested example "+example+" that does not exist");
          return;
        }
        String     command    = null,options = "";
        if (properties.getProperty("COMMAND").equals("make")) {
          command = "petscmake ";
        }
        if (properties.getProperty("COMMAND").equals("maketest")) {
          command = "petscmake ";
          example = "run"+example;
        } else {
	  /* make sure number of processors is reasonable */
	  int maxnp = Integer.parseInt((String)systems[MAXNP].get(arch));
	  if (maxnp < np) np = maxnp;
          command = "petscrun "+np+" ";
          if (properties.getProperty("OPTIONS") != null) {
	      options = properties.getProperty("OPTIONS");
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
      } catch (java.io.IOException ex) {System.out.println("bad");}
        catch (java.lang.ClassNotFoundException ex) {System.out.println("bad");} 
    }
  }

  /*
        Loads information about the available systems
  */
  public Hashtable[] load() throws java.io.IOException
  {
    LineNumberReader input = new LineNumberReader(new InputStreamReader(new FileInputStream("PETScRundrc")));
    String           edirectory = null;
    ArrayList        examples = null;
       
    systems = new Hashtable[4];
    systems[MACHINE]   = new Hashtable();
    systems[DIRECTORY] = new Hashtable();
    systems[MAXNP]     = new Hashtable();
    systems[EXAMPLES]  = new Hashtable();

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
      } else if (cnt == 2) { /* found a new directory */
        if (examples != null) {
          System.out.println("putting directory "+edirectory);
          systems[EXAMPLES].put(edirectory,examples);
        };
        examples = new ArrayList();
        edirectory = toke.nextToken();
        examples.add(toke.nextToken());
      } else if (cnt == 1) { /* found a new example */
        examples.add(toke.nextToken());
      }
    }
    if (examples != null) {
          System.out.println("putting directory "+edirectory);
      systems[EXAMPLES].put(edirectory,examples);
    } 
    return systems;
  }

  public static void main(String s[])
  {
    PETScRund prun = new PETScRund();
    try {
      prun.start();
    } catch (java.io.IOException ex) {;}
  }
}




