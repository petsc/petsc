/*$Id: PETScRun.java,v 1.16 2001/02/15 19:02:41 bsmith Exp $*/
/*
     Compiles and runs a PETSc program
*/

import java.lang.Thread;
import java.io.*;
import java.net.*;
import java.util.*;

import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import java.awt.print.*;

public class PETScRun extends JApplet implements Pageable, Printable
{
  static final int MACHINE = 0,DIRECTORY = 1,MAXNP = 2,EXAMPLES = 3,EXAMPLESHELP = 4;
  Hashtable        systems[];
  Socket csock = null;

  java.applet.AppletContext appletcontext;
  PETScRun applet;

  JPanel     tpanel;
  JTextField options,help;

  JCheckBox toptions;
  JCheckBox tbopt;
  JCheckBox snoop;

    JComboBox    server;
    JComboBox    arch;
    JComboBox    dir;
    JComboBox    example;
    JComboBox    np;

  JTextArea   opanel,epanel;

    Container japplet;

  String servername;

  public void init() {
    appletcontext = getAppletContext();
    applet        = this;
    japplet       = this.getContentPane();

    System.out.println("Codebase"+this.getDocumentBase());
    setserver(null);
  }

  public void setserver(String servern   ) { 
    try {
      if (servern == null) servername = this.getParameter("server");
      else servername = servern;

      System.out.println("parameter"+servername);
      Socket sock = null;
      try {
        sock = new Socket(servername,2000);
      } catch(java.net.ConnectException oops) {
        getserver();
        return;
      }
      sock.setSoLinger(true,5);

      /* construct properties to send to server */
      Properties   properties = new Properties();
      properties.setProperty("Loadsystems","Yes");
      (new ObjectOutputStream(sock.getOutputStream())).writeObject(properties);

      ObjectInputStream os = new ObjectInputStream(sock.getInputStream());
      systems                = new Hashtable[5];
      systems[MACHINE]       = (Hashtable) os.readObject();
      systems[DIRECTORY]     = (Hashtable) os.readObject();
      systems[MAXNP]         = (Hashtable) os.readObject();
      systems[EXAMPLES]      = (Hashtable) os.readObject();
      systems[EXAMPLESHELP]  = (Hashtable) os.readObject();
      sock.close();
    } catch (java.io.IOException ex) {ex.printStackTrace(); System.out.println("no systems");}
      catch (ClassNotFoundException ex) {    System.out.println("no class");}
 
    japplet.removeAll();
    japplet.setVisible(false);

    japplet.setLayout(new FlowLayout());

    tpanel = new JPanel(new GridLayout(3,4));
    japplet.add(tpanel, BorderLayout.NORTH);
      
    help = new JTextField(50);
    japplet.add(help, BorderLayout.NORTH);

    JPanel grid = new JPanel(new GridBagLayout());
    japplet.add(grid,BorderLayout.NORTH);
    options = new JTextField(50);
    grid.add(new JLabel("Options"));
    grid.add(options);

    server = new JComboBox();
    server.addItem(servername);
    server.setEditable(true);
    server.addItemListener(new ItemListener() {
                          public void itemStateChanged(ItemEvent e) {
                            if (e.getStateChange() == ItemEvent.SELECTED) {
                              final JComboBox choice = (JComboBox) e.getItemSelectable();
          System.out.println("slected server"+(String)choice.getSelectedItem());

                              (new Thread()  {
                                public void run() {
                                  setserver((String)choice.getSelectedItem());
                                }
                              }).start();
                            }}
			   });
    tpanel.add(server);

    arch = new JComboBox();
    Enumeration keys = systems[MAXNP].keys();
    while (keys.hasMoreElements()) {
      arch.addItem((String)keys.nextElement());
    }
    arch.addItemListener(new ItemListener() {
                          public void itemStateChanged(ItemEvent e) {
                            if (e.getStateChange() == ItemEvent.SELECTED) {
                              JComboBox choice = (JComboBox) e.getItemSelectable();
          System.out.println("set np"+(String)choice.getSelectedItem());
                              setnp(np,(String)choice.getSelectedItem());}}
	                    }); 
    tpanel.add(arch);
        
    dir = new JComboBox();
    String pexampled = this.getParameter("EXAMPLEDIRECTORY");
    if (pexampled == null) {
      keys = systems[EXAMPLES].keys();
      while (keys.hasMoreElements()) {
        dir.addItem((String)keys.nextElement());
      }
      dir.addItemListener(new ItemListener() {
                          public void itemStateChanged(ItemEvent e) {
                            if (e.getStateChange() == ItemEvent.SELECTED) {
                              JComboBox choice = (JComboBox) e.getItemSelectable();
                              setexamples(example,(String)choice.getSelectedItem());
                            }
                          }
	                });
    } else { 
      dir.addItem((String)pexampled);
    }
    tpanel.add(dir);

    String phelp = this.getParameter("HELP"); 
    if (phelp != null) {
      this.help.setText(phelp);
    }

    np = new JComboBox();
    setnp(np,(String)systems[MAXNP].keys().nextElement());
    tpanel.add(np);

    example = new JComboBox();
    String pexample = this.getParameter("EXAMPLE");
    if (pexample == null) {
      setexamples(example,(String)systems[EXAMPLES].keys().nextElement());
      example.addItemListener(new ItemListener() {
                            public void itemStateChanged(ItemEvent e) {
                              if (e.getStateChange() == ItemEvent.SELECTED) {
                                JComboBox choice = (JComboBox) e.getItemSelectable();
                                String directory = (String)dir.getSelectedItem();
                                ArrayList ehelp = (ArrayList)systems[EXAMPLESHELP].get(directory);
                                ArrayList ex = (ArrayList)systems[EXAMPLES].get(directory);
                                int index = ex.indexOf(choice.getSelectedItem());
                                help.setText((String)ehelp.get(index));
                              }
			    }
	                  });
    } else {  
      example.addItem((String)pexample);
    }
    tpanel.add(example);


    JButton mbutton = new JButton("Compile");
    tpanel.add(mbutton);
    mbutton.addActionListener(new ActionListener(){
      public void actionPerformed(ActionEvent e) { 
        runprogram("make");
      }
    }); 

    JButton tbutton = new JButton("Test");
    tpanel.add(tbutton);
    tbutton.addActionListener(new ActionListener(){
      public void actionPerformed(ActionEvent e) { 
        runprogram("maketest");
      }
    }); 

    JButton rbutton = new JButton("Run");
    tpanel.add(rbutton);
    rbutton.addActionListener(new ActionListener(){
      public void actionPerformed(ActionEvent e) { 
        runprogram("mpirun");
      }
    }); 

    /*
          Opens new window that displays the source 
    */
    JButton sbutton = new JButton("View source");
    tpanel.add(sbutton);
    sbutton.addActionListener(new ActionListener(){
      public void actionPerformed(ActionEvent e) { 
        String ex = (String)example.getSelectedItem();
        URL    urlb = applet.getDocumentBase();

        try {
          String s = null;
          if (ex.endsWith("f90") || ex.endsWith("f")) s = ".F";
          else s = ".c";
          final URL url = new URL(""+urlb+"../"+dir.getSelectedItem()+"/"+ex+s+".html");  
          System.out.println("showing"+url);
          appletcontext.showDocument(url,"Source");
        } catch (MalformedURLException oops) { System.out.println("bad:showing"+urlb);;} 
      }
    }); 

    toptions = new JCheckBox("Set options graphically");
    tpanel.add(toptions);
    snoop = new JCheckBox("Snoop on program");
    tpanel.add(snoop);
    tbopt = new JCheckBox("Compile debug version");
    tpanel.add(tbopt);


    epanel = new JTextArea(4,60);
    japplet.add(new JScrollPane(epanel), BorderLayout.NORTH); 
    epanel.setLineWrap(true);
    epanel.setWrapStyleWord(true);

    opanel = new JTextArea(20,60);
    japplet.add(new JScrollPane(opanel), BorderLayout.NORTH); 
    opanel.setLineWrap(true);
    opanel.setWrapStyleWord(true);

    japplet.setVisible(true);
    japplet.validate(); 
    japplet.repaint(); 
  }

  /*
      Fills in the examples pull down menu based on the directory selected
  */
  public void setexamples(JComboBox example,String dir) {
    ArrayList ex = (ArrayList)systems[EXAMPLES].get(dir);
    Iterator its = ex.iterator();
    example.removeAllItems();
    while (its.hasNext()) {
      example.addItem(""+its.next());
    }
    ArrayList ehelp = (ArrayList)systems[EXAMPLESHELP].get(dir);
    this.help.setText((String)ehelp.get(0));
  }

  /*
     Fills in the number processors pull down menu based on the arch selected
  */
  public void setnp(JComboBox np,String arch){
    int onp = Integer.parseInt((String)systems[MAXNP].get(arch)),i;
    np.removeAllItems();
    for (i=1; i<=onp;i++) {
      np.addItem(""+i);
    }
  }

  public void stop() {
    System.out.println("Called stop");
  }

  public void runprogram(String what)
  {
    try {
      csock = new Socket(servername,2000);
      final Socket sock = csock;
      sock.setSoLinger(true,5);
      final InputStream sstream = sock.getInputStream();

      /* construct properties to send to server */
      final Properties   properties = new Properties();
      properties.setProperty("PETSC_ARCH",(String)arch.getSelectedItem());
      properties.setProperty("DIRECTORY",(String)dir.getSelectedItem());
      properties.setProperty("EXAMPLE",(String)example.getSelectedItem());
      properties.setProperty("NUMBERPROCESSORS",(String)np.getSelectedItem());
      properties.setProperty("COMMAND",what);
      if (tbopt.isSelected()) {
        properties.setProperty("BOPT","g");
      } else {
        properties.setProperty("BOPT","O");
      }
      String soptions = options.getText();
      if (toptions.isSelected() && what.equals("mpirun")) {
	URL urlb = this.getDocumentBase();
        soptions += " -ams_publish_options";
	try {
	  final URL url = new URL(""+urlb+"PETScOptions.html");  
          System.out.println("loading url"+url);
          (new Thread() {
            public void run() {
              System.out.println("open new");
              appletcontext.showDocument(url,"AMSOptions");
              System.out.println("done open new");
            }
	  }).start();
        } catch (MalformedURLException ex) {System.out.println("bad luck");;} 
      } 
      if (snoop.isSelected() && what.equals("mpirun")) {
	URL urlb = this.getDocumentBase();
        soptions += " -ams_publish_objects";
	try {
	  final URL url = new URL(""+urlb+"PETScView.html");  
          System.out.println("loading url"+url);
          (new Thread() {
            public void run() {
              System.out.println("open new");
              appletcontext.showDocument(url,"PETScView");
              System.out.println("done open new");
            }
	  }).start();
        } catch (MalformedURLException ex) {System.out.println("bad luck");;} 
      } 
      properties.setProperty("OPTIONS",soptions);
      System.out.println("getting back");

      /* get output and print to screen */
      (new Thread() {
        public void run() {
          try {
            (new ObjectOutputStream(sock.getOutputStream())).writeObject(properties);

            InputStreamReader stream = new InputStreamReader(sstream);
            char[] results = new char[128];
            int    fd      = 0,cnt = 0;

            opanel.setText(null);
            while (true) {
              fd  = stream.read(results);
              if (fd == -1) {break;}
              opanel.append(new String(results,0,fd));
	      cnt += fd;
            }
          } catch (java.io.IOException ex) {;}
          opanel.append("----DONE-----");
          donerun();
        }
      }).start();

      /* get stderr and popup */
      (new Thread() {
        public void run() {
          try {
            Socket nsock = new Socket(servername,2002);
            nsock.setSoLinger(true,5);
            InputStreamReader stream = new InputStreamReader(nsock.getInputStream());
            char[] results = new char[128];
            int    fd      = 0;

            epanel.setText(null);
            while (true) {
              fd  = stream.read(results);
              if (fd == -1) {break;}
              System.out.println("PETScRun:got stderr output"+results);
              epanel.append(new String(results,0,fd));
            }
            stream.close();
            nsock.close();
          } catch (java.io.IOException ex) {System.out.println("PETScRun: error getting stderr");}
          System.out.println("PETScRun: finished checking for stderr");
        }
      }).start();

    } catch (java.io.IOException ex) {;}
  }


  public void donerun() 
  {
    /* Uncomment this to print the applet screen dump
    PrinterJob job = PrinterJob.getPrinterJob();
    format = job.pageDialog(job.defaultPage());
    job.setPageable(applet);
    job.printDialog();
    try {
      job.print();
    } catch (java.awt.print.PrinterException e) {System.out.println("problem printing");}
    */
  }

  PageFormat        format;
  public int        getNumberOfPages()         {return 1;}
  public PageFormat getPageFormat(int pagenum) {return format;}
  public Printable  getPrintable(int pagenum)  {return this;}

  public int print(Graphics g,PageFormat format,int pagenum) {
    Graphics2D g2 = (Graphics2D) g;
    g2.translate(format.getImageableX(),format.getImageableY());
    g2.scale(.6,.6);
    japplet.printComponents(g2);
    return Printable.PAGE_EXISTS;
  }

  public void getserver() { /* ------------------------------------------*/

    japplet.removeAll();
    japplet.setVisible(false);
    /*

         Make GUI to get host and port number from user 
    */
    japplet.setLayout(new FlowLayout());
        
    tpanel = new JPanel(new GridLayout(3,1));
    japplet.add(tpanel, BorderLayout.NORTH);
        
    final JTextField inputserver = new JTextField(servername,32);
    JPanelSimplePack text = new JPanelSimplePack("AMS Client machine",inputserver);
    tpanel.add(text);
    JTextField inputport = new JTextField(2000+"",8);
    text = new JPanelSimplePack("AMS Client port",inputport);
    tpanel.add(text);
    System.out.println("put up server and port");
    
    /*--------------------- */
    JButton button = new JButton("Continue");
    tpanel.add(button);
    button.addActionListener(new ActionListener(){
      public void actionPerformed(ActionEvent e) { 
        System.out.println("User selected continue");
        setserver(inputserver.getText());
      }
    }); 
    System.out.println("put up continue");

    japplet.setVisible(true);
    japplet.validate(); 
    japplet.repaint(); 
    System.out.println("put up continue done");
    return;
  }

  public class JPanelSimplePack extends JPanel { /*-----------------------------------*/
    public JPanelSimplePack(String text,Component c1) {
      super( new GridBagLayout());
      add(new JLabel(text));
      add(c1);
    }
  }

}




