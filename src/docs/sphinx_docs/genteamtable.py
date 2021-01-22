#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 14:09:21 2020

@author: jacobfaibussowitsch
"""
import os
import requests

imDir = os.path.join("..", "_static", "images")

ownerURL = "https://gitlab.com/api/v4/groups/petsc/members/all"
integratorURL = "https://gitlab.com/api/v4/groups/5583565/members/all"
devURL = "https://gitlab.com/api/v4/groups/5981367/members/all"

def createDevDicts(imDirPath):
    emeritus = {
        # Keys (i.e. usernames) here are just lower full names with underscores. Not really
        # needed, just so that I don't have to write an extra "writeRst" function :)
        "william_gropp" : {
            "web_url" : "https://cs.illinois.edu/directory/profile/wgropp",
            "avatar_url" : os.path.join(imDirPath, "bill.gif"),
            "name" : "William Gropp"
        },
        "victor_eijkhout" : {
            "web_url" : "https://www.tacc.utexas.edu/staff/victor-eijkhout",
            "avatar_url" : os.path.join(imDirPath, "victor.jpg"),
            "name" : "Victor Eijkhout"
        },
        "peter_brune" : {
            "web_url" : "",
            "avatar_url" : os.path.join(imDirPath, "peter.jpg"),
            "name" : "Peter Brune"
        },
        "kris_buschelman" : {
            "web_url" : "",
            "avatar_url" : os.path.join(imDirPath, "buschelman.jpg"),
            "name" : "Kris Buschelman"
        },
        "sean_farley" : {
            "web_url" : "https://farley.io/",
            "avatar_url" : os.path.join(imDirPath, "sean.jpg"),
            "name" : "Sean Farley"
        },
        "dmitry_karpeev" : {
            "web_url" : "https://www.ci.uchicago.edu/profile/224",
            "avatar_url" : os.path.join(imDirPath, "dmitry.jpg"),
            "name" : "Dmitry Karpeev"
        },
        "dinesh_kaushik" : {
            "web_url" : "",
            "avatar_url" : os.path.join(imDirPath, "dinesh.jpg"),
            "name" : "Dinesh Kaushik"
        },
        "jason_sarich" : {
            "web_url" : "https://www.anl.gov/mcs/person/jason-sarich",
            "avatar_url" : os.path.join(imDirPath, "sarich.jpg"),
            "name" : "Jason Sarich"
        },
        "victor_minden" : {
            "web_url" : "",
            "avatar_url" : os.path.join(imDirPath, "victorminden.jpg"),
            "name" : "Victor Minden"
        }
    }
    # List of devs who will go into the table. Key must be all-lowercase Gitlab username.
    activeCoreDevs = {
        "lois.curfman.mcinnes" : {
            "web_url" : "https://press3.mcs.anl.gov/curfman/",
            "avatar_url" : os.path.join(imDirPath, "lois.gif"),
            "name" : "Lois Curfman McInnes"
        },
        "sbalay" : {
            "web_url" : None,
            "avatar_url" : None,
            "name" : None
        },
        "jedbrown" : {
            "web_url" : None,
            "avatar_url" : None,
            "name" : None
        },
        "adener" : {
            "web_url" : None,
            "avatar_url" : None,
            "name" : None
        },
        "blaisebourdin" : {
	    "web_url" : None,
	    "avatar_url" : None,
	    "name" : None
        },
        "dmay" : {
	    "web_url" : None,
	    "avatar_url" : os.path.join(imDirPath, "dave.jpg"),
	    "name" : None
        },
        "fdkong" : {
	    "web_url" : None,
	    "avatar_url" : None,
	    "name" : None
        },
        "ghammond" : {
	    "web_url" : None,
	    "avatar_url" : None,
	    "name" : None
        },
        "hannah_mairs" : {
	    "web_url" : None,
	    "avatar_url" : None,
	    "name" : None
        },
        "hongzhangsun" : {
	    "web_url" : None,
	    "avatar_url" : None,
	    "name" : None
        },
        "jfaibussowitsch" : {
	    "web_url" : None,
	    "avatar_url" : None,
	    "name" : None
        },
        "caidao22" : {
	    "web_url" : None,
	    "avatar_url" : None,
	    "name" : None
        },
        "jczhang07" : {
	    "web_url" : None,
	    "avatar_url" : None,
	    "name" : None
        },
        "karlrupp" : {
	    "web_url" : None,
	    "avatar_url" : None,
	    "name" : None
        },
        "markadams4" : {
	    "web_url" : None,
	    "avatar_url" : None,
	    "name" : "Mark Adams"
        },
        "knepley" : {
	    "web_url" : None,
	    "avatar_url" : None,
	    "name" : None
        },
        "oanam198" : {
	    "web_url" : None,
	    "avatar_url" : None,
	    "name" : None
        },
        "psanan" : {
	    "web_url" : None,
	    "avatar_url" : None,
	    "name" : None
        },
        "rtmills" : {
	    "web_url" : None,
	    "avatar_url" : None,
	    "name" : None
        },
        "abhyshr" : {
	    "web_url" : None,
	    "avatar_url" : None,
	    "name" : None
        },
        "stefanozampini" : {
	    "web_url" : None,
	    "avatar_url" : None,
	    "name" : None
        },
        "tmunson" : {
	    "web_url" : None,
	    "avatar_url" : os.path.join(imDirPath, "todd.jpg"),
	    "name" : None
        },
        "haplav" : {
	    "web_url" : None,
	    "avatar_url" : None,
	    "name" : None
        },
        "prj-" : {
	    "web_url" : None,
	    "avatar_url" : None,
	    "name" : None
        },
        "wence" : {
	    "web_url" : None,
	    "avatar_url" : None,
	    "name" : None
        },
        "tisaac" : {
	    "web_url" : None,
	    "avatar_url" : None,
	    "name" : None
        },
        "krugers" : {
	    "web_url" : None,
	    "avatar_url" : None,
	    "name" : None
        },
        "dalcinl" : {
	    "web_url" : None,
	    "avatar_url" : None,
	    "name" : None
        },
        "joseroman" : {
	    "web_url" : None,
	    "avatar_url" : None,
	    "name" : None
        },
        "bwhitchurch" : {
	    "web_url" : None,
	    "avatar_url" : None,
	    "name" : None
        },
        "barrysmith" : {
	    "web_url" : None,
	    "avatar_url" : None,
	    "name" : None
        }
    }
    return emeritus, activeCoreDevs

def getJson(url, token):
    headers = {"PRIVATE-TOKEN" : token}
    params = {"per_page" : 100, "page" : 1}
    rpage = requests.get(url, headers = headers, params = params)
    rpage.raise_for_status()
    numPages = int(rpage.headers["X-Total-Pages"])
    lst = []
    for i in range(numPages):
        r = requests.get(url, headers = headers, params = params)
        r.raise_for_status()
        lst.extend(r.json())
        params["page"] += 1
    return lst

def makeActiveDevDict(devJson, activeCoreDevs):
    for dev in devJson:
        ldev = dev["username"].lower()
        if ldev in activeCoreDevs:
            for key, val in activeCoreDevs[ldev].items():
                if val is None:
                    activeCoreDevs[ldev][key] = dev[key]
    return activeCoreDevs

def writeRst(fname, devs):
    with open(fname, "w+") as f:
        lines = [".. raw:: html\n\n",
                 "   <!-- Generated by %s -->\n" % (__file__),
                 "   <div class=\"petsc-team-container\">\n",
                 "   <style>\n",
                 "     img.avatar {border-radius: 10px;width: 60px;height: 60px;}\n",
                 "   </style>\n"]
        for dev in sorted(devs.items(), key = lambda item: item[1]["name"].split(" ")[-1]):
            lines.append("    <div>\n")
            lines.append("    <a href='%s'><img src='%s' class='avatar' /></a> <br />\n" %
                         (dev[1]["web_url"], dev[1]["avatar_url"]))
            lines.append("    <p>%s</p>\n" % (dev[1]["name"]))
            lines.append("    </div>\n")
        lines.append("    </div>\n")
        f.writelines(lines)
    print("Wrote table to "+fname)

def writeWarnRst(fname):
    import warnings
    with open(fname, "w+") as f:
        f.writelines([".. warning::\n\n",
                      "   ``$PETSC_GITLAB_PRIVATE_TOKEN`` was not defined in your environment. Please generate a gitlab private token and export it to your environment. See https://docs.gitlab.com/ee/user/profile/personal_access_tokens.html for more information."])

def main(writeDirPath, token, builderName=None):
    print("Running from %s" % (os.path.realpath(__file__)))
    try:
        os.mkdir(writeDirPath)
        print("Generate directory created at %s" % (writeDirPath))
    except OSError as e:
        import errno
        if e.errno != errno.EEXIST:
            raise
        else:
            print("Generate directory already exists at %s" % (writeDirPath))
            print("Assuming table is up to date, skipping! Use 'make clean' to clear existing table")
            return

    currentFile = os.path.join(writeDirPath, "petsc-team-table.inc")
    emeritusFile = os.path.join(writeDirPath, "petsc-emeritus-table.inc")
    if "CI_JOB_TOKEN" in os.environ:
        token = os.environ["CI_JOB_TOKEN"]
        print("Using CI_JOB_TOKEN as auth token")
    if token is None and "READTHEDOCS" not in os.environ:
        writeWarnRst(currentFile)
        writeWarnRst(emeritusFile)
    else:
        devJson = getJson(devURL, token)
        ownerJson = getJson(ownerURL, token)
        integratorJson = getJson(integratorURL, token)
        megaJson = devJson+ownerJson+integratorJson
        if builderName is not None:
            # dirhtml makes it so every rst file is built as __file__/index.html, so we must
            # prepend ".." so image paths are correct
            print("Using builder %s" % (builderName))
            if builderName == "dirhtml":
                global imDir
                imDir = os.path.join("..", imDir)
        print("Image directory (relative to contact/petsc_team.rst) at %s" % (imDir))
        emeritus, activeCoreDevs = createDevDicts(imDir)
        tierlist = makeActiveDevDict(megaJson, activeCoreDevs)
        writeRst(currentFile, tierlist)
        writeRst(emeritusFile, emeritus)

if __name__ == "__main__":
    import argparse
    import pathlib

    string = "<string>"
    path = "<path/to/dir>"
    parser = argparse.ArgumentParser(description = "Build Developer Table", formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-t", "--gitlab-token", required = True, metavar = string, help = "Specify your private Gitlab Authentican Token", dest = "token")
    parser.add_argument("-o", "--output-dir", required = True, metavar = path, type = pathlib.Path, help = "Specify the output directory", dest = "writeDir")
    args = parser.parse_args()
    writeDirPath = os.path.realpath(args.writeDir)
    print("============================================")
    print("  GENERATING TEAM TABLE FROM COMMAND LINE   ")
    print("============================================")
    main(writeDirPath, args.token)
