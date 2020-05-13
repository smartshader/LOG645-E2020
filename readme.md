# Repository for LOG645 labs

#### Setting up permanent SSH auth between your personal computer and ETS' server
Creating a permanent SSH auth link between your workstation and the server will make it easier to run your own custom .sh scripts without having to constantly log into the remote server manually each time.
1. Launch `Git Bash` command line
1. (If you haven't generated an SSH key for Git) 
    * `ssh-keygen -t rsa -b 4096`
1. ssh copy to the server (it'll prompt for password after this which you only have to enter **once**)
    * `ssh-copy-id AJxxxxx@log645-srv-1.logti.etsmtl.ca`
1. Test logging into the server with 
    * `ssh AJxxxxx@log645-srv-1.logti.etsmtl.ca`
    
#### Lab 1 - Intro to MPI scripts
To make it easier to transfer changes from local PC-> remote server, utilize the `prepXXX.sh` script.
#### **Instructions**
1. Launch `Git Bash` command line
1. change directory to `/l1/` (for Lab1)
1. For **Sequential**, execute `./prepSeq.sh AAXXXXX` (AAXXXXX is your studentID (all caps))
1. For **Parallel**, execute `./prepPar.sh AAXXXXX`
