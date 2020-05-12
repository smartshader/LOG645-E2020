Repository for LOG645 labs

# Setting up SSH between your personal computer and ETS' server
1. (If you haven't generated an SSH key for Git) ssh-keygen -t rsa -b 4096
1. ssh copy to the server (it'll prompt for PW after this which you only have to enter once)
 . ssh-copy-id AJxxxxx@log645-srv-1.logti.etsmtl.ca
1. Test logging into the server with ssh AJxxxxx@log645-srv-1.logti.etsmtl.ca