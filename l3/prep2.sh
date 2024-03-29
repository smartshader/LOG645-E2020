## this script transfers the necessary files to the ETS server and executes run.sh ##

## to execute, run the script by doing ./prepSeq.sh $1 ##
## where $1 is your student ID in caps ##
## EXAMPLE : ./prepSeq.sh AJ12345 ##

## make sure your SSH keys are loaded on the server first ##
## generate ssh key (if you haven't done it yet) ##
##          ssh-keygen -t rsa -b 4096 ##
## ssh copy to the server (it'll prompt for a PW later) ##
##          ssh-copy-id AJ12345@log645-srv-1.logti.etsmtl.ca ##

## removes trailings r from scripts ##
sed -i""  's/\r$//' run2.sh
sed -i""  's/\r$//' prep2.sh

## variables ##
_mydir="$(pwd)"
_studentID=$1
_serverID="log645-srv-1.logti.etsmtl.ca"
# _studentID=howardphieu
# _serverID="35.193.40.96"
chmod 1744 *

echo "Current working directory: $PWD"
echo ">>> [SERVER] : deleting old contents"
## replace w/ your student ID ##
ssh $_studentID@$_serverID 'rm -r src/ Makefile run2.sh'
echo ">>> [SERVER] : transferring new files"
echo ''
scp -p -r $_mydir'/src' $_mydir'/Makefile' $_mydir'/run2.sh' $_studentID@$_serverID:
ssh $_studentID@$_serverID 'chmod +x run2.sh'
ssh $_studentID@$_serverID './run2.sh'
echo ''
echo ">>> [shell process complete]"
echo ''
