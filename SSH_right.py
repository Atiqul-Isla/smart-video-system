import paramiko
import time



# Update the next three lines with your
# server's information

def connectRight(hst,usr,pwd):
    host = "10.0.0.83"
    username = "gurkaran"
    password = "gurkaran"

    client = paramiko.client.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=username, password=password)

    _stdin, _stdout,_stderr = client.exec_command("cd Desktop;python right.py")
    #print(_stdout.read().decode())
    print (_stdout.read().decode())
    client.close()
    client.close()