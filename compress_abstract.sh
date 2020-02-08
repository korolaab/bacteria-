#!/bin/bash
if [ -z $1]
    then
    echo "No key found."
else
    $(mv ./abstract/*.pdf ./)
    $(tar -cf - abstract/ | xz -9 -c | openssl aes-256-cbc -salt -a -e -out abstract.enc -pass file:./$1)
fi

