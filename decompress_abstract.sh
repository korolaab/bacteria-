#!/bin/bash
if [ -z $1 ]
    then
    echo "No key found"
else
    $(openssl aes-256-cbc -salt -a -d -in abstract.enc  -pass file:./$1 | tar -xJf -)
fi

