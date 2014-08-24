#!/bin/bash

# Simple script to create the Makefile and build

# export PATH="$PATH:/usr/local/cuda/bin/"

make clean || echo clean

rm -f Makefile.in
rm -f config.status
./autogen.sh || echo done

CFLAGS="-O2" ./configure --with-mpir-src=../mpir-2.6.0

make
