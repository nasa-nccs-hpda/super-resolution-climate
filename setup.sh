#!/bin/bash
cd notebooks;  ln -s ../sres ./sres
cd ../tests;   ln -s ../sres ./sres
cd ../scripts/inference; ln -s ../../sres ./sres
cd ../scripts/train; ln -s ../../sres ./sres
