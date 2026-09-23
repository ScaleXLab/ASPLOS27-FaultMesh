#!/bin/bash
cd inputgen
chmod +x graphgen
./graphgen 25164824 24M
mv graph24M.txt ../
