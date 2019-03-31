#!/bin/bash
echo "date,name, symbol,industry,open,high,low,close,volume" > all_stocks_5yr.csv
mkdir -p merged
cd merged
files=$(ls ../*.csv)
for file in $files
do
	tail -n +2 $file >> all_stocks_5yr.csv
done
