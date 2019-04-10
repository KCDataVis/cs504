#!/bin/bash
mkdir -p merged
cd merged
files=$(ls ../*.csv)
echo "date,name,symbol,sector,open,high,low,close,volume" > all_stocks_5yr.csv
for file in $files
do
	tail -n +2 $file >> all_stocks_5yr.csv
done
