#!/bin/sh

for f in `ls | grep $1`; do 
	echo $f; 
	cat $f | grep -E -o "roc_auc.{0,21}" | cut -d' ' -f2; 
done
