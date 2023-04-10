#!/bin/bash

CONFIGFILE=$(realpath $1)
OLDPWD=$(pwd)

BFROOT="/research"
SCRIPTSPATH="/research/sharedresources/cbi/public/bigdata/bioformats-io/shell_scripts"
SIFPATH="/research/sharedresources/cbi/public/bigdata/bioformats-io/bioimage-io.sif"

cd "$SCRIPTSPATH"

echo $CONFIGFILE

ONCLUSTER=True
( bhosts 2> /dev/null ) || export ONCLUSTER=False
echo "ONCLUSTER=$ONCLUSTER"

if [ $ONCLUSTER ]
then
    module load jq
fi

FILENAME=$(jq -r '.FILENAME' $CONFIGFILE)
DESTDIR=$(jq -r '.DESTDIR // empty' $CONFIGFILE)

cd "/research/sharedresources/cbi/public/bigdata/bioformats-io"

BASEFUN="singularity exec --bind $BFROOT $SIFPATH python bdv2ims.py $FILENAME $DESTDIR"
job="bf2ims_$(printf '%x' $(date +%s%N))_$i"

if [ $ONCLUSTER ]
then
    outdir=$( jq -r '.BSUB_OUT // empty' $CONFIGFILE)
    n=$( jq '.BSUB_N // 4' $CONFIGFILE )
    q=$( jq -r '.BSUB_Q // "standard"' $CONFIGFILE )
    mem=$( jq -r '.BSUB_MEM // "128MB"' $CONFIGFILE )

    if [[ -z "$outdir" ]]; then outdir="$(dirname $FILENAME)"; fi
    outdir="$outdir/bf2ims_log"
    mkdir -p $outdir

    bsub -J $job -q $q -n $n -R'rusage[mem='"$mem"']' -o "$outdir"'/'"$job.out" -e "$outdir"'/'"$job.err" $BASEFUN
else
    eval $BASEFUN
fi