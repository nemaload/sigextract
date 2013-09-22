#!/bin/sh
#
# Convert a list of tab-separated coordinates to JSON format suitable
# e.g. for nemashow's consumption.
#
# Usage: tsv2json.sh [INPUT [OUTPUT]]

if [ -n "$1" ]; then
	exec <"$1"
fi
if [ -n "$2" ]; then
	exec >"$2"
elif [ -n "$1" ]; then
	exec >"${1%.tsv}-backbone.json"
fi

echo '{"bbpoints":['
delim=""
while read z y x d; do
	echo " $delim[$x,$y,$z,$d]"
	delim=,
done
echo ']}'
