#!/bin/bash

set -e

conda env export -p conda | awk -F= '
    BEGIN {
        mode = "chn";
    }
    /^(prefix|name):/ {
        next;
    }
    /^dependencies:$/ {
        mode = "dep";
        for (i = chn_idx - 1; i >= 0; ) {
            print chn_arr[i--];
        }
    }
    /- (cudatoolkit|cudnn|scglue)/ {
        print "# "$0;
        next;
    }
    /^[^=]+=[^=]+=[^=]+$/ {
        print $1"="$2;
        next;
    }
    /- .+$/ {
        if (mode == "chn") {
            chn_arr[chn_idx++] = $0;
            next;
        }
    }
    {
        print $0;
    }
' > env.yaml
