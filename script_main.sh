#!/bin/bash

EXP="BraninMultiple"
COSTS="0.2"
METHODS="SF-MES"
COND_VAR="0.1"
COND_IG="0.1"
N_REP=100
BUDGET=5
SEED=3000
RESULTFOLDER="results"
MOGP="lineartruncated"
NAME="exp_branmul_lt_sfmes"
VERBOSE=0


echo -e "FUNCTIONS=$EXP\nCOSTS=$COSTS\nMETHODS=$METHODS\nCONDITION_VAR=$COND_VAR\nCONDITION_IG=$COND_IG\nN_REPS=$N_REP\nBUDGET=$BUDGET\nSEED=$SEED" > "$RESULTFOLDER/config_$NAME.txt"
if [[ $EXP == *"Multiple"* ]]
    then
        # cost parameter has no impact in multiple case, it is hardcoded to 0.2 for all AIS and 1.0 for target!
        python3 main_multiple_debug.py -n $N_REP -b $BUDGET -s "$RESULTFOLDER/$NAME" -se $SEED -m $METHODS -v $VERBOSE -cv $COND_VAR -ci $COND_IG -co $COSTS -e $EXP -j $MOGP
    else
        python3 main.py -n $N_REP -b $BUDGET -s "$RESULTFOLDER/$NAME" -se $SEED -m $METHODS -v $VERBOSE -cv $COND_VAR -ci $COND_IG -co $COSTS -e $EXP -j $MOGP
fi