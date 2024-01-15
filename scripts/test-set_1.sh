#!/bin/sh
set -x

bleu -vr ./data/set_1/ref_1_MTmedicine.txt -h data/set_1/hyp_1_1_MTmedicineyandex.txt -h ./data/set_1/hyp_1_2_MTmedicinedeep.txt
