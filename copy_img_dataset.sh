#!/bin/bash
# change p to i or o
iter1=("0.01_exponential_h" "0.01_exponential_hv" "0.01_exponential_o" "0.01_exponential_rl" "0.01_exponential_rr" "0.01_exponential_v" "0.01_quadratic_h" "0.01_quadratic_hv" "0.01_quadratic_o" "0.01_quadratic_rl" "0.01_quadratic_rr" "0.01_quadratic_v" "0.05_exponential_h" "0.05_exponential_hv" "0.05_exponential_o" "0.05_exponential_rl" "0.05_exponential_rr" "0.05_exponential_v" "0.05_quadratic_h" "0.05_quadratic_hv" "0.05_quadratic_o" "0.05_quadratic_rl" "0.05_quadratic_rr" "0.05_quadratic_v" "0.1_exponential_h" "0.1_exponential_hv" "0.1_exponential_o" "0.1_exponential_rl" "0.1_exponential_rr" "0.1_exponential_v" "0.1_quadratic_h" "0.1_quadratic_hv" "0.1_quadratic_o" "0.1_quadratic_rl" "0.1_quadratic_rr" "0.1_quadratic_v" "1_exponential_h" "1_exponential_hv" "1_exponential_o" "1_exponential_rl" "1_exponential_rr" "1_exponential_v" "1_quadratic_h" "1_quadratic_hv" "1_quadratic_o" "1_quadratic_rl" "1_quadratic_rr" "1_quadratic_v" "original_h" "original_hv" "original_o" "original_rl" "original_rr" "original_v" "output_h" "output_hv" "output_o" "output_rl" "output_rr" "output_v")

iter_test=("100.png" "115.png" "127.png" "153.png" "025.png" "003.png" "059.png" "007.png" "109.png" "116.png" "140.png" "156.png" "027.png" "054.png" "066.png" "082.png" "111.png" "121.png" "146.png" "017.png" "031.png" "056.png" "069.png" "008.png" "113.png" "125.png" "149.png" "001.png" "035.png" "057.png" "006.png" "098.png")

iter_train=("101.png" "122.png" "013.png" "015.png" "033.png" "050.png" "072.png" "090.png" "102.png" "123.png" "141.png" "160.png" "034.png" "051.png" "073.png" "091.png" "103.png" "124.png" "142.png" "161.png" "036.png" "052.png" "074.png" "092.png" "104.png" "126.png" "143.png" "016.png" "037.png" "053.png" "075.png" "093.png" "105.png" "128.png" "144.png" "018.png" "038.png" "055.png" "076.png" "094.png" "106.png" "129.png" "145.png" "019.png" "039.png" "058.png" "077.png" "095.png" "107.png" "012.png" "147.png" "020.png" "040.png" "005.png" "078.png" "096.png" "108.png" "130.png" "148.png" "021.png" "041.png" "060.png" "079.png" "097.png" "010.png" "131.png" "014.png" "022.png" "042.png" "061.png" "080.png" "099.png" "110.png" "132.png" "150.png" "023.png" "043.png" "062.png" "081.png" "009.png" "112.png" "133.png" "151.png" "024.png" "044.png" "063.png" "083.png" "114.png" "134.png" "152.png" "026.png" "045.png" "064.png" "084.png" "117.png" "135.png" "154.png" "028.png" "046.png" "065.png" "085.png" "118.png" "136.png" "155.png" "029.png" "047.png" "067.png" "086.png" "119.png" "137.png" "157.png" "002.png" "048.png" "068.png" "087.png" "011.png" "138.png" "158.png" "030.png" "049.png" "070.png" "088.png" "120.png" "139.png" "159.png" "032.png" "004.png" "071.png" "089.png")

#make directory
dir1=("i" "o" "p")
dir2=("train" "test" "train_output" "test_output")
dir3=("h" "hv" "o" "rl" "rr" "v")

cd /home/yeonjee/lv_challenge/data/dataset/
mkdir dataset04

for i in "${dir1[@]}"
do
cd /home/yeonjee/lv_challenge/data/dataset/dataset04/
mkdir $i
  for j in "${dir2[@]}"
  do
  cd /home/yeonjee/lv_challenge/data/dataset/dataset04/$i/
  mkdir $j
    for k in "${dir3[@]}"
    do
    cd /home/yeonjee/lv_challenge/data/dataset/dataset04/$i/$j/
    mkdir $k
    cd /home/yeonjee/lv_challenge/data/dataset/dataset04/$i/$j/$k/
    if [[ "$j" == *"out"* ]]
    then
      mkdir output_$k
    else
      mkdir 0.01_exponential_$k 0.01_quadratic_$k 0.05_exponential_$k 0.05_quadratic_$k 0.1_exponential_$k 0.1_quadratic_$k 1_exponential_$k 1_quadratic_$k original_$k
    fi
    done
  done
done

#test
for i in "${iter1[@]}"
do
  for j in "${iter_test[@]}"
  do
    if [ "${i:0:6}" = "output" ]
    then
      k=`echo $i|cut -d"_" -f2`
      cp /home/yeonjee/lv_challenge/data/Anisotropic/poriginal_A_png/$i/$j /home/yeonjee/lv_challenge/data/dataset/dataset04/p/test_output/$k/$i/
    elif [ "${i:0:8}" = "original" ]
    then
      k=`echo $i|cut -d"_" -f2`
      cp /home/yeonjee/lv_challenge/data/Anisotropic/poriginal_A_png/$i/$j /home/yeonjee/lv_challenge/data/dataset/dataset04/p/test/$k/$i/
    else
      k=`echo $i|cut -d"_" -f3`
      cp /home/yeonjee/lv_challenge/data/Anisotropic/poriginal_A_png/$i/$j /home/yeonjee/lv_challenge/data/dataset/dataset04/p/test/$k/$i/
    fi 
  done
done

#train
for i in "${iter1[@]}"
do
  for j in "${iter_train[@]}"
  do
    if [ "${i:0:6}" = "output" ]
    then
      k=`echo $i|cut -d"_" -f2`
      cp /home/yeonjee/lv_challenge/data/Anisotropic/poriginal_A_png/$i/$j /home/yeonjee/lv_challenge/data/dataset/dataset04/p/train_output/$k/$i/
    elif [ "${i:0:8}" = "original" ]
    then
      k=`echo $i|cut -d"_" -f2`
      cp /home/yeonjee/lv_challenge/data/Anisotropic/poriginal_A_png/$i/$j /home/yeonjee/lv_challenge/data/dataset/dataset04/p/train/$k/$i/
    else
      k=`echo $i|cut -d"_" -f3`
      cp /home/yeonjee/lv_challenge/data/Anisotropic/poriginal_A_png/$i/$j /home/yeonjee/lv_challenge/data/dataset/dataset04/p/train/$k/$i/
    fi  
  done
done
