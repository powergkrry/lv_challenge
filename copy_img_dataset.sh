#!/bin/bash
# change p to i or o
iter1=("0.01_exponential_h" "0.01_exponential_hv" "0.01_exponential_o" "0.01_exponential_rl" "0.01_exponential_rr" "0.01_exponential_v" "0.01_quadratic_h" "0.01_quadratic_hv" "0.01_quadratic_o" "0.01_quadratic_rl" "0.01_quadratic_rr" "0.01_quadratic_v" "0.05_exponential_h" "0.05_exponential_hv" "0.05_exponential_o" "0.05_exponential_rl" "0.05_exponential_rr" "0.05_exponential_v" "0.05_quadratic_h" "0.05_quadratic_hv" "0.05_quadratic_o" "0.05_quadratic_rl" "0.05_quadratic_rr" "0.05_quadratic_v" "0.1_exponential_h" "0.1_exponential_hv" "0.1_exponential_o" "0.1_exponential_rl" "0.1_exponential_rr" "0.1_exponential_v" "0.1_quadratic_h" "0.1_quadratic_hv" "0.1_quadratic_o" "0.1_quadratic_rl" "0.1_quadratic_rr" "0.1_quadratic_v" "1_exponential_h" "1_exponential_hv" "1_exponential_o" "1_exponential_rl" "1_exponential_rr" "1_exponential_v" "1_quadratic_h" "1_quadratic_hv" "1_quadratic_o" "1_quadratic_rl" "1_quadratic_rr" "1_quadratic_v" "original_h" "original_hv" "original_o" "original_rl" "original_rr" "original_v" "output_h" "output_hv" "output_o" "output_rl" "output_rr" "output_v")

iter_test=("069.png" "292.png" "411.png" "392.png" "033.png" "131.png" "061.png" "254.png" "390.png" "231.png" "242.png" "334.png" "195.png" "404.png" "108.png" "049.png" "250.png" "015.png" "200.png" "222.png" "312.png" "391.png" "393.png" "002.png" "357.png" "229.png" "137.png" "370.png" "118.png" "303.png" "053.png" "163.png" "016.png" "012.png" "014.png" "333.png" "278.png" "005.png" "196.png" "352.png" "111.png" "217.png" "372.png" "403.png" "271.png" "114.png" "225.png" "413.png" "284.png" "120.png" "177.png" "119.png" "347.png" "113.png" "236.png" "149.png" "387.png" "214.png" "285.png" "329.png" "052.png" "096.png" "323.png" "152.png" "062.png" "171.png" "257.png" "379.png" "260.png" "344.png" "098.png" "156.png" "146.png" "301.png" "256.png" "259.png" "202.png" "302.png" "018.png" "246.png" "125.png" "207.png" "213.png" "089.png" "188.png" "281.png" "192.png" "045.png" "374.png" "261.png" "056.png" "084.png" "267.png" "351.png" "190.png" "251.png" "388.png" "241.png" "023.png" "158.png" "315.png" "304.png" "297.png" "327.png" "088.png" "087.png" "258.png" "117.png" "007.png" "103.png" "277.png" "335.png" "369.png" "208.png" "264.png" "398.png" "296.png" "181.png" "366.png" "138.png" "309.png" "003.png" "197.png" "263.png" "067.png" "266.png" "288.png" "106.png")

iter_train=("219.png" "029.png" "247.png" "187.png" "378.png" "311.png" "345.png" "212.png" "249.png" "183.png" "338.png" "178.png" "001.png" "276.png" "310.png" "170.png" "235.png" "377.png" "417.png" "091.png" "093.png" "047.png" "415.png" "017.png" "037.png" "043.png" "009.png" "232.png" "008.png" "144.png" "128.png" "368.png" "057.png" "095.png" "305.png" "365.png" "036.png" "328.png" "041.png" "066.png" "136.png" "244.png" "044.png" "169.png" "070.png" "166.png" "283.png" "076.png" "313.png" "180.png" "083.png" "262.png" "122.png" "030.png" "400.png" "080.png" "099.png" "316.png" "406.png" "204.png" "405.png" "319.png" "028.png" "065.png" "289.png" "270.png" "054.png" "349.png" "380.png" "210.png" "006.png" "058.png" "383.png" "102.png" "038.png" "010.png" "185.png" "042.png" "115.png" "399.png" "130.png" "174.png" "110.png" "140.png" "381.png" "162.png" "205.png" "300.png" "133.png" "116.png" "221.png" "135.png" "167.png" "306.png" "419.png" "173.png" "148.png" "410.png" "294.png" "339.png" "223.png" "324.png" "189.png" "077.png" "416.png" "055.png" "013.png" "079.png" "019.png" "020.png" "237.png" "326.png" "373.png" "107.png" "145.png" "395.png" "034.png" "299.png" "318.png" "353.png" "396.png" "330.png" "348.png" "274.png" "407.png" "160.png" "227.png" "298.png" "097.png" "360.png" "337.png" "026.png" "412.png" "147.png" "224.png" "172.png" "050.png" "127.png" "027.png" "100.png" "245.png" "362.png" "358.png" "220.png" "361.png" "287.png" "073.png" "354.png" "243.png" "389.png" "382.png" "035.png" "320.png" "409.png" "228.png" "420.png" "025.png" "350.png" "193.png" "340.png" "414.png" "269.png" "011.png" "022.png" "141.png" "129.png" "155.png" "325.png" "394.png" "230.png" "150.png" "201.png" "255.png" "282.png" "314.png" "234.png" "336.png" "048.png" "168.png" "375.png" "275.png" "218.png" "031.png" "078.png" "153.png" "092.png" "063.png" "342.png" "075.png" "071.png" "343.png" "386.png" "215.png" "401.png" "402.png" "165.png" "265.png" "184.png" "290.png" "367.png" "112.png" "216.png" "157.png" "359.png" "164.png" "074.png" "159.png" "291.png" "384.png" "151.png" "240.png" "376.png" "321.png" "142.png" "124.png" "179.png" "059.png" "346.png" "101.png" "198.png" "182.png" "397.png" "086.png" "364.png" "121.png" "308.png" "273.png" "252.png" "139.png" "051.png" "072.png" "154.png" "408.png" "199.png" "085.png" "060.png" "272.png" "191.png" "307.png" "363.png" "238.png" "123.png" "194.png" "082.png" "286.png" "104.png" "094.png" "064.png" "021.png" "143.png" "209.png" "355.png" "175.png" "248.png" "280.png" "068.png" "211.png" "317.png" "161.png" "081.png" "176.png" "040.png" "385.png" "226.png" "341.png" "418.png" "004.png" "295.png" "371.png" "293.png" "105.png" "331.png" "109.png" "332.png" "253.png" "024.png" "279.png" "032.png" "132.png" "126.png" "039.png" "206.png" "322.png" "090.png" "203.png" "186.png" "134.png" "239.png" "268.png" "233.png" "046.png" "356.png")

:<<'END'
#make directory
dir1=("i" "o" "p")
dir2=("train" "test" "train_output" "test_output")
dir3=("h" "hv" "o" "rl" "rr" "v")

cd /home/image/lv_challenge/data/dataset/
mkdir dataset04

for i in "${dir1[@]}"
do
cd /home/image/lv_challenge/data/dataset/dataset04/
mkdir $i
  for j in "${dir2[@]}"
  do
  cd /home/image/lv_challenge/data/dataset/dataset04/$i/
  mkdir $j
    for k in "${dir3[@]}"
    do
    cd /home/image/lv_challenge/data/dataset/dataset04/$i/$j/
    mkdir $k
    cd /home/image/lv_challenge/data/dataset/dataset04/$i/$j/$k/
    if [[ "$j" == *"out"* ]]
    then
      mkdir output_$k
    else
      mkdir 0.01_exponential_$k 0.01_quadratic_$k 0.05_exponential_$k 0.05_quadratic_$k 0.1_exponential_$k 0.1_quadratic_$k 1_exponential_$k 1_quadratic_$k original_$k
    fi
    done
  done
done
END

#test
for i in "${iter1[@]}"
do
  for j in "${iter_test[@]}"
  do
    if [ "${i:0:6}" = "output" ]
    then
      k=`echo $i|cut -d"_" -f2`
      cp /home/image/lv_challenge/data/Anisotropic/ooriginal_A_png/$i/$j /home/image/lv_challenge/data/dataset/dataset04/o/test_output/$k/$i/
    elif [ "${i:0:8}" = "original" ]
    then
      k=`echo $i|cut -d"_" -f2`
      cp /home/image/lv_challenge/data/Anisotropic/ooriginal_A_png/$i/$j /home/image/lv_challenge/data/dataset/dataset04/o/test/$k/$i/
    else
      k=`echo $i|cut -d"_" -f3`
      cp /home/image/lv_challenge/data/Anisotropic/ooriginal_A_png/$i/$j /home/image/lv_challenge/data/dataset/dataset04/o/test/$k/$i/
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
      cp /home/image/lv_challenge/data/Anisotropic/ooriginal_A_png/$i/$j /home/image/lv_challenge/data/dataset/dataset04/o/train_output/$k/$i/
    elif [ "${i:0:8}" = "original" ]
    then
      k=`echo $i|cut -d"_" -f2`
      cp /home/image/lv_challenge/data/Anisotropic/ooriginal_A_png/$i/$j /home/image/lv_challenge/data/dataset/dataset04/o/train/$k/$i/
    else
      k=`echo $i|cut -d"_" -f3`
      cp /home/image/lv_challenge/data/Anisotropic/ooriginal_A_png/$i/$j /home/image/lv_challenge/data/dataset/dataset04/o/train/$k/$i/
    fi  
  done
done
