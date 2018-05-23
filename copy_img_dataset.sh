#!/bin/bash
# change p to i or o
iter1=("0.01_exponential_h" "0.05_exponential_v" "0.1_quadratic_o" "0.01_exponential_hv" "0.05_quadratic_h" "0.1_quadratic_v" "0.01_exponential_o" "0.05_quadratic_hv" "1_exponential_h" "0.01_exponential_v" "0.05_quadratic_o" "1_exponential_hv" "0.01_quadratic_h" "0.05_quadratic_v" "1_exponential_o" "0.01_quadratic_hv" "0.1_exponential_h" "1_exponential_v" "0.01_quadratic_o" "0.1_exponential_hv" "1_quadratic_h" "0.01_quadratic_v" "0.1_exponential_o" "1_quadratic_hv" "0.05_exponential_h" "0.1_exponential_v" "1_quadratic_o" "0.05_exponential_hv" "0.1_quadratic_h" "1_quadratic_v" "0.05_exponential_o" "0.1_quadratic_hv" "original" "output")

iter_test=("100.png" "115.png" "127.png" "153.png" "25.png" "3.png" "59.png" "7.png" "109.png" "116.png" "140.png" "156.png" "27.png" "54.png" "66.png" "82.png" "111.png" "121.png" "146.png" "17.png" "31.png" "56.png" "69.png" "8.png" "113.png" "125.png" "149.png" "1.png" "35.png" "57.png" "6.png" "98.png")

iter_train=("101.png" "122.png" "13.png" "15.png" "33.png" "50.png" "72.png" "90.png" "102.png" "123.png" "141.png" "160.png" "34.png" "51.png" "73.png" "91.png" "103.png" "124.png" "142.png" "161.png" "36.png" "52.png" "74.png" "92.png" "104.png" "126.png" "143.png" "16.png" "37.png" "53.png" "75.png" "93.png" "105.png" "128.png" "144.png" "18.png" "38.png" "55.png" "76.png" "94.png" "106.png" "129.png" "145.png" "19.png" "39.png" "58.png" "77.png" "95.png" "107.png" "12.png" "147.png" "20.png" "40.png" "5.png" "78.png" "96.png" "108.png" "130.png" "148.png" "21.png" "41.png" "60.png" "79.png" "97.png" "10.png" "131.png" "14.png" "22.png" "42.png" "61.png" "80.png" "99.png" "110.png" "132.png" "150.png" "23.png" "43.png" "62.png" "81.png" "9.png" "112.png" "133.png" "151.png" "24.png" "44.png" "63.png" "83.png" "114.png" "134.png" "152.png" "26.png" "45.png" "64.png" "84.png" "117.png" "135.png" "154.png" "28.png" "46.png" "65.png" "85.png" "118.png" "136.png" "155.png" "29.png" "47.png" "67.png" "86.png" "119.png" "137.png" "157.png" "2.png" "48.png" "68.png" "87.png" "11.png" "138.png" "158.png" "30.png" "49.png" "70.png" "88.png" "120.png" "139.png" "159.png" "32.png" "4.png" "71.png" "89.png")

#test
for i in "${iter1[@]}"
do
  for j in "${iter_test[@]}"
  do
  if [ "$i" = "output" ]
  then
    cp /hoem04/powergkrry/lv_challenge/data/pcontour_png/$j /hoem04/powergkrry/lv_challenge/data/dataset/dataset02/p/test/$i/folder/
  elif [ "$i" = "original" ]
  then
    cp /hoem04/powergkrry/lv_challenge/data/poriginal_png/$j /hoem04/powergkrry/lv_challenge/data/dataset/dataset02/p/test/$i/folder/
  else
    cp /hoem04/powergkrry/lv_challenge/data/Anisotropic/poriginal_A_png/$i/$j /hoem04/powergkrry/lv_challenge/data/dataset/dataset02/p/test/$i/folder/
  fi    
  done
done

#train
for i in "${iter1[@]}"
do
  for j in "${iter_train[@]}"
  do
  if [ "$i" = "output" ]
  then
    cp /hoem04/powergkrry/lv_challenge/data/pcontour_png/$j /hoem04/powergkrry/lv_challenge/data/dataset/dataset02/p/train/$i/folder/
  elif [ "$i" = "original" ]
  then
    cp /hoem04/powergkrry/lv_challenge/data/poriginal_png/$j /hoem04/powergkrry/lv_challenge/data/dataset/dataset02/p/train/$i/folder/
  else
    cp /hoem04/powergkrry/lv_challenge/data/Anisotropic/poriginal_A_png/$i/$j /hoem04/powergkrry/lv_challenge/data/dataset/dataset02/p/train/$i/folder/
  fi    
  done
done

