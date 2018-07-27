#!/bin/bash
# change p to i or o
iter1=("output_o" "output_rl2" "stack_o" "stack_rl2" "output_ov" "output_rl2v" "stack_ov" "stack_rl2v" "output_rl1" "output_rl3" "stack_rl1" "stack_rl3" "output_rl1v" "output_rl3v" "stack_rl1v" "stack_rl3v")

iter_test=("138.pt" "583.pt" "783.pt" "065.pt" "262.pt" "121.pt" "508.pt" "780.pt" "461.pt" "484.pt" "668.pt" "389.pt" "215.pt" "097.pt" "500.pt" "030.pt" "400.pt" "444.pt" "623.pt" "781.pt" "003.pt" "713.pt" "457.pt" "273.pt" "739.pt" "235.pt" "606.pt" "105.pt" "326.pt" "032.pt" "023.pt" "027.pt" "666.pt" "555.pt" "010.pt" "391.pt" "703.pt" "222.pt" "433.pt" "744.pt" "790.pt" "541.pt" "228.pt" "449.pt" "799.pt" "567.pt" "239.pt" "354.pt" "237.pt" "694.pt" "225.pt" "471.pt" "297.pt" "775.pt" "427.pt" "570.pt" "658.pt" "103.pt" "191.pt" "645.pt" "742.pt" "304.pt" "124.pt" "341.pt" "786.pt" "729.pt" "513.pt" "767.pt" "520.pt" "687.pt" "195.pt" "311.pt" "291.pt" "602.pt" "512.pt" "518.pt" "403.pt" "604.pt" "036.pt" "492.pt" "249.pt" "414.pt" "425.pt" "681.pt" "178.pt" "376.pt" "562.pt" "691.pt" "384.pt" "089.pt" "450.pt" "680.pt" "521.pt" "111.pt" "168.pt" "534.pt" "740.pt" "380.pt" "502.pt" "031.pt" "481.pt" "045.pt" "316.pt" "630.pt" "608.pt" "593.pt" "404.pt" "663.pt" "175.pt" "173.pt" "515.pt" "233.pt" "013.pt" "205.pt" "553.pt" "719.pt" "238.pt" "415.pt" "527.pt" "353.pt" "592.pt" "362.pt" "754.pt" "276.pt" "676.pt" "690.pt" "624.pt" "006.pt" "393.pt" "525.pt" "133.pt" "532.pt" "575.pt" "211.pt" "437.pt" "058.pt" "493.pt" "374.pt" "584.pt" "568.pt" "692.pt" "517.pt" "424.pt" "497.pt" "366.pt" "723.pt" "355.pt" "002.pt" "552.pt" "554.pt" "639.pt" "628.pt" "340.pt" "470.pt" "615.pt" "029.pt" "236.pt" "182.pt" "564.pt" "599.pt" "186.pt" "094.pt" "565.pt" "801.pt" "034.pt" "073.pt" "086.pt" "018.pt" "464.pt" "015.pt" "288.pt" "256.pt" "682.pt" "113.pt" "190.pt" "686.pt" "298.pt" "072.pt" "172.pt" "164.pt" "642.pt" "764.pt" "696.pt" "280.pt" "302.pt" "466.pt" "330.pt" "509.pt" "486.pt" "117.pt" "025.pt" "320.pt" "396.pt" "352.pt" "432.pt" "193.pt" "265.pt" "112.pt" "260.pt" "523.pt" "793.pt" "443.pt" "022.pt" "231.pt" "019.pt" "407.pt" "150.pt" "037.pt" "165.pt" "803.pt" "519.pt" "671.pt" "558.pt" "226.pt" "529.pt" "462.pt" "229.pt" "537.pt" "776.pt" "405.pt" "329.pt" "594.pt" "061.pt" "306.pt" "129.pt" "218.pt" "049.pt" "314.pt" "640.pt" "079.pt" "318.pt" "582.pt" "163.pt" "751.pt" "259.pt" "134.pt" "009.pt" "039.pt" "223.pt" "472.pt" "176.pt" "522.pt" "773.pt" "388.pt" "206.pt" "356.pt" "102.pt" "672.pt" "788.pt" "199.pt" "505.pt" "107.pt" "789.pt" "766.pt" "664.pt" "731.pt")

iter_train=("638.pt" "334.pt" "412.pt" "289.pt" "601.pt" "161.pt" "561.pt" "336.pt" "139.pt" "348.pt" "440.pt" "219.pt" "782.pt" "099.pt" "794.pt" "702.pt" "662.pt" "241.pt" "067.pt" "042.pt" "087.pt" "137.pt" "174.pt" "171.pt" "538.pt" "275.pt" "745.pt" "595.pt" "625.pt" "377.pt" "347.pt" "349.pt" "616.pt" "299.pt" "674.pt" "501.pt" "679.pt" "656.pt" "597.pt" "021.pt" "209.pt" "038.pt" "735.pt" "557.pt" "699.pt" "076.pt" "660.pt" "802.pt" "697.pt" "059.pt" "315.pt" "301.pt" "401.pt" "475.pt" "194.pt" "040.pt" "293.pt" "282.pt" "115.pt" "290.pt" "530.pt" "488.pt" "528.pt" "187.pt" "596.pt" "152.pt" "546.pt" "274.pt" "474.pt" "791.pt" "798.pt" "460.pt" "142.pt" "056.pt" "709.pt" "024.pt" "698.pt" "796.pt" "007.pt" "499.pt" "344.pt" "008.pt" "047.pt" "212.pt" "480.pt" "423.pt" "454.pt" "586.pt" "510.pt" "792.pt" "123.pt" "496.pt" "498.pt" "216.pt" "083.pt" "060.pt" "732.pt" "655.pt" "730.pt" "743.pt" "082.pt" "381.pt" "738.pt" "053.pt" "760.pt" "495.pt" "413.pt" "278.pt" "419.pt" "151.pt" "726.pt" "130.pt" "365.pt" "245.pt" "162.pt" "052.pt" "550.pt" "548.pt" "573.pt" "797.pt" "014.pt" "678.pt" "458.pt" "566.pt" "372.pt" "574.pt" "626.pt" "453.pt" "201.pt" "544.pt" "665.pt" "033.pt" "418.pt" "431.pt" "308.pt" "234.pt" "670.pt" "581.pt" "712.pt" "402.pt" "317.pt" "399.pt" "442.pt" "710.pt" "695.pt" "339.pt" "183.pt" "675.pt" "644.pt" "600.pt" "447.pt" "158.pt" "559.pt" "127.pt" "185.pt" "489.pt" "144.pt" "046.pt" "386.pt" "230.pt" "467.pt" "611.pt" "295.pt" "619.pt" "736.pt" "551.pt" "200.pt" "398.pt" "603.pt" "711.pt" "096.pt" "416.pt" "753.pt" "156.pt" "126.pt" "627.pt" "434.pt" "279.pt" "578.pt" "737.pt" "473.pt" "048.pt" "375.pt" "632.pt" "011.pt" "125.pt" "543.pt" "761.pt" "805.pt" "283.pt" "598.pt" "563.pt" "012.pt" "777.pt" "428.pt" "149.pt" "184.pt" "253.pt" "590.pt" "576.pt" "373.pt" "257.pt" "708.pt" "494.pt" "261.pt" "779.pt" "716.pt" "092.pt" "077.pt" "577.pt" "637.pt" "157.pt" "055.pt" "264.pt" "309.pt" "539.pt" "707.pt" "106.pt" "653.pt" "622.pt" "017.pt" "435.pt" "614.pt" "284.pt" "332.pt" "342.pt" "154.pt" "768.pt" "633.pt" "081.pt" "651.pt" "613.pt" "438.pt" "417.pt" "589.pt" "221.pt" "439.pt" "547.pt" "718.pt" "755.pt" "321.pt" "756.pt" "659.pt" "203.pt" "385.pt" "088.pt" "410.pt" "725.pt" "693.pt" "214.pt" "669.pt" "771.pt" "587.pt" "648.pt" "071.pt" "333.pt" "387.pt" "303.pt" "477.pt" "204.pt" "483.pt" "363.pt" "343.pt" "392.pt" "120.pt" "621.pt" "004.pt" "091.pt" "271.pt" "378.pt" "540.pt" "310.pt" "116.pt" "459.pt" "420.pt" "254.pt" "246.pt" "277.pt" "620.pt" "787.pt" "141.pt" "661.pt" "319.pt" "359.pt" "784.pt" "189.pt" "762.pt" "609.pt" "778.pt" "160.pt" "153.pt" "323.pt" "747.pt" "085.pt" "180.pt" "255.pt" "542.pt" "119.pt" "571.pt" "448.pt" "220.pt" "370.pt" "390.pt" "720.pt" "132.pt" "147.pt" "478.pt" "286.pt" "346.pt" "294.pt" "110.pt" "476.pt" "146.pt" "395.pt" "506.pt" "421.pt" "429.pt" "673.pt" "514.pt" "101.pt" "406.pt" "772.pt" "090.pt" "536.pt" "436.pt" "217.pt" "043.pt" "140.pt" "617.pt" "667.pt" "135.pt" "705.pt" "207.pt" "066.pt" "524.pt" "026.pt" "069.pt" "612.pt" "250.pt" "727.pt" "556.pt" "382.pt" "169.pt" "643.pt" "482.pt" "430.pt" "114.pt" "062.pt" "098.pt" "411.pt" "397.pt" "445.pt" "084.pt" "268.pt" "769.pt" "607.pt" "364.pt" "337.pt" "706.pt" "383.pt" "258.pt" "487.pt" "345.pt" "503.pt" "170.pt" "504.pt" "654.pt" "064.pt" "243.pt" "677.pt" "749.pt" "579.pt" "136.pt" "196.pt" "758.pt" "307.pt" "468.pt" "605.pt" "588.pt" "063.pt" "531.pt" "446.pt" "704.pt" "331.pt" "684.pt" "741.pt" "451.pt" "070.pt" "080.pt" "646.pt" "240.pt" "491.pt" "044.pt" "322.pt" "324.pt" "367.pt" "108.pt" "248.pt" "054.pt" "244.pt" "685.pt" "647.pt" "166.pt" "028.pt" "683.pt" "746.pt" "360.pt" "526.pt" "652.pt" "035.pt" "020.pt" "312.pt" "379.pt" "795.pt" "701.pt" "361.pt" "155.pt" "455.pt" "001.pt" "426.pt" "369.pt" "272.pt" "167.pt" "351.pt" "328.pt" "770.pt" "338.pt" "247.pt" "533.pt" "657.pt" "074.pt" "296.pt" "636.pt" "734.pt" "721.pt" "208.pt" "774.pt" "572.pt" "131.pt" "210.pt" "263.pt" "469.pt" "093.pt" "095.pt" "800.pt" "456.pt" "300.pt" "463.pt" "251.pt" "224.pt" "148.pt" "188.pt" "292.pt" "281.pt" "227.pt" "700.pt" "728.pt" "177.pt" "452.pt" "068.pt" "748.pt" "535.pt" "325.pt" "631.pt" "050.pt" "075.pt" "192.pt" "715.pt" "335.pt" "181.pt" "197.pt" "569.pt" "118.pt" "409.pt" "511.pt" "785.pt" "422.pt" "763.pt" "051.pt" "733.pt" "752.pt" "629.pt" "179.pt" "232.pt" "641.pt" "479.pt" "689.pt" "327.pt" "610.pt" "722.pt" "591.pt" "765.pt" "714.pt" "585.pt" "269.pt" "560.pt" "408.pt" "507.pt" "252.pt" "549.pt" "109.pt" "145.pt" "350.pt" "757.pt" "005.pt" "213.pt" "104.pt" "650.pt" "100.pt" "057.pt" "750.pt" "724.pt" "305.pt" "287.pt" "634.pt" "618.pt" "688.pt" "358.pt" "270.pt" "394.pt" "285.pt" "078.pt" "516.pt" "635.pt" "202.pt" "649.pt" "371.pt" "357.pt" "717.pt" "580.pt" "804.pt" "041.pt" "266.pt" "465.pt" "122.pt" "267.pt" "128.pt" "242.pt" "198.pt" "159.pt" "143.pt" "368.pt" "490.pt" "485.pt" "545.pt" "016.pt" "441.pt" "759.pt" "313.pt")

#make directory
#dir1=("i" "o" "p")
dir1=("i")
dir2=("train" "test" "train_output" "test_output")
dir3=("o" "rl1" "rl2" "rl3" "ov" "rl1v" "rl2v" "rl3v")

cd /home/yeonjee/lv_challenge/data/dataset/
mkdir dataset08_tensor

for i in "${dir1[@]}"
do
cd /home/yeonjee/lv_challenge/data/dataset/dataset08_tensor/
mkdir $i
  for j in "${dir2[@]}"
  do
  cd /home/yeonjee/lv_challenge/data/dataset/dataset08_tensor/$i/
  mkdir $j
    for k in "${dir3[@]}"
    do
    cd /home/yeonjee/lv_challenge/data/dataset/dataset08_tensor/$i/$j/
    mkdir $k
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
      cp /home/yeonjee/lv_challenge/data/dataset/dataset08/ioriginal_TV_png/$i/$j /home/yeonjee/lv_challenge/data/dataset/dataset08_tensor/i/test_output/$k/
    else
      k=`echo $i|cut -d"_" -f2`
      cp /home/yeonjee/lv_challenge/data/dataset/dataset08/ioriginal_TV_png/$i/$j /home/yeonjee/lv_challenge/data/dataset/dataset08_tensor/i/test/$k/
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
      cp /home/yeonjee/lv_challenge/data/dataset/dataset08/ioriginal_TV_png/$i/$j /home/yeonjee/lv_challenge/data/dataset/dataset08_tensor/i/train_output/$k/
    else
      k=`echo $i|cut -d"_" -f2`
      cp /home/yeonjee/lv_challenge/data/dataset/dataset08/ioriginal_TV_png/$i/$j /home/yeonjee/lv_challenge/data/dataset/dataset08_tensor/i/train/$k/
    fi  
  done
done
