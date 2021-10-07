# -*- coding: utf-8 -*-
# python3 python3 mktimeseries_StellarMotion.py

import astropy.io.fits as fits
import numpy as np
import random
import math

# 追加パッケージ
import os
import sys
import shutil

# pixel size: 0.536 arc second = 2.600E-06 radian = dpixel
# world: approximately l/b = +/- 0.5 (unit = degree)
#     1.5010 < lambda < 1.5184, -0.1047 < beta < -0.0873,
#       =lmin             =lmax      =bmin          =bmax
# Plate left low end: within the circle with radius 0.1 deg
#   1.5026 < lambda0 < 1.5061, -0.1031 < beta0 < -0.0996
#     =l0min             =l0max   =b0min            =b0max
# Plate size: 4096 x 4096 pixel for both direction.
# Plate orientation: plus / minus 10 degree = 0.1745 rad. (anglemin / anglemax)
#
# stellar parameter, lambda_0, beta_0, mu_lambda, mu_beta, pi
# plate parameter, center_lambda, center_beta, orientation

numStar = 1000
numPlate = 1 # 1フレームにつき1プレートを作成する

lmin = 0.5010
lmax = 0.5184
bmin = -0.1047
bmax = -0.0873

# 規定範囲に修正
# 上の範囲 (radian)
lmin = 1.5010 + math.radians(180)
lmax = 1.5184 + math.radians(180)
bmin = -0.1047
bmax = -0.0873
# 銀河中心方向 (degree)
#lmin = 266.5010
#lmax = 266.5184
#bmin = -5.5047
#bmax = -5.4873
##

l0min = 0.5026
l0max = 0.5061
b0min = -0.1031
b0max = -0.0996

# 規定範囲に修正
# 上の範囲 (radian)
l0min = 1.5026 + math.radians(180)
l0max = 1.5061 + math.radians(180)
b0min = -0.1031
b0max = -0.0996
# 銀河中心方向 (degree)
#l0min = 266.5026 
#l0max = 266.5061
#b0min = -5.5031
#b0max = -5.4996
##

l1min = -0.00175
l1max = 0.00175
b1min = -0.00175
b1max = 0.00175

# stellar magnitude in Hw band
magmin = 12.0 
magmax = 12.0

lsunmin = -9.4
lsunmax = 9.4
anglemin = -0.1745
anglemax = 0.1745

dpixel = 2.1E-6 # rad/pix (10um/pix; EFL=4.86m)
array_nx = 1920 
array_ny = 1920
gauss_sigma = 0.74 # 1.028 lambda/ApDia/PxScale/2.35

# 作成するフレーム数
numframe = 1

# ランダムで作成する星の位置を固定する
random.seed(100)

### create star data set 
stars = np.arange(numStar * 6, dtype='float64').reshape(numStar, 6)
for i in range(numStar):
    stars[i][0] = lmin + random.random() * (lmax - lmin)
    stars[i][1] = bmin + random.random() * (bmax - bmin)
    stars[i][2] = random.random() * 5E-5
    stars[i][3] = random.random() * 5E-5
    stars[i][4] = random.random() * 5E-5
######
## 方針
# [x] 動かす星以外の星の位置は固定
# [x] frameごとに星の位置をずらしていく
# [x] 1星の動きをいれる
# [x] 星の等級の情報もいれる(明るい星と暗い星を区別する)。
# [x] frames.csvとKnown_starsの情報を残すようにする
# [ ] マージンをつけた（余裕を持たせた）imageのstar_plate.csvを作成する
# [ ] 動く星の数を増やす
# [ ] プレートを別の場所にする、プレートを大きくする、フィールドを増やす
# [ ] 観測スケジュールをいれる
##

### create plate parameters 
plates = np.arange(numPlate * 4, dtype='float64').reshape(numPlate, 4)
for i in range(numPlate):
    plates[i][0] = l0min + random.random() * (l1max - l1min)
    plates[i][1] = b0min + random.random() * (b1max - b1min)
    plates[i][2] = anglemin + random.random() * (anglemax - anglemin)
    plates[i][3] = lsunmin + (lsunmax - lsunmin) * (i + 0.5) / numPlate
######
## 仮定
# データをとる位置は毎回全く同じと仮定 => フィールドを固定
##

# 簡単のために、ランダムで星の等級をばらつかせる
for i in range(numStar):
    stars[i][5] = magmin + random.random() * (magmax - magmin)
## 課題
# [ ] 星等級の全天平均の頻度分布に合わせる 
# [ ] 明るさの時間変化をランダムで入れるか？今はすべての画像で同じフォトン数
# [ ] 伴星による運動を入れるために、星質量を入れる。
##

with open('stars.csv', 'w') as files:
    files.write('star index,lambda,beta,mu lambda,mu beta,pi,Hwmag\n') 
    for i in range(numStar):
        files.write(str(i)+","+str(stars[i][0])+","+str(stars[i][1])+","+str(stars[i][2])+","+str(stars[i][3])+","+str(stars[i][4])+","+str(stars[i][5])+"\n")

# 簡単のために、maglimit_known_star より明るい星は既知の星とする 
maglimit_known_star = 12
maglimit_star_catalog = 12.5
with open('known_stars.csv', 'w') as files:
    files.write('star index,lambda,beta,mu lambda,mu beta,pi,Hwmag\n') 
    for i in range(numStar):
        if stars[i][5] < maglimit_known_star:
            files.write(str(i)+","+str(stars[i][0])+","+str(stars[i][1])+","+str(stars[i][2])+","+str(stars[i][3])+","+str(stars[i][4])+","+str(stars[i][5])+"\n")

with open('star_catalogue.csv', 'w') as files:
    files.write('star index,lambda,beta,mu lambda,mu beta,pi,Hwmag\n') 
    for i in range(numStar):
        if stars[i][5] >= maglimit_known_star and stars[i][5] < maglimit_star_catalog:
            files.write(str(i)+","+str(stars[i][0])+","+str(stars[i][1])+","+str(stars[i][2])+","+str(stars[i][3])+","+str(stars[i][4])+","+str(stars[i][5])+"\n")

### バックグラウンドの値を指定する
bg_level = 435
bg_random = 37
## 
# [ ] バックグラウンドのレベルがこれで良いか確認する
##

### 1星の動き（固有運動（ulambda, ubeta)と年周視差(ppi)）
ulambda = 1000 # mas/yr 
ubeta = 1000 # mas/yr
ppi = 1000 # mas
###

ulambda = ulambda / 3600 / 1000 # arc degree / year
ubeta = ubeta / 3600 / 1000 # arc degree / year
ppi = ppi / 3600 / 1000 # arc degree)

### フォトンカウントと露出@1プレート
nphoton127 = 3140 # photons/sec @ star with Hw2 (1.1~1.6um) = 12.7 for an astrometric precition of ~ 25uas
exptime = 12.5 # sec for 1 plate
###

fbase = "plate"
cen = np.arange(numStar * 2, dtype='float64').reshape(numStar, 2)

frame_info = open("frames.csv","w")
frame_info.write('directory,time,longitude\n')

#######

for frame_id in range(numframe):
    framename = "frame%05d" %(frame_id)
    print("# create directory " + framename)
    os.makedirs(framename, exist_ok=True)
    
    # L = 0 から始めて、interval (year) ごとにフレームを作成する
    interval = 0.1
    t_yr = frame_id * interval
    L = (- 1.5 + frame_id * interval) * 2 * math.pi
    print(t_yr, L)
    frame_info.write("./%s,%f,%f\n" %(framename,t_yr*365.25,L))

    frame_plate_csv = framename + "/plates.csv"
    with open(frame_plate_csv, 'w') as filep:
        filep.write('l0,b0,angle,lsun\n')
        for i in range(numPlate):
            filep.write(str(plates[i][0]) + "," + str(plates[i][1]) + "," + str(plates[i][2]) + "," + str(L) + "\n")

    star_plate_csv = framename + "/star_plate.csv"
    with open(star_plate_csv, 'w') as star_plate_file:

        star_plate_file.write('plate index,star index,x pixel,y pixel,lambda,beta,Hwmag,nphoton\n')
        for i in range(numPlate):

            fnameo = framename + "/" + '{}{:03}{}'.format(fbase, i, ".fits")
            array_data = np.random.randint(bg_level, bg_level + bg_random, array_nx * array_ny).reshape(array_ny, array_nx)

            for j in range(numStar):
                x0 = stars[j][0]
                y0 = stars[j][1]

                # star IDが、640番の星だけを動かす
                if j == 640:
                    starlambda = stars[j][0]
                    starbeta = stars[j][1]

                    x0 = starlambda  +  math.radians(ulambda) * math.cos(math.radians(starbeta)) * t_yr  +  math.radians(ppi) * math.sin(L-math.radians(starlambda))
                    y0 = starbeta  +  math.radians(ubeta) * t_yr  -  math.radians(ppi) * math.cos(L-math.radians(starlambda)) * math.sin(math.radians(starbeta))

                wlambda = x0
                wbeta = y0

                x0 = x0 - plates[i][0]
                y0 = y0 - plates[i][1]

                cen[j][0] = (math.sin(plates[i][2]) * y0 + math.cos(plates[i][2]) * x0) / dpixel
                cen[j][1] = (math.cos(plates[i][2]) * y0 - math.sin(plates[i][2]) * x0) / dpixel

                nph = 30000
                nph = int(nphoton127 * math.pow(10, (stars[j][5] - 12.7) / (-2.5)) * exptime)
                px = np.array(np.random.normal(cen[j][0], gauss_sigma, nph)+1.5, dtype=np.int)-1
                py = np.array(np.random.normal(cen[j][1], gauss_sigma, nph)+1.5, dtype=np.int)-1

                in_plate = np.all(np.stack([px >= 0+100, px < array_nx-100, py >= 0+100, py < array_ny-100], axis=1), axis=1)
                pxy = np.stack([px, py], axis=1)[in_plate]

                for k in pxy: 
                    array_data[k[1]][k[0]] += 1
                if len(pxy) > 0:
                    print(i, j, x0, y0, cen[j][0], cen[j][1], stars[j][0], stars[j][1], plates[i][0], plates[i][1], stars[j][5], nph)
                    star_plate_file.write(f'{i},{j},{cen[j][0]},{cen[j][1]},{wlambda},{wbeta},{stars[j][5]},{nph}\n')

            hdu = fits.PrimaryHDU()
            hdu.data = array_data
            hdu.writeto(fnameo, overwrite=True)

frame_info.close()

if len(sys.argv) > 1: dataset = sys.argv[1]
else: dataset = "dataset"
if os.path.exists(dataset): shutil.rmtree(dataset)
os.makedirs(dataset, exist_ok=True)
for frame_id in range(numframe): shutil.move("frame%05d" %(frame_id), dataset)
shutil.move("stars.csv", dataset)
shutil.move("known_stars.csv", dataset)
shutil.move("star_catalogue.csv", dataset)
shutil.move("frames.csv", dataset)
