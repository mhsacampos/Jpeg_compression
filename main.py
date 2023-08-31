# python 3.9.5
from math import ceil

import cv2
import numpy as np
import os
from functions import *

filename = 'marbles.bmp'

# define quantization tables
QTY_0 = np.array([[16, 11, 10, 16, 24, 40, 51, 61],  # luminance quantization table
                [12, 12, 14, 19, 26, 48, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99]])

QTC_0 = np.array([[17, 18, 24, 47, 99, 99, 99, 99],  # chrominance quantization table
                [18, 21, 26, 66, 99, 99, 99, 99],
                [24, 26, 56, 99, 99, 99, 99, 99],
                [47, 66, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99]])

# Other quantizations

# An improved detection model for DCT coefficient quantization (1993) Peterson, Ahumada and Watson.
QTY_1 = np.array([[14, 10, 11, 14, 19, 25, 34, 45],  # chrominance quantization table (STANDARD)
                [10, 11, 11, 12, 15, 20, 26, 33],
                [11, 11, 15, 18, 21, 25, 31, 38],
                [14, 12, 18, 24, 28, 33, 39, 47],
                [19, 15, 21, 28, 36, 43, 51, 59],
                [25, 20, 25, 33, 43, 54, 64, 74],
                [34, 26, 31, 39, 51, 64, 77, 91],
                [45, 33, 38, 47, 59, 74, 91, 108]])

QTC_1Cb = np.array([[29, 49, 101, 132, 179, 243, 325, 428],
                    [49, 110, 101, 114, 144, 188, 245, 319],
                    [101, 101, 148, 170, 197, 237, 294, 367],
                    [132, 114, 170, 227, 272, 318, 376, 451],
                    [179, 144, 197, 272, 347, 415, 486, 569],
                    [243, 188, 237, 318, 415, 514, 611, 713],
                    [325, 245, 294, 376, 486, 611, 741, 873],
                    [428, 319, 367, 451, 569, 713, 873, 1040]])    

QTC_1Cr = np.array([[20, 34, 39, 52, 70, 95, 127, 168],
                    [34, 43, 40, 45, 57, 74, 96, 125],
                    [39, 40, 58, 67, 77, 93, 115, 144],
                    [52, 45, 67, 89, 107, 125, 147, 177],
                    [70, 57, 77, 107, 136, 163, 191, 223],
                    [95, 74, 93, 125, 163, 202, 240, 280],
                    [127, 96, 115, 147, 191, 240, 291, 342],
                    [168, 125, 144, 177, 223, 280, 342, 408]])   

# Relevance of human vision to JPEG-DCT compression (1992) Klein, Silverstein and Carney.
# 10 13 12 13 14 16 19 22        10 13 12 14 17 21 26 34       10 12 14 19 26 38 57 86
#    18 17 18 18 17 17 19           17 18 20 22 22 25 29          18 21 28 35 41 54 76
#       18 19 21 24 28 33              19 22 26 32 40 51             25 32 44 63 92 136
#          20 23 26 30 35                 25 29 36 44 56                41 54 75 107 157
#             25 28 32 38                    34 41 50 63                    70 95 132 190
#                32 36 42                       48 59 72                       125 170 239
#                   41 47                          70 85                            227 312
#                      53                             103                               419

QTY_2 = np.array([[10, 13, 12, 13, 14, 16, 19, 22],     
                    [13, 18, 17, 18, 18, 17, 17, 19],        
                    [12, 17, 18, 19, 21, 24, 28, 33],              
                    [13, 18, 19, 20, 23, 26, 30, 35],               
                    [14, 18, 21, 23, 25, 28, 32, 38],                    
                    [16, 17, 24, 26, 28, 32, 36, 42],                    
                    [19, 17, 28, 30, 32, 36, 41, 47],                      
                    [22, 19, 33, 35, 38, 42, 47, 53]]) 


QTC_2Cb = np.array([[10, 13, 12, 14, 17, 21, 26, 34],      
                    [13, 17, 18, 20, 22, 22, 25, 29],         
                    [12, 18, 19, 22, 26, 32, 40, 51],            
                    [14, 20, 22, 25, 29, 36, 44, 56],                
                    [17, 22, 26, 29, 34, 41, 50, 63],                   
                    [21, 22, 32, 36, 41, 48, 59, 72],                      
                    [26, 25, 40, 44, 50, 59, 70, 85],                            
                    [34, 29, 51, 56, 63, 72, 85, 103]])                                       
                    
QTC_2Cr = np.array([[10, 12, 14, 19, 26, 38, 57, 86],
                    [12, 18, 21, 28, 35, 41, 54, 76],
                    [14, 21, 25, 32, 44, 63, 92, 136],
                    [19, 28, 32, 41, 54, 75, 107, 157],
                    [26, 35, 44, 54, 70, 95, 132, 190],
                    [38, 41, 63, 75, 95, 125, 170, 239],
                    [57, 54, 92, 107, 132, 170, 227, 312],
                    [86, 76, 136, 157, 190, 239, 312, 419]])
    




# define window size
# QTY = QTY_0
# QTCb = QTC_0
# QTCr = QTC_0



# QTY = QTY_1
# QTCb = QTC_1Cb
# QTCr = QTC_1Cr


QTY = QTY_2
QTCb = QTC_2Cb
QTCr = QTC_2Cr



windowSize = len(QTY)

# read image
imgOriginal = cv2.imread(filename, cv2.IMREAD_COLOR)
# convert BGR to YCrCb
img = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2YCR_CB)
width = len(img[0])
height = len(img)
y = np.zeros((height, width), np.float32) + img[:, :, 0]
cr = np.zeros((height, width), np.float32) + img[:, :, 1]
cb = np.zeros((height, width), np.float32) + img[:, :, 2]
# size of the image in bits before compression
totalNumberOfBitsWithoutCompression = len(y) * len(y[0]) * 8 + len(cb) * len(cb[0]) * 8 + len(cr) * len(cr[0]) * 8
# channel values should be normalized, hence subtract 128
y = y - 128
cr = cr - 128
cb = cb - 128
# 4: 2: 2 subsampling is used # another subsampling scheme can be used
# thus chrominance channels should be sub-sampled
# define subsampling factors in both horizontal and vertical directions
SSH, SSV = 2, 2
# filter the chrominance channels using a 2x2 averaging filter # another type of filter can be used
crf = cv2.boxFilter(cr, ddepth=-1, ksize=(2, 2))
cbf = cv2.boxFilter(cb, ddepth=-1, ksize=(2, 2))
crSub = crf[::SSV, ::SSH]
cbSub = cbf[::SSV, ::SSH]

# check if padding is needed,
# if yes define empty arrays to pad each channel DCT with zeros if necessary
yWidth, yLength = ceil(len(y[0]) / windowSize) * windowSize, ceil(len(y) / windowSize) * windowSize
if (len(y[0]) % windowSize == 0) and (len(y) % windowSize == 0):
    yPadded = y.copy()
else:
    yPadded = np.zeros((yLength, yWidth))
    for i in range(len(y)):
        for j in range(len(y[0])):
            yPadded[i, j] += y[i, j]

# chrominance channels have the same dimensions, meaning both can be padded in one loop
cWidth, cLength = ceil(len(cbSub[0]) / windowSize) * windowSize, ceil(len(cbSub) / windowSize) * windowSize
if (len(cbSub[0]) % windowSize == 0) and (len(cbSub) % windowSize == 0):
    crPadded = crSub.copy()
    cbPadded = cbSub.copy()
# since chrominance channels have the same dimensions, one loop is enough
else:
    crPadded = np.zeros((cLength, cWidth))
    cbPadded = np.zeros((cLength, cWidth))
    for i in range(len(crSub)):
        for j in range(len(crSub[0])):
            crPadded[i, j] += crSub[i, j]
            cbPadded[i, j] += cbSub[i, j]

# get DCT of each channel
# define three empty matrices
yDct, crDct, cbDct = np.zeros((yLength, yWidth)), np.zeros((cLength, cWidth)), np.zeros((cLength, cWidth))

# number of iteration on x axis and y axis to calculate the luminance cosine transform values
hBlocksForY = int(len(yDct[0]) / windowSize)  # number of blocks in the horizontal direction for luminance
vBlocksForY = int(len(yDct) / windowSize)  # number of blocks in the vertical direction for luminance
# number of iteration on x axis and y axis to calculate the chrominance channels cosine transforms values
hBlocksForC = int(len(crDct[0]) / windowSize)  # number of blocks in the horizontal direction for chrominance
vBlocksForC = int(len(crDct) / windowSize)  # number of blocks in the vertical direction for chrominance

# define 3 empty matrices to store the quantized values
yq, crq, cbq = np.zeros((yLength, yWidth)), np.zeros((cLength, cWidth)), np.zeros((cLength, cWidth))
# and another 3 for the zigzags
yZigzag = np.zeros(((vBlocksForY * hBlocksForY), windowSize * windowSize))
crZigzag = np.zeros(((vBlocksForC * hBlocksForC), windowSize * windowSize))
cbZigzag = np.zeros(((vBlocksForC * hBlocksForC), windowSize * windowSize))


yCounter = 0
for i in range(vBlocksForY):
    for j in range(hBlocksForY):
        yDct[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize] = cv2.dct(
            yPadded[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize])
        yq[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize] = np.ceil(
            yDct[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize] / QTY)
        yZigzag[yCounter] += zigzag(
            yq[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize])
        yCounter += 1
yZigzag = yZigzag.astype(np.int16)

# either crq or cbq can be used to compute the number of blocks


cCounter = 0
for i in range(vBlocksForC):
    for j in range(hBlocksForC):
        crDct[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize] = cv2.dct(
            crPadded[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize])
        crq[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize] = np.ceil(
            crDct[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize] / QTCb)
        crZigzag[cCounter] += zigzag(
            crq[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize])
        cbDct[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize] = cv2.dct(
            cbPadded[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize])
        cbq[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize] = np.ceil(
            cbDct[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize] / QTCr)
        cbZigzag[cCounter] += zigzag(
            cbq[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize])
        cCounter += 1
crZigzag = crZigzag.astype(np.int16)
cbZigzag = cbZigzag.astype(np.int16)

# find the run length encoding for each channel
# then get the frequency of each component in order to form a Huffman dictionary
yEncoded = run_length_encoding(yZigzag)
yFrequencyTable = get_freq_dict(yEncoded)
yHuffman = find_huffman(yFrequencyTable)

crEncoded = run_length_encoding(crZigzag)
crFrequencyTable = get_freq_dict(crEncoded)
crHuffman = find_huffman(crFrequencyTable)

cbEncoded = run_length_encoding(cbZigzag)
cbFrequencyTable = get_freq_dict(cbEncoded)
cbHuffman = find_huffman(cbFrequencyTable)

# calculate the number of bits to transmit for each channel
# and write them to an output file
file = open("CompressedImage.asfh", "w")
yBitsToTransmit = str()
for value in yEncoded:
    yBitsToTransmit += yHuffman[value]

crBitsToTransmit = str()
for value in crEncoded:
    crBitsToTransmit += crHuffman[value]

cbBitsToTransmit = str()
for value in cbEncoded:
    cbBitsToTransmit += cbHuffman[value]

if file.writable():
    file.write(yBitsToTransmit + "\n" + crBitsToTransmit + "\n" + cbBitsToTransmit)
file.close()

totalNumberOfBitsAfterCompression = len(yBitsToTransmit) + len(crBitsToTransmit) + len(cbBitsToTransmit)
print("CR: taxa de compressão: " + str(
        np.round(totalNumberOfBitsWithoutCompression / totalNumberOfBitsAfterCompression, 1)))
print("RD: redundância relativa dos dados (%) : " + str(
        100*np.round(1 - totalNumberOfBitsAfterCompression/totalNumberOfBitsWithoutCompression, 3)))

print("Tamanho Original (bits): ", totalNumberOfBitsWithoutCompression)
print("Tamanho pós Compressao (bits): ", totalNumberOfBitsAfterCompression)
print("Diferença em bits (Sem Compressao - Pós Compressao): ", totalNumberOfBitsWithoutCompression - totalNumberOfBitsAfterCompression)

name = filename.split('.')
outImgType = '.jpg'
outImg = name[0]+outImgType
Quality = 15

cv2.imwrite(outImg,imgOriginal, [cv2.IMWRITE_JPEG_QUALITY, Quality])
imgjpeg = cv2.imread(outImg, cv2.IMREAD_COLOR)
# convert BGR to YCrCb
img = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2RGB)
width = len(img[0])
height = len(img)
y = np.zeros((height, width), np.float32) + img[:, :, 0]
cr = np.zeros((height, width), np.float32) + img[:, :, 1]
cb = np.zeros((height, width), np.float32) + img[:, :, 2]
# size of the image in bits before compression
totalNumberOfBitsWithoutCompression = len(y) * len(y[0]) * 8 + len(cb) * len(cb[0]) * 8 + len(cr) * len(cr[0]) * 8

     

