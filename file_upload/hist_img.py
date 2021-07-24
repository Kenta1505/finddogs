import cv2
import matplotlib.pyplot as plt #matplotlib.pyplotのインポート

def hist_img(file_1, file_2):
    img_1 = cv2.imread(file_1) #1つ目の画像を呼び出し、オブジェクトimg_1に代入
    img_2=cv2.imread(file_2) #2つ目
    
    hist_g_1 = cv2.calcHist([img_1], [2], None, [256], [0,256]) #img_1のR(赤)のヒストグラムを計算
    plt.plot(hist_g_1, color = "r") #ヒストグラムをプロット
    plt.show()
    
    hist_g_2=cv2.calcHist([img_2], [2], None, [256], [0,256]) #img_2のR(赤)のヒストグラムを計算
    plt.plot(hist_g_2, color='r') #プロット
    plt.show()
    
    comp_hist = cv2.compareHist(hist_g_1, hist_g_2, cv2.HISTCMP_CORREL) #ヒストグラムの比較
    return comp_hist

# if __name__ == "__main__":
#     img1="1.2.jpg"
#     img2="25.0.jpg"
#     x = hist_img(img1, img2)
#     print(x + 1)
