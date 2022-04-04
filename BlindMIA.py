# Sobel算子学习
def sobel_test(img_set):
    ret = np.empty(img_set.shape)
    for i, img in enumerate(img_set):
        grad_x = cv.Sobel(np.float32(img), cv.CV_32F, 1, 0) # 水平方向的一阶差分矩阵
        grad_y = cv.Sobel(np.float32(img), cv.CV_32F, 0, 1) # 垂直方向的一阶差分矩阵
        gradx = cv.convertScaleAbs(grad_x) # 参数形如src(输入数组)，alpha(乘数因子)，beta(偏移量)，该方法对src做形如：dst=|alpha*src+beta|的操作
        grady = cv.convertScaleAbs(grad_y)
        gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0) # 对gradx与grady做加权和，权值为自定义
        ret[i, :] = gradxy
    return ret
