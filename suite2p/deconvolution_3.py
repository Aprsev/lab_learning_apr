#import page
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from scipy.ndimage import gaussian_filter
from skimage import restoration

#get_PSF function
def get_psf(px, sigma):
    half_px = px // 2
    x, y = np.mgrid[-half_px:half_px+1:, -half_px:half_px+1:]
    pos = np.dstack((x, y))
    mean = np.array([0, 0])  # 分布の平均
    cov = np.array([[sigma**2, 0], [0, sigma**2]])  # 分布の分散共分散行列
    rv = multivariate_normal(mean, cov)
    psf = rv.pdf(pos)
    return psf


input(0)
choose_mode = int(input("Gray:1 Colored:2 "))
if(choose_mode == 1):
    #gray_image_process
    # 参数设置
    true_img_fname = 'd:\\Desktop\\test\\Lenna_(test_image).png'  # 用于模拟的真实图像
    psf_px = 51  # PSF 高斯滤波器的像素大小（仅限奇数）
    psf_sigma = 7  # PSF 高斯滤波器的 σ
    noise_sigma = 0.5  # 高斯噪声 σ
    iterations = 50  # RL方法的迭代次数

    # 加载真实图像
    true_img = cv2.imread(true_img_fname)
    true_img = cv2.cvtColor(true_img, cv2.COLOR_BGR2GRAY)

    # PSF
    psf = get_psf(psf_px, psf_sigma)

    # 将真实图像与 PSF 进行卷积
    blur_img = cv2.filter2D(true_img, -1, psf)

    # 通过添加高斯噪声创建捕获的图像
    np.random.seed(2022)
    blurNoisy_img = blur_img + np.round(np.random.normal(0, noise_sigma, true_img.shape))
    blurNoisy_img[blurNoisy_img < 0] = 0
    blurNoisy_img[blurNoisy_img > 255] = 255
    blurNoisy_img = blurNoisy_img.astype(np.uint8)

    """deconvolution"""
    balance = np.sum(np.abs(blurNoisy_img-blur_img)) / np.sum(true_img)  # 计算S/N的倒数

    # 通过将Freeier变换后的数组个数全部补为1，实现了不使用reg的维纳反卷积
    no_reg = np.ones((blurNoisy_img.shape[0], blurNoisy_img.shape[1]//2+1), dtype=np.complex64)
    no_reg_wiener_img = restoration.wiener(blurNoisy_img/255., psf, balance, no_reg)
    no_reg_wiener_img *= 1+balance  # 反卷积前保持亮度的一项
    no_reg_wiener_img *= 255.
    no_reg_wiener_img = no_reg_wiener_img.astype(np.uint8)

    # regに4近傍ラプラシアンフィルタを使用(デフォルトのreg=Noneと同一のフィルタ)
    laplacian4_reg = np.array([[0, 1, 0], [1, -4, 1],  [0, 1, 0]], dtype=np.float64)
    laplacian4_reg_wiener_img = restoration.wiener(blurNoisy_img/255., psf, balance, laplacian4_reg)
    laplacian4_reg_wiener_img *= 255.
    laplacian4_reg_wiener_img = laplacian4_reg_wiener_img.astype(np.uint8)

    # regに8近傍ラプラシアンフィルタを使用
    laplacian8_reg = np.array([[1, 1, 1], [1, -8, 1],  [1, 1, 1]], dtype=np.float64)
    laplacian8_reg_wiener_img = restoration.wiener(blurNoisy_img/255., psf, balance, laplacian8_reg)
    laplacian8_reg_wiener_img *= 255.
    laplacian8_reg_wiener_img = laplacian8_reg_wiener_img.astype(np.uint8)

    # 比較用にRichardson-Lucy deconvolution
    rl_img = restoration.richardson_lucy(blurNoisy_img/255., psf, iterations)
    rl_img *= 255.
    rl_img = rl_img.astype(np.uint8)

    """表示"""
    fig, axs = plt.subplots(2, 4, figsize=(28, 14))
    axs[0, 0].set_title('The true image')
    axs[0, 0].imshow(true_img, cmap='gray')
    axs[0, 1].set_title(f'PSF psf_sigma={psf_sigma}')
    axs[0, 1].imshow(psf, cmap='gray')
    axs[0, 2].set_title(f'PSF convolved image')
    axs[0, 2].imshow(blur_img, cmap='gray')
    axs[0, 3].set_title(f'The obtained image noise_sigma={noise_sigma}(PSF convolved + noise)')
    axs[0, 3].imshow(blurNoisy_img, cmap='gray')
    axs[1, 0].set_title(f'Wiener deconvolution\ninverse of S/N {balance:.5f} no reg')
    axs[1, 0].imshow(no_reg_wiener_img, cmap='gray')
    axs[1, 1].set_title(f'Wiener deconvolution\ninverse of S/N {balance:.5f} 4-Laplacian reg')
    axs[1, 1].imshow(laplacian4_reg_wiener_img, cmap='gray')
    axs[1, 2].set_title(f'Wiener deconvolution\ninverse of S/N {balance:.5f} 8-Laplacian reg')
    axs[1, 2].imshow(laplacian8_reg_wiener_img, cmap='gray')
    axs[1, 3].set_title(f'Richardson-Lucy deconvolution\niterations={iterations}')
    axs[1, 3].imshow(rl_img, cmap='gray')

    plt.show()
elif (choose_mode == 2):
    #colored_image_process

    # 各パラメータ
    true_img_fname = 'd:\\Desktop\\test\\he.webp'  # シミュレーションに使う真の画像
    psf_px = 51  # PSF用のガウシアンフィルタのピクセルサイズ(奇数のみ)
    psf_sigma = 7  # PSF用のガウシアンフィルタのσ
    noise_sigma = 0.5  # ガウシアンノイズのσ
    iterations = 50  # RL法の反復回数

    # 真の画像の読み込み
    true_img = cv2.imread(true_img_fname)
    true_img = cv2.cvtColor(true_img, cv2.COLOR_BGR2RGB)

    # PSF
    psf = get_psf(psf_px, psf_sigma)

    # 真の画像をPSFで畳み込み
    blur_img = cv2.filter2D(true_img, -1, psf)
    # ガウシアンノイズを加えて撮像画像を作成
    np.random.seed(2022)
    blurNoisy_img = blur_img + np.round(np.random.normal(0, noise_sigma, true_img.shape))
    blurNoisy_img[blurNoisy_img < 0] = 0
    blurNoisy_img[blurNoisy_img > 255] = 255
    blurNoisy_img = blurNoisy_img.astype(np.uint8)

    """deconvolution"""
    balance = np.sum(np.abs(blurNoisy_img-blur_img)) / np.sum(true_img)  # S/Nの逆数を計算

    # フリーエ変換後の配列数において全て1埋めすることでregを使わないwiener deconvolutionを実装
    no_reg = np.ones((blurNoisy_img.shape[0], blurNoisy_img.shape[1]//2+1), dtype=np.complex64)
    no_reg_wiener_img = np.zeros_like(blurNoisy_img, dtype=np.float64)
    no_reg_wiener_img[..., 0] = restoration.wiener(blurNoisy_img[..., 0]/255., psf, balance, no_reg)
    no_reg_wiener_img[..., 1] = restoration.wiener(blurNoisy_img[..., 1]/255., psf, balance, no_reg)
    no_reg_wiener_img[..., 2] = restoration.wiener(blurNoisy_img[..., 2]/255., psf, balance, no_reg)
    no_reg_wiener_img *= 1+balance  # deconvolution前の輝度を保つための項
    no_reg_wiener_img *= 255.
    no_reg_wiener_img = no_reg_wiener_img.astype(np.uint8)

    # regに4近傍ラプラシアンフィルタを使用(デフォルトのreg=Noneと同一のフィルタ)
    laplacian4_reg = np.array([[0, 1, 0], [1, -4, 1],  [0, 1, 0]], dtype=np.float64)
    laplacian4_reg_wiener_img = np.zeros_like(blurNoisy_img, dtype=np.float64)
    laplacian4_reg_wiener_img[..., 0] = restoration.wiener(blurNoisy_img[..., 0]/255., psf, balance, laplacian4_reg)
    laplacian4_reg_wiener_img[..., 1] = restoration.wiener(blurNoisy_img[..., 1]/255., psf, balance, laplacian4_reg)
    laplacian4_reg_wiener_img[..., 2] = restoration.wiener(blurNoisy_img[..., 2]/255., psf, balance, laplacian4_reg)
    laplacian4_reg_wiener_img *= 255.
    laplacian4_reg_wiener_img = laplacian4_reg_wiener_img.astype(np.uint8)

    # regに8近傍ラプラシアンフィルタを使用
    laplacian8_reg = np.array([[1, 1, 1], [1, -8, 1],  [1, 1, 1]], dtype=np.float64)
    laplacian8_reg_wiener_img = np.zeros_like(blurNoisy_img, dtype=np.float64)
    laplacian8_reg_wiener_img[..., 0] = restoration.wiener(blurNoisy_img[..., 0]/255., psf, balance, laplacian8_reg)
    laplacian8_reg_wiener_img[..., 1] = restoration.wiener(blurNoisy_img[..., 1]/255., psf, balance, laplacian8_reg)
    laplacian8_reg_wiener_img[..., 2] = restoration.wiener(blurNoisy_img[..., 2]/255., psf, balance, laplacian8_reg)
    laplacian8_reg_wiener_img *= 255.
    laplacian8_reg_wiener_img = laplacian8_reg_wiener_img.astype(np.uint8)

    # 比較用にRichardson-Lucy deconvolution
    rl_img = np.zeros_like(blurNoisy_img, dtype=np.float64)
    rl_img[..., 0] = restoration.richardson_lucy(blurNoisy_img[..., 0]/255., psf, iterations)
    rl_img[..., 1] = restoration.richardson_lucy(blurNoisy_img[..., 1]/255., psf, iterations)
    rl_img[..., 2] = restoration.richardson_lucy(blurNoisy_img[..., 2]/255., psf, iterations)
    rl_img *= 255.
    rl_img = rl_img.astype(np.uint8)

    """表示"""
    fig, axs = plt.subplots(2, 4, figsize=(28, 14))
    axs[0, 0].set_title('The true image')
    axs[0, 0].imshow(true_img)
    axs[0, 1].set_title(f'PSF psf_sigma={psf_sigma}')
    axs[0, 1].imshow(psf, cmap='gray')
    axs[0, 2].set_title(f'PSF convolved image')
    axs[0, 2].imshow(blur_img)
    axs[0, 3].set_title(f'The obtained image noise_sigma={noise_sigma}(PSF convolved + noise)')
    axs[0, 3].imshow(blurNoisy_img)
    axs[1, 0].set_title(f'Wiener deconvolution\ninverse of S/N {balance:.5f} no reg')
    axs[1, 0].imshow(no_reg_wiener_img)
    axs[1, 1].set_title(f'Wiener deconvolution\ninverse of S/N {balance:.5f} 4-Laplacian reg')
    axs[1, 1].imshow(laplacian4_reg_wiener_img)
    axs[1, 2].set_title(f'Wiener deconvolution\ninverse of S/N {balance:.5f} 8-Laplacian reg')
    axs[1, 2].imshow(laplacian8_reg_wiener_img)
    axs[1, 3].set_title(f'Richardson-Lucy deconvolution\niterations={iterations}')
    axs[1, 3].imshow(rl_img)

    plt.show()
else:
    print("Wrong input!")
    exit()