# ComputerVision 學習日誌
## W2 電腦視覺入門
1. 使用 Keras建立全連接神經網路 (MLP) 對 CIFAR-10彩色圖片資料集進行分類
## W4 卷積神經網路 (CNN)
1. 分別用 Keras 和 PyTorch 從頭搭建基礎的 CNN 模型，使用 MNIST 手寫數字資料集
## W5 模型性能分析與正規化
1. 資料增強 (Data Augmentation)、L2 正規化、批次正規化 (Batch Normalization) 和 Dropout
1. 混淆矩陣 (Confusion Matrix) 和分類報告 (Classification Report) 
## W7 CNN架構與預訓練模型
1. 從零開始實作了 LeNet 和 AlexNet，CNN 從淺層到深層、從 tanh 到 ReLU、從「平均池化」到「最大池化」
1.  ImageNet 大型資料集上訓練好的模型（VGG16, ResNet）進行推論 (Inference)
## W8 進階模型評估
1. Rank-N (或 Top-N) 準確率
1. Keras/NumPy (argsort) 和 PyTorch (topk) 中計算指標技巧
## W10 遷移學習
1. 特徵提取 (Feature Extraction)：凍結預訓練模型的卷積層，只訓練自己添加的分類層，適用於自有資料集較小的場景
1. 微調 (Fine-Tuning)：在特徵提取的基礎上，解凍部分預訓練模型的頂層網路，並用極低的學習率進行共同訓練
1. Keras Callbacks：使用 ModelCheckpoint 和 EarlyStopping
## W11&12 物件偵測與視覺 Transformer
1. 經典的單階偵測器 YOLO 和 SSD基本流程，包括邊界框 (Bounding Box) 預測和非極大值抑制 (NMS)
1. ViT 如何將 Transformer 架構應用於影像分類， YOLO5 和 DETR 模型如何將其用於物件偵測
## W14 語意分割
1. 語意分割，即對影像中的每一個像素進行分類
1. 從零開始搭建 U-Net 模型
1. 使用 Hugging Face transformers 函式庫調用預訓練的 SegFormer 模型
## W15  生成式與非監督式模型
1. 自編碼器 (Autoencoder)，透過編碼器-解碼器結構進行資料壓縮與重建，將其應用於圖像去噪
1. 生成對抗網路 (GAN)：對抗式訓練，並實作了一個能夠生成 MNIST 手寫數字的 DCGAN 模型
## W16 人臉辨識
1. 度量學習 (Metric Learning)：學習一個嵌入向量空間 (Embedding Space)，讓同類樣本距離近，異類樣本距離遠
1. 孿生網路 (Siamese Network) 搭配對比損失 (Contrastive Loss)，以及 FaceNet 所使用的三元組損失 (Triplet Loss)
1. 使用 deepface 函式庫

